#!/usr/bin/env python3
"""
Train/evaluate one or two model conditions and report stratified test performance by age and sex.
Includes per-class F1 for each adverse event.
"""

import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset

REPRO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reproduction"))
sys.path.append(REPRO_DIR)
import augment_utils  # noqa: E402


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=1, batch_first=True, dropout=0.1
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding_layer(x)
        attn_output, _ = self.self_attention(embedded, embedded, embedded)
        pooled = attn_output.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


class EHRDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def tokenize_sample(sample):
    tokens = []
    for field in sample:
        for token in str(field).split(","):
            token = token.strip()
            if token:
                tokens.extend(token.split())
    return tokens


def process_labels(label_str: str) -> List[str]:
    return [l.strip() for l in str(label_str).split(",") if l.strip()]


def encode_sequences(tokenized_samples, w2i):
    sequences = []
    for tokens in tokenized_samples:
        seq = [w2i[token] for token in tokens if token in w2i]
        sequences.append(seq)
    return sequences


def pad_sequences(sequences: List[List[int]], max_len: int) -> np.ndarray:
    return np.array([seq + [0] * (max_len - len(seq)) for seq in sequences], dtype=np.int64)


def compute_metrics(all_labels: np.ndarray, all_preds: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    binary_preds = (all_preds >= 0.5).astype(int)
    metrics = {}

    # --- Aggregate metrics ---
    for avg in ["micro", "macro"]:
        metrics[f"f1_{avg}"] = f1_score(all_labels, binary_preds, average=avg, zero_division=0)
        metrics[f"prec_{avg}"] = precision_score(all_labels, binary_preds, average=avg, zero_division=0)
        metrics[f"rec_{avg}"] = recall_score(all_labels, binary_preds, average=avg, zero_division=0)
        try:
            metrics[f"auc_{avg}"] = roc_auc_score(all_labels, all_preds, average=avg)
        except Exception:
            metrics[f"auc_{avg}"] = float("nan")

    minority_indices = [i for i, label in enumerate(class_names) if label != "no"]
    if minority_indices:
        metrics["minority_f1_macro"] = f1_score(
            all_labels[:, minority_indices],
            binary_preds[:, minority_indices],
            average="macro",
            zero_division=0,
        )
        metrics["minority_prec_macro"] = precision_score(
            all_labels[:, minority_indices],
            binary_preds[:, minority_indices],
            average="macro",
            zero_division=0,
        )
        metrics["minority_rec_macro"] = recall_score(
            all_labels[:, minority_indices],
            binary_preds[:, minority_indices],
            average="macro",
            zero_division=0,
        )
    else:
        metrics["minority_f1_macro"] = float("nan")
        metrics["minority_prec_macro"] = float("nan")
        metrics["minority_rec_macro"] = float("nan")

    # --- Per-class F1, Precision, Recall, and positive count for every class ---
    per_class_f1 = f1_score(all_labels, binary_preds, average=None, zero_division=0)
    per_class_prec = precision_score(all_labels, binary_preds, average=None, zero_division=0)
    per_class_rec = recall_score(all_labels, binary_preds, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        metrics[f"f1_{name}"] = per_class_f1[i]
        metrics[f"prec_{name}"] = per_class_prec[i]
        metrics[f"rec_{name}"] = per_class_rec[i]
        # Number of positive (ground-truth) samples for this class in this subgroup
        metrics[f"n_pos_{name}"] = int(all_labels[:, i].sum())
        # Per-class AUC (may fail if only one class present in subgroup)
        try:
            metrics[f"auc_{name}"] = roc_auc_score(all_labels[:, i], all_preds[:, i])
        except Exception:
            metrics[f"auc_{name}"] = float("nan")

    return metrics


def parse_age_bins(spec: str) -> List[Tuple[int, int]]:
    bins = []
    for part in spec.split(","):
        part = part.strip()
        lo, hi = part.split("-")
        bins.append((int(lo), int(hi)))
    return bins


def resolve_seeded_path(path_template: str, seed: int) -> str:
    if "{seed}" in path_template:
        return path_template.format(seed=seed)
    return path_template.replace("seed6.csv", f"seed{seed}.csv")


def normalize_sex(raw: str) -> str:
    """Normalize gender/sex values from various EHR coding conventions."""
    v = str(raw).strip().upper()
    if v in {"M", "MALE", "1", "1.0"}:
        return "Male"
    if v in {"F", "FEMALE", "2", "2.0"}:
        return "Female"
    return "Other/Unknown"


def run_one_seed(args, seed: int, train_file: str) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    augment_utils.seed_everything(seed)
    global device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[seed={seed}] device={device}")

    train_df = pd.read_csv(train_file).fillna("")
    test_df = pd.read_csv(args.test_file).fillna("")

    y_train_raw = train_df["labels"].astype(str).tolist()
    y_test_raw = test_df["labels"].astype(str).tolist()

    x_train_raw = train_df.drop(columns=["labels"]).astype(str).to_numpy()
    x_test_raw = test_df.drop(columns=["labels"]).astype(str).to_numpy()

    y_train_processed = [process_labels(x) for x in y_train_raw]
    y_test_processed = [process_labels(x) for x in y_test_raw]

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train_processed)
    y_test = mlb.transform(y_test_processed)
    class_names = mlb.classes_.tolist()
    print(f"[seed={seed}] Class names: {class_names}")

    tokenized_train = [tokenize_sample(sample) for sample in x_train_raw]
    tokenized_test = [tokenize_sample(sample) for sample in x_test_raw]

    vocab = sorted(set(tok for row in tokenized_train for tok in row))
    w2i = {word: idx for idx, word in enumerate(vocab)}
    train_seqs = encode_sequences(tokenized_train, w2i)
    test_seqs = encode_sequences(tokenized_test, w2i)

    max_train = max((len(s) for s in train_seqs), default=0)
    max_test = max((len(s) for s in test_seqs), default=0)
    max_seq_length = max(max_train, max_test)
    x_train = pad_sequences(train_seqs, max_seq_length)
    x_test = pad_sequences(test_seqs, max_seq_length)

    if args.val_file and os.path.exists(args.val_file):
        val_df = pd.read_csv(args.val_file).fillna("")
        y_val_raw = val_df["labels"].astype(str).tolist()
        y_val = mlb.transform([process_labels(x) for x in y_val_raw])
        x_val_raw = val_df.drop(columns=["labels"]).astype(str).to_numpy()
        tokenized_val = [tokenize_sample(sample) for sample in x_val_raw]
        val_seqs = encode_sequences(tokenized_val, w2i)
        x_val = pad_sequences(val_seqs, max_seq_length)
        x_train_fit, y_train_fit = x_train, y_train
    else:
        x_train_fit, x_val, y_train_fit, y_val = train_test_split(
            x_train,
            y_train,
            test_size=0.1,
            random_state=seed,
            stratify=y_train_raw,
        )

    train_loader = DataLoader(
        EHRDataset(x_train_fit, y_train_fit),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        EHRDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        EHRDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = TransformerModel(len(vocab), 64, y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    minority_indices = [i for i, label in enumerate(class_names) if label != "no"]
    best_minority_f1 = -1.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_binary = (val_preds >= 0.5).astype(int)
        val_minority_f1 = f1_score(
            val_targets[:, minority_indices],
            val_binary[:, minority_indices],
            average="macro",
            zero_division=0,
        )

        if val_minority_f1 > best_minority_f1:
            best_minority_f1 = val_minority_f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(batch_y.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    global_metrics = compute_metrics(all_labels, all_preds, class_names)

    # ---- Subgroup stratification ----
    subgroup_rows = []
    age_series = pd.to_numeric(test_df["ageYear"], errors="coerce")
    sex_norm = test_df["gender"].astype(str).apply(normalize_sex)

    age_bins = parse_age_bins(args.age_bins)

    # 1. Age-only stratification
    for lo, hi in age_bins:
        mask = (age_series >= lo) & (age_series <= hi)
        idx = np.where(mask.to_numpy())[0]
        if len(idx) == 0:
            continue
        m = compute_metrics(all_labels[idx], all_preds[idx], class_names)
        subgroup_rows.append(
            {"seed": seed, "subgroup_type": "age", "subgroup": f"{lo}-{hi}",
             "n_samples": len(idx), **m}
        )

    # 2. Sex-only stratification
    sex_categories = sorted(sex_norm.unique())
    print(f"[seed={seed}] Normalized sex categories found: {sex_categories}")
    for sex_name in sex_categories:
        mask = sex_norm == sex_name
        idx = np.where(mask.to_numpy())[0]
        if len(idx) == 0:
            continue
        m = compute_metrics(all_labels[idx], all_preds[idx], class_names)
        subgroup_rows.append(
            {"seed": seed, "subgroup_type": "sex", "subgroup": sex_name,
             "n_samples": len(idx), **m}
        )

    # 3. Sex x Age combined stratification
    for lo, hi in age_bins:
        age_mask = (age_series >= lo) & (age_series <= hi)
        for sex_name in sex_categories:
            sex_mask = sex_norm == sex_name
            combo_mask = age_mask & sex_mask
            idx = np.where(combo_mask.to_numpy())[0]
            if len(idx) == 0:
                continue
            m = compute_metrics(all_labels[idx], all_preds[idx], class_names)
            subgroup_rows.append(
                {"seed": seed, "subgroup_type": "sex_age", "subgroup": f"{sex_name}|{lo}-{hi}",
                 "n_samples": len(idx), **m}
            )

    return global_metrics, subgroup_rows


def main():
    parser = argparse.ArgumentParser(description="Run stratified performance analysis by age and sex.")
    parser.add_argument("--train_file", type=str, default=None,
                        help="Single condition mode: one train CSV.")
    parser.add_argument("--train_file_no_aug", type=str, default=None,
                        help="Two-condition mode: train CSV for non-augmented transformer.")
    parser.add_argument("--train_file_taser", type=str, default=None,
                        help="Two-condition mode: train CSV for TASER-AE condition.")
    parser.add_argument("--test_file", type=str, required=True, help="Held-out test CSV.")
    parser.add_argument("--val_file", type=str, default=None, help="Optional validation CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for outputs.")
    parser.add_argument("--seeds", type=str, default="6,42,81", help="Comma-separated seeds.")
    parser.add_argument("--age_bins", type=str, default="0-40,41-50,51-60,61-70,71-80,81-120",
                        help="Age bins as lo-hi list.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    conditions = []
    if args.train_file_no_aug and args.train_file_taser:
        conditions = [
            ("transformer_no_aug", args.train_file_no_aug),
            ("taser_ae", args.train_file_taser),
        ]
    elif args.train_file:
        conditions = [("single_condition", args.train_file)]
    else:
        raise ValueError(
            "Provide either --train_file for single-condition mode, "
            "or both --train_file_no_aug and --train_file_taser for two-condition mode."
        )

    all_global = []
    all_subgroups = []
    for condition_name, train_file_template in conditions:
        for seed in seeds:
            train_file = train_file_template
            if condition_name == "taser_ae":
                train_file = resolve_seeded_path(train_file_template, seed)
            if not os.path.exists(train_file):
                raise FileNotFoundError(
                    f"Missing train file for condition={condition_name}, seed={seed}: {train_file}"
                )
            global_metrics, subgroup_rows = run_one_seed(args, seed, train_file)
            all_global.append({"condition": condition_name, "seed": seed, **global_metrics})
            all_subgroups.extend([{**r, "condition": condition_name} for r in subgroup_rows])

    global_df = pd.DataFrame(all_global).sort_values(["condition", "seed"])
    subgroup_df = pd.DataFrame(all_subgroups).sort_values(
        ["condition", "subgroup_type", "subgroup", "seed"]
    )

    global_df.to_csv(os.path.join(args.output_dir, "global_metrics_by_seed.csv"), index=False)
    subgroup_df.to_csv(os.path.join(args.output_dir, "stratified_metrics_by_seed.csv"), index=False)

    # Build aggregation dict dynamically based on available columns
    # Always aggregate these core columns
    agg_dict = {
        "n_samples": [("n_samples_mean", "mean")],
        "f1_macro": [("f1_macro_mean", "mean"), ("f1_macro_std", "std")],
        "auc_macro": [("auc_macro_mean", "mean"), ("auc_macro_std", "std")],
        "minority_f1_macro": [("minority_f1_macro_mean", "mean"), ("minority_f1_macro_std", "std")],
    }
    # Add per-class columns if they exist
    for col in subgroup_df.columns:
        if col.startswith("f1_") and col not in ["f1_macro", "f1_micro"]:
            class_name = col  # e.g. "f1_falls"
            agg_dict[col] = [(f"{col}_mean", "mean"), (f"{col}_std", "std")]
        if col.startswith("prec_") and col not in ["prec_macro", "prec_micro"]:
            agg_dict[col] = [(f"{col}_mean", "mean"), (f"{col}_std", "std")]
        if col.startswith("rec_") and col not in ["rec_macro", "rec_micro"]:
            agg_dict[col] = [(f"{col}_mean", "mean"), (f"{col}_std", "std")]
        if col.startswith("auc_") and col not in ["auc_macro", "auc_micro"]:
            agg_dict[col] = [(f"{col}_mean", "mean"), (f"{col}_std", "std")]

    # Flatten for pandas named agg
    flat_agg = {}
    for src_col, agg_list in agg_dict.items():
        if src_col in subgroup_df.columns:
            for out_name, func in agg_list:
                flat_agg[out_name] = (src_col, func)

    agg = (
        subgroup_df.groupby(["condition", "subgroup_type", "subgroup"])
        .agg(**flat_agg)
        .reset_index()
    )
    agg.to_csv(os.path.join(args.output_dir, "stratified_metrics_aggregated.csv"), index=False)

    # Global aggregation (same approach)
    global_flat_agg = {}
    for src_col, agg_list in agg_dict.items():
        if src_col in global_df.columns:
            for out_name, func in agg_list:
                global_flat_agg[out_name] = (src_col, func)

    global_agg = (
        global_df.groupby("condition")
        .agg(**global_flat_agg)
        .reset_index()
    )
    global_agg.to_csv(os.path.join(args.output_dir, "global_metrics_aggregated.csv"), index=False)

    if len(conditions) == 2:
        two_row = global_agg[global_agg["condition"].isin(["transformer_no_aug", "taser_ae"])].copy()
        two_row.to_csv(os.path.join(args.output_dir, "two_row_comparison.csv"), index=False)

    print(f"\nSaved outputs in: {args.output_dir}")
    print(f"Subgroup types found: {subgroup_df['subgroup_type'].unique().tolist()}")
    print(f"Subgroups: {subgroup_df['subgroup'].unique().tolist()}")
    print(f"Per-class columns: {[c for c in subgroup_df.columns if c.startswith('f1_') and c not in ['f1_macro','f1_micro']]}")


if __name__ == "__main__":
    main()