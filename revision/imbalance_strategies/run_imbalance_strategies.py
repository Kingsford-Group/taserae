#!/usr/bin/env python3
"""
Train/evaluate transformer on non-augmented data with three imbalance strategies:
1) undersampling to minority count
2) oversampling to majority count
3) class-weighted BCE loss
Reports per-class precision, recall, F1 for all labels + macro and minority averages.
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
    return [[w2i[token] for token in tokens if token in w2i] for tokens in tokenized_samples]


def pad_sequences(seqs: List[List[int]], max_len: int) -> np.ndarray:
    return np.array([s + [0] * (max_len - len(s)) for s in seqs], dtype=np.int64)


def compute_metrics(all_labels: np.ndarray, all_preds: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    binary_preds = (all_preds >= 0.5).astype(int)
    metrics = {}

    # Aggregate metrics
    for avg in ["micro", "macro"]:
        metrics[f"f1_{avg}"] = f1_score(all_labels, binary_preds, average=avg, zero_division=0)
        metrics[f"prec_{avg}"] = precision_score(all_labels, binary_preds, average=avg, zero_division=0)
        metrics[f"rec_{avg}"] = recall_score(all_labels, binary_preds, average=avg, zero_division=0)
        try:
            metrics[f"auc_{avg}"] = roc_auc_score(all_labels, all_preds, average=avg)
        except Exception:
            metrics[f"auc_{avg}"] = float("nan")

    # Minority average
    minority_indices = [i for i, label in enumerate(class_names) if label != "no"]
    if minority_indices:
        metrics["minority_f1_macro"] = f1_score(
            all_labels[:, minority_indices],
            binary_preds[:, minority_indices],
            average="macro", zero_division=0,
        )
        metrics["minority_prec_macro"] = precision_score(
            all_labels[:, minority_indices],
            binary_preds[:, minority_indices],
            average="macro", zero_division=0,
        )
        metrics["minority_rec_macro"] = recall_score(
            all_labels[:, minority_indices],
            binary_preds[:, minority_indices],
            average="macro", zero_division=0,
        )
    else:
        metrics["minority_f1_macro"] = float("nan")
        metrics["minority_prec_macro"] = float("nan")
        metrics["minority_rec_macro"] = float("nan")

    # Per-class metrics
    per_class_f1 = f1_score(all_labels, binary_preds, average=None, zero_division=0)
    per_class_prec = precision_score(all_labels, binary_preds, average=None, zero_division=0)
    per_class_rec = recall_score(all_labels, binary_preds, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        metrics[f"f1_{name}"] = per_class_f1[i]
        metrics[f"prec_{name}"] = per_class_prec[i]
        metrics[f"rec_{name}"] = per_class_rec[i]

    return metrics


def rebalance_df(df: pd.DataFrame, strategy: str, seed: int) -> pd.DataFrame:
    counts = df["labels"].astype(str).value_counts()
    grouped = df.groupby(df["labels"].astype(str), group_keys=False)
    if strategy == "undersample":
        target = int(counts.min())
        out = grouped.apply(lambda g: g.sample(n=target, random_state=seed, replace=False))
    elif strategy == "oversample":
        target = int(counts.max())
        out = grouped.apply(lambda g: g.sample(n=target, random_state=seed, replace=(len(g) < target)))
    else:
        raise ValueError(f"Unknown rebalance strategy: {strategy}")
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def run_one_seed(args, seed: int, strategy: str) -> Dict[str, float]:
    augment_utils.seed_everything(seed)
    global device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[{strategy}] seed={seed}, device={device}")

    train_df = pd.read_csv(args.train_file).fillna("")
    test_df = pd.read_csv(args.test_file).fillna("")
    if strategy in {"undersample", "oversample"}:
        train_df = rebalance_df(train_df, strategy=strategy, seed=seed)

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
    print(f"  Class names: {class_names}")

    tokenized_train = [tokenize_sample(s) for s in x_train_raw]
    tokenized_test = [tokenize_sample(s) for s in x_test_raw]
    vocab = sorted(set(tok for row in tokenized_train for tok in row))
    w2i = {w: i for i, w in enumerate(vocab)}

    train_seqs = encode_sequences(tokenized_train, w2i)
    test_seqs = encode_sequences(tokenized_test, w2i)
    max_len = max(max((len(s) for s in train_seqs), default=0), max((len(s) for s in test_seqs), default=0))
    x_train = pad_sequences(train_seqs, max_len)
    x_test = pad_sequences(test_seqs, max_len)

    # Stratifying on raw multi-label strings needs enough val rows for each unique string.
    # After undersampling, 10% val can be < n_unique_labels → sklearn raises
    # "test size = k should be greater or equal to the number of classes".
    n_train = len(x_train)
    n_unique_label_strings = len(set(y_train_raw))
    val_count = int(np.ceil(n_train * 0.1)) if n_train else 0
    stratify_arg = (
        y_train_raw
        if n_unique_label_strings > 1 and val_count >= n_unique_label_strings
        else None
    )
    if stratify_arg is None and n_unique_label_strings > 1:
        print(
            f"  [train/val split] stratify=off (n_train={n_train}, val≈{val_count}, "
            f"unique label strings={n_unique_label_strings})"
        )

    x_train_fit, x_val, y_train_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=seed, stratify=stratify_arg,
    )

    train_loader = DataLoader(
        EHRDataset(x_train_fit, y_train_fit),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        EHRDataset(x_val, y_val),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        EHRDataset(x_test, y_test),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = TransformerModel(len(vocab), 64, y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if strategy == "class_weighted":
        pos_counts = np.sum(y_train_fit, axis=0)
        neg_counts = y_train_fit.shape[0] - pos_counts
        pos_weight = (neg_counts + 1e-6) / (pos_counts + 1e-6)
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    minority_indices = [i for i, label in enumerate(class_names) if label != "no"]
    best_minority_f1 = -1.0
    patience_counter = 0
    best_model_state = None

    for _ in range(args.epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(by.cpu().numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_binary = (val_preds >= 0.5).astype(int)
        val_minority_f1 = f1_score(
            val_targets[:, minority_indices],
            val_binary[:, minority_indices],
            average="macro", zero_division=0,
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
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(by.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(all_labels, all_preds, class_names)
    metrics["strategy"] = strategy
    metrics["seed"] = seed
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Imbalance strategy experiment.")
    parser.add_argument("--train_file", type=str, default="data/processed/train_pure.csv")
    parser.add_argument("--test_file", type=str, default="data/processed/test_split.csv")
    parser.add_argument("--seeds", type=str, default="6,42,81")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="revision/imbalance_strategies/outputs")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reproduction"))
    # Resolve paths relative to repo root's parent (project root)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    args.train_file = os.path.join(project_root, args.train_file)
    args.test_file = os.path.join(project_root, args.test_file)
    out_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    strategies = ["undersample", "oversample", "class_weighted"]

    records = []
    for strategy in strategies:
        for seed in seeds:
            m = run_one_seed(args, seed=seed, strategy=strategy)
            records.append(m)

    raw_df = pd.DataFrame(records).sort_values(["strategy", "seed"])
    raw_path = os.path.join(out_dir, "imbalance_strategy_metrics_by_seed.csv")
    raw_df.to_csv(raw_path, index=False)

    # Build aggregation dynamically over all numeric columns
    non_metric_cols = {"strategy", "seed"}
    metric_cols = [c for c in raw_df.columns if c not in non_metric_cols and raw_df[c].dtype in [np.float64, np.int64, float]]

    agg_dict = {}
    for col in metric_cols:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"] = (col, "std")

    agg_df = raw_df.groupby("strategy").agg(**agg_dict).reset_index()
    agg_path = os.path.join(out_dir, "imbalance_strategy_metrics_aggregated.csv")
    agg_df.to_csv(agg_path, index=False)

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(
            {"train_file": args.train_file, "test_file": args.test_file,
             "seeds": seeds, "strategies": strategies},
            f, indent=2,
        )

    print(f"\nSaved: {raw_path}")
    print(f"Saved: {agg_path}")
    print(f"Columns in raw output: {sorted(raw_df.columns.tolist())}")


if __name__ == "__main__":
    main()