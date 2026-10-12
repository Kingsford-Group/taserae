#!/usr/bin/env python3
"""
Train/evaluate Bio_ClinicalBERT for multilabel ADE prediction.
"""

import argparse
import copy
import json
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def process_labels(label_str: str) -> List[str]:
    return [x.strip() for x in str(label_str).split(",") if x.strip()]


def row_to_text(row: pd.Series) -> str:
    drugs = str(row.get("administered_drugs", "")).replace(",", " ; ")
    comps = str(row.get("complications", "")).replace(",", " ; ")
    age = str(row.get("ageYear", ""))
    gender = str(row.get("gender", ""))
    return f"age {age} gender {gender} drugs {drugs} complications {comps}"


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class ClinicalBERTClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.2):
        super().__init__()
        # Prefer safetensors to avoid torch.load restrictions on older torch versions.
        try:
            self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True)
        except Exception as e:
            raise RuntimeError(
                "Failed to load pretrained model with safetensors. "
                "Please use a model repo that provides .safetensors weights "
                f"or upgrade torch to >=2.6. Original error: {e}"
            ) from e
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return self.fc(cls)


def evaluate(model, loader, device, class_names: List[str]) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    binary = (all_preds >= 0.5).astype(int)

    metrics = {}
    for avg in ["micro", "macro"]:
        metrics[f"f1_{avg}"] = f1_score(all_labels, binary, average=avg, zero_division=0)
        metrics[f"prec_{avg}"] = precision_score(all_labels, binary, average=avg, zero_division=0)
        metrics[f"rec_{avg}"] = recall_score(all_labels, binary, average=avg, zero_division=0)
        try:
            metrics[f"auc_{avg}"] = roc_auc_score(all_labels, all_preds, average=avg)
        except Exception:
            metrics[f"auc_{avg}"] = float("nan")

    minority_indices = [i for i, label in enumerate(class_names) if label != "no"]
    if minority_indices:
        metrics["minority_f1_macro"] = f1_score(
            all_labels[:, minority_indices],
            binary[:, minority_indices],
            average="macro",
            zero_division=0,
        )
        metrics["minority_prec_macro"] = precision_score(
            all_labels[:, minority_indices],
            binary[:, minority_indices],
            average="macro",
            zero_division=0,
        )
        metrics["minority_rec_macro"] = recall_score(
            all_labels[:, minority_indices],
            binary[:, minority_indices],
            average="macro",
            zero_division=0,
        )

    for i, label in enumerate(class_names):
        y_true = all_labels[:, i]
        y_pred = binary[:, i]
        y_prob = all_preds[:, i]
        metrics[f"f1_{label}"] = f1_score(y_true, y_pred, zero_division=0)
        metrics[f"prec_{label}"] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f"rec_{label}"] = recall_score(y_true, y_pred, zero_division=0)
        try:
            metrics[f"auc_{label}"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics[f"auc_{label}"] = float("nan")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Bio_ClinicalBERT on ADE labels.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_metrics", type=str, required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(args.train_file).fillna("")
    test_df = pd.read_csv(args.test_file).fillna("")

    y_train_raw = train_df["labels"].astype(str).tolist()
    y_test_raw = test_df["labels"].astype(str).tolist()
    y_train_proc = [process_labels(x) for x in y_train_raw]
    y_test_proc = [process_labels(x) for x in y_test_raw]

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train_proc)
    y_test = mlb.transform(y_test_proc)
    class_names = mlb.classes_.tolist()

    train_texts_all = train_df.apply(row_to_text, axis=1).tolist()
    test_texts = test_df.apply(row_to_text, axis=1).tolist()

    if args.val_file and os.path.exists(args.val_file):
        val_df = pd.read_csv(args.val_file).fillna("")
        val_texts = val_df.apply(row_to_text, axis=1).tolist()
        y_val = mlb.transform([process_labels(x) for x in val_df["labels"].astype(str).tolist()])
        train_texts = train_texts_all
        y_train_fit = y_train
    else:
        train_texts, val_texts, y_train_fit, y_val = train_test_split(
            train_texts_all,
            y_train,
            test_size=0.1,
            random_state=args.seed,
            stratify=y_train_raw,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = TextDataset(train_texts, y_train_fit, tokenizer, args.max_len)
    val_ds = TextDataset(val_texts, y_val, tokenizer, args.max_len)
    test_ds = TextDataset(test_texts, y_test, tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ClinicalBERTClassifier(args.model_name, num_classes=y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_val = -1.0
    wait = 0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()

        val_metrics = evaluate(model, val_loader, device, class_names)
        val_min = val_metrics.get("minority_f1_macro", val_metrics.get("f1_macro", 0.0))
        print(f"Epoch {epoch+1}/{args.epochs} loss={running/len(train_loader):.4f} val_min_f1={val_min:.4f}")
        if val_min > best_val:
            best_val = val_min
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, class_names)
    test_metrics["model_name"] = args.model_name
    test_metrics["seed"] = args.seed

    os.makedirs(os.path.dirname(os.path.abspath(args.save_metrics)), exist_ok=True)
    with open(args.save_metrics, "w") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in test_metrics.items()}, f, indent=2)
    print(f"Saved metrics: {args.save_metrics}")


if __name__ == "__main__":
    main()
