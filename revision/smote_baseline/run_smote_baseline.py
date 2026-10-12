#!/usr/bin/env python3
"""
SMOTE baseline for TASER-AE revision experiments.
Supports multi-seed runs with --seeds for aggregated results.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

REPRO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reproduction"))
sys.path.append(REPRO_DIR)
import augment_utils  # noqa: E402


def tokenize_sample(sample):
    tokens = []
    for field in sample:
        for token in str(field).split(","):
            token = token.strip()
            if token:
                tokens.extend(token.split())
    return tokens


def is_single_label(label_str: str) -> bool:
    labels = [l.strip() for l in str(label_str).split(",") if l.strip()]
    return len(labels) == 1


def encode_and_pad(train_raw, test_raw):
    tokenized_train = [tokenize_sample(s) for s in train_raw]
    tokenized_test = [tokenize_sample(s) for s in test_raw]
    vocab = sorted(set(tok for row in tokenized_train for tok in row))
    w2i = {w: i for i, w in enumerate(vocab)}

    def enc(samples):
        return [[w2i.get(t, 0) for t in toks] for toks in samples]

    tr_s, te_s = enc(tokenized_train), enc(tokenized_test)
    ml = max(max((len(s) for s in tr_s), default=0), max((len(s) for s in te_s), default=0))
    pad = lambda s: s + [0] * (ml - len(s))
    return np.array([pad(s) for s in tr_s], dtype=np.int64), np.array([pad(s) for s in te_s], dtype=np.int64)


def compute_metrics(y_test, y_pred, y_prob, class_names):
    m = {
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_test, y_pred, average="micro", zero_division=0),
        "prec_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "rec_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
    }
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_test)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    try:
        m["auc_macro"] = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
        m["auc_micro"] = roc_auc_score(y_bin, y_prob, average="micro", multi_class="ovr")
    except Exception:
        m["auc_macro"] = m["auc_micro"] = float("nan")
    for idx, label in enumerate(class_names):
        yt = (y_test == idx).astype(int)
        yp = (y_pred == idx).astype(int)
        m[f"f1_{label}"] = f1_score(yt, yp, zero_division=0)
        m[f"prec_{label}"] = precision_score(yt, yp, zero_division=0)
        m[f"rec_{label}"] = recall_score(yt, yp, zero_division=0)
    minority_names = [n for n in class_names if n != "no"]
    if minority_names:
        m["minority_f1_macro"] = float(np.mean([m[f"f1_{n}"] for n in minority_names]))
        m["minority_prec_macro"] = float(np.mean([m[f"prec_{n}"] for n in minority_names]))
        m["minority_rec_macro"] = float(np.mean([m[f"rec_{n}"] for n in minority_names]))
    return m


def build_model(name, seed, n_est):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=n_est, random_state=seed, n_jobs=4)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=n_est, random_state=seed)
    raise ValueError(f"Unsupported model: {name}")


def run_one_seed(args, seed):
    augment_utils.seed_everything(seed)
    train_df = pd.read_csv(args.train_file).fillna("")
    test_df = pd.read_csv(args.test_file).fillna("")

    y_tr_raw = train_df["labels"].astype(str).tolist()
    y_te_raw = test_df["labels"].astype(str).tolist()
    x_tr_raw = train_df.drop(columns=["labels"]).astype(str).to_numpy()
    x_te_raw = test_df.drop(columns=["labels"]).astype(str).to_numpy()

    tr_mask = [is_single_label(l) for l in y_tr_raw]
    te_mask = [is_single_label(l) for l in y_te_raw]
    x_tr_raw = x_tr_raw[tr_mask]
    x_te_raw = x_te_raw[te_mask]
    y_tr_str = [y_tr_raw[i].strip() for i in range(len(y_tr_raw)) if tr_mask[i]]
    y_te_str = [y_te_raw[i].strip() for i in range(len(y_te_raw)) if te_mask[i]]
    print(f"[seed={seed}] Train: {len(y_tr_str)}, Test: {len(y_te_str)}")

    x_train, x_test = encode_and_pad(x_tr_raw, x_te_raw)
    le = LabelEncoder()
    y_train = le.fit_transform(y_tr_str)
    y_test = le.transform(y_te_str)

    cls_counts = np.bincount(y_train)
    min_c = int(np.min(cls_counts[cls_counts > 0]))
    k = min(args.k_neighbors, max(1, min_c - 1))
    print(f"[seed={seed}] SMOTE k_neighbors={k}")
    smote = SMOTE(random_state=seed, k_neighbors=k)
    x_sm, y_sm = smote.fit_resample(x_train, y_train)

    model = build_model(args.model, seed, args.n_estimators)
    model.fit(x_sm, y_sm)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    metrics = compute_metrics(y_test, y_pred, y_prob, le.classes_)
    metrics["seed"] = seed
    metrics["model"] = args.model
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SMOTE baseline.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42, help="Single-seed mode")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (multi-seed mode)")
    parser.add_argument("--save_metrics", type=str, default=None, help="Output JSON (single-seed)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir (multi-seed)")
    parser.add_argument("--model", type=str, default="random_forest",
                        choices=["random_forest", "decision_tree", "gradient_boosting"])
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--k_neighbors", type=int, default=5)
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(out_dir, exist_ok=True)

        records = []
        for seed in seeds:
            m = run_one_seed(args, seed)
            path = os.path.join(out_dir, f"metrics_smote_{args.model}_seed{seed}.json")
            with open(path, "w") as f:
                json.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else v
                           for k, v in m.items()}, f, indent=4)
            print(f"  Saved: {path}")
            records.append(m)

        raw_df = pd.DataFrame(records).sort_values("seed")
        raw_df.to_csv(os.path.join(out_dir, f"smote_{args.model}_metrics_by_seed.csv"), index=False)

        num_cols = [c for c in raw_df.columns if c not in {"seed", "model"}
                    and raw_df[c].dtype in [np.float64, float]]
        agg = {"model": args.model}
        for c in num_cols:
            agg[f"{c}_mean"] = raw_df[c].mean()
            agg[f"{c}_std"] = raw_df[c].std()
        agg_df = pd.DataFrame([agg])
        agg_csv = os.path.join(out_dir, f"smote_{args.model}_metrics_aggregated.csv")
        agg_df.to_csv(agg_csv, index=False)

        print("-" * 50)
        for _, r in raw_df.iterrows():
            print(f"  seed={int(r['seed'])}: F1={r['f1_macro']:.4f} AUC={r.get('auc_macro', float('nan')):.4f}")
        print(f"Aggregated -> {agg_csv}")
    else:
        m = run_one_seed(args, args.seed)
        path = args.save_metrics or "metrics.json"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else v
                       for k, v in m.items()}, f, indent=4)
        print(f"Macro F1: {m['f1_macro']:.4f} | Saved: {path}")


if __name__ == "__main__":
    main()
