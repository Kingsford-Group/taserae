#!/usr/bin/env python3
"""
Run TASER-AE threshold sensitivity experiment:
- augment with different cosine thresholds
- train classifier on each augmented set
- evaluate on held-out test set
- export per-label precision/recall/F1 tables
- plot F1 vs threshold
"""

import argparse
import json
import os
import subprocess
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fmt_thr(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def _run_cmd(cmd: List[str], cwd: str, env: Dict[str, str]) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{result.stderr}")


def _discover_labels(metrics: Dict[str, float]) -> List[str]:
    labels = []
    for k in metrics.keys():
        if k.startswith("f1_") and k not in {"f1_micro", "f1_macro"}:
            labels.append(k.replace("f1_", "", 1))
    return sorted(labels)


def _plot_f1_by_threshold(df: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(10, 6))
    for label, grp in df.groupby("label"):
        plt.plot(
            grp["threshold"],
            grp["f1_mean"],
            marker="o",
            linewidth=2,
            label=label,
        )
    plt.xlabel("Cosine Similarity Threshold")
    plt.ylabel("F1 Score (mean across seeds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def _plot_minority_f1_by_threshold(df: pd.DataFrame, out_png: str) -> None:
    minority_df = df[df["label"] != "no"].copy()
    if minority_df.empty:
        return
    plt.figure(figsize=(10, 6))
    for label, grp in minority_df.groupby("label"):
        plt.plot(
            grp["threshold"],
            grp["f1_mean"],
            marker="o",
            linewidth=2,
            label=label,
        )
    plt.xlabel("Cosine Similarity Threshold")
    plt.ylabel("F1 Score (mean across seeds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Threshold sensitivity for TASER-AE.")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
        help="Comma-separated cosine thresholds.",
    )
    parser.add_argument("--seeds", type=str, default="6,42,81", help="Comma-separated seeds.")
    parser.add_argument("--multiplier", type=float, default=1.0, help="Target_Multiplier value (1.0 => 1X).")
    parser.add_argument(
        "--strategy_prefix",
        type=str,
        default="Target_Multiplier_1X",
        help="Strategy prefix used in output naming.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/processed/train_pure.csv",
        help="Path to real train split.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/processed/test_split.csv",
        help="Path to held-out test split.",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/processed/embeddings.npy",
        help="Path to embeddings.npy.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for classifier training.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="revision/threshold_sensitivity/outputs",
        help="Output directory for tables/plots.",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    repro_dir = os.path.join(repo_root, "reproduction")
    out_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"

    rows = []
    global_rows = []
    for thr in thresholds:
        thr_tag = _fmt_thr(thr)
        for seed in seeds:
            strategy = f"{args.strategy_prefix}_thr{thr_tag}_seed{seed}"
            aug_file = os.path.join(
                repro_dir,
                "augmented_ehr_data",
                f"augmented_ehr_data_taser_ae_{strategy}.csv",
            )
            metrics_file = os.path.join(out_dir, f"metrics_taser_ae_{strategy}.json")

            if not os.path.exists(aug_file):
                aug_cmd = [
                    "python3",
                    os.path.join(repro_dir, "augment_taser_ae.py"),
                    "--input",
                    os.path.join(repo_root, args.train_file),
                    "--strategy",
                    strategy,
                    "--multiplier",
                    str(args.multiplier),
                    "--seed",
                    str(seed),
                    "--embeddings",
                    os.path.join(repo_root, args.embeddings),
                    "--threshold",
                    str(thr),
                ]
                print(f"Augmenting threshold={thr}, seed={seed}")
                _run_cmd(aug_cmd, cwd=repro_dir, env=env)
            else:
                print(f"Using existing augmentation for threshold={thr}, seed={seed}")

            if not os.path.exists(metrics_file):
                train_cmd = [
                    "python3",
                    os.path.join(repro_dir, "train_classifier.py"),
                    "--train_file",
                    aug_file,
                    "--test_file",
                    os.path.join(repo_root, args.test_file),
                    "--epochs",
                    str(args.epochs),
                    "--batch_size",
                    str(args.batch_size),
                    "--patience",
                    str(args.patience),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--save_metrics",
                    metrics_file,
                ]
                print(f"Training threshold={thr}, seed={seed}")
                _run_cmd(train_cmd, cwd=repro_dir, env=env)
            else:
                print(f"Using existing metrics for threshold={thr}, seed={seed}")

            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            labels = _discover_labels(metrics)
            for label in labels:
                rows.append(
                    {
                        "threshold": thr,
                        "seed": seed,
                        "label": label,
                        "precision": metrics.get(f"prec_{label}", np.nan),
                        "recall": metrics.get(f"rec_{label}", np.nan),
                        "f1": metrics.get(f"f1_{label}", np.nan),
                    }
                )
            global_rows.append({
                "threshold": thr, "seed": seed,
                "f1_macro": metrics.get("f1_macro", np.nan),
                "prec_macro": metrics.get("prec_macro", np.nan),
                "rec_macro": metrics.get("rec_macro", np.nan),
                "minority_f1_macro": metrics.get("minority_f1_macro", np.nan),
                "auc_macro": metrics.get("auc_macro", np.nan),
            })

    raw_df = pd.DataFrame(rows).sort_values(["threshold", "seed", "label"])
    raw_csv = os.path.join(out_dir, "per_label_metrics_by_threshold_seed.csv")
    raw_df.to_csv(raw_csv, index=False)

    agg_df = (
        raw_df.groupby(["threshold", "label"])
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .reset_index()
        .sort_values(["label", "threshold"])
    )
    agg_csv = os.path.join(out_dir, "per_label_metrics_by_threshold_aggregated.csv")
    agg_df.to_csv(agg_csv, index=False)

    # Wide table for manuscript convenience: one row per threshold+label
    wide_df = agg_df[
        ["threshold", "label", "precision_mean", "recall_mean", "f1_mean", "precision_std", "recall_std", "f1_std"]
    ].copy()
    wide_csv = os.path.join(out_dir, "table_precision_recall_f1_by_label_threshold.csv")
    wide_df.to_csv(wide_csv, index=False)

    plot_png = os.path.join(out_dir, "f1_vs_threshold_by_label.png")
    _plot_f1_by_threshold(agg_df, plot_png)
    minority_plot_png = os.path.join(out_dir, "f1_vs_threshold_minority_labels.png")
    _plot_minority_f1_by_threshold(agg_df, minority_plot_png)

    # Optional macro F1 over threshold (from per-label mean)
    macro_df = (
        agg_df.groupby("threshold")
        .agg(f1_macro_from_labels=("f1_mean", "mean"))
        .reset_index()
        .sort_values("threshold")
    )
    macro_csv = os.path.join(out_dir, "f1_macro_over_threshold.csv")
    macro_df.to_csv(macro_csv, index=False)

    # Global macro/minority metrics by threshold
    global_df = pd.DataFrame(global_rows).sort_values(["threshold", "seed"])
    global_csv = os.path.join(out_dir, "global_metrics_by_threshold_seed.csv")
    global_df.to_csv(global_csv, index=False)
    global_agg = (
        global_df.groupby("threshold")
        .agg(**{f"{c}_mean": (c, "mean") for c in ["f1_macro", "prec_macro", "rec_macro", "minority_f1_macro", "auc_macro"]},
             **{f"{c}_std": (c, "std") for c in ["f1_macro", "prec_macro", "rec_macro", "minority_f1_macro", "auc_macro"]})
        .reset_index().sort_values("threshold")
    )
    global_agg_csv = os.path.join(out_dir, "global_metrics_by_threshold_aggregated.csv")
    global_agg.to_csv(global_agg_csv, index=False)

    print("Done.")
    print(f"Raw per-seed table: {raw_csv}")
    print(f"Aggregated per-label table: {agg_csv}")
    print(f"Manuscript table: {wide_csv}")
    print(f"Plot: {plot_png}")
    print(f"Minority plot: {minority_plot_png}")
    print(f"Macro trend: {macro_csv}")


if __name__ == "__main__":
    main()
