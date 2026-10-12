#!/usr/bin/env python3
"""
Aggregate bin-level utility/intrinsic metrics and produce plots.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _explode_labels(df: pd.DataFrame, label_col: str = "labels") -> pd.Series:
    labels = (
        df[label_col]
        .astype(str)
        .apply(lambda x: [z.strip() for z in x.split(",") if z.strip()])
        .explode()
        .dropna()
    )
    return labels


def _prob_dist(labels: pd.Series) -> pd.Series:
    vc = labels.value_counts().sort_index()
    p = vc / vc.sum()
    return p


def _js_divergence(p: pd.Series, q: pd.Series) -> float:
    idx = sorted(set(p.index) | set(q.index))
    p = p.reindex(idx, fill_value=0.0).to_numpy(dtype=np.float64)
    q = q.reindex(idx, fill_value=0.0).to_numpy(dtype=np.float64)
    m = 0.5 * (p + q)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = np.clip(m, eps, 1.0)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def main():
    parser = argparse.ArgumentParser(description="Analyze similarity-performance relation by bins.")
    parser.add_argument("--bin_summary", type=str, required=True, help="Path to bin_summary.csv")
    parser.add_argument("--generated_with_bins", type=str, required=True, help="Path to generated_with_bins.csv")
    parser.add_argument("--bin_seed_metrics", type=str, required=True, help="Path to bin_seed_metrics.csv")
    parser.add_argument("--train_file", type=str, required=True, help="Path to train_pure.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir for aggregated results and figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    bin_summary = pd.read_csv(args.bin_summary).sort_values("bin_id")
    gen = pd.read_csv(args.generated_with_bins).fillna("")
    metrics = pd.read_csv(args.bin_seed_metrics)
    train_df = pd.read_csv(args.train_file).fillna("")

    train_row_keys = set(train_df.astype(str).agg("|".join, axis=1).tolist())
    p_train_labels = _prob_dist(_explode_labels(train_df))

    intrinsic_rows = []
    for bin_id, g in gen.groupby("similarity_bin"):
        row_keys = g.drop(columns=["similarity_bin"], errors="ignore").astype(str).agg("|".join, axis=1)
        duplicate_ratio = float(np.mean(row_keys.isin(train_row_keys)))
        nn_distance_mean = float(g["nn_distance"].mean()) if "nn_distance" in g.columns else float("nan")
        nn_distance_median = float(g["nn_distance"].median()) if "nn_distance" in g.columns else float("nan")

        p_bin_labels = _prob_dist(_explode_labels(g))
        js_label_shift = _js_divergence(p_bin_labels, p_train_labels)

        intrinsic_rows.append(
            {
                "bin_id": int(bin_id),
                "duplicate_ratio": duplicate_ratio,
                "nn_distance_mean": nn_distance_mean,
                "nn_distance_median": nn_distance_median,
                "js_label_shift": js_label_shift,
                "similarity_mean": float(g["similarity_max"].mean()),
            }
        )

    intrinsic_df = pd.DataFrame(intrinsic_rows).sort_values("bin_id")

    agg_dict = {
        "f1_macro_mean": ("f1_macro", "mean"),
        "f1_macro_std": ("f1_macro", "std"),
        "auc_macro_mean": ("auc_macro", "mean"),
        "auc_macro_std": ("auc_macro", "std"),
    }
    if "minority_f1_macro" in metrics.columns:
        agg_dict["minority_f1_macro_mean"] = ("minority_f1_macro", "mean")
        agg_dict["minority_f1_macro_std"] = ("minority_f1_macro", "std")

    perf = (
        metrics.groupby("bin_id")
        .agg(**agg_dict)
        .reset_index()
        .sort_values("bin_id")
    )

    merged = perf.merge(intrinsic_df, on="bin_id", how="left").merge(
        bin_summary[["bin_id", "score_mean", "score_min", "score_max", "n_generated"]],
        on="bin_id",
        how="left",
    )

    # Spearman trend (bin order vs f1_macro)
    rho, pval = spearmanr(merged["bin_id"], merged["f1_macro_mean"], nan_policy="omit")
    merged["spearman_rho_bin_vs_f1"] = rho
    merged["spearman_pvalue_bin_vs_f1"] = pval

    merged.to_csv(os.path.join(args.output_dir, "similarity_performance_summary.csv"), index=False)
    intrinsic_df.to_csv(os.path.join(args.output_dir, "intrinsic_by_bin.csv"), index=False)

    # Plot 1: utility vs similarity bin
    plt.figure(figsize=(9, 6))
    plt.errorbar(
        merged["bin_id"],
        merged["f1_macro_mean"],
        yerr=merged["f1_macro_std"].fillna(0.0),
        marker="o",
        linewidth=2,
        capsize=4,
        label="Macro F1",
    )
    plt.plot(merged["bin_id"], merged["auc_macro_mean"], marker="s", linewidth=2, label="Macro AUC")
    if "minority_f1_macro_mean" in merged.columns:
        plt.plot(
            merged["bin_id"],
            merged["minority_f1_macro_mean"],
            marker="^",
            linewidth=2,
            label="Minority Macro F1",
        )
    plt.xlabel("Similarity Bin (low to high)")
    plt.ylabel("Performance")
    plt.title("Performance vs Similarity Bin")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "performance_vs_similarity_bin.png"), dpi=300)
    plt.close()

    # Plot 2: intrinsic metrics vs similarity bin
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(merged["bin_id"], merged["duplicate_ratio"], marker="o", label="Duplicate Ratio")
    ax1.plot(merged["bin_id"], merged["js_label_shift"], marker="s", label="JS Label Shift")
    ax1.set_xlabel("Similarity Bin (low to high)")
    ax1.set_ylabel("Intrinsic Metric Value")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        merged["bin_id"],
        merged["nn_distance_mean"],
        marker="^",
        color="tab:red",
        label="NN Distance Mean",
    )
    ax2.set_ylabel("NN Distance")
    ax2.legend(loc="upper right")
    plt.title("Intrinsic Quality vs Similarity Bin")
    fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "intrinsic_vs_similarity_bin.png"), dpi=300)
    plt.close()

    print(f"Saved analysis outputs to: {args.output_dir}")
    print(f"Spearman rho (bin vs f1_macro): {rho:.4f}, p={pval:.4g}")


if __name__ == "__main__":
    main()
