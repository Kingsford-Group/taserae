#!/usr/bin/env python3
"""
Size vs per-class F1 for TASER-AE (revision experiment).
Runs with 3 seeds and reports mean ± std across seeds.
"""

import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MULTIPLIERS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
SEEDS = [6, 42, 81]
DEVICE = "cuda:0"
MINORITY_CLASSES = ["Falls", "Fractures", "GI_Bleed", "Stroke"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPRO_DIR = os.path.join(REPO_ROOT, "reproduction")
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")

TRAIN_FILE = os.path.join(DATA_DIR, "train_pure.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_split.csv")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
FIGS_DIR = os.path.join(OUTPUT_DIR, "figs")
RESULTS_JSON = os.path.join(OUTPUT_DIR, "size_vs_performance_per_class_results.json")


def run_augmentation(multiplier, seed):
    mult_str = f"{int(multiplier)}" if float(multiplier).is_integer() else f"{multiplier}"
    strategy = f"Target_Multiplier_{mult_str}X_size_seed{seed}"
    aug_file = os.path.join(
        REPRO_DIR, "augmented_ehr_data",
        f"augmented_ehr_data_taser_ae_{strategy}.csv",
    )
    if os.path.exists(aug_file):
        print(f"  Augmented file exists: {aug_file}")
        return aug_file, strategy

    print(f"  Running TASER-AE augmentation multiplier={multiplier}, seed={seed}...")
    cmd = [
        "python3", os.path.join(REPRO_DIR, "augment_taser_ae.py"),
        "--input", TRAIN_FILE, "--strategy", strategy,
        "--multiplier", str(multiplier), "--seed", str(seed),
        "--embeddings", EMBEDDINGS_FILE,
    ]
    result = subprocess.run(cmd, cwd=REPRO_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return None, strategy
    return aug_file, strategy


def train_and_evaluate(aug_file, strategy, multiplier, seed):
    if multiplier == 1.0:
        existing = os.path.join(
            REPRO_DIR, "log",
            f"metrics_taser_ae_Target_Multiplier_1X_seed{seed}.json",
        )
        if os.path.exists(existing):
            print(f"  Using existing benchmark metrics: {existing}")
            with open(existing, "r") as f:
                return json.load(f)

    metrics_dir = os.path.join(REPRO_DIR, "log", "size_performance_analysis")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_dir, f"metrics_{strategy}.json")

    if os.path.exists(metrics_file):
        print(f"  Metrics file exists: {metrics_file}")
        with open(metrics_file, "r") as f:
            return json.load(f)

    print(f"  Training classifier strategy={strategy}...")
    cmd = [
        "python3", os.path.join(REPRO_DIR, "train_classifier.py"),
        "--train_file", aug_file, "--test_file", TEST_FILE,
        "--save_metrics", metrics_file, "--seed", str(seed), "--device", DEVICE,
    ]
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    result = subprocess.run(cmd, cwd=REPRO_DIR, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return None
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return json.load(f)
    return None


def plot_one_class(multipliers, f1_means, f1_stds, class_name):
    plt.rcParams.update({
        "font.size": 20, "axes.labelsize": 26, "axes.titlesize": 26,
        "xtick.labelsize": 22, "ytick.labelsize": 22,
    })
    x = np.arange(1, len(multipliers) + 1)
    y = np.array(f1_means, dtype=float)
    xlabels = [f"{m}\u00d7" for m in multipliers]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, marker="o", markersize=10, linewidth=2.5, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Generated Sample Multiplier")
    ax.set_ylabel(f"F1 ({class_name})")
    ax.set_title(f"Test F1 vs Augmentation Size \u2014 {class_name}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.8)

    yr = y.max() - y.min() if y.max() != y.min() else 0.1
    ax.set_ylim(y.min() - 0.10 * yr, y.max() + 0.15 * yr)
    ax.margins(x=0.04)

    for xi, yi in zip(x, y):
        if np.isnan(yi):
            continue
        ax.annotate(f"{yi:.3f}", (xi, yi),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", va="bottom", fontsize=13,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.25))

    fig.tight_layout(pad=0.2)
    safe = class_name.replace(" ", "_")
    out_path = os.path.join(FIGS_DIR, f"size_vs_performance_{safe}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    print("=" * 60)
    print("Size vs Per-Class F1 (TASER-AE, multi-seed)")
    print(f"Multipliers: {MULTIPLIERS}")
    print(f"Seeds: {SEEDS}  |  Device: {DEVICE}")
    print("=" * 60)

    records = []
    for i, mult in enumerate(MULTIPLIERS):
        for seed in SEEDS:
            print(f"\n[{i+1}/{len(MULTIPLIERS)}] mult={mult}X, seed={seed}")
            aug_file, strategy = run_augmentation(mult, seed)
            row = {"multiplier": mult, "seed": seed, "strategy": strategy}
            if aug_file is None:
                for c in MINORITY_CLASSES:
                    row[f"f1_{c}"] = None
                records.append(row)
                continue
            metrics = train_and_evaluate(aug_file, strategy, mult, seed)
            if metrics is None:
                for c in MINORITY_CLASSES:
                    row[f"f1_{c}"] = None
                records.append(row)
                continue
            for k, v in metrics.items():
                row[k] = v
            for c in MINORITY_CLASSES:
                print(f"  {c} F1: {row.get(f'f1_{c}')}")
            records.append(row)

    with open(RESULTS_JSON, "w") as f:
        json.dump(records, f, indent=2)

    raw_df = pd.DataFrame(records)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, "size_vs_performance_per_class_by_seed.csv"), index=False)

    # Aggregate mean ± std per multiplier
    # Determine all metric keys (everything except multiplier/seed/strategy)
    skip_keys = {"multiplier", "seed", "strategy"}
    all_metric_keys = sorted(set(k for r in records for k in r if k not in skip_keys and isinstance(r.get(k), (int, float))))

    agg_rows = []
    for mult in MULTIPLIERS:
        sub = [r for r in records if r["multiplier"] == mult]
        agg = {"multiplier": mult}
        for mk in all_metric_keys:
            vals = [r[mk] for r in sub if r.get(mk) is not None]
            agg[f"{mk}_mean"] = float(np.mean(vals)) if vals else None
            agg[f"{mk}_std"] = float(np.std(vals)) if vals else None
        agg_rows.append(agg)

    agg_df = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(OUTPUT_DIR, "size_vs_performance_per_class_aggregated.csv")
    agg_df.to_csv(agg_csv, index=False)

    valid = [a for a in agg_rows if a.get("f1_Falls_mean") is not None]
    if not valid:
        print("No valid results to plot.")
        sys.exit(1)
    mults = [a["multiplier"] for a in valid]
    for c in MINORITY_CLASSES:
        means = [a[f"f1_{c}_mean"] for a in valid]
        stds = [a.get(f"f1_{c}_std", 0) or 0 for a in valid]
        plot_one_class(mults, means, stds, c)

    print("\n" + "=" * 60)
    print("SUMMARY (mean \u00b1 std across seeds)")
    print("=" * 60)
    for a in agg_rows:
        parts = [f"{a['multiplier']}\u00d7"]
        for c in MINORITY_CLASSES:
            m, s = a.get(f"f1_{c}_mean"), a.get(f"f1_{c}_std")
            parts.append(f"{c}={m:.4f}\u00b1{s:.4f}" if m is not None else f"{c}=N/A")
        print("  ".join(parts))


if __name__ == "__main__":
    main()
