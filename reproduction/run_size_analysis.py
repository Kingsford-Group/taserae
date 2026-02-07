#!/usr/bin/env python3
"""
Size vs Performance Analysis for TASER-AE
Generates augmented data at different multipliers, trains classifiers, and evaluates performance.
Multipliers: 0.01X, 0.05X, 0.1X, 0.5X, 1X, 2X, 5X, 10X
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
MULTIPLIERS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
SEED = 6
DEVICE = "cuda:0"

# Use script directory as base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")

# File paths
TRAIN_FILE = os.path.join(DATA_DIR, "train_pure.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_split.csv")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")

def run_augmentation(multiplier):
    """Run TASER-AE augmentation with given multiplier"""
    # Strategy must contain 'Target_Multiplier' for augment_utils.get_augment_targets to work
    # Format multiplier to match benchmark (e.g. 1.0 -> 1)
    mult_str = f"{int(multiplier)}" if multiplier.is_integer() else f"{multiplier}"
    strategy = f"Target_Multiplier_{mult_str}X_size_seed{SEED}"
    # Output file pattern: augmented_ehr_data_{method}_{strategy}.csv
    aug_file = os.path.join(SCRIPT_DIR, "augmented_ehr_data", f"augmented_ehr_data_taser_ae_native_{strategy}.csv")
    
    if os.path.exists(aug_file):
        print(f"  Augmented file exists: {aug_file}")
        return aug_file, strategy
    
    print(f"  Running TASER-AE augmentation with multiplier={multiplier}...")
    cmd = [
        "python3", os.path.join(SCRIPT_DIR, "augment_taser_ae_native.py"),
        "--input", TRAIN_FILE,
        "--strategy", strategy,
        "--multiplier", str(multiplier),
        "--seed", str(SEED),
        "--embeddings", EMBEDDINGS_FILE
    ]
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return None, strategy
    
    return aug_file, strategy

def train_and_evaluate(aug_file, strategy, multiplier):
    """Train classifier and evaluate on test set"""
    
    # Special case: use existing benchmark metrics for 1X
    if multiplier == 1.0:
        existing_metrics = os.path.join(SCRIPT_DIR, "log", f"metrics_taser_ae_native_Target_Multiplier_1X_seed{SEED}.json")
        if os.path.exists(existing_metrics):
            print(f"  Using existing benchmark metrics: {existing_metrics}")
            with open(existing_metrics, 'r') as f:
                return json.load(f)
    
    metrics_file = os.path.join(SCRIPT_DIR, "log", "size_performance_analysis", f"metrics_{strategy}.json")
    
    if os.path.exists(metrics_file):
        print(f"  Metrics file exists: {metrics_file}")
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    print(f"  Training classifier with strategy={strategy}...")
    cmd = [
        "python3", os.path.join(SCRIPT_DIR, "train_classifier.py"),
        "--train_file", aug_file,
        "--test_file", TEST_FILE,
        "--save_metrics", metrics_file,
        "--device", DEVICE
    ]
    
    # Set environment to avoid MKL threading issues
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return None
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def plot_results(multipliers, f1_scores):
    """Create size vs performance plot"""
    plt.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 26,
        "axes.titlesize": 26,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
    })
    
    x = np.arange(1, len(multipliers) + 1)
    y = np.array(f1_scores)
    xlabels = [f"{m}×" if m >= 1 else f"{m}×" for m in multipliers]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, marker="o", markersize=10, linewidth=2.5, color='#1f77b4')
    
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Generated Sample Multiplier")
    ax.set_ylabel("Macro F1")
    ax.grid(True, which="both", linestyle="--", linewidth=0.8)
    
    # Add headroom for labels
    yr = y.max() - y.min() if y.max() != y.min() else 0.1
    ax.set_ylim(y.min() - 0.10 * yr, y.max() + 0.15 * yr)
    ax.margins(x=0.04)
    
    # Add value labels
    for xi, yi in zip(x, y):
        ax.annotate(
            f"{yi:.4f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            va="bottom",
            fontsize=16,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.25),
        )
    
    fig.tight_layout(pad=0.2)
    output_file = os.path.join(SCRIPT_DIR, "figs", "size_vs_performance.png")
    fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"\nPlot saved to: {output_file}")
    return output_file

def main():
    os.chdir(SCRIPT_DIR)
    
    print("=" * 60)
    print("Size vs Performance Analysis for TASER-AE")
    print("=" * 60)
    print(f"Multipliers: {MULTIPLIERS}")
    print(f"Seed: {SEED}")
    print(f"Device: {DEVICE}")
    print()
    
    results = []
    
    for i, mult in enumerate(MULTIPLIERS):
        print(f"\n[{i+1}/{len(MULTIPLIERS)}] Processing multiplier={mult}X")
        print("-" * 40)
        
        # Step 1: Augmentation
        aug_file, strategy = run_augmentation(mult)
        if aug_file is None:
            print(f"  Skipping due to augmentation error")
            results.append({"multiplier": mult, "macro_f1": None})
            continue
        
        # Step 2: Train and evaluate
        metrics = train_and_evaluate(aug_file, strategy, mult)
        if metrics is None:
            print(f"  Skipping due to training error")
            results.append({"multiplier": mult, "macro_f1": None})
            continue
        
        macro_f1 = metrics.get("f1_macro", 0)
        print(f"  Macro F1: {macro_f1:.4f}")
        results.append({"multiplier": mult, "macro_f1": macro_f1})
    
    # Save results to JSON
    results_file = os.path.join(SCRIPT_DIR, "log", "size_performance_analysis", "size_vs_performance_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Filter valid results and plot
    valid_results = [r for r in results if r["macro_f1"] is not None]
    if valid_results:
        multipliers = [r["multiplier"] for r in valid_results]
        f1_scores = [r["macro_f1"] for r in valid_results]
        plot_results(multipliers, f1_scores)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Multiplier':<12} {'Macro F1':<10}")
    print("-" * 22)
    for r in results:
        f1 = f"{r['macro_f1']:.4f}" if r['macro_f1'] else "N/A"
        print(f"{r['multiplier']}×{'':<8} {f1}")

if __name__ == "__main__":
    main()
