#!/usr/bin/env python3
"""
End-to-end similarity-performance pipeline across multiple seeds.

For each augmentation seed:
  1) compute_similarity_scores.py  -> per-sample scores
  2) build_bin_datasets.py         -> quantile bins
  3) run_bin_benchmark.py          -> train classifier (matching seed)
Then:
  4) Merge all bin_seed_metrics across augmentation seeds
  5) analyze_similarity_performance.py -> aggregated tables + plots

Usage:
    python3 run_full_pipeline.py --seeds 6,42,81 --device cuda:0
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPRO_DIR = os.path.join(REPO_ROOT, "reproduction")
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")


def run_cmd(cmd, cwd=None, env=None):
    print(f"  CMD: {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Failed:\n{result.stderr[-1000:]}")
    for line in result.stdout.strip().split("\n")[-3:]:
        print(f"    {line}")


def main():
    parser = argparse.ArgumentParser(description="Full similarity-performance pipeline (multi-seed).")
    parser.add_argument("--seeds", type=str, default="6,42,81")
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--score_column", type=str, default="similarity_max")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=os.path.join(SCRIPT_DIR, "outputs"))
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    train_file = os.path.join(DATA_DIR, "train_pure.csv")
    test_file = os.path.join(DATA_DIR, "test_split.csv")
    embeddings = os.path.join(DATA_DIR, "embeddings.npy")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"

    all_bin_metrics = []

    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")

        aug_file = os.path.join(
            REPRO_DIR, "augmented_ehr_data",
            f"augmented_ehr_data_taser_ae_Target_Multiplier_1X_seed{seed}.csv",
        )
        if not os.path.exists(aug_file):
            raise FileNotFoundError(
                f"Missing augmented file for seed {seed}: {aug_file}\n"
                "Run the benchmark first to generate it."
            )

        # Step 1: Compute Similarity Scores
        sim_csv = os.path.join(out_dir, f"generated_similarity_scores_seed{seed}.csv")
        if not os.path.exists(sim_csv):
            print(f"\n[Step 1] Computing similarity scores (seed={seed})...")
            run_cmd([
                "python3", os.path.join(SCRIPT_DIR, "compute_similarity_scores.py"),
                "--train_file", train_file, "--augmented_file", aug_file,
                "--embeddings", embeddings, "--output_csv", sim_csv, "--seed", str(seed),
            ], cwd=SCRIPT_DIR, env=env)
        else:
            print(f"[Step 1] Exists: {sim_csv}")

        # Step 2: Build Bin Datasets
        bin_dir = os.path.join(out_dir, f"bin_datasets_seed{seed}")
        bin_summary = os.path.join(bin_dir, "bin_summary.csv")
        if not os.path.exists(bin_summary):
            print(f"\n[Step 2] Building bin datasets (seed={seed})...")
            run_cmd([
                "python3", os.path.join(SCRIPT_DIR, "build_bin_datasets.py"),
                "--train_file", train_file, "--similarity_csv", sim_csv,
                "--num_bins", str(args.num_bins), "--score_column", args.score_column,
                "--output_dir", bin_dir,
            ], cwd=SCRIPT_DIR, env=env)
        else:
            print(f"[Step 2] Exists: {bin_dir}")

        # Step 3: Run Bin Benchmark (matching seed)
        bench_dir = os.path.join(out_dir, f"bin_benchmark_seed{seed}")
        bench_csv = os.path.join(bench_dir, "bin_seed_metrics.csv")
        if not os.path.exists(bench_csv):
            print(f"\n[Step 3] Running bin benchmark (seed={seed})...")
            run_cmd([
                "python3", os.path.join(SCRIPT_DIR, "run_bin_benchmark.py"),
                "--bin_summary", bin_summary, "--test_file", test_file,
                "--reproduction_dir", REPRO_DIR, "--output_dir", bench_dir,
                "--seeds", str(seed), "--device", args.device,
                "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
                "--patience", str(args.patience),
            ], cwd=SCRIPT_DIR, env=env)
        else:
            print(f"[Step 3] Exists: {bench_csv}")

        seed_metrics = pd.read_csv(bench_csv)
        seed_metrics["aug_seed"] = seed
        all_bin_metrics.append(seed_metrics)

    # Step 4: Merge all seeds
    print(f"\n{'='*60}\nMerging across seeds\n{'='*60}")
    merged = pd.concat(all_bin_metrics, ignore_index=True)
    merged_csv = os.path.join(out_dir, "bin_seed_metrics_all_seeds.csv")
    merged.to_csv(merged_csv, index=False)
    print(f"Merged metrics: {merged_csv}")

    # Step 5: Analyze
    first_bin_dir = os.path.join(out_dir, f"bin_datasets_seed{seeds[0]}")
    analysis_dir = os.path.join(out_dir, "analysis")
    print(f"\n[Step 4] Running analysis...")
    run_cmd([
        "python3", os.path.join(SCRIPT_DIR, "analyze_similarity_performance.py"),
        "--bin_summary", os.path.join(first_bin_dir, "bin_summary.csv"),
        "--generated_with_bins", os.path.join(first_bin_dir, "generated_with_bins.csv"),
        "--bin_seed_metrics", merged_csv,
        "--train_file", train_file, "--output_dir", analysis_dir,
    ], cwd=SCRIPT_DIR, env=env)

    print(f"\nDone. All outputs in: {out_dir}")
    print(f"Analysis: {analysis_dir}")


if __name__ == "__main__":
    main()
