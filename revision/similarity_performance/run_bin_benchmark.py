#!/usr/bin/env python3
"""
Train/evaluate classifier per similarity bin using existing reproduction pipeline.
"""

import argparse
import json
import os
import subprocess

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Run bin-level benchmark with train_classifier.py")
    parser.add_argument("--bin_summary", type=str, required=True, help="Path to bin_summary.csv")
    parser.add_argument("--test_file", type=str, required=True, help="Path to held-out test_split.csv")
    parser.add_argument("--reproduction_dir", type=str, required=True, help="Path to reproduction directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write metrics and summary JSON")
    parser.add_argument("--seeds", type=str, default="6,42,81", help="Comma-separated seeds")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for classifier training")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    bins = pd.read_csv(args.bin_summary)
    records = []
    total_jobs = len(bins) * len(seeds)
    pbar = tqdm(total=total_jobs, desc="Bin benchmark jobs")

    for _, row in bins.sort_values("bin_id").iterrows():
        bin_id = int(row["bin_id"])
        train_file = row["train_file"]
        for seed in seeds:
            metrics_file = os.path.join(args.output_dir, f"metrics_similarity_bin_{bin_id}_seed{seed}.json")
            if not os.path.exists(metrics_file):
                cmd = [
                    "python3",
                    os.path.join(args.reproduction_dir, "train_classifier.py"),
                    "--train_file",
                    train_file,
                    "--test_file",
                    args.test_file,
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
                env = os.environ.copy()
                env["MKL_THREADING_LAYER"] = "GNU"
                print(f"Running bin={bin_id}, seed={seed}")
                subprocess.run(cmd, check=True, env=env)
            else:
                print(f"Skipping existing metrics: {metrics_file}")

            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            record = {"bin_id": bin_id, "seed": seed, "metrics_file": metrics_file}
            record.update(metrics)
            records.append(record)
            pbar.update(1)

    pbar.close()

    result_df = pd.DataFrame(records).sort_values(["bin_id", "seed"])
    result_csv = os.path.join(args.output_dir, "bin_seed_metrics.csv")
    result_df.to_csv(result_csv, index=False)
    print(f"Wrote benchmark summary: {result_csv}")


if __name__ == "__main__":
    main()
