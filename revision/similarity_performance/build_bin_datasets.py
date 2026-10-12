#!/usr/bin/env python3
"""
Build quantile-bin datasets from generated sample similarity scores.
"""

import argparse
import os

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Build training datasets per similarity bin.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to original train_pure.csv")
    parser.add_argument("--similarity_csv", type=str, required=True, help="Path to generated similarity scores CSV")
    parser.add_argument("--num_bins", type=int, default=5, help="Number of quantile bins")
    parser.add_argument(
        "--score_column",
        type=str,
        default="similarity_max",
        choices=["similarity_max", "similarity_topk_mean", "similarity_labelaware_max"],
        help="Similarity column for binning",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for bin datasets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_df = pd.read_csv(args.train_file).fillna("")
    sim_df = pd.read_csv(args.similarity_csv).fillna("")

    if args.score_column not in sim_df.columns:
        raise ValueError(f"Missing score column: {args.score_column}")

    # Quantile bins (Q1 low similarity -> QN high similarity)
    sim_df = sim_df.copy()
    sim_df["similarity_bin"] = pd.qcut(
        sim_df[args.score_column],
        q=args.num_bins,
        labels=False,
        duplicates="drop",
    )
    if sim_df["similarity_bin"].isna().all():
        raise ValueError("Failed to assign bins. Check similarity values.")

    sim_df["similarity_bin"] = sim_df["similarity_bin"].astype(int) + 1
    bin_summary = []

    grouped = list(sim_df.groupby("similarity_bin"))
    for bin_id, group in tqdm(grouped, total=len(grouped), desc="Building bin datasets"):
        combined = pd.concat([
            train_df, 
            group.drop(columns=[
                "similarity_bin", "generated_index", "similarity_max", 
                "similarity_topk_mean", "similarity_labelaware_max", "nn_distance"
            ], errors="ignore")
        ], ignore_index=True)
        out_file = os.path.join(args.output_dir, f"train_with_similarity_bin_{bin_id}.csv")
        combined.to_csv(out_file, index=False)

        bin_summary.append(
            {
                "bin_id": int(bin_id),
                "n_generated": int(len(group)),
                "score_min": float(group[args.score_column].min()),
                "score_mean": float(group[args.score_column].mean()),
                "score_max": float(group[args.score_column].max()),
                "train_file": out_file,
            }
        )
        print(f"Saved bin {bin_id}: +{len(group)} generated -> {out_file}")

    summary_df = pd.DataFrame(bin_summary).sort_values("bin_id")
    summary_file = os.path.join(args.output_dir, "bin_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Wrote bin summary: {summary_file}")

    tagged_file = os.path.join(args.output_dir, "generated_with_bins.csv")
    sim_df.to_csv(tagged_file, index=False)
    print(f"Wrote generated samples with bin labels: {tagged_file}")


if __name__ == "__main__":
    main()
