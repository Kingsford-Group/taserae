#!/usr/bin/env python3
"""
Run TASER-AE augmentation on real-name train split.
Uses realname embeddings and unique 'realname_' strategy prefix
to avoid collision with benchmark coded-data augmented files.
"""

import argparse
import os
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Run TASER-AE augmentation for real-name train split.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to real-name train_pure.csv")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to realname_embeddings.npy")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--strategy_prefix", type=str, default="Target_Multiplier_1X")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Copy augmented CSV here (defaults to clinicalbert outputs)")
    args = parser.parse_args()

    # Resolve to absolute paths since subprocess runs in a different cwd
    args.train_file = os.path.abspath(args.train_file)
    args.embeddings = os.path.abspath(args.embeddings)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    reproduction_dir = os.path.join(repo_root, "reproduction")

    # Unique strategy name with realname_ prefix to avoid collision with coded-data files
    strategy = f"realname_{args.strategy_prefix}_seed{args.seed}"

    cmd = [
        "python3",
        os.path.join(reproduction_dir, "augment_taser_ae.py"),
        "--input", args.train_file,
        "--strategy", strategy,
        "--multiplier", str(args.multiplier),
        "--seed", str(args.seed),
        "--embeddings", args.embeddings,
        "--threshold", str(args.threshold),
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=reproduction_dir)
    if result.returncode != 0:
        raise RuntimeError("TASER-AE augmentation failed.")

    # augment_utils.save_augmented_data saves to reproduction/augmented_ehr_data/
    aug_file = os.path.join(
        reproduction_dir, "augmented_ehr_data",
        f"augmented_ehr_data_taser_ae_{strategy}.csv",
    )
    if not os.path.exists(aug_file):
        raise FileNotFoundError(f"Expected augmented file not found: {aug_file}")
    print(f"Augmented file: {aug_file}")

    # Also copy to clinicalbert output dir for easy reference
    out_dir = args.output_dir or os.path.join(script_dir, "outputs", "augmented")
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, os.path.basename(aug_file))
    shutil.copy2(aug_file, dest)
    print(f"Saved copy: {dest}")


if __name__ == "__main__":
    main()
