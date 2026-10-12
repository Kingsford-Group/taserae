#!/usr/bin/env python3
"""
Create train/test splits for revision real-name CSV.
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Prepare stratified train/test split for real-name CSV.")
    parser.add_argument("--input_csv", type=str, required=True, help="Input real-name CSV with labels column")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for train/test splits")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv).fillna("")
    if "labels" not in df.columns:
        raise ValueError("Input CSV must contain 'labels' column")

    counts = df["labels"].value_counts()
    valid_labels = counts[counts > 1].index
    df = df[df["labels"].isin(valid_labels)].copy()

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["labels"],
    )
    train_path = os.path.join(args.output_dir, "train_pure.csv")
    test_path = os.path.join(args.output_dir, "test_split.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Wrote train split: {train_path} ({len(train_df)} rows)")
    print(f"Wrote test split:  {test_path} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
