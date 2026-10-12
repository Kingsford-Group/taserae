#!/usr/bin/env python3
"""
Train/evaluate Decision Tree, Random Forest, and Gradient Boosting baselines
on TASER-AE augmented train data and evaluate on held-out test data.
"""

import argparse
import json
import os
import subprocess
from typing import Dict, List

import numpy as np
import pandas as pd


def resolve_seeded_path(path_template: str, seed: int) -> str:
    if "{seed}" in path_template:
        return path_template.format(seed=seed)
    return path_template.replace("seed6.csv", f"seed{seed}.csv")


def run_cmd(cmd: List[str], cwd: str, env: Dict[str, str]) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Run tree/boosting baselines on TASER-AE augmented data.")
    parser.add_argument(
        "--train_file_taser",
        type=str,
        default="reproduction/augmented_ehr_data/augmented_ehr_data_taser_ae_Target_Multiplier_1X_seed{seed}.csv",
        help="TASER-AE train file template, supports {seed}.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/processed/test_split.csv",
        help="Held-out test CSV path.",
    )
    parser.add_argument("--seeds", type=str, default="6,42,81", help="Comma-separated seeds.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="revision/tree_baselines_on_taser/outputs",
        help="Output directory.",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    repro_dir = os.path.join(repo_root, "reproduction")
    out_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    test_file = os.path.join(repo_root, args.test_file)
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Missing test file: {test_file}")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    methods = [
        ("decision_tree", "baseline_decision_tree.py", ["--no_gridsearch"]),
        ("random_forest", "baseline_random_forest.py", ["--no_gridsearch"]),
        ("gradient_boosting", "baseline_gradient_boosting.py", []),
    ]

    env = os.environ.copy()
    records = []

    for seed in seeds:
        train_file = os.path.join(repo_root, resolve_seeded_path(args.train_file_taser, seed))
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Missing TASER train file for seed {seed}: {train_file}")

        for method_name, script_name, extra_args in methods:
            metrics_path = os.path.join(out_dir, f"metrics_{method_name}_taser_seed{seed}.json")
            if not os.path.exists(metrics_path):
                cmd = [
                    "python3",
                    os.path.join(repro_dir, script_name),
                    "--train_file",
                    train_file,
                    "--test_file",
                    test_file,
                    "--seed",
                    str(seed),
                    "--save_metrics",
                    metrics_path,
                    *extra_args,
                ]
                print(f"Running {method_name} on TASER (seed={seed})")
                run_cmd(cmd, cwd=repro_dir, env=env)
            else:
                print(f"Using existing metrics: {metrics_path}")

            with open(metrics_path, "r") as f:
                m = json.load(f)

            row = {
                "method": method_name,
                "seed": seed,
                "f1_macro": m.get("f1_macro", np.nan),
                "f1_micro": m.get("f1_micro", np.nan),
                "prec_macro": m.get("prec_macro", np.nan),
                "rec_macro": m.get("rec_macro", np.nan),
                "auc_macro": m.get("auc_macro", np.nan),
                "minority_f1_macro": m.get("minority_f1_macro", np.nan),
                "minority_prec_macro": m.get("minority_prec_macro", np.nan),
                "minority_rec_macro": m.get("minority_rec_macro", np.nan),
                "metrics_file": metrics_path,
            }
            for key, val in m.items():
                if key.startswith("f1_") and key not in {"f1_macro", "f1_micro"}:
                    label = key.replace("f1_", "", 1)
                    row[f"f1_{label}"] = val
                    row[f"prec_{label}"] = m.get(f"prec_{label}", np.nan)
                    row[f"rec_{label}"] = m.get(f"rec_{label}", np.nan)
            records.append(row)

    raw_df = pd.DataFrame(records).sort_values(["method", "seed"])
    raw_csv = os.path.join(out_dir, "tree_baselines_taser_metrics_by_seed.csv")
    raw_df.to_csv(raw_csv, index=False)

    # Overall aggregate
    agg_cols = ["f1_macro", "f1_micro", "prec_macro", "rec_macro", "auc_macro",
                "minority_f1_macro", "minority_prec_macro", "minority_rec_macro"]
    agg = (
        raw_df.groupby("method")[agg_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["method"] + [f"{a}_{b}" for a, b in agg.columns.tolist()[1:]]
    agg_csv = os.path.join(out_dir, "tree_baselines_taser_metrics_aggregated.csv")
    agg.to_csv(agg_csv, index=False)

    # Per-label table
    label_rows = []
    per_label_cols = [c for c in raw_df.columns if c.startswith("f1_") and c not in {"f1_macro", "f1_micro"}]
    for col in sorted(per_label_cols):
        label = col.replace("f1_", "", 1)
        prec_col = f"prec_{label}"
        rec_col = f"rec_{label}"
        g = raw_df.groupby("method").agg(
            f1_mean=(col, "mean"),
            f1_std=(col, "std"),
            precision_mean=(prec_col, "mean"),
            precision_std=(prec_col, "std"),
            recall_mean=(rec_col, "mean"),
            recall_std=(rec_col, "std"),
        )
        g = g.reset_index()
        g["label"] = label
        label_rows.append(g)

    if label_rows:
        label_df = pd.concat(label_rows, ignore_index=True).sort_values(["label", "method"])
    else:
        label_df = pd.DataFrame(columns=["method", "label"])
    label_csv = os.path.join(out_dir, "tree_baselines_taser_per_label_table.csv")
    label_df.to_csv(label_csv, index=False)

    print("Done.")
    print(f"Per-seed metrics: {raw_csv}")
    print(f"Aggregated metrics: {agg_csv}")
    print(f"Per-label table: {label_csv}")


if __name__ == "__main__":
    main()
