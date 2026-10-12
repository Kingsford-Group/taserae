#!/usr/bin/env python3
"""
Orchestrate Bio_ClinicalBERT no-augmentation vs TASER-AE comparison.

Pipeline:
  1) Generate real-name embeddings (if needed)
  2) For each seed: augment with TASER-AE using real-name embeddings
  3) Train/evaluate ClinicalBERT on both no-aug and TASER-AE conditions
"""

import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd


def run_cmd(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Run ClinicalBERT no-aug vs TASER-AE comparison.")
    parser.add_argument("--train_file", type=str, required=True, help="Real-name train_pure.csv")
    parser.add_argument("--test_file", type=str, required=True, help="Real-name test_split.csv")
    parser.add_argument("--seeds", type=str, default="6,42,81")
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--strategy_prefix", type=str, default="Target_Multiplier_1X")
    parser.add_argument("--model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--output_dir", type=str,
        default="revision/pretrained_clinicalbert/outputs/noaug_vs_taser",
    )
    args = parser.parse_args()

    # Resolve to absolute paths since subprocesses run in different cwds
    args.train_file = os.path.abspath(args.train_file)
    args.test_file = os.path.abspath(args.test_file)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    repro_dir = os.path.join(repo_root, "reproduction")
    out_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    # --- Step 0: Generate real-name embeddings if needed ---
    realname_emb = os.path.join(script_dir, "outputs", "realname_embeddings.npy")
    if not os.path.exists(realname_emb):
        print("=" * 60)
        print("[Step 0] Generating real-name embeddings...")
        print("=" * 60)
        run_cmd(
            ["python3", os.path.join(script_dir, "generate_realname_embeddings.py")],
            cwd=repo_root,
        )
    else:
        print(f"Real-name embeddings exist: {realname_emb}")

    records = []
    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")

        # --- No-augmentation condition ---
        # User requested to skip no_aug training because they already have it.
        # We will only train and collect TASER-AE metrics.

        # --- TASER-AE condition ---
        # Use realname_ prefix to avoid collision with coded-data benchmark files
        strategy = f"realname_{args.strategy_prefix}_seed{seed}"
        aug_file = os.path.join(
            repro_dir, "augmented_ehr_data",
            f"augmented_ehr_data_taser_ae_{strategy}.csv",
        )

        if not os.path.exists(aug_file):
            print(f"\n[TASER-AE] Augmenting real-name data (seed={seed})...")
            run_cmd([
                "python3", os.path.join(script_dir, "augment_taser_realname.py"),
                "--train_file", args.train_file,
                "--embeddings", realname_emb,
                "--seed", str(seed),
                "--multiplier", str(args.multiplier),
                "--strategy_prefix", args.strategy_prefix,
                "--threshold", str(args.threshold),
                "--output_dir", os.path.join(out_dir, "augmented"),
            ], cwd=repo_root)
        else:
            print(f"[TASER-AE] Using existing aug file: {aug_file}")

        taser_metrics = os.path.join(out_dir, f"metrics_clinicalbert_taser_ae_seed{seed}.json")
        if not os.path.exists(taser_metrics):
            print(f"\n[TASER-AE] Training ClinicalBERT (seed={seed})...")
            run_cmd([
                "python3", os.path.join(script_dir, "train_eval_clinicalbert.py"),
                "--train_file", aug_file,
                "--test_file", args.test_file,
                "--model_name", args.model_name,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--max_len", str(args.max_len),
                "--patience", str(args.patience),
                "--seed", str(seed),
                "--device", args.device,
                "--save_metrics", taser_metrics,
            ], cwd=repo_root)
        else:
            print(f"[TASER-AE] Using existing: {taser_metrics}")

        # Collect results
        with open(taser_metrics, "r") as f:
            m_ta = json.load(f)

        for cond, m in [("clinicalbert_taser_ae", m_ta)]:
            row = {"condition": cond, "seed": seed}
            row.update({k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in m.items()})
            records.append(row)

    # Save per-seed and aggregated results
    by_seed = pd.DataFrame(records).sort_values(["condition", "seed"])
    by_seed_path = os.path.join(out_dir, "clinicalbert_noaug_vs_taser_by_seed.csv")
    by_seed.to_csv(by_seed_path, index=False)

    metric_cols = [c for c in by_seed.columns if c not in {"condition", "seed", "model_name"}
                   and by_seed[c].dtype in [np.float64, float, np.int64, int]]
    agg = by_seed.groupby("condition")[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["condition"] + [f"{a}_{b}" for a, b in agg.columns.tolist()[1:]]
    agg_path = os.path.join(out_dir, "clinicalbert_noaug_vs_taser_aggregated.csv")
    agg.to_csv(agg_path, index=False)

    print(f"\nSaved: {by_seed_path}")
    print(f"Saved: {agg_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for _, r in by_seed.iterrows():
        print(f"  {r['condition']} seed={int(r['seed'])}: "
              f"F1_macro={r.get('f1_macro', float('nan')):.4f} "
              f"minority_F1={r.get('minority_f1_macro', float('nan')):.4f}")


if __name__ == "__main__":
    main()
