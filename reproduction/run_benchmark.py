#!/usr/bin/env python3
"""
TASER-AE Reproduction Benchmark Runner
Runs 1X augmentation benchmark with seeds [6, 35, 81] on a single GPU.

Usage:
    python3 run_benchmark.py --data /path/to/mimic4_label.csv --device cuda:0
"""

import subprocess
import os
import sys
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="Run TASER-AE Benchmark")
    parser.add_argument("--data", type=str, required=True, help="Path to mimic4_label.csv")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device (e.g., cuda:0)")
    parser.add_argument("--seeds", type=str, default="6,35,81", help="Comma-separated seeds")
    parser.add_argument("--multiplier", type=int, default=1, help="Augmentation multiplier (1 for 1X/Doubling)")
    parser.add_argument("--strategy", type=str, default="Target_Multiplier_1X", help="Strategy name")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--skip_prep", action="store_true", help="Skip data preparation if already done")
    parser.add_argument("--cv", action="store_true", help="Run with 5-fold CV (5 seeds: 6,35,81,42,100)")
    args = parser.parse_args()
    
    if args.cv:
        seeds = [6, 35, 81]
        print(f"Running in 5-fold CV mode with seeds: {seeds}")
    else:
        seeds = [6]
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=" * 60)
    print("TASER-AE Reproduction Benchmark")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Device: {args.device}")
    print(f"Seeds: {seeds}")
    print(f"Strategy: {args.strategy}")
    print(f"Multiplier: {args.multiplier}")
    print("=" * 60)
    
    # Configure paths
    data_dir = os.path.join(script_dir, "..", "data", "processed")
    aug_dir = os.path.join(script_dir, "augmented_ehr_data")
    log_dir = os.path.join(script_dir, "log")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(aug_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    train_file = os.path.join(data_dir, "train_pure.csv")
    test_file = os.path.join(data_dir, "test_split.csv")
    embed_file = os.path.join(data_dir, "embeddings.npy")
    
    # Step 1: Data Preparation
    if not args.skip_prep:
        print("\n[Step 1] Preparing data splits...")
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            prep_script = f'''
import pandas as pd
from sklearn.model_selection import train_test_split
try:
    df = pd.read_csv("{args.data}")
    # Rename if needed
    if "adverse_events" in df.columns:
        df = df.rename(columns={{"adverse_events": "labels"}})
    
    # Stratified split
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["labels"])
    train.to_csv("{train_file}", index=False)
    test.to_csv("{test_file}", index=False)
    print(f"Created train ({{len(train)}}) and test ({{len(test)}}) splits.")
except Exception as e:
    print(f"Error preparing data: {{e}}")
'''
            subprocess.run(["python3", "-c", prep_script], check=True)
        else:
            print("  Data splits already exist, skipping.")
    
    # Step 2: Generate Embeddings
    print("\n[Step 2] Generating embeddings...")
    if not os.path.exists(embed_file):
         subprocess.run(["python3", "generate_embeddings.py"], check=True)
    else:
         print("  Embeddings already exist, skipping.")
    
    # Define all methods
    # Include: No Augmentation, Baselines (XGB, DT), Augmentation Baselines, TASER-AE, Generative
    AUGMENTATION_METHODS = [
        "random_swap", "shuffle", "random_drop",
        "synonym_replacement", "synonym_insertion",
        "ishikawa", "composite_insertion",
        "taser_ae_random", "taser_ae_native",
        "gan", "vae"
    ]
    
    print("\n[Step 3] Running benchmark...")
    
    total_jobs = len(seeds) * (len(AUGMENTATION_METHODS) + 4) # +3 for No_Aug, XGB, DT
    job_idx = 1
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        
        # 1. No Augmentation Baseline (Transformer)
        print(f"  [{job_idx}/{total_jobs}] Running no_augmentation (seed={seed})...")
        metrics_file = os.path.join(log_dir, f"metrics_no_augmentation_{args.strategy}_seed{seed}.json")
        if not os.path.exists(metrics_file):
            train_cmd = [
                "python3", "train_classifier.py",
                "--train_file", train_file,
                "--test_file", test_file,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--patience", str(args.patience),
                "--seed", str(seed),
                "--device", args.device,
                "--save_metrics", metrics_file
            ]
            try:
                subprocess.run(train_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"    ERROR during training: {e}")
        else:
            print("    SKIPPED (exists)")
        job_idx += 1
            
        # 2. XGBoost Baseline
        print(f"  [{job_idx}/{total_jobs}] Running baseline_xgboost (seed={seed})...")
        metrics_file_xgb = os.path.join(log_dir, f"metrics_baseline_xgboost_{args.strategy}_seed{seed}.json")
        if not os.path.exists(metrics_file_xgb):
            # Pass save_metrics argument
            subprocess.run([
                "python3", "baseline_xgboost.py",
                "--train_file", train_file,
                "--test_file", test_file,
                "--seed", str(seed),
                "--save_metrics", metrics_file_xgb
            ], check=True)
        else:
            print("    SKIPPED (exists)")
        job_idx += 1
        
        # 3. Decision Tree Baseline
        print(f"  [{job_idx}/{total_jobs}] Running baseline_decision_tree (seed={seed})...")
        metrics_file_dt = os.path.join(log_dir, f"metrics_baseline_decision_tree_{args.strategy}_seed{seed}.json")
        if not os.path.exists(metrics_file_dt):
            subprocess.run([
                "python3", "baseline_decision_tree.py",
                "--train_file", train_file,
                "--test_file", test_file,
                "--seed", str(seed),
                "--save_metrics", metrics_file_dt
            ], check=True)
        else:
            print("    SKIPPED (exists)")
        job_idx += 1

        # 4. Random Forest Baseline
        print(f"  [{job_idx}/{total_jobs}] Running baseline_random_forest (seed={seed})...")
        metrics_file_rf = os.path.join(log_dir, f"metrics_baseline_random_forest_{args.strategy}_seed{seed}.json")
        if not os.path.exists(metrics_file_rf):
            subprocess.run([
                "python3", "baseline_random_forest.py",
                "--train_file", train_file,
                "--test_file", test_file,
                "--seed", str(seed),
                "--save_metrics", metrics_file_rf
            ], check=True)
        else:
            print("    SKIPPED (exists)")
        job_idx += 1

        # 5. Augmentation Methods
        for method in AUGMENTATION_METHODS:
            print(f"  [{job_idx}/{total_jobs}] Running {method} (seed={seed})...")
            
            unique_strat = f"{args.strategy}_seed{seed}"
            aug_filename = f"augmented_ehr_data_{method}_{unique_strat}.csv"
            aug_file = os.path.join(aug_dir, aug_filename)
            metrics_file = os.path.join(log_dir, f"metrics_{method}_{unique_strat}.json")
            
            # Run Augmentation if needed
            if not os.path.exists(aug_file):
                print("    Augmenting...")
                script = f"augment_{method}.py"
                if method == "ishikawa": script = "ishikawa_augment.py"
                
                # Build command based on method requirements
                aug_cmd = ["python3", script]
                aug_cmd.extend(["--input", train_file])
                aug_cmd.extend(["--strategy", unique_strat])
                aug_cmd.extend(["--multiplier", str(args.multiplier)])
                aug_cmd.extend(["--seed", str(seed)])
                
                # Embeddings required?
                if method in ["synonym_replacement", "synonym_insertion", "ishikawa", 
                              "composite_insertion", "taser_ae_random", "taser_ae_native"]:
                    aug_cmd.extend(["--embeddings", embed_file])
                    
                # GAN/VAE require device (epochs default)
                if method in ["gan", "vae"]:
                    aug_cmd.extend(["--device", args.device])
                
                try:
                    subprocess.run(aug_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"    ERROR during augmentation: {e}")
                    job_idx += 1
                    continue
            else:
                 print("    SKIPPED (augmentation exists)")
            
            # Run Training if needed
            if not os.path.exists(metrics_file):
                print("    Training...")
                if not os.path.exists(aug_file):
                    print(f"    ERROR: Aug file {aug_file} missing. Skipping training.")
                    job_idx += 1
                    continue
                    
                train_cmd = [
                    "python3", "train_classifier.py",
                    "--train_file", aug_file,
                    "--test_file", test_file,
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--patience", str(args.patience),
                    "--seed", str(seed),
                    "--device", args.device,
                    "--save_metrics", metrics_file
                ]
                try:
                    subprocess.run(train_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"    ERROR during training: {e}")
            else:
                print("    SKIPPED (training exists)")
            
            job_idx += 1
            
    print("\n" + "=" * 60)
    print("Benchmark Completed.")
    print("Run 'python3 aggregate_results.py' to see aggregated results.")
    print("=" * 60)

if __name__ == "__main__":
    main()
