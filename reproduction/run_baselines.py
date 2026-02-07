#!/usr/bin/env python3
"""
Run All Baselines (Non-Augmented Evaluation)

This script runs:
1. Transformer on non-augmented data (same as augmented methods but without augmentation)
2. Decision Tree baseline
3. Random Forest baseline
4. XGBoost baseline

Usage:
    python3 run_baselines.py --train_file train_pure.csv --test_file test_split.csv \
        --device cuda:0 --seeds 6,35,81
"""

import subprocess
import os
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="Run All Baselines")
    parser.add_argument("--train_file", type=str, default="train_pure.csv", help="Training data")
    parser.add_argument("--test_file", type=str, default="test_split.csv", help="Test data")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device for Transformer")
    parser.add_argument("--seeds", type=str, default="6,35,81", help="Comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=15, help="Transformer epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(",")]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=" * 60)
    print("Running All Baselines (Non-Augmented)")
    print("=" * 60)
    
    # Define baseline jobs
    baselines = []
    
    for seed in seeds:
        # 1. Transformer on non-augmented data
        baselines.append({
            "name": f"transformer_no_aug (seed={seed})",
            "cmd": [
                "python3", "train_classifier.py",
                "--train_file", args.train_file,
                "--test_file", args.test_file,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--patience", str(args.patience),
                "--seed", str(seed),
                "--device", args.device,
                "--save_metrics", f"metrics_transformer_no_aug_seed{seed}.json"
            ],
            "metrics_file": f"metrics_transformer_no_aug_seed{seed}.json"
        })
        
        # 2. Decision Tree
        baselines.append({
            "name": f"decision_tree (seed={seed})",
            "cmd": [
                "python3", "baseline_decision_tree.py",
                "--train_file", args.train_file,
                "--test_file", args.test_file,
                "--seed", str(seed),
                "--save_metrics", f"metrics_decision_tree_seed{seed}.json"
            ],
            "metrics_file": f"metrics_decision_tree_seed{seed}.json"
        })
        
        # 3. Random Forest
        baselines.append({
            "name": f"random_forest (seed={seed})",
            "cmd": [
                "python3", "baseline_random_forest.py",
                "--train_file", args.train_file,
                "--test_file", args.test_file,
                "--seed", str(seed),
                "--save_metrics", f"metrics_random_forest_seed{seed}.json"
            ],
            "metrics_file": f"metrics_random_forest_seed{seed}.json"
        })
        
        # 4. XGBoost
        baselines.append({
            "name": f"xgboost (seed={seed})",
            "cmd": [
                "python3", "baseline_xgboost.py",
                "--train_file", args.train_file,
                "--test_file", args.test_file,
                "--seed", str(seed),
                "--save_metrics", f"metrics_xgboost_seed{seed}.json"
            ],
            "metrics_file": f"metrics_xgboost_seed{seed}.json"
        })
    
    total = len(baselines)
    completed = 0
    
    for job in baselines:
        # Skip if already done
        if os.path.exists(job["metrics_file"]):
            print(f"[{completed+1}/{total}] {job['name']} - SKIPPED (exists)")
            completed += 1
            continue
        
        print(f"\n[{completed+1}/{total}] Running {job['name']}...")
        start = time.time()
        
        result = subprocess.run(job["cmd"], capture_output=True, text=True)
        
        # Print last part of output
        output = result.stdout
        if len(output) > 1500:
            output = "..." + output[-1500:]
        print(output)
        
        if result.returncode != 0:
            print(f"ERROR: {result.stderr[:500]}")
        else:
            elapsed = time.time() - start
            print(f"Completed in {elapsed:.1f}s")
        
        completed += 1
    
    print("\n" + "=" * 60)
    print(f"Baselines completed: {completed}/{total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
