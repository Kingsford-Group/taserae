#!/usr/bin/env python3
"""
Run reproduction/train_classifier.py and collect --save_timing JSON (training vs test inference).
"""

import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Measure transformer training and test-inference wall time.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_metrics", type=str, default=None, help="Optional metrics JSON path")
    parser.add_argument("--save_timing", type=str, required=True, help="Output timing JSON path")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    repro_dir = os.path.join(repo_root, "reproduction")
    train_script = os.path.join(repro_dir, "train_classifier.py")

    save_metrics = args.save_metrics
    if save_metrics is None:
        save_metrics = os.path.splitext(args.save_timing)[0] + "_metrics.json"

    cmd = [
        sys.executable,
        train_script,
        "--train_file",
        os.path.abspath(args.train_file),
        "--test_file",
        os.path.abspath(args.test_file),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--patience",
        str(args.patience),
        "--num_workers",
        str(args.num_workers),
        "--save_metrics",
        os.path.abspath(save_metrics),
        "--save_timing",
        os.path.abspath(args.save_timing),
    ]

    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")

    timing_dir = os.path.dirname(os.path.abspath(args.save_timing))
    if timing_dir:
        os.makedirs(timing_dir, exist_ok=True)
    subprocess.run(cmd, cwd=repro_dir, env=env, check=True)

    with open(args.save_timing) as f:
        timing = json.load(f)
    timing["component"] = "transformer_classifier"
    timing["train_file"] = os.path.abspath(args.train_file)
    timing["test_file"] = os.path.abspath(args.test_file)
    timing["seed"] = args.seed
    with open(args.save_timing, "w") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
