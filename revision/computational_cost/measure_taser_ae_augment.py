#!/usr/bin/env python3
"""
Wall-clock time for TASER-AE augmentation only (reproduction/augment_taser_ae.py).
Does not include embedding generation or classifier training.
"""

import argparse
import json
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Measure TASER-AE augmentation wall time.")
    parser.add_argument("--input", type=str, required=True, help="Train CSV (e.g. data/processed/train_pure.csv)")
    parser.add_argument("--strategy", type=str, required=True, help="e.g. Target_Multiplier_1X_seed6")
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings.npy")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Write timing + metadata JSON (default: stdout only)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    repro_dir = os.path.join(repo_root, "reproduction")
    aug_script = os.path.join(repro_dir, "augment_taser_ae.py")

    cmd = [
        sys.executable,
        aug_script,
        "--input",
        os.path.abspath(args.input),
        "--strategy",
        args.strategy,
        "--multiplier",
        str(args.multiplier),
        "--seed",
        str(args.seed),
        "--embeddings",
        os.path.abspath(args.embeddings),
        "--threshold",
        str(args.threshold),
    ]

    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=repro_dir, env=env, capture_output=True, text=True)
    t1 = time.perf_counter()
    wall_sec = t1 - t0

    out_name = f"augmented_ehr_data_taser_ae_{args.strategy}.csv"
    out_path = os.path.join(repro_dir, "augmented_ehr_data", out_name)

    record = {
        "component": "taser_ae_augmentation",
        "wall_sec": wall_sec,
        "returncode": result.returncode,
        "command": cmd,
        "cwd": repro_dir,
        "expected_output_csv": out_path,
        "output_csv_exists": os.path.isfile(out_path),
    }
    if result.returncode != 0:
        record["stderr_tail"] = (result.stderr or "")[-4000:]

    print(json.dumps(record, indent=2))
    if result.returncode != 0:
        sys.exit(result.returncode)

    if args.output_json:
        out_dir = os.path.dirname(os.path.abspath(args.output_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Wrote {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
