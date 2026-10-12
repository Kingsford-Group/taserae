#!/usr/bin/env python3
"""
Wall-clock time for Word2Vec embedding generation (reproduction/generate_embeddings.py).

Reads data/processed/train_pure.csv and writes data/processed/embeddings.npy
(paths are fixed inside generate_embeddings.py).
"""

import argparse
import json
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Measure Word2Vec / generate_embeddings.py wall time.")
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
    gen_script = os.path.join(repro_dir, "generate_embeddings.py")

    expected_out = os.path.join(repo_root, "data", "processed", "embeddings.npy")
    expected_in = os.path.join(repo_root, "data", "processed", "train_pure.csv")

    cmd = [sys.executable, gen_script]
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=repro_dir, env=env, capture_output=True, text=True)
    t1 = time.perf_counter()
    wall_sec = t1 - t0

    record = {
        "component": "word2vec_generate_embeddings",
        "wall_sec": wall_sec,
        "returncode": result.returncode,
        "command": cmd,
        "cwd": repro_dir,
        "script": "reproduction/generate_embeddings.py",
        "expected_input_csv": expected_in,
        "expected_output_npy": expected_out,
        "output_npy_exists": os.path.isfile(expected_out),
        "input_csv_exists": os.path.isfile(expected_in),
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
