#!/usr/bin/env python3
"""
Re-run every (method, seed) training job and dump the held-out test-set predictions.

This lets the statistical comparison in revision/statistical_comparison
(per-seed results + paired bootstrap CIs on the fixed test set) be computed
from the same runs that produced the reported metrics. Training is fully
seeded (`augment_utils.seed_everything`, `cudnn.deterministic=True`), so a
re-run reproduces the logged metrics bit-for-bit; `--verify_dir` checks that.

Nothing in `log/` is overwritten: metrics and predictions
go to `--out_dir`. Runs are skipped when their .npz already exists, so the
script is resumable.

Outputs per run:
    <out_dir>/preds_<method>_<strategy>_seed<SEED>.npz     probs/labels/classes
    <out_dir>/metrics_<method>_<strategy>_seed<SEED>.json  same metrics as log/

Example:
    python3 save_test_predictions.py --aug_dir augmented_ehr_data \
        --out_dir preds --verify_dir log --device cuda:0
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

DEFAULT_METHODS = [
    "random_swap", "shuffle", "random_drop",
    "synonym_replacement", "synonym_insertion",
    "ishikawa", "composite_insertion",
    "taser_ae",
    "gan", "vae",
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aug_dir", default="augmented_ehr_data",
                    help="Folder holding augmented_ehr_data_<method>_<strategy>_seed<N>.csv")
    ap.add_argument("--train_pure", default="../data/processed/train_pure.csv",
                    help="Un-augmented training CSV (the no_augmentation run).")
    ap.add_argument("--test_file", default="../data/processed/test_split.csv")
    ap.add_argument("--out_dir", default="preds")
    ap.add_argument("--verify_dir", default=None,
                    help="Existing metrics folder to check the re-run against.")
    ap.add_argument("--strategy", default="Target_Multiplier_1X")
    ap.add_argument("--seeds", default="6,42,81")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--include_no_aug", action="store_true", default=True)
    ap.add_argument("--no_include_no_aug", dest="include_no_aug", action="store_false")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--jobs", type=int, default=1, help="Concurrent training runs.")
    ap.add_argument("--notebook_eval_compat", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    methods = [m for m in args.methods.split(",") if m.strip()]
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(here, args.out_dir)
    aug_dir = args.aug_dir if os.path.isabs(args.aug_dir) else os.path.join(here, args.aug_dir)
    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    for seed in seeds:
        tag = f"{args.strategy}_seed{seed}"
        if args.include_no_aug:
            jobs.append(("no_augmentation", seed, args.train_pure, tag))
        for method in methods:
            aug = os.path.join(aug_dir, f"augmented_ehr_data_{method}_{tag}.csv")
            if not os.path.isfile(aug):
                print(f"[skip] missing augmented CSV: {aug}")
                continue
            jobs.append((method, seed, aug, tag))

    pending = []
    for method, seed, train_file, tag in jobs:
        preds = os.path.join(out_dir, f"preds_{method}_{tag}.npz")
        metrics = os.path.join(out_dir, f"metrics_{method}_{tag}.json")
        if os.path.isfile(preds) and os.path.isfile(metrics):
            print(f"[have] {method} seed={seed}")
            continue
        pending.append((method, seed, train_file, preds, metrics))

    print(f"{len(pending)} run(s) to execute on {args.device} "
          f"({args.jobs} concurrent), {len(jobs) - len(pending)} already present.")
    if args.dry_run:
        for method, seed, train_file, _, _ in pending:
            print(f"  would run {method} seed={seed} <- {train_file}")
        return

    def run_one(job):
        method, seed, train_file, preds, metrics = job
        cmd = [sys.executable, os.path.join(here, "train_classifier.py"),
               "--train_file", train_file, "--test_file", args.test_file,
               "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
               "--patience", str(args.patience), "--seed", str(seed),
               "--device", args.device,
               "--save_metrics", metrics, "--save_preds", preds]
        if args.notebook_eval_compat:
            cmd.append("--notebook_eval_compat")
        print(f">>> {method} seed={seed}", flush=True)
        r = subprocess.run(cmd, cwd=here, stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)
        status = "ok" if r.returncode == 0 else f"FAILED({r.returncode})"
        print(f"<<< {method} seed={seed} {status}", flush=True)
        return method, seed, r.returncode

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for method, seed, rc in pool.map(run_one, pending):
            if rc != 0:
                failures.append((method, seed))
    if failures:
        print(f"\n{len(failures)} run(s) failed: {failures}")

    if args.verify_dir:
        verify(out_dir, args.verify_dir if os.path.isabs(args.verify_dir)
               else os.path.join(here, args.verify_dir), jobs, args.strategy)


def verify(out_dir, ref_dir, jobs, strategy):
    """Confirm the re-runs reproduce the previously logged metrics."""
    print(f"\n=== Verifying against {ref_dir} ===")
    worst, checked, missing = 0.0, 0, []
    for method, seed, _, tag in jobs:
        new_p = os.path.join(out_dir, f"metrics_{method}_{tag}.json")
        ref_p = os.path.join(ref_dir, f"metrics_{method}_{tag}.json")
        if not (os.path.isfile(new_p) and os.path.isfile(ref_p)):
            missing.append((method, seed))
            continue
        new, ref = json.load(open(new_p)), json.load(open(ref_p))
        for k in set(new) & set(ref):
            worst = max(worst, abs(float(new[k]) - float(ref[k])))
        checked += 1
    print(f"compared {checked} run(s); max |metric difference| = {worst:.3e}")
    if missing:
        print(f"no reference metrics for: {missing}")
    print("EXACT MATCH" if worst == 0 else
          ("within 1e-6" if worst < 1e-6 else "MISMATCH -- investigate before using"))


if __name__ == "__main__":
    main()
