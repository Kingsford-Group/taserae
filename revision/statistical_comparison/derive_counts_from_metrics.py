#!/usr/bin/env python3
"""
Recover per-seed TP / FP / predicted-positive counts from metrics files that are
ALREADY on disk, without retraining anything.

The counts behind the reported per-class precision and recall are fully
determined by the logged metrics plus the number of positive instances of each
class in the fixed test set:

    recall    = TP / P          =>  TP = recall * P            (P = test positives)
    precision = TP / (TP + FP)  =>  FP = TP * (1 - p) / p      (when p > 0)

so nothing has to be re-run. The only case that is not recoverable is a run with
TP = 0: precision is then 0 under the `zero_division=0` convention whether the run
predicted a few wrong positives or none at all, and the logged metrics cannot
distinguish those. Such rows are reported with FP = NA and flagged.

Every derived row is verified by recomputing precision, recall and F1 from the
integer counts and comparing against the logged values.

    python3 derive_counts_from_metrics.py \
        --log_dir ../../reproduction/log \
        --test_file ../../data/processed/test_split.csv \
        --strategy Target_Multiplier_1X --seeds 6,42,81 \
        --out outputs/prediction_counts_derived.csv
"""

import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

DISPLAY = {"no_augmentation": "No Augmentation", "random_drop": "Random Drop",
           "random_swap": "Random Swap", "synonym_replacement": "Random Synonym Repl.",
           "synonym_insertion": "Random Synonym Insert.", "shuffle": "Shuffle",
           "composite_insertion": "Composite Insertion", "vae": "Conditional VAE",
           "gan": "Conditional GAN", "ishikawa": "Ishikawa et al.",
           "taser_ae": "TASER-AE"}
CLASS_DISPLAY = {"GI_Bleed": "GI Bleed", "gibleed": "GI Bleed", "falls": "Falls",
                 "fractures": "Fractures", "stroke": "Stroke", "no": "No AE"}


def test_positives(test_file):
    df = pd.read_csv(test_file)
    col = "labels" if "labels" in df.columns else "adverse_events"
    counts = Counter()
    for s in df[col].astype(str):
        for lab in s.split(","):
            lab = lab.strip()
            if lab:
                counts[lab] += 1
    return counts, len(df), col


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--strategy", default="Target_Multiplier_1X")
    ap.add_argument("--seeds", default="6,42,81")
    ap.add_argument("--methods", default="", help="Comma-separated subset; empty = all found.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seeds = {int(s) for s in args.seeds.split(",") if s.strip()}
    want = {m for m in args.methods.split(",") if m.strip()}
    pos, n_test, label_col = test_positives(args.test_file)
    print(f"test set: {n_test} rows, label column {label_col!r}")
    print("positives per class:", dict(pos))

    rows, bad, micro_prec = [], [], {}
    for f in sorted(glob.glob(os.path.join(args.log_dir, f"metrics_*_{args.strategy}_seed*.json"))):
        base = os.path.basename(f).replace("metrics_", "").replace(f"_{args.strategy}", "")
        m = re.match(r"(.+)_seed(\d+)\.json$", base)
        if not m:
            continue
        method, seed = m.group(1), int(m.group(2))
        if seed not in seeds or (want and method not in want):
            continue
        d = json.load(open(f))
        micro_prec[(method, seed)] = float(d.get("prec_micro", float("nan")))
        classes = sorted({k.split("_", 1)[1] for k in d if k.startswith("f1_")}
                         - {"macro", "micro"})
        for c in classes:
            if c not in pos or f"prec_{c}" not in d:
                continue
            p, r, f1 = float(d[f"prec_{c}"]), float(d[f"rec_{c}"]), float(d.get(f"f1_{c}", np.nan))
            P = pos[c]
            tp_f = r * P
            tp = int(round(tp_f))
            if abs(tp_f - tp) > 1e-6:
                bad.append((method, seed, c, "recall*P is not an integer", tp_f))
            if p > 0:
                fp_f = tp * (1.0 - p) / p
                fp = int(round(fp_f))
                if abs(fp_f - fp) > 1e-6:
                    bad.append((method, seed, c, "TP(1-p)/p is not an integer", fp_f))
                pp, fp_out = tp + fp, fp
            else:
                fp, pp, fp_out = np.nan, np.nan, np.nan
            # verify against the logged values
            ok = True
            if p > 0:
                ok &= abs(tp / (tp + fp) - p) < 5e-4 if (tp + fp) else False
                denom = 2 * tp + fp + (P - tp)
                ok &= abs((2 * tp / denom if denom else 0.0) - f1) < 5e-4
            ok &= abs((tp / P if P else 0.0) - r) < 5e-4
            rows.append({
                "Method": DISPLAY.get(method, method), "method_key": method, "Seed": seed,
                "Class": CLASS_DISPLAY.get(c, c), "class_key": c,
                "TP": tp, "FP": fp_out, "FN": P - tp,
                "predicted_positives": pp, "actual_positives": P,
                "precision": p, "recall": r, "f1": f1,
                "precision_undefined_zero_division": bool(p == 0 and tp == 0),
                "fp_not_recoverable": bool(p == 0),
                "fp_source": "per_class_precision" if p > 0 else "unresolved",
                "verified_against_logged_metrics": bool(ok),
            })

    df = pd.DataFrame(rows)

    # Second pass: micro-averaged precision pins down the TOTAL false positives of a run,
    #   prec_micro = sum(TP) / sum(TP + FP),
    # so any run in which exactly one class has an unresolved FP is fully determined; when
    # several are unresolved we can still report their combined total.
    df["fp_group_total"] = np.nan
    for (method, seed), g in df.groupby(["method_key", "Seed"]):
        pm = micro_prec.get((method, seed), np.nan)
        if not np.isfinite(pm) or pm <= 0:
            continue
        sum_tp = g["TP"].sum()
        total_pp = sum_tp / pm
        sum_fp = total_pp - sum_tp
        known = g.loc[~g["FP"].isna(), "FP"].sum()
        unknown_total = sum_fp - known
        idx = g.index[g["FP"].isna()]
        if len(idx) == 0:
            continue
        if abs(unknown_total - round(unknown_total)) > 1e-4:
            bad.append((method, seed, "*", "micro-implied FP total is not an integer",
                        unknown_total))
        if len(idx) == 1:
            i = idx[0]
            fp = int(round(unknown_total))
            df.loc[i, "FP"] = fp
            df.loc[i, "predicted_positives"] = df.loc[i, "TP"] + fp
            df.loc[i, "fp_not_recoverable"] = False
            df.loc[i, "fp_source"] = "micro_precision"
            df.loc[i, "precision_undefined_zero_division"] = bool(df.loc[i, "TP"] + fp == 0)
        else:
            df.loc[idx, "fp_group_total"] = round(unknown_total)
            df.loc[idx, "fp_source"] = "group_total_only"
            if round(unknown_total) == 0:      # none of them predicted anything
                df.loc[idx, "FP"] = 0
                df.loc[idx, "predicted_positives"] = df.loc[idx, "TP"]
                df.loc[idx, "fp_not_recoverable"] = False
                df.loc[idx, "fp_source"] = "micro_precision (group total = 0)"
                df.loc[idx, "precision_undefined_zero_division"] = True

    df = df.sort_values(["method_key", "class_key", "Seed"])
    df.to_csv(args.out, index=False)
    nver = int(df["verified_against_logged_metrics"].sum())
    print(f"\nwrote {args.out}: {len(df)} rows, {nver}/{len(df)} verified against logged metrics")
    if bad:
        print(f"{len(bad)} non-integer reconstruction(s) -- inspect these:")
        for b in bad[:10]:
            print("   ", b)
    print("\nFP provenance:", dict(df["fp_source"].value_counts()))
    nrec = int(df["fp_not_recoverable"].sum())
    if nrec:
        print(f"{nrec} row(s) still unresolved: precision = 0 and several classes share one "
              f"micro-precision constraint; their combined FP total is in 'fp_group_total'.")


if __name__ == "__main__":
    main()
