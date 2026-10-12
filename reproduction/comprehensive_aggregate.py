#!/usr/bin/env python3
"""
Comprehensive aggregation: mean & std (sample, ddof=1) of EVERY metric present in
a folder of metrics_<method>_<suffix>_seed<N>.json files, grouped by method.

Outputs per-class metrics (f1/prec/rec/auc/ap for Falls,Fractures,GI_Bleed,Stroke,no),
the micro/macro averages, and derived minority averages over the 4 minority classes
(f1/prec/rec/auc/ap). Missing metrics (e.g. AUPR not computed) appear as empty.
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd

MINORITY = ["Falls", "Fractures", "GI_Bleed", "Stroke"]
CLASSES = MINORITY + ["no"]
FAMILIES = ["f1", "prec", "rec", "auc", "ap"]   # ap = AUPR (average precision)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--suffix", default="Target_Multiplier_1X")
    ap.add_argument("--seeds", default="6,42,81")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = {int(s) for s in args.seeds.split(",") if s.strip()}

    rows = []
    for f in glob.glob(os.path.join(args.log_dir, f"metrics_*_{args.suffix}_seed*.json")):
        base = os.path.basename(f).replace("metrics_", "").replace(f"_{args.suffix}", "")
        m = re.match(r"(.+)_seed(\d+)\.json$", base)
        if not m:
            continue
        method, seed = m.group(1), int(m.group(2))
        if seed not in seeds:
            continue
        d = json.load(open(f))
        row = {"Method": method, "Seed": seed}
        # micro/macro averages
        for fam in FAMILIES:
            for avg in ["macro", "micro"]:
                row[f"{fam}_{avg}"] = d.get(f"{fam}_{avg}", np.nan)
        # per-class
        for fam in FAMILIES:
            for c in CLASSES:
                row[f"{fam}_{c}"] = d.get(f"{fam}_{c}", np.nan)
        # derived minority averages (mean over the 4 minority classes)
        for fam in FAMILIES:
            vals = [d.get(f"{fam}_{c}") for c in MINORITY]
            vals = [v for v in vals if v is not None]
            row[f"{fam}_minority"] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)

    if not rows:
        print("No data found.")
        return
    df = pd.DataFrame(rows)
    n_seeds = df.groupby("Method")["Seed"].nunique()
    metric_cols = [c for c in df.columns if c not in ("Method", "Seed")]
    agg = df.groupby("Method")[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.insert(0, "n_seeds", n_seeds.values)
    agg = agg.reset_index().sort_values("f1_macro_mean", ascending=False, na_position="last")
    agg.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(agg)} methods, {len(metric_cols)} metrics x (mean,std)")
    # quick preview
    def f(r, k):
        m, s = r.get(f"{k}_mean", np.nan), r.get(f"{k}_std", np.nan)
        return "NA" if pd.isna(m) else (f"{m:.4f}±{s:.4f}" if not pd.isna(s) else f"{m:.4f}")
    prev = pd.DataFrame({"Method": agg["Method"]})
    for k in ["f1_macro", "f1_minority", "auc_macro", "auc_minority", "ap_macro"]:
        prev[k] = agg.apply(lambda r: f(r, k), axis=1)
    print(prev.to_markdown(index=False))


if __name__ == "__main__":
    main()
