#!/usr/bin/env python3
"""
Statistical comparison of augmentation methods on a FIXED test set.

Per-seed results and a paired comparison between methods, with bootstrap
confidence intervals for the performance *differences*.

Inputs: the per-run test-set prediction dumps written by
`save_test_predictions.py` (which calls train_classifier.py --save_preds):

    <preds_dir>/preds_<method>_<strategy>_seed<SEED>.npz
        probs      float32 (N, C)   sigmoid outputs on the held-out test set
        labels     int8    (N, C)   ground truth (identical across all runs)
        classes    str     (C,)     class names
        threshold  float32          decision threshold used for binarisation

Two complementary analyses are produced:

1. Seed-level paired analysis (variability due to random initialisation).
   For each competing method we take the 3 paired per-seed differences
   (same seed -> same initialisation/shuffling) and run a paired t-test.
   With n_seeds = 3 this has very low power and is reported for completeness.

2. Patient-level paired bootstrap on the fixed test set (variability due to
   finite test-set sampling -- the dominant source of uncertainty here).
   The same bootstrap resample of test patients is applied to every method and
   every seed, so comparisons stay paired. Within each resample, the metric is
   averaged over seeds, and the difference (reference - competitor) is recorded.
   We report the percentile 95% CI of the difference and a two-sided bootstrap
   p-value, Holm-corrected across competing methods.

F1 is recomputed from bootstrap-resampled TP/FP/FN counts, which is exact and
identical to sklearn's `f1_score(..., zero_division=0)` on the resampled rows.

Usage
-----
python3 paired_bootstrap.py \
    --preds_dir ../../reproduction/preds \
    --strategy Target_Multiplier_1X --seeds 6,42,81 \
    --reference taser_ae --dataset MIMIC-IV \
    --out_dir outputs/mimic
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

# Paper display names (unknown methods fall back to a prettified key).
DISPLAY_NAMES = {
    "no_augmentation": "No Augmentation",
    "random_drop": "Random Drop",
    "random_swap": "Random Swap",
    "synonym_replacement": "Random Synonym Repl.",
    "synonym_insertion": "Random Synonym Insert.",
    "shuffle": "Shuffle",
    "composite_insertion": "Composite Insertion",
    "vae": "Conditional VAE",
    "gan": "Conditional GAN",
    "ishikawa": "Ishikawa et al.",
    "taser_ae": "TASER-AE",
}

# Order used in the paper tables; methods not listed are appended alphabetically.
METHOD_ORDER = [
    "no_augmentation", "random_drop", "random_swap", "synonym_replacement",
    "synonym_insertion", "shuffle", "composite_insertion", "vae", "gan",
    "ishikawa", "taser_ae",
]

CLASS_DISPLAY = {"GI_Bleed": "GI Bleed", "gibleed": "GI Bleed", "falls": "Falls",
                 "fractures": "Fractures", "stroke": "Stroke", "no": "No AE"}


def display_name(method):
    return DISPLAY_NAMES.get(method, method.replace("_", " ").title())


def load_runs(preds_dir, strategy, seeds):
    """-> dict[(method, seed)] = (probs, binarised preds), plus labels/classes."""
    pattern = os.path.join(preds_dir, f"preds_*_{strategy}_seed*.npz")
    runs, labels, classes = {}, None, None
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)[len("preds_"):-len(".npz")]
        base = base.replace(f"_{strategy}", "")
        m = re.match(r"(.+)_seed(\d+)$", base)
        if not m:
            continue
        method, seed = m.group(1), int(m.group(2))
        if seeds and seed not in seeds:
            continue
        d = np.load(path, allow_pickle=True)
        probs = d["probs"].astype(np.float32)
        y = d["labels"].astype(np.int8)
        cls = [str(c) for c in d["classes"]]
        thr = float(d["threshold"]) if "threshold" in d else 0.5
        if labels is None:
            labels, classes = y, cls
        else:
            if cls != classes:
                raise ValueError(f"class set mismatch in {path}: {cls} vs {classes}")
            if not np.array_equal(y, labels):
                raise ValueError(f"test labels differ in {path} -- not a fixed test set")
        runs[(method, seed)] = (probs >= thr).astype(np.int8)
    if labels is None:
        raise SystemExit(f"No prediction files matched {pattern}")
    return runs, labels, classes


def f1_from_counts(tp, fp, fn):
    """Elementwise F1 with the sklearn zero_division=0 convention."""
    denom = 2.0 * tp + fp + fn
    out = np.zeros_like(denom, dtype=np.float64)
    np.divide(2.0 * tp, denom, out=out, where=denom > 0)
    return out


def per_row_indicators(pred, labels):
    """TP/FP/FN indicator matrices, each (N, C) float32."""
    tp = (pred & labels).astype(np.float32)
    fp = (pred & (1 - labels)).astype(np.float32)
    fn = ((1 - pred) & labels).astype(np.float32)
    return tp, fp, fn


def metric_from_counts(tp, fp, fn, minority_idx, all_idx):
    """-> dict of metric name -> value(s); counts broadcast over leading axes."""
    f1 = f1_from_counts(tp, fp, fn)                       # (..., C)
    return {
        "minority_f1": f1[..., minority_idx].mean(axis=-1),
        "macro_f1": f1[..., all_idx].mean(axis=-1),
        "per_class_f1": f1,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--strategy", default="Target_Multiplier_1X")
    ap.add_argument("--seeds", default="6,42,81")
    ap.add_argument("--reference", default="taser_ae",
                    help="Method all others are compared against (TASER-AE).")
    ap.add_argument("--exclude", default="",
                    help="Comma-separated methods to drop from the comparison set.")
    ap.add_argument("--dataset", default="MIMIC-IV", help="Label used in captions.")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--boot_seed", type=int, default=20250913)
    ap.add_argument("--chunk", type=int, default=250,
                    help="Bootstrap replicates per block (memory/speed trade-off).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tex_color", default="",
                    help="Colour for the generated LaTeX tables. "
                         "Empty string disables colouring.")
    args = ap.parse_args()

    seeds = sorted({int(s) for s in args.seeds.split(",") if s.strip()})
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    os.makedirs(args.out_dir, exist_ok=True)

    runs, labels, classes = load_runs(args.preds_dir, args.strategy, seeds)
    n_rows, n_cls = labels.shape
    minority_idx = [i for i, c in enumerate(classes) if c.lower() != "no"]
    all_idx = list(range(n_cls))

    methods = sorted({m for m, _ in runs})
    have = {m: sorted(s for mm, s in runs if mm == m) for m in methods}
    methods = [m for m in methods if have[m] == seeds]
    dropped = {m: have[m] for m in have if have[m] != seeds}
    if dropped:
        print(f"[warn] skipping methods without all seeds {seeds}: {dropped}")
    if args.reference not in methods:
        raise SystemExit(f"reference method '{args.reference}' missing seeds {seeds}")
    methods.sort(key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else 99, m))
    competitors = [m for m in methods if m != args.reference and m not in exclude]

    print(f"Dataset          : {args.dataset}")
    print(f"Test rows        : {n_rows}   classes: {classes}")
    print(f"Methods          : {len(methods)} x {len(seeds)} seeds")
    print(f"Reference        : {args.reference} ({display_name(args.reference)})")
    print(f"Competitors      : {len(competitors)}")
    print(f"Bootstrap        : B={args.n_boot}, paired over test patients\n")

    # ---------------------------------------------------------------- point estimates
    # Stack per-row indicators: (M, S, N, C) would be large, so keep (M*S, C, N).
    keys = [(m, s) for m in methods for s in seeds]
    TP = np.empty((len(keys), n_cls, n_rows), dtype=np.float32)
    FP = np.empty_like(TP)
    FN = np.empty_like(TP)
    for k, key in enumerate(keys):
        tp, fp, fn = per_row_indicators(runs[key], labels)
        TP[k], FP[k], FN[k] = tp.T, fp.T, fn.T

    obs = metric_from_counts(TP.sum(axis=2), FP.sum(axis=2), FN.sum(axis=2),
                             minority_idx, all_idx)
    per_seed = []
    for k, (m, s) in enumerate(keys):
        row = {"Method": display_name(m), "method_key": m, "Seed": s,
               "minority_f1": obs["minority_f1"][k], "macro_f1": obs["macro_f1"][k]}
        for ci, c in enumerate(classes):
            row[f"f1_{c}"] = obs["per_class_f1"][k, ci]
        per_seed.append(row)
    per_seed = pd.DataFrame(per_seed)
    per_seed.to_csv(os.path.join(args.out_dir, "per_seed_metrics.csv"), index=False)

    metric_cols = ["minority_f1", "macro_f1"] + [f"f1_{c}" for c in classes]
    summary = (per_seed.groupby(["method_key", "Method"], sort=False)[metric_cols]
               .agg(["mean", "std"]))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(args.out_dir, "per_seed_summary.csv"), index=False)

    # ------------------------------------------- predicted-positive / TP-FP counts
    # For near-zero-recall cases (e.g. CGAN on Fractures) the
    # precision denominator (TP+FP) can be a handful of predictions, so the
    # per-seed spread is dominated by denominator instability. These are the raw
    # counts behind every precision/recall/F1 in the paper.
    count_rows = []
    for m, sd in keys:
        pred = runs[(m, sd)]
        for ci, c in enumerate(classes):
            p_c, y_c = pred[:, ci].astype(int), labels[:, ci].astype(int)
            tp = int((p_c & y_c).sum())
            fp = int((p_c & (1 - y_c)).sum())
            fn = int(((1 - p_c) & y_c).sum())
            tn = int(((1 - p_c) & (1 - y_c)).sum())
            pp, ap_ = tp + fp, tp + fn
            prec = tp / pp if pp else 0.0        # sklearn zero_division=0
            rec = tp / ap_ if ap_ else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            count_rows.append({
                "Method": display_name(m), "method_key": m, "Seed": sd,
                "Class": CLASS_DISPLAY.get(c, c), "class_key": c,
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "predicted_positives": pp, "actual_positives": ap_,
                "precision": prec, "recall": rec, "f1": f1,
                "precision_undefined_zero_division": bool(pp == 0),
            })
    counts = pd.DataFrame(count_rows)
    counts.to_csv(os.path.join(args.out_dir, "prediction_counts.csv"), index=False)

    csum = (counts.groupby(["method_key", "Method", "class_key", "Class"], sort=False)
            .agg(predicted_positives_mean=("predicted_positives", "mean"),
                 predicted_positives_min=("predicted_positives", "min"),
                 predicted_positives_max=("predicted_positives", "max"),
                 TP_mean=("TP", "mean"), FP_mean=("FP", "mean"),
                 precision_mean=("precision", "mean"), precision_sd=("precision", "std"),
                 recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
                 seeds_with_no_positive_prediction=("precision_undefined_zero_division", "sum"))
            .reset_index())
    csum.to_csv(os.path.join(args.out_dir, "prediction_counts_summary.csv"), index=False)

    # ------------------------------------------------- seed-level paired t-tests
    wide = per_seed.pivot(index="Seed", columns="method_key", values="minority_f1")
    ref_seed_vals = wide[args.reference].to_numpy()
    seed_rows = []
    if len(seeds) < 2:
        print("[warn] fewer than 2 seeds -- skipping the seed-level paired t-tests.")
    for m in (competitors if len(seeds) >= 2 else []):
        diffs = ref_seed_vals - wide[m].to_numpy()
        t, p = stats.ttest_rel(ref_seed_vals, wide[m].to_numpy())
        sd = diffs.std(ddof=1)
        se = sd / np.sqrt(len(diffs))
        tcrit = stats.t.ppf(0.975, len(diffs) - 1)
        seed_rows.append({
            "Method": display_name(m), "method_key": m,
            "ref_mean": ref_seed_vals.mean(), "cmp_mean": wide[m].mean(),
            "mean_diff": diffs.mean(), "sd_diff": sd,
            "ci_low": diffs.mean() - tcrit * se, "ci_high": diffs.mean() + tcrit * se,
            "t_stat": t, "p_value": p, "n_seeds": len(diffs),
        })
    seed_df = pd.DataFrame(seed_rows, columns=[
        "Method", "method_key", "ref_mean", "cmp_mean", "mean_diff", "sd_diff",
        "ci_low", "ci_high", "t_stat", "p_value", "n_seeds"])
    seed_df["p_holm"] = holm(seed_df["p_value"].to_numpy())
    seed_df.to_csv(os.path.join(args.out_dir, "seed_paired_ttest.csv"), index=False)

    # ------------------------------------------------------- paired bootstrap
    rng = np.random.default_rng(args.boot_seed)
    n_m, n_s = len(methods), len(seeds)
    m_index = {m: i for i, m in enumerate(methods)}
    boot_minority = np.empty((args.n_boot, n_m), dtype=np.float64)
    boot_macro = np.empty((args.n_boot, n_m), dtype=np.float64)
    boot_class = np.empty((args.n_boot, n_m, n_cls), dtype=np.float64)
    # per-seed (not averaged over seeds) minority F1, for the per-seed CIs
    boot_minority_seed = np.empty((args.n_boot, n_m, n_s), dtype=np.float64)

    probs = np.full(n_rows, 1.0 / n_rows)
    done = 0
    while done < args.n_boot:
        b = min(args.chunk, args.n_boot - done)
        # Multiplicity vector of a with-replacement resample of all N test rows.
        W = rng.multinomial(n_rows, probs, size=b).astype(np.float32)   # (b, N)
        tp = np.einsum("bn,kcn->bkc", W, TP)
        fp = np.einsum("bn,kcn->bkc", W, FP)
        fn = np.einsum("bn,kcn->bkc", W, FN)
        f1 = f1_from_counts(tp, fp, fn).reshape(b, n_m, n_s, n_cls)
        boot_minority_seed[done:done + b] = f1[:, :, :, minority_idx].mean(axis=3)
        f1 = f1.mean(axis=2)                                    # average over seeds
        boot_class[done:done + b] = f1
        boot_minority[done:done + b] = f1[:, :, minority_idx].mean(axis=2)
        boot_macro[done:done + b] = f1[:, :, all_idx].mean(axis=2)
        done += b
        print(f"  bootstrap {done}/{args.n_boot}", end="\r", flush=True)
    print()

    obs_seedavg = {
        "minority_f1": obs["minority_f1"].reshape(n_m, n_s).mean(axis=1),
        "macro_f1": obs["macro_f1"].reshape(n_m, n_s).mean(axis=1),
        "per_class_f1": obs["per_class_f1"].reshape(n_m, n_s, n_cls).mean(axis=1),
    }

    boot_tables = {}
    for metric, boot_mat, obs_vec in (
        ("minority_f1", boot_minority, obs_seedavg["minority_f1"]),
        ("macro_f1", boot_macro, obs_seedavg["macro_f1"]),
    ):
        rows = []
        ri = m_index[args.reference]
        for m in competitors:
            mi = m_index[m]
            d_boot = boot_mat[:, ri] - boot_mat[:, mi]
            d_obs = obs_vec[ri] - obs_vec[mi]
            row = bootstrap_row(m, d_obs, d_boot, obs_vec[ri], obs_vec[mi],
                                boot_mat[:, ri], boot_mat[:, mi], args.n_boot)
            if metric == "minority_f1":
                # Per-seed verdict: in how many individual seeds does the paired
                # bootstrap CI of the difference exclude zero in the reference's favour?
                favour = adverse = 0
                for si in range(n_s):
                    ds = boot_minority_seed[:, ri, si] - boot_minority_seed[:, mi, si]
                    lo_s, hi_s = np.percentile(ds, [2.5, 97.5])
                    if lo_s > 0:
                        favour += 1
                    elif hi_s < 0:
                        adverse += 1
                row["seeds_ci_favour_ref"] = favour
                row["seeds_ci_favour_cmp"] = adverse
                row["n_seeds"] = n_s
            rows.append(row)
        df = pd.DataFrame(rows)
        df["p_holm"] = holm(df["p_boot"].to_numpy())
        df.to_csv(os.path.join(args.out_dir, f"bootstrap_{metric}.csv"), index=False)
        boot_tables[metric] = df

    # per-class bootstrap differences (secondary)
    rows = []
    ri = m_index[args.reference]
    for ci, c in enumerate(classes):
        for m in competitors:
            mi = m_index[m]
            d_boot = boot_class[:, ri, ci] - boot_class[:, mi, ci]
            d_obs = obs_seedavg["per_class_f1"][ri, ci] - obs_seedavg["per_class_f1"][mi, ci]
            r = bootstrap_row(m, d_obs, d_boot,
                              obs_seedavg["per_class_f1"][ri, ci],
                              obs_seedavg["per_class_f1"][mi, ci],
                              boot_class[:, ri, ci], boot_class[:, mi, ci], args.n_boot)
            r["Class"] = CLASS_DISPLAY.get(c, c)
            rows.append(r)
    per_class_df = pd.DataFrame(rows)
    per_class_df["p_holm"] = (per_class_df.groupby("Class")["p_boot"]
                              .transform(lambda s: holm(s.to_numpy())))
    per_class_df.to_csv(os.path.join(args.out_dir, "bootstrap_per_class_f1.csv"), index=False)

    # reference CI for its own score (not a difference)
    ref_ci = np.percentile(boot_minority[:, ri], [2.5, 97.5])
    # P(reference is the best method) over the comparison set, per resample
    pool = [m_index[m] for m in competitors] + [ri]
    pool_scores = boot_minority[:, pool]
    p_best = float(np.mean(pool_scores.argmax(axis=1) == len(pool) - 1))
    edit_like = [m for m in competitors
                 if m not in ("no_augmentation", "gan", "vae")]
    if edit_like:
        pool_e = [m_index[m] for m in edit_like] + [ri]
        p_best_edit = float(np.mean(boot_minority[:, pool_e].argmax(axis=1) == len(pool_e) - 1))
    else:
        p_best_edit = float("nan")
    meta = {
        "dataset": args.dataset, "preds_dir": os.path.abspath(args.preds_dir),
        "strategy": args.strategy, "seeds": seeds, "n_test_rows": int(n_rows),
        "classes": classes, "reference": args.reference, "n_boot": args.n_boot,
        "boot_seed": args.boot_seed,
        "zero_division_convention": (
            "precision, recall and F1 are computed per class at threshold 0.5 with "
            "sklearn's zero_division=0: when a run predicts no positives for a class "
            "(TP+FP=0) its precision is recorded as 0 rather than undefined, and when a "
            "class has no positives in the test set (TP+FN=0) recall is recorded as 0. "
            "Per-seed TP/FP/FN counts are in prediction_counts.csv."),
        "reference_minority_f1": float(obs_seedavg["minority_f1"][ri]),
        "reference_minority_f1_ci": [float(ref_ci[0]), float(ref_ci[1])],
        "p_reference_ranks_first": p_best,
        "p_reference_ranks_first_vs_edit_based": p_best_edit,
        "edit_based_methods": edit_like,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    write_latex(args, classes, per_seed, summary, boot_tables, per_class_df,
                meta, methods, competitors)
    report(args, meta, seed_df, boot_tables)


def bootstrap_row(method, d_obs, d_boot, ref_obs, cmp_obs, ref_boot, cmp_boot, n_boot):
    lo, hi = np.percentile(d_boot, [2.5, 97.5])
    # Two-sided bootstrap p-value by CI inversion; floored at 1/B (resolution limit).
    frac_le = float(np.mean(d_boot <= 0.0))
    frac_ge = float(np.mean(d_boot >= 0.0))
    p = min(1.0, 2.0 * min(frac_le, frac_ge))
    p = max(p, 1.0 / n_boot)
    return {
        "Method": display_name(method), "method_key": method,
        "ref_score": ref_obs, "cmp_score": cmp_obs, "diff": d_obs,
        "ci_low": lo, "ci_high": hi, "p_boot": p,
        "p_floor": 1.0 / n_boot,
        "prob_ref_better": float(np.mean(d_boot > 0.0)),
        "ref_ci_low": float(np.percentile(ref_boot, 2.5)),
        "ref_ci_high": float(np.percentile(ref_boot, 97.5)),
        "cmp_ci_low": float(np.percentile(cmp_boot, 2.5)),
        "cmp_ci_high": float(np.percentile(cmp_boot, 97.5)),
    }


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values. NaN in -> NaN out."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0 or np.isnan(p).all():
        return np.full(n, np.nan)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        if np.isnan(val):
            adj[idx] = np.nan
            continue
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def fmt_p(p, floor):
    """Report p conservatively: below the bootstrap's 1/B resolution we only claim
    the conventional '< 0.001' rather than the literal 1/B value."""
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def write_latex(args, classes, per_seed, summary, boot_tables, per_class_df,
                meta, methods, competitors):
    """Emit ready-to-paste supplementary tables."""
    ds = args.dataset
    tag = ds.lower().replace("-", "").replace(" ", "")
    colour_cmd = [rf"    \color{{{args.tex_color}}}"] if args.tex_color else []
    out = []

    # --- per-seed table (minority + macro F1) ---
    n_seed = len(meta["seeds"])
    block = n_seed + 1                       # seed columns + mean/SD column
    ncol = 2 * block + 1
    lines = [
        r"\begin{table}[htbp]", r"    \centering", *colour_cmd,
        rf"    \caption{{Per-seed Minority-average and Macro-average F1 on the {ds} test set "
        rf"(seeds {', '.join(str(s) for s in meta['seeds'])}). "
        rf"All runs share the same fixed stratified test split.}}",
        rf"    \label{{tab:perseed_{tag}}}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{|l|" + "c|" * (ncol - 1) + "}", r"    \hline",
        r"    \multirow{2}{*}{\textbf{Method}} & "
        rf"\multicolumn{{{block}}}{{c|}}{{\textbf{{Minority Average F1}}}} & "
        rf"\multicolumn{{{block}}}{{c|}}{{\textbf{{Macro Average F1}}}} \\ \cline{{2-{ncol}}}",
        r"    & Seed " + " & Seed ".join(str(s) for s in meta["seeds"]) + r" & Mean$\pm$SD"
        r" & Seed " + " & Seed ".join(str(s) for s in meta["seeds"]) + r" & Mean$\pm$SD \\ \hline",
    ]
    for _, r in summary.iterrows():
        k = r["method_key"]
        sub = per_seed[per_seed["method_key"] == k].set_index("Seed")
        cells = [f"{sub.loc[s, 'minority_f1']:.4f}" for s in meta["seeds"]]
        cells.append(f"{r['minority_f1_mean']:.4f}{{\\scriptsize$\\pm${r['minority_f1_std']:.4f}}}")
        cells += [f"{sub.loc[s, 'macro_f1']:.4f}" for s in meta["seeds"]]
        cells.append(f"{r['macro_f1_mean']:.4f}{{\\scriptsize$\\pm${r['macro_f1_std']:.4f}}}")
        name = r["Method"]
        if k == args.reference:
            name = rf"\textbf{{{name}}}"
        lines.append("    " + name + " & " + " & ".join(cells) + r" \\ \hline")
    lines += [r"    \end{tabular}%", r"    }", r"\end{table}", ""]
    out.append("\n".join(lines))

    # --- per-seed per-class F1 table ---
    cls_cols = [c for c in classes if c.lower() != "no"] + [c for c in classes if c.lower() == "no"]
    lines = [
        r"\begin{table}[htbp]", r"    \centering", *colour_cmd,
        rf"    \caption{{Per-seed per-class F1 on the {ds} test set. Each row is a single "
        rf"training run; all runs share the same fixed stratified test split.}}",
        rf"    \label{{tab:perseedclass_{tag}}}",
        r"    \begin{tabular}{|l|c|" + "c|" * len(cls_cols) + "}", r"    \hline",
        r"    \textbf{Method} & \textbf{Seed} & " +
        " & ".join(rf"\textbf{{{CLASS_DISPLAY.get(c, c)}}}" for c in cls_cols) + r" \\ \hline",
    ]
    for _, r in summary.iterrows():
        k = r["method_key"]
        sub = per_seed[per_seed["method_key"] == k].set_index("Seed")
        name = rf"\textbf{{{r['Method']}}}" if k == args.reference else r["Method"]
        for i, sd in enumerate(meta["seeds"]):
            label = rf"\multirow{{{len(meta['seeds'])}}}{{*}}{{{name}}}" if i == 0 else ""
            cells = " & ".join(f"{sub.loc[sd, f'f1_{c}']:.4f}" for c in cls_cols)
            end = r" \\ \hline" if i == len(meta["seeds"]) - 1 else r" \\ \cline{2-" + str(2 + len(cls_cols)) + "}"
            lines.append(f"    {label} & {sd} & {cells}" + end)
    lines += [r"    \end{tabular}", r"\end{table}", ""]
    out.append("\n".join(lines))

    # --- bootstrap difference table ---
    df = boot_tables["minority_f1"]
    ref_name = display_name(args.reference)
    lines = [
        r"\begin{table}[htbp]", r"    \centering", *colour_cmd,
        rf"    \caption{{Paired bootstrap comparison of {ref_name} against each competing "
        rf"augmentation method on the fixed {ds} test set "
        rf"($B={meta['n_boot']}$ resamples of the {meta['n_test_rows']} test patients, "
        rf"metric averaged over the {len(meta['seeds'])} seeds within each resample). "
        r"$\Delta$ is the difference in Minority-average F1 "
        rf"({ref_name} $-$ competitor); positive favours {ref_name}. "
        r"$p_{\text{Holm}}$ is Holm-corrected across the "
        rf"{len(competitors)} comparisons. \emph{{Seeds}} counts the individual seeds in "
        rf"which the per-seed bootstrap CI of $\Delta$ excludes zero in favour of "
        rf"{ref_name}.}}",
        rf"    \label{{tab:bootstrap_{tag}}}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{|l|c|c|c|c|c|c|}", r"    \hline",
        r"    \textbf{Method} & \textbf{Minority F1} & \textbf{$\Delta$} & "
        r"\textbf{95\% CI of $\Delta$} & \textbf{$p$} & \textbf{$p_{\text{Holm}}$} & "
        r"\textbf{Seeds} \\ \hline",
        rf"    \textbf{{{ref_name}}} & {meta['reference_minority_f1']:.4f} "
        rf"[{meta['reference_minority_f1_ci'][0]:.4f}, {meta['reference_minority_f1_ci'][1]:.4f}] "
        r"& --- & --- & --- & --- & --- \\ \hline",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"    {r['Method']} & {r['cmp_score']:.4f} & {r['diff']:+.4f} & "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] & "
            f"{fmt_p(r['p_boot'], r['p_floor'])} & {fmt_p(r['p_holm'], r['p_floor'])} & "
            f"{int(r['seeds_ci_favour_ref'])}/{int(r['n_seeds'])} \\\\ \\hline")
    lines += [r"    \end{tabular}%", r"    }", r"\end{table}", ""]
    out.append("\n".join(lines))

    # --- seed-level paired t-test table ---
    seed_df = pd.read_csv(os.path.join(args.out_dir, "seed_paired_ttest.csv"))
    if seed_df.empty:
        path = os.path.join(args.out_dir, "tables.tex")
        with open(path, "w") as f:
            f.write("\n".join(out))
        print(f"LaTeX tables -> {path} (seed-level table omitted: <2 seeds)")
        return
    lines = [
        r"\begin{table}[htbp]", r"    \centering", *colour_cmd,
        rf"    \caption{{Seed-level paired $t$-tests on {ds} (Minority-average F1, "
        rf"$n={len(meta['seeds'])}$ paired seeds). Each seed fixes the initialisation and "
        rf"batch order for every method, so the comparison is paired. With {len(meta['seeds'])} seeds these "
        r"tests have very low power and are reported for completeness alongside the "
        r"patient-level bootstrap.}",
        rf"    \label{{tab:seedtest_{tag}}}",
        r"    \begin{tabular}{|l|c|c|c|c|}", r"    \hline",
        r"    \textbf{Method} & \textbf{Mean $\Delta$} & \textbf{SD of $\Delta$} & "
        r"\textbf{95\% CI of $\Delta$} & \textbf{$p$} \\ \hline",
    ]
    for _, r in seed_df.iterrows():
        lines.append(
            f"    {r['Method']} & {r['mean_diff']:+.4f} & {r['sd_diff']:.4f} & "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] & "
            f"{'$<$0.001' if r['p_value'] < 0.001 else format(r['p_value'], '.3f')} \\\\ \\hline")
    lines += [r"    \end{tabular}", r"\end{table}", ""]
    out.append("\n".join(lines))

    path = os.path.join(args.out_dir, "tables.tex")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"LaTeX tables -> {path}")


def report(args, meta, seed_df, boot_tables):
    df = boot_tables["minority_f1"]
    print("\n=== Minority-average F1: paired bootstrap on the fixed test set ===")
    show = df[["Method", "cmp_score", "diff", "ci_low", "ci_high", "p_boot", "p_holm",
               "seeds_ci_favour_ref"]].copy()
    show.columns = ["Method", "Minority F1", "Delta", "CI low", "CI high", "p", "p (Holm)",
                    "seeds CI>0"]
    print(show.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nReference {display_name(args.reference)}: "
          f"{meta['reference_minority_f1']:.4f} "
          f"[{meta['reference_minority_f1_ci'][0]:.4f}, {meta['reference_minority_f1_ci'][1]:.4f}]")
    n_sig = int((df["p_holm"] < 0.05).sum())
    print(f"Significant after Holm correction: {n_sig}/{len(df)} comparisons")
    print(f"P(reference ranks first among all methods)        = {meta['p_reference_ranks_first']:.4f}")
    print(f"P(reference ranks first among edit-based methods) = "
          f"{meta['p_reference_ranks_first_vs_edit_based']:.4f}")
    if seed_df.empty:
        print(f"\nOutputs written to {os.path.abspath(args.out_dir)}")
        return
    print(f"\n=== Minority-average F1: seed-level paired t-tests "
          f"(n={int(seed_df['n_seeds'].iloc[0])}) ===")
    s = seed_df[["Method", "mean_diff", "sd_diff", "p_value", "p_holm"]].copy()
    s.columns = ["Method", "Mean Delta", "SD", "p", "p (Holm)"]
    print(s.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nOutputs written to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
