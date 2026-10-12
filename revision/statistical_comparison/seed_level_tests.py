#!/usr/bin/env python3
"""
Per-seed results and seed-level paired tests, computed from metrics files that are
already on disk -- no retraining and no saved predictions required.

The patient-level bootstrap needs per-patient predictions and lives in
paired_bootstrap.py; this script needs only log/*.json.

Because a given seed is applied identically to every method, the per-seed
differences are paired, and a paired t-test across seeds is the appropriate
comparison. With three seeds it has low power, which is reported alongside.

    python3 seed_level_tests.py --log_dir ../../reproduction/log \
        --strategy Target_Multiplier_1X --seeds 6,42,81 \
        --reference taser_ae --dataset MIMIC-IV --out_dir outputs
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

DISPLAY = {"no_augmentation": "No Augmentation", "random_drop": "Random Drop",
           "random_swap": "Random Swap", "synonym_replacement": "Random Synonym Repl.",
           "synonym_insertion": "Random Synonym Insert.", "shuffle": "Shuffle",
           "composite_insertion": "Composite Insertion", "vae": "Conditional VAE",
           "gan": "Conditional GAN", "ishikawa": "Ishikawa et al.",
           "taser_ae": "TASER-AE"}
ORDER = ["no_augmentation", "random_drop", "random_swap", "synonym_replacement",
         "synonym_insertion", "shuffle", "composite_insertion", "vae", "gan",
         "ishikawa", "taser_ae"]


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    adj, running = np.empty(n), 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--strategy", default="Target_Multiplier_1X")
    ap.add_argument("--strategy_override", default="",
                    help="method=strategy pairs for runs logged under a different tag, "
                         "e.g. gan=Target_Multiplier_1X,vae=Target_Multiplier_1X")
    ap.add_argument("--seeds", default="6,42,81")
    ap.add_argument("--reference", default="taser_ae")
    ap.add_argument("--exclude", default="baseline_decision_tree,"
                                         "baseline_random_forest,baseline_gradient_boosting")
    ap.add_argument("--dataset", default="MIMIC-IV")
    ap.add_argument("--tex_color", default="")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    seeds = sorted({int(s) for s in args.seeds.split(",") if s.strip()})
    exclude = {m.strip() for m in args.exclude.split(",") if m.strip()}
    override = {}
    for kv in args.strategy_override.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            override[k.strip()] = v.strip()
    os.makedirs(args.out_dir, exist_ok=True)

    tags = {args.strategy} | set(override.values())
    runs = {}
    for tag in tags:
        for f in glob.glob(os.path.join(args.log_dir, f"metrics_*_{tag}_seed*.json")):
            base = os.path.basename(f).replace("metrics_", "").replace(f"_{tag}", "")
            m = re.match(r"(.+)_seed(\d+)\.json$", base)
            if not m:
                continue
            method, seed = m.group(1), int(m.group(2))
            if seed not in seeds or method in exclude:
                continue
            if override.get(method, args.strategy) != tag:
                continue
            runs.setdefault(method, {})[seed] = json.load(open(f))

    rows = []
    for method, by_seed in runs.items():
        if sorted(by_seed) != seeds:
            print(f"[warn] {method}: seeds {sorted(by_seed)} != {seeds}; skipped")
            continue
        for seed, d in sorted(by_seed.items()):
            classes = sorted({k.split("_", 1)[1] for k in d if k.startswith("f1_")}
                             - {"macro", "micro"})
            minority = [d[f"f1_{c}"] for c in classes if c.lower() != "no"]
            row = {"Method": DISPLAY.get(method, method), "method_key": method, "Seed": seed,
                   "minority_f1": float(np.mean(minority)),
                   "macro_f1": float(d.get("f1_macro", np.nan))}
            for c in classes:
                row[f"f1_{c}"] = d[f"f1_{c}"]
            rows.append(row)

    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        raise SystemExit("no runs found -- check --log_dir / --strategy / --seeds")
    present = [m for m in ORDER if m in set(per_seed.method_key)]
    present += sorted(set(per_seed.method_key) - set(present))
    per_seed["__o"] = per_seed.method_key.map({m: i for i, m in enumerate(present)})
    per_seed = per_seed.sort_values(["__o", "Seed"]).drop(columns="__o")
    per_seed.to_csv(os.path.join(args.out_dir, "per_seed_metrics_from_log.csv"), index=False)

    metric_cols = [c for c in per_seed.columns if c not in ("Method", "method_key", "Seed")]
    summary = per_seed.groupby(["method_key", "Method"], sort=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(args.out_dir, "per_seed_summary_from_log.csv"), index=False)

    if args.reference not in set(per_seed.method_key):
        raise SystemExit(f"reference {args.reference!r} not found")
    wide = per_seed.pivot(index="Seed", columns="method_key", values="minority_f1")
    ref = wide[args.reference].to_numpy()
    trows = []
    for m in present:
        if m == args.reference:
            continue
        diffs = ref - wide[m].to_numpy()
        t, p = stats.ttest_rel(ref, wide[m].to_numpy())
        sd = diffs.std(ddof=1)
        se = sd / np.sqrt(len(diffs))
        crit = stats.t.ppf(0.975, len(diffs) - 1)
        trows.append({"Method": DISPLAY.get(m, m), "method_key": m,
                      "ref_mean": ref.mean(), "cmp_mean": wide[m].mean(),
                      "mean_diff": diffs.mean(), "sd_diff": sd,
                      "ci_low": diffs.mean() - crit * se, "ci_high": diffs.mean() + crit * se,
                      "t_stat": t, "p_value": p, "n_seeds": len(diffs)})
    tt = pd.DataFrame(trows)
    tt["p_holm"] = holm(tt["p_value"].to_numpy())
    tt.to_csv(os.path.join(args.out_dir, "seed_paired_ttest_from_log.csv"), index=False)

    write_latex(args, per_seed, summary, tt, seeds)
    print(f"\n=== {args.dataset}: per-seed Minority-average F1 ===")
    print(per_seed.pivot(index="Method", columns="Seed", values="minority_f1")
          .reindex([DISPLAY.get(m, m) for m in present]).round(4).to_string())
    print(f"\n=== {args.dataset}: seed-level paired t-tests (n={len(seeds)}), "
          f"{DISPLAY.get(args.reference, args.reference)} vs each method ===")
    print(tt[["Method", "cmp_mean", "mean_diff", "sd_diff", "p_value", "p_holm"]]
          .round(4).to_string(index=False))
    sig = int((tt.p_holm < 0.05).sum())
    print(f"\nsignificant after Holm correction: {sig}/{len(tt)}")
    print(f"outputs -> {os.path.abspath(args.out_dir)}")


def write_latex(args, per_seed, summary, tt, seeds):
    ds, tag = args.dataset, args.dataset.lower().replace("-", "").replace(" ", "")
    col = [rf"    \color{{{args.tex_color}}}"] if args.tex_color else []
    block, ncol = len(seeds) + 1, 2 * (len(seeds) + 1) + 1
    out = []

    lines = [r"\begin{table}[htbp]", r"    \centering", *col,
             rf"    \caption{{Per-seed Minority-average and Macro-average F1 on the {ds} test "
             rf"set for each augmentation method (seeds {', '.join(map(str, seeds))}). All "
             rf"methods share the same fixed stratified test split, and a given seed is applied "
             rf"identically to every method.}}",
             rf"    \label{{tab:perseed_{tag}}}",
             r"    \resizebox{\textwidth}{!}{%",
             r"    \begin{tabular}{|l|" + "c|" * (ncol - 1) + "}", r"    \hline",
             r"    \multirow{2}{*}{\textbf{Method}} & "
             rf"\multicolumn{{{block}}}{{c|}}{{\textbf{{Minority Average F1}}}} & "
             rf"\multicolumn{{{block}}}{{c|}}{{\textbf{{Macro Average F1}}}} \\ \cline{{2-{ncol}}}",
             r"    & Seed " + " & Seed ".join(map(str, seeds)) + r" & Mean$\pm$SD & Seed "
             + " & Seed ".join(map(str, seeds)) + r" & Mean$\pm$SD \\ \hline"]
    for _, r in summary.iterrows():
        sub = per_seed[per_seed.method_key == r.method_key].set_index("Seed")
        cells = [f"{sub.loc[s, 'minority_f1']:.4f}" for s in seeds]
        cells.append(f"{r.minority_f1_mean:.4f}{{\\scriptsize$\\pm${r.minority_f1_std:.4f}}}")
        cells += [f"{sub.loc[s, 'macro_f1']:.4f}" for s in seeds]
        cells.append(f"{r.macro_f1_mean:.4f}{{\\scriptsize$\\pm${r.macro_f1_std:.4f}}}")
        name = rf"\textbf{{{r.Method}}}" if r.method_key == args.reference else r.Method
        lines.append("    " + name + " & " + " & ".join(cells) + r" \\ \hline")
    lines += [r"    \end{tabular}%", r"    }", r"\end{table}", ""]
    out.append("\n".join(lines))

    ref_name = DISPLAY.get(args.reference, args.reference)
    lines = [r"\begin{table}[htbp]", r"    \centering", *col,
             rf"    \caption{{Seed-level paired $t$-tests on {ds}: difference in "
             rf"Minority-average F1 between {ref_name} and each competing method "
             rf"($\Delta = $ {ref_name} $-$ competitor), over {len(seeds)} paired seeds. "
             r"Each seed fixes the initialisation and batch ordering for every method, so the "
             r"comparison is paired. $p_{\text{Holm}}$ is corrected across the "
             rf"{len(tt)} comparisons. With {len(seeds)} seeds these tests have low power, so a "
             r"non-significant result is not evidence that two methods are equivalent.}",
             rf"    \label{{tab:seedtest_{tag}}}",
             r"    \resizebox{\textwidth}{!}{%",
             r"    \begin{tabular}{|l|c|c|c|c|c|}", r"    \hline",
             r"    \textbf{Method} & \textbf{Minority F1} & \textbf{$\Delta$} & "
             r"\textbf{SD of $\Delta$} & \textbf{$p$} & \textbf{$p_{\text{Holm}}$} \\ \hline"]
    for _, r in tt.iterrows():
        fp = lambda v: "$<$0.001" if v < 0.001 else f"{v:.3f}"
        lines.append(f"    {r.Method} & {r.cmp_mean:.4f} & {r.mean_diff:+.4f} & "
                     f"{r.sd_diff:.4f} & {fp(r.p_value)} & {fp(r.p_holm)} \\\\ \\hline")
    lines += [r"    \end{tabular}%", r"    }", r"\end{table}", ""]
    out.append("\n".join(lines))

    path = os.path.join(args.out_dir, "tables_seed_level.tex")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"LaTeX tables -> {path}")


if __name__ == "__main__":
    main()
