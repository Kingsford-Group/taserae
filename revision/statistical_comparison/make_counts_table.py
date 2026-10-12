#!/usr/bin/env python3
"""
Build the per-seed TP / FP / predicted-positive table,
in the same layout as the existing per-class tables in the manuscript
(two class panels, \\resizebox'd tabular, one \\hline per row).

Input:  prediction_counts.csv written by paired_bootstrap.py
Output: a LaTeX table (optionally coloured with --color).

    python3 make_counts_table.py --counts outputs/prediction_counts.csv \
        --dataset MIMIC-IV --out outputs/table_counts_mimiciv.tex
"""

import argparse
import os

import pandas as pd

# Non-augmented and generative baselines (low-recall cases), plus TASER-AE for reference.
DEFAULT_METHODS = ["no_augmentation", "vae", "gan", "taser_ae"]


def panel(df, classes, methods, seeds, ncol=7):
    lines = [rf"    \multicolumn{{{ncol}}}{{|c|}}{{\textbf{{{' and '.join(classes)}}}}} \\ \hline",
             r"    \textbf{Method (seed)} & " +
             " & ".join(rf"\multicolumn{{3}}{{c|}}{{\textbf{{{c}}}}}" for c in classes) +
             rf" \\ \cline{{2-{ncol}}}",
             r"     & " + " & ".join(["TP & FP & Pred.\\ pos."] * len(classes)) + r" \\ \hline"]
    for mk in methods:
        sub = df[df.method_key == mk]
        if sub.empty:
            continue
        name = sub["Method"].iloc[0]
        for sd in seeds:
            cells = []
            for c in classes:
                r = sub[(sub.Class == c) & (sub.Seed == sd)]
                if r.empty:
                    cells += ["--", "--", "--"]
                    continue
                r = r.iloc[0]
                pp = int(r.predicted_positives)
                cells += [str(int(r.TP)), str(int(r.FP)),
                          (rf"\textbf{{{pp}}}" if pp <= 10 else str(pp))]
            lines.append(f"    {name} (seed {sd}) & " + " & ".join(cells) + r" \\ \hline")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--dataset", default="MIMIC-IV")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--color", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.counts)
    df = df[df.class_key.str.lower() != "no"]          # adverse-event classes only
    seeds = sorted(df.Seed.unique())
    order = [m for m in args.methods.split(",") if m in set(df.method_key)]
    classes = list(dict.fromkeys(df.Class))            # preserve class order
    pairs = [classes[i:i + 2] for i in range(0, len(classes), 2)]
    first = df.iloc[0]
    n_test = (int(first.TP + first.FP + first.FN + first.TN)
              if "TN" in df.columns else None)
    pos = (df.drop_duplicates(["class_key"])[["Class", "actual_positives"]]
           .astype({"actual_positives": int}))
    pos_tex = ", ".join(f"{int(r.actual_positives):,} {r.Class}".replace(",", "{,}")
                        for _, r in pos.iterrows())
    tag = args.dataset.lower().replace("-", "").replace(" ", "")
    out = [r"\begin{table}[htbp]", r"    \centering"]
    if args.color:
        out.append(rf"    \color{{{args.color}}}")
    out += [
        rf"    \caption{{Per-seed true positives (TP), false positives (FP) and total "
        rf"predicted positives (TP\,+\,FP) on the {args.dataset} test set, for the "
        rf"non-augmented and generative baselines together with \tool for reference. "
        rf"The fixed test set contains {pos_tex} positive instances. "
        rf"Counts are taken at the same fixed decision "
        rf"threshold of 0.5 used throughout. Precision, recall and F1 are computed per class "
        rf"with the convention \texttt{{zero\_division}}\,=\,0, so a run that predicts no "
        rf"positives for a class (TP\,+\,FP\,=\,0) is assigned a precision of 0 rather than "
        rf"an undefined value; cells with ten or fewer predicted positives, for which "
        rf"precision is dominated by denominator instability, are shown in bold.}}",
        rf"    \label{{tab:counts_{tag}}}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{|l|c|c|c|c|c|c|}",
        r"    \hline",
    ]
    for i, pr in enumerate(pairs):
        out += panel(df, pr, order, seeds)
        if i != len(pairs) - 1:
            out.append(r"    \hline")
    out += [r"    \end{tabular}%", r"    }", r"\end{table}", ""]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {args.out}  ({len(order)} methods x {len(seeds)} seeds, "
          f"{len(pairs)} panels, test n={n_test})")


if __name__ == "__main__":
    main()
