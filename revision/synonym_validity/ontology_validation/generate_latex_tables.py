#!/usr/bin/env python3
"""Step 5: render LaTeX tables + narrative numbers from the summary CSV."""
from __future__ import annotations

import json

import pandas as pd

import common as C

summ = pd.read_csv(C.OUT / "ontology_validation_summary.csv")
GROUP_ORDER = ["[0.95,1.00]", "[0.90,0.95)", "[0.85,0.90)", "[0.80,0.85)",
               "All", "Random baseline"]


def g(ttype, group):
    r = summ[(summ.token_type == ttype) & (summ.group == group)]
    return r.iloc[0] if len(r) else None


def cell(row, col):
    v = row.get(col)
    if pd.isna(v):
        return "--"
    std = row.get(col + "_std")
    if pd.notna(std) and row["group"] == "Random baseline":
        return f"{v:.1f}\\,$\\pm$\\,{std:.1f}"
    return f"{v:.1f}"


def fmt_group(group):
    return "Random" if group == "Random baseline" else (
        "\\textbf{All}" if group == "All" else group.replace("[", "[").replace(",", ",\\,"))


def drug_table():
    cols = ["L5_same_substance", "L4_same_chem", "L3_same_pharm",
            "L2_same_therapeutic", "L1_same_anatomical", "no_match", "unmapped"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{Ontology-based validation of \emph{drug} synonym pairs against the "
        r"ATC classification (via RxNorm). Each pair is assigned to its deepest shared "
        r"ATC level; columns are mutually exclusive and sum to 100\%. Unmapped pairs "
        r"(one or both drug names unresolvable to ATC) are retained in the denominator. "
        r"The random baseline draws, for each of 5 seeds, as many in-vocabulary drug "
        r"pairs as the thesaurus contains (mean\,$\pm$\,s.d.).}",
        r"\label{tab:drug_ontology}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Cosine bin & $N$ & L5 & L4 & L3 & L2 & L1 & No & Un- \\",
        r" & & subst. & chem. & pharm. & ther. & anat. & match & mapped \\",
        r"\midrule",
    ]
    for grp in GROUP_ORDER:
        r = g("drug", grp)
        if r is None:
            continue
        if grp == "All":
            lines.append(r"\midrule")
        vals = " & ".join(cell(r, c) for c in cols)
        lines.append(f"{fmt_group(grp)} & {int(r['N'])} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def comp_table():
    cols = ["same_block", "same_chapter", "no_match", "unmapped"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{Ontology-based validation of \emph{complication} synonym pairs against "
        r"the ICD-10-CM hierarchy. Complication descriptions were resolved to ICD-10-CM "
        r"codes via the NLM Clinical Tables service; agreement is scored at the block "
        r"(3-character) and chapter (1-character) levels. Columns are mutually exclusive "
        r"and sum to 100\%; unmapped pairs are retained in the denominator. Random "
        r"baseline as in Table~\ref{tab:drug_ontology}.}",
        r"\label{tab:compl_ontology}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Cosine bin & $N$ & Same & Same & No & Un- \\",
        r" & & block & chapter & match & mapped \\",
        r"\midrule",
    ]
    for grp in GROUP_ORDER:
        r = g("complication", grp)
        if r is None:
            continue
        if grp == "All":
            lines.append(r"\midrule")
        vals = " & ".join(cell(r, c) for c in cols)
        lines.append(f"{fmt_group(grp)} & {int(r['N'])} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def narrative():
    out = {}
    for tt in ["drug", "complication"]:
        a = g(tt, "All")
        b = g(tt, "Random baseline")
        if tt == "drug":
            valid_cols = ["L5_same_substance", "L4_same_chem", "L3_same_pharm",
                          "L2_same_therapeutic", "L1_same_anatomical"]
            related_cols = ["L5_same_substance", "L4_same_chem", "L3_same_pharm"]
        else:
            valid_cols = ["same_block", "same_chapter"]
            related_cols = ["same_block"]
        a_valid = sum(a[c] for c in valid_cols)
        b_valid = sum(b[c] for c in valid_cols)
        a_rel = sum(a[c] for c in related_cols)
        b_rel = sum(b[c] for c in related_cols)
        out[tt] = {
            "thes_any_pct": round(a_valid, 1),
            "base_any_pct": round(b_valid, 1),
            "any_fold": round(a_valid / b_valid, 1) if b_valid else None,
            "thes_related_pct": round(a_rel, 1),
            "base_related_pct": round(b_rel, 1),
            "related_fold": round(a_rel / b_rel, 1) if b_rel else None,
            "unmapped_pct": round(a["unmapped"], 1),
            "N": int(a["N"]),
        }
    return out


if __name__ == "__main__":
    dt, ct = drug_table(), comp_table()
    (C.OUT / "table_drug.tex").write_text(dt + "\n")
    (C.OUT / "table_complication.tex").write_text(ct + "\n")
    nar = narrative()
    (C.OUT / "narrative_stats.json").write_text(json.dumps(nar, indent=2))
    print(dt, "\n\n", ct, "\n")
    print("=== NARRATIVE STATS ===")
    print(json.dumps(nar, indent=2))
