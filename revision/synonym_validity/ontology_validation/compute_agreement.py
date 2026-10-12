#!/usr/bin/env python3
"""Step 4: score ontology agreement, bin by cosine, and build the random baseline.

Drug pairs: ATC agreement (best of 5 levels).
  L5 same substance  = share an ingredient RxCUI
  L4 same chem subgroup = share a 5-char ATC code (e.g. C09AA)
  L3 same pharm subgroup = share 4-char ATC prefix (C09A)
  L2 same therapeutic   = share 3-char ATC prefix (C09)
  L1 same anatomical    = share 1-char ATC prefix (C)
  0  no match           = both mapped, no shared level
Complication pairs: ICD-10-CM agreement.
  block   = share 3-char code (e.g. N17)
  chapter = share 1st character (e.g. N), different block
  0       = both mapped, different chapter
Each pair gets exactly one best category; Unmapped = >=1 token unresolved.
Unmapped pairs stay in the denominator (full-picture reporting).

Random baseline: for each type, draw N random in-vocab pairs not in the
thesaurus (N = #thesaurus pairs of that type), 5 seeds, same scoring.
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import common as C

DRUG_CACHE = json.loads((C.CACHE / "drug_atc_mapping.json").read_text())
COMP_CACHE = json.loads((C.CACHE / "complication_icd_mapping.json").read_text())

DRUG_CATS = ["L5_same_substance", "L4_same_chem", "L3_same_pharm",
             "L2_same_therapeutic", "L1_same_anatomical", "no_match", "unmapped"]
COMP_CATS = ["same_block", "same_chapter", "no_match", "unmapped"]


# ---------- drug scoring ----------
def drug_mapped(tid: str) -> bool:
    e = DRUG_CACHE.get(str(tid))
    return bool(e) and (bool(e.get("atc")) or bool(e.get("rxcuis")))


def _prefixes(codes, n):
    return {c[:n] for c in codes if len(c) >= n}


def drug_level(a: str, b: str) -> int:
    ea, eb = DRUG_CACHE.get(str(a), {}), DRUG_CACHE.get(str(b), {})
    ra, rb = set(ea.get("rxcuis", [])), set(eb.get("rxcuis", []))
    if ra and rb and (ra & rb):
        return 5
    A, B = set(ea.get("atc", [])), set(eb.get("atc", []))
    if A & B:
        return 4
    if _prefixes(A, 4) & _prefixes(B, 4):
        return 3
    if _prefixes(A, 3) & _prefixes(B, 3):
        return 2
    if _prefixes(A, 1) & _prefixes(B, 1):
        return 1
    return 0


def drug_category(a, b):
    if not (drug_mapped(a) and drug_mapped(b)):
        return "unmapped"
    return {5: "L5_same_substance", 4: "L4_same_chem", 3: "L3_same_pharm",
            2: "L2_same_therapeutic", 1: "L1_same_anatomical",
            0: "no_match"}[drug_level(a, b)]


def drug_code(tid):
    e = DRUG_CACHE.get(str(tid), {})
    return "|".join(e.get("atc", [])) or ("rxcui:" + ",".join(e.get("rxcuis", [])) if e.get("rxcuis") else "")


# ---------- complication scoring ----------
def _icd(tid):
    e = COMP_CACHE.get(str(tid))
    if e and e.get("mapped") and e.get("icd"):
        return e["icd"].replace(".", "")
    return None


def comp_category(a, b):
    ca, cb = _icd(a), _icd(b)
    if ca is None or cb is None:
        return "unmapped"
    if ca[:3] == cb[:3]:
        return "same_block"
    if ca[:1] == cb[:1]:
        return "same_chapter"
    return "no_match"


def comp_code(tid):
    e = COMP_CACHE.get(str(tid), {})
    return e.get("icd") or ""


# ---------- generic ----------
SCORERS = {
    "drug": (drug_category, drug_code, DRUG_CATS),
    "complication": (comp_category, comp_code, COMP_CATS),
}


def score_pairs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cats, codes, valids = [], [], []
    for _, r in df.iterrows():
        catfn, codefn, _ = SCORERS[r.token_type]
        cat = catfn(r.id_a, r.id_b)
        cats.append(cat)
        codes.append(f"{codefn(r.id_a)} || {codefn(r.id_b)}")
        valids.append(cat not in ("no_match", "unmapped"))
    out["agreement_category"] = cats
    out["codes"] = codes
    out["is_valid"] = valids
    out["bin"] = out["cosine"].apply(C.bin_label)
    return out


def category_rates(sub: pd.DataFrame, cats: list[str]) -> dict:
    n = len(sub)
    vc = sub["agreement_category"].value_counts()
    d = {"N": n}
    for c in cats:
        d[c] = 100.0 * vc.get(c, 0) / n if n else 0.0
    return d


# ---------- baseline ----------
def random_pairs(vocab, n, exclude, rng):
    vocab = list(vocab)
    out, seen = [], set()
    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        i, j = rng.integers(0, len(vocab), size=2)
        if i == j:
            continue
        a, b = vocab[i], vocab[j]
        key = (a, b) if a < b else (b, a)
        if key in exclude or key in seen:
            continue
        seen.add(key)
        out.append({"token_type": None, "id_a": a, "id_b": b, "cosine": np.nan})
    return out


def baseline(ttype, vocab, thes_df, n, cats, seeds=(0, 1, 2, 3, 4)):
    exclude = {
        (a, b) if a < b else (b, a)
        for a, b in zip(thes_df.id_a.astype(str), thes_df.id_b.astype(str))
    }
    rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        rp = random_pairs([str(v) for v in vocab], n, exclude, rng)
        for r in rp:
            r["token_type"] = ttype
        scored = score_pairs(pd.DataFrame(rp))
        rows.append(category_rates(scored, cats))
    agg = {"N": int(np.mean([r["N"] for r in rows]))}
    for c in cats:
        vals = [r[c] for r in rows]
        agg[c + "_mean"] = float(np.mean(vals))
        agg[c + "_std"] = float(np.std(vals))
    # valid-rate per seed for significance
    agg["_valid_rates"] = [
        100 - r["no_match"] - r["unmapped"] for r in rows
    ]
    return agg


def main():
    pairs = pd.read_csv(C.OUT / "thesaurus_pairs.csv",
                        dtype={"id_a": str, "id_b": str})
    scored = score_pairs(pairs)
    scored.to_csv(C.OUT / "ontology_validation_results.csv", index=False)

    summary_rows = []
    sig_lines = []
    for ttype in ["drug", "complication"]:
        cats = SCORERS[ttype][2]
        sub = scored[scored.token_type == ttype]
        vocab = pd.read_csv(C.OUT / f"vocab_{ttype}.csv")["id"].astype(str).tolist()

        for lab in C.BIN_LABELS:
            b = sub[sub.bin == lab]
            if len(b):
                row = {"token_type": ttype, "group": lab}
                row.update(category_rates(b, cats))
                summary_rows.append(row)
        allrow = {"token_type": ttype, "group": "All"}
        allrow.update(category_rates(sub, cats))
        summary_rows.append(allrow)

        bl = baseline(ttype, vocab, sub, len(sub), cats)
        blrow = {"token_type": ttype, "group": "Random baseline",
                 "N": bl["N"]}
        for c in cats:
            blrow[c] = bl[c + "_mean"]
            blrow[c + "_std"] = bl[c + "_std"]
        summary_rows.append(blrow)

        # significance: thesaurus valid vs baseline-mean valid (Fisher on counts)
        t_valid = int(sub["is_valid"].sum())
        t_tot = len(sub)
        b_valid = int(round(np.mean(bl["_valid_rates"]) / 100 * t_tot))
        try:
            _, p = fisher_exact([[t_valid, t_tot - t_valid],
                                 [b_valid, t_tot - b_valid]])
        except Exception:
            p = float("nan")
        sig_lines.append(
            f"{ttype}: thesaurus valid={100*t_valid/t_tot:.1f}% vs "
            f"baseline valid={np.mean(bl['_valid_rates']):.1f}% "
            f"(Fisher p={p:.2e})"
        )

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(C.OUT / "ontology_validation_summary.csv", index=False)

    # unmapped tokens for inspection
    unmapped = []
    drug_map = pd.read_csv(C.DRUG_MAP)
    comp_map = pd.read_csv(C.COMP_MAP)
    dn = dict(zip(drug_map.drug_id.astype(str), drug_map.drug_name))
    cn = dict(zip(comp_map.complication_id.astype(str), comp_map.complication_name))
    seen = set()
    for ttype, sub in scored.groupby("token_type"):
        for tid in set(sub.id_a) | set(sub.id_b):
            mapped = drug_mapped(tid) if ttype == "drug" else (_icd(tid) is not None)
            if not mapped and (ttype, tid) not in seen:
                seen.add((ttype, tid))
                nm = dn.get(tid, tid) if ttype == "drug" else cn.get(tid, tid)
                unmapped.append({"token_type": ttype, "id": tid, "name": nm})
    pd.DataFrame(unmapped).to_csv(C.OUT / "ontology_validation_unmapped.csv", index=False)

    print("=== SUMMARY ===")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(summ.round(1).to_string(index=False))
    print("\n=== SIGNIFICANCE ===")
    print("\n".join(sig_lines))
    print(f"\nUnmapped tokens: {len(unmapped)} "
          f"(written to ontology_validation_unmapped.csv)")
    print("Wrote results, summary, unmapped CSVs ->", C.OUT)


if __name__ == "__main__":
    main()
