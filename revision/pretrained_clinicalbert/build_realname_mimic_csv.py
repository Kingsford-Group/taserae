#!/usr/bin/env python3
"""
Build a revision-only patient-level CSV with real drug/complication names.
Does not modify files under /data.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
sys.path.append(str(DATA_DIR))
from mimic_all_ade import (  # noqa: E402
    ADE_ICD10GM,
    build_drug_lists,
    derive_complications,
    expand_icd10_descendants,
    load_athena,
    tag_adverse_events_icd10_desc,
)


def _sanitize_item(x: str) -> str:
    # Avoid embedded commas that would break downstream list parsing.
    return str(x).replace(",", " ").strip()


def _list_to_csv_string(items):
    cleaned = []
    for x in items:
        # Skip None, NaN, and empty values
        if x is None or (isinstance(x, float) and x != x):
            continue
        s = _sanitize_item(x)
        if s and s.lower() != "nan":
            cleaned.append(s)
    return ",".join(cleaned)


def main():
    parser = argparse.ArgumentParser(description="Create real-name MIMIC CSV for ClinicalBERT revision experiments.")
    parser.add_argument("--patients", type=str, required=True, help="Path to patients.csv.gz")
    parser.add_argument("--diagnoses", type=str, required=True, help="Path to diagnoses_icd.csv.gz")
    parser.add_argument("--emar", type=str, required=True, help="Path to emar.csv.gz")
    parser.add_argument("--athena_dir", type=str, required=True, help="Path to Athena folder")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    concept, crel, cans = load_athena(Path(args.athena_dir))

    patients = pd.read_csv(args.patients, usecols=["subject_id", "gender", "anchor_age"]).rename(
        columns={"anchor_age": "ageYear"}
    )
    diag = pd.read_csv(args.diagnoses, usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    try:
        emar = pd.read_csv(args.emar, usecols=["subject_id", "medication"])
    except Exception:
        emar = pd.DataFrame(columns=["subject_id", "medication"])

    ade_cat_order = list(ADE_ICD10GM.keys())
    ade_sets = expand_icd10_descendants(concept, ADE_ICD10GM)
    ae = tag_adverse_events_icd10_desc(diag, ade_sets, ade_cat_order)
    ade_exclude_set = set().union(*[ade_sets[c] for c in ade_cat_order])

    drugs = build_drug_lists(emar)
    comp = derive_complications(diag, concept, crel, cans, exclude_icd10_set=ade_exclude_set)

    out = (
        patients.merge(drugs, on="subject_id", how="left")
        .merge(comp, on="subject_id", how="left")
        .merge(ae, on="subject_id", how="left")
    )
    out["drugs"] = out["drugs"].apply(lambda x: x if isinstance(x, list) else [])
    out["complications"] = out["complications"].apply(lambda x: x if isinstance(x, list) else [])
    out["adverse_events"] = out["adverse_events"].apply(lambda x: x if isinstance(x, list) else ["no"])

    out = out[(out["drugs"].str.len() > 0) & (out["complications"].str.len() > 0)].copy()

    out["gender"] = out["gender"].astype(str).str.strip().str.lower()
    out["administered_drugs"] = out["drugs"].apply(_list_to_csv_string)
    out["complications"] = out["complications"].apply(_list_to_csv_string)
    out["labels"] = out["adverse_events"].apply(_list_to_csv_string)

    final = out[["ageYear", "gender", "administered_drugs", "complications", "labels"]].copy()

    # Drop rows where drugs or complications became empty after NaN filtering
    final = final[
        (final["administered_drugs"].astype(str).str.strip() != "")
        & (final["complications"].astype(str).str.strip() != "")
    ].copy()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    final.to_csv(args.out_csv, index=False)
    print(f"Wrote real-name CSV: {args.out_csv}")
    print(f"Rows kept: {len(final)}")


if __name__ == "__main__":
    main()
