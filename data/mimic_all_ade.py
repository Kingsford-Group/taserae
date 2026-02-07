#!/usr/bin/env python3
"""
MIMIC-IV Adverse Event Dataset Preprocessing Script

This script processes raw MIMIC-IV data to create the patient-level dataset
with adverse event labels used in TASER-AE experiments.

Prerequisites:
- MIMIC-IV data files (patients.csv.gz, diagnoses_icd.csv.gz, emar.csv.gz)
- Athena vocabulary files (CONCEPT.csv, CONCEPT_RELATIONSHIP.csv, CONCEPT_ANCESTOR.csv)

Usage:
    python3 mimic_all_ade.py --patients data/mimic/patients.csv.gz \
                              --diagnoses data/mimic/diagnoses_icd.csv.gz \
                              --emar data/mimic/emar.csv.gz \
                              --athena_dir data/athena \
                              --out_csv data/mimic4_patient_level.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import csv
import re
from typing import Dict, List, Set

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# --- Expand ADE ICD-10 roots to full descendant sets using Athena CONCEPT.csv ---
def _norm_icd(s: str) -> str:
    return str(s).replace(".", "").upper()

def expand_icd10_descendants(concept: pd.DataFrame, ade_roots: dict) -> dict:
    """
    Given ADE roots (category -> list of ICD-10 strings), return
    category -> SET of normalized ICD-10 codes found in Athena (descendants included).
    Descendants are taken as any code whose concept_code starts with the root (after removing '.').
    We expand across both ICD10CM and ICD10GM vocabularies to be robust.
    """
    icd10 = concept[concept["vocabulary_id"].isin(["ICD10CM", "ICD10GM"])].copy()
    icd10["code_norm"] = icd10["concept_code"].astype(str).str.replace(".", "", regex=False).str.upper()

    expanded = {}
    for cat, roots in ade_roots.items():
        roots_norm = tuple(_norm_icd(r) for r in roots)
        # any ICD-10 concept whose normalized code starts with any root -> descendant
        mask = icd10["code_norm"].str.startswith(roots_norm)
        expanded[cat] = set(icd10.loc[mask, "code_norm"].unique().tolist())
    return expanded

def tag_adverse_events_icd10_desc(diag_df: pd.DataFrame, ade_sets: dict, cat_order: list[str]) -> pd.DataFrame:
    """
    Label patients by ADE categories using expanded ICD-10 descendant SETs.
    Returns DataFrame[subject_id, adverse_events] with list[str] (or ["no"]).
    """
    d = diag_df[["subject_id","icd_code","icd_version"]].dropna(subset=["subject_id","icd_code"]).copy()
    d = d[d["icd_version"] == 10]  # only ICD-10
    d["code_norm"] = d["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)

    codes_by_pat = d.groupby("subject_id", sort=False)["code_norm"].apply(list)

    def cats_for_codes(codes):
        found = []
        for cat in cat_order:
            if any(code in ade_sets[cat] for code in codes):
                found.append(cat)
        return found if found else ["no"]

    out = codes_by_pat.apply(cats_for_codes).reset_index(name="adverse_events")
    return out

def load_icd_codes():
    """Load ICD codes for specific conditions from CSV files."""
    code_map = {}
    
    # 1. Falls
    try:
        df_falls = pd.read_csv(SCRIPT_DIR / "Falls_ICD.csv")
        code_map["Falls"] = df_falls["Code"].dropna().astype(str).tolist()
    except Exception as e:
        print(f"Error loading Falls_ICD.csv: {e}")
        code_map["Falls"] = []

    # 2. Fractures
    try:
        df_frac = pd.read_csv(SCRIPT_DIR / "Fractures_ICD.csv")
        code_map["Fractures"] = df_frac["Code"].dropna().astype(str).tolist()
    except Exception as e:
        print(f"Error loading Fractures_ICD.csv: {e}")
        code_map["Fractures"] = []

    # 3. Stroke (semicolon sep, column 'ICD-10')
    try:
        df_stroke = pd.read_csv(SCRIPT_DIR / "Stroke_ICD.csv", sep=";")
        code_map["Stroke"] = df_stroke["ICD-10"].dropna().astype(str).tolist()
    except Exception as e:
        print(f"Error loading Stroke_ICD.csv: {e}")
        code_map["Stroke"] = []

    # 4. GI Bleed
    try:
        df_gi = pd.read_csv(SCRIPT_DIR / "Gi_Bleed_ICD.csv")
        code_map["GI_Bleed"] = df_gi["Code"].dropna().astype(str).tolist()
    except Exception as e:
        print(f"Error loading Gi_Bleed_ICD.csv: {e}")
        code_map["GI_Bleed"] = []

    return code_map

ADE_ICD10GM = load_icd_codes()

# Normalize to prefix strings (remove dots, uppercase)
def _norm_prefixes(d):
    return {cat: [c.replace(".", "").upper() for c in codes] for cat, codes in d.items()}

ADE_PREFIXES = _norm_prefixes(ADE_ICD10GM)
ADE_CAT_ORDER = list(ADE_ICD10GM.keys())


def tag_adverse_events_icd10gm(diag_df: pd.DataFrame, prefixes: dict) -> pd.DataFrame:
    """
    Label patients by ADE categories using ICD-10 codes (prefix match, dots removed).
    Returns DataFrame[subject_id, adverse_events] where adverse_events is list[str] (or ["no"]).
    """
    d = diag_df[["subject_id","icd_code","icd_version"]].dropna(subset=["subject_id","icd_code"]).copy()
    d = d[d["icd_version"] == 10]
    d["code_norm"] = d["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)

    codes_by_pat = d.groupby("subject_id", sort=False)["code_norm"].apply(list)

    def cats_for_codes(codes):
        found = []
        for cat in ADE_CAT_ORDER:
            pref = prefixes[cat]
            if any(any(code.startswith(p) for p in pref) for code in codes):
                found.append(cat)
        return found if found else ["no"]

    out = codes_by_pat.apply(cats_for_codes).reset_index(name="adverse_events")
    return out


def _norm_code(x: str) -> str:
    return re.sub(r"\.", "", str(x)).upper()

def load_athena(athena_dir: Path):
    """Load Athena vocab CSVs (TAB-delimited)."""
    concept = pd.read_csv(
        athena_dir / "CONCEPT.csv",
        sep="\t", engine="python", encoding="utf-8",
        quoting=csv.QUOTE_NONE, on_bad_lines="warn",
        dtype={"concept_code": "string"}
    )
    crel = pd.read_csv(
        athena_dir / "CONCEPT_RELATIONSHIP.csv",
        sep="\t", engine="python", encoding="utf-8",
        quoting=csv.QUOTE_NONE, on_bad_lines="warn"
    )
    cans = pd.read_csv(
        athena_dir / "CONCEPT_ANCESTOR.csv",
        sep="\t", engine="python", encoding="utf-8",
        quoting=csv.QUOTE_NONE, on_bad_lines="warn",
        dtype={"ancestor_concept_id": "int64", "descendant_concept_id": "int64"}
    )
    return concept, crel, cans

def build_ae_codebook(athena_dir: Path, ae_snomed: Dict[str, List[int]], include_descendants: bool = True):
    """From SNOMED targets -> collect all ICD9/10 source codes that map to those SNOMED concepts."""
    concept, crel, cans = load_athena(athena_dir)

    ae_sets = {}
    for label, roots in ae_snomed.items():
        roots_set = set(roots)
        if include_descendants:
            desc = cans.loc[cans["ancestor_concept_id"].isin(roots_set), "descendant_concept_id"]
            snomed_all = roots_set.union(set(desc.tolist()))
        else:
            snomed_all = roots_set

        crel_maps = crel[crel["relationship_id"].isin(["Maps to", "Maps to value"])]
        icd_src = concept[concept["vocabulary_id"].isin(["ICD9CM", "ICD10CM"])][["concept_id","concept_code","vocabulary_id"]]
        m = icd_src.merge(crel_maps, left_on="concept_id", right_on="concept_id_1", how="inner")
        m = m[m["concept_id_2"].isin(snomed_all)]
        m["code_norm"] = m["concept_code"].astype(str).str.replace(".", "", regex=False).str.upper()

        ae_sets[label] = {
            "ICD9": set(m.loc[m["vocabulary_id"]=="ICD9CM","code_norm"].unique().tolist()),
            "ICD10": set(m.loc[m["vocabulary_id"]=="ICD10CM","code_norm"].unique().tolist()),
        }
    return ae_sets, (concept, crel, cans)

def tag_adverse_events(diag_df: pd.DataFrame, ae_codebook: Dict[str, Dict[str, Set[str]]]) -> pd.DataFrame:
    """
    Returns DataFrame[subject_id, adverse_events] where adverse_events is list[str] (or ["no"]).
    """
    d = diag_df[["subject_id","icd_code","icd_version"]].dropna(subset=["subject_id","icd_code"]).copy()
    d["code_norm"] = d["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)

    def labels_for_patient(g: pd.DataFrame):
        found = set()
        for code, ver in zip(g["code_norm"], g["icd_version"].fillna(0)):
            try:
                ver = int(ver)
            except Exception:
                ver = 0
            for label, cb in ae_codebook.items():
                if (ver == 9 and code in cb["ICD9"]) or \
                   (ver == 10 and code in cb["ICD10"]) or \
                   (ver == 0 and (code in cb["ICD9"] or code in cb["ICD10"])):
                    found.add(label)
        return sorted(found) if found else ["no"]

    gb = d.groupby("subject_id", sort=False)
    try:
        out = gb.apply(labels_for_patient, include_groups=False).reset_index(name="adverse_events")
    except TypeError:
        out = gb.apply(labels_for_patient).reset_index(name="adverse_events")
    return out

def derive_complications(
    diag_df: pd.DataFrame,
    concept: pd.DataFrame,
    crel: pd.DataFrame,
    cans: pd.DataFrame,
    exclude_icd10_set: set[str] | None = None
) -> pd.DataFrame:
    """
    Build patient-level 'complications' from diagnoses by mapping ICD -> SNOMED Condition names,
    excluding any diagnoses whose ICD-10 normalized code is in exclude_icd10_set.
    Returns DataFrame[subject_id, complications] as list[str].
    """
    d = diag_df[["subject_id","icd_code","icd_version"]].dropna(subset=["subject_id","icd_code"]).copy()
    d["code_norm"] = d["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)

    if exclude_icd10_set:
        mask_excl = (d["icd_version"] == 10) & (d["code_norm"].isin(exclude_icd10_set))
        d = d[~mask_excl].copy()

    icd_concepts = concept[concept["vocabulary_id"].isin(["ICD9CM","ICD10CM"])].copy()
    icd_concepts["code_norm"] = icd_concepts["concept_code"].astype(str).str.replace(".", "", regex=False).str.upper()

    crel_maps = crel[crel["relationship_id"].isin(["Maps to","Maps to value"])]
    icd_to_std = (icd_concepts[["concept_id","code_norm"]]
                  .merge(crel_maps[["concept_id_1","concept_id_2"]],
                         left_on="concept_id", right_on="concept_id_1", how="inner")
                  .merge(concept[["concept_id","concept_name","domain_id","vocabulary_id"]],
                         left_on="concept_id_2", right_on="concept_id", how="left"))

    icd_to_snomed_cond = icd_to_std[(icd_to_std["vocabulary_id"]=="SNOMED") &
                                    (icd_to_std["domain_id"]=="Condition")].copy()
    icd_to_snomed_cond = icd_to_snomed_cond.rename(columns={
        "concept_id_2": "snomed_id",
        "concept_name": "snomed_name"
    })[["code_norm","snomed_id","snomed_name"]].drop_duplicates()

    d_with_snomed = d.merge(icd_to_snomed_cond, on="code_norm", how="left")

    comp_per_patient = (d_with_snomed.groupby("subject_id")["snomed_name"]
                        .apply(lambda s: sorted(set([x for x in s.dropna().tolist()])))
                        .reset_index(name="complications"))
    return comp_per_patient




def build_drug_lists(emar_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized EMAR medication names per patient."""
    if emar_df is None or emar_df.empty:
        return pd.DataFrame({"subject_id": [], "drugs": []})
    e = emar_df.copy()
    e["med_norm"] = e["medication"].astype(str).str.lower().str.strip()
    drugs = (e.groupby("subject_id")["med_norm"]
               .apply(lambda s: sorted(set([x for x in s.dropna().tolist()])))
               .reset_index(name="drugs"))
    return drugs

def make_id_map(values: List[str], start: int = 1) -> Dict[str, int]:
    """Deterministic mapping (sorted) from string values to integer IDs."""
    uniq = sorted(set(values))
    return {v: i for i, v in enumerate(uniq, start=start)}

def apply_list_id_map(lst: List[str], id_map: Dict[str, int]) -> List[int]:
    return [id_map[x] for x in lst if x in id_map]

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Build patient-level table with numeric encodings from MIMIC-IV + Athena.")
    ap.add_argument("--patients", default="mimic/patients.csv.gz", help="Path to MIMIC patients.csv.gz")
    ap.add_argument("--diagnoses", default="mimic/diagnoses_icd.csv.gz", help="Path to MIMIC diagnoses_icd.csv.gz")
    ap.add_argument("--emar", default="mimic/emar.csv.gz", help="Path to MIMIC emar.csv.gz")
    ap.add_argument("--athena_dir", default="athena", help="Folder with CONCEPT.csv, CONCEPT_RELATIONSHIP.csv, CONCEPT_ANCESTOR.csv")
    ap.add_argument("--include_descendants", action="store_true", help="Include descendants for AE and AE-exclusion in complications")
    ap.add_argument("--out_csv", default="mimic4_patient_level_full_all.csv", help="Output patient-level CSV")
    ap.add_argument("--drug_map_csv", default="drug_id_map.csv", help="Drug ID map output")
    ap.add_argument("--comp_map_csv", default="complication_id_map.csv", help="Complication ID map output")
    ap.add_argument("--gender_map_csv", default="gender_id_map.csv", help="Gender ID map output")
    args = ap.parse_args()


    concept, crel, cans = load_athena(Path(args.athena_dir))
    # --- Load MIMIC CSVs ---
    patients = pd.read_csv(args.patients, usecols=["subject_id","gender","anchor_age"])
    patients = patients.rename(columns={"anchor_age":"ageYear"})
    diag = pd.read_csv(args.diagnoses, usecols=["subject_id","hadm_id","icd_code","icd_version"])
    try:
        emar = pd.read_csv(args.emar, usecols=["subject_id","medication"])
    except Exception:
        emar = pd.DataFrame(columns=["subject_id","medication"])
    ADE_CAT_ORDER = list(ADE_ICD10GM.keys())
    ade_sets = expand_icd10_descendants(concept, ADE_ICD10GM)
    # --- Build per-patient aggregates ---
    drugs = build_drug_lists(emar)                           # drugs list[str]
    # Tag AEs using your ADE ICD-10 categories
    
    ae = tag_adverse_events_icd10_desc(diag, ade_sets, ADE_CAT_ORDER)
    ADE_EXCLUDE_SET = set().union(*[ade_sets[c] for c in ADE_CAT_ORDER])
    # Flatten all ADE prefixes once for exclusion from complications
    ADE_ALL_PREFIXES = [p for cat in ADE_CAT_ORDER for p in ADE_PREFIXES[cat]]

    # Build complications, excluding ADE-coded diagnoses
    comp = derive_complications(diag, concept, crel, cans, exclude_icd10_set=ADE_EXCLUDE_SET)

    # --- Merge ---
    out = (patients.assign(rowId=patients["subject_id"].astype("int64"))
                   .merge(drugs, left_on="subject_id", right_on="subject_id", how="left")
                   .merge(comp, on="subject_id", how="left")
                   .merge(ae, on="subject_id", how="left"))

    # Ensure lists
    out["drugs"] = out["drugs"].apply(lambda x: x if isinstance(x, list) else [])
    out["complications"] = out["complications"].apply(lambda x: x if isinstance(x, list) else [])
    out["adverse_events"] = out["adverse_events"].apply(lambda x: x if isinstance(x, list) else ["no"])

    # --- Drop rows with no drugs OR no complications (require both non-empty) ---
    out = out[(out["drugs"].str.len() > 0) & (out["complications"].str.len() > 0)].copy()

    # --- Build and apply ID maps for gender, drugs, complications ---
    # Gender map (deterministic)
    gender_values = sorted(out["gender"].astype(str).unique().tolist())
    gender_map = {g: i for i, g in enumerate(gender_values, start=1)}  # e.g., F->1, M->2, etc.
    out["gender"] = out["gender"].map(gender_map).astype("int64")

    # Drug map
    all_drugs = [d for lst in out["drugs"] for d in lst]
    drug_map = make_id_map(all_drugs, start=1)
    out["administered_drugs"] = out["drugs"].apply(lambda lst: "|".join(str(x) for x in apply_list_id_map(lst, drug_map)))

    # Complication map (use SNOMED names)
    all_comps = [c for lst in out["complications"] for c in lst]
    comp_map = make_id_map(all_comps, start=1)
    out["complications_ids"] = out["complications"].apply(lambda lst: "|".join(str(x) for x in apply_list_id_map(lst, comp_map)))

    # --- Final shaping & column order ---
    # Keep AE labels as strings (pipe-delimited)
    out["adverse_events"] = out["adverse_events"].apply(lambda xs: "|".join(xs) if isinstance(xs, list) else str(xs))

    final = out[["rowId", "ageYear", "gender", "administered_drugs", "complications_ids", "adverse_events"]].rename(
        columns={"complications_ids": "complications"}
    ).sort_values("rowId")

    # --- Save outputs ---
    final.to_csv(args.out_csv, index=False)

    # Save maps
    pd.DataFrame([{"gender": k, "gender_id": v} for k, v in gender_map.items()]) \
      .sort_values("gender_id") \
      .to_csv(args.gender_map_csv, index=False)

    pd.DataFrame([{"drug_name": k, "drug_id": v} for k, v in drug_map.items()]) \
      .sort_values("drug_id") \
      .to_csv(args.drug_map_csv, index=False)

    pd.DataFrame([{"complication_name": k, "complication_id": v} for k, v in comp_map.items()]) \
      .sort_values("complication_id") \
      .to_csv(args.comp_map_csv, index=False)

    print(f"Wrote: {args.out_csv}")
    print(f"Maps:  {args.gender_map_csv}, {args.drug_map_csv}, {args.comp_map_csv}")
    print(f"Rows kept (both drugs & complications present): {len(final)}")

if __name__ == "__main__":
    main()
