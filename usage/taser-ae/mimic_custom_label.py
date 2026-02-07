#!/usr/bin/env python3
"""
MIMIC-IV Custom Label Dataset Preprocessing Script

This script processes raw MIMIC-IV data to create a patient-level dataset
with CUSTOM adverse event labels defined by the user.

Prerequisites:
- MIMIC-IV data files (patients.csv.gz, diagnoses_icd.csv.gz, emar.csv.gz)
- Athena vocabulary files (CONCEPT.csv, CONCEPT_RELATIONSHIP.csv, CONCEPT_ANCESTOR.csv)
- A Label Map CSV (headers: Label, Path) pointing to CSVs with ICD codes.

Usage:
    python3 mimic_custom_label.py --patients data/mimic/patients.csv.gz \
                                  --diagnoses data/mimic/diagnoses_icd.csv.gz \
                                  --emar data/mimic/emar.csv.gz \
                                  --athena_dir data/athena \
                                  --label_map my_labels.csv \
                                  --out_csv data/mimic4_custom.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import csv
import re
import sys
from typing import Dict, List, Set

# --- Expand ADE ICD-10 roots to full descendant sets using Athena CONCEPT.csv ---
def _norm_icd(s: str) -> str:
    return str(s).replace(".", "").upper()

def expand_icd10_descendants(concept: pd.DataFrame, ade_roots: dict) -> dict:
    """
    Given ADE roots (category -> list of ICD-10 strings), return
    category -> SET of normalized ICD-10 codes found in Athena (descendants included).
    """
    icd10 = concept[concept["vocabulary_id"].isin(["ICD10CM", "ICD10GM"])].copy()
    icd10["code_norm"] = icd10["concept_code"].astype(str).str.replace(".", "", regex=False).str.upper()

    expanded = {}
    for cat, roots in ade_roots.items():
        roots_norm = tuple(_norm_icd(r) for r in roots)
        mask = icd10["code_norm"].str.startswith(roots_norm)
        expanded[cat] = set(icd10.loc[mask, "code_norm"].unique().tolist())
    return expanded

def tag_adverse_events_icd10_desc(diag_df: pd.DataFrame, ade_sets: dict, cat_order: list[str]) -> pd.DataFrame:
    """
    Label patients by ADE categories using expanded ICD-10 descendant SETs.
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

def load_custom_labels(map_file: str):
    """
    Load custom label definitions from a CSV map file.
    Expected Map File Format:
        Label, Path
        MyLabel, /path/to/icd_codes.csv
        
    Expected ICD File Format:
        CSV with a column named 'Code' or 'ICD-10' or simply the first column.
    """
    map_path = Path(map_file)
    if not map_path.exists():
        print(f"Error: Label map file not found: {map_file}")
        sys.exit(1)
        
    try:
        df_map = pd.read_csv(map_path)
    except Exception as e:
        print(f"Error reading label map: {e}")
        sys.exit(1)
        
    if "Label" not in df_map.columns or "Path" not in df_map.columns:
        print(f"Error: Label map must have 'Label' and 'Path' columns. Found: {df_map.columns.tolist()}")
        sys.exit(1)
        
    code_map = {}
    base_dir = map_path.parent
    
    for _, row in df_map.iterrows():
        label = str(row["Label"]).strip()
        path_str = str(row["Path"]).strip()
        
        # Resolve path relative to map file if not absolute
        file_path = Path(path_str)
        if not file_path.is_absolute():
            file_path = base_dir / file_path
            
        if not file_path.exists():
            print(f"Warning: ICD file for '{label}' not found at {file_path}. Skipping.")
            code_map[label] = []
            continue
            
        try:
            sep = ";" if file_path.suffix == ".csv" and "Stroke" in str(file_path) else "," # Legacy hack logic, let's be smarter
            try:
                df_codes = pd.read_csv(file_path, sep=None, engine='python')
            except:
                df_codes = pd.read_csv(file_path)
            
            # Find the code column
            if "Code" in df_codes.columns:
                codes = df_codes["Code"]
            elif "ICD-10" in df_codes.columns:
                codes = df_codes["ICD-10"]
            else:
                # Use first column
                codes = df_codes.iloc[:, 0]
                
            code_list = codes.dropna().astype(str).tolist()
            code_map[label] = code_list
            print(f"Loaded {len(code_list)} codes for label '{label}' from {file_path.name}")
            
        except Exception as e:
            print(f"Error loading codes for '{label}': {e}")
            code_map[label] = []
            
    return code_map

# Normalize to prefix strings (remove dots, uppercase)
def _norm_prefixes(d):
    return {cat: [c.replace(".", "").upper() for c in codes] for cat, codes in d.items()}

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

def derive_complications(
    diag_df: pd.DataFrame,
    concept: pd.DataFrame,
    crel: pd.DataFrame,
    cans: pd.DataFrame,
    exclude_icd10_set: set[str] or None = None
) -> pd.DataFrame:
    """
    Build patient-level 'complications' from diagnoses by mapping ICD -> SNOMED Condition names,
    excluding any diagnoses whose ICD-10 normalized code is in exclude_icd10_set.
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
    ap = argparse.ArgumentParser(description="Build patient-level table with Custom Labels from MIMIC-IV.")
    ap.add_argument("--patients", default="mimic/patients.csv.gz", help="Path to MIMIC patients.csv.gz")
    ap.add_argument("--diagnoses", default="mimic/diagnoses_icd.csv.gz", help="Path to MIMIC diagnoses_icd.csv.gz")
    ap.add_argument("--emar", default="mimic/emar.csv.gz", help="Path to MIMIC emar.csv.gz")
    ap.add_argument("--athena_dir", default="athena", help="Folder with Athena vocab files")
    ap.add_argument("--label_map", required=True, help="CSV file mapping Labels to ICD Code files (Cols: Label, Path)")
    ap.add_argument("--out_csv", default="mimic4_custom.csv", help="Output patient-level CSV")
    ap.add_argument("--drug_map_csv", default="drug_id_map.csv", help="Drug ID map output")
    ap.add_argument("--comp_map_csv", default="complication_id_map.csv", help="Complication ID map output")
    ap.add_argument("--gender_map_csv", default="gender_id_map.csv", help="Gender ID map output")
    args = ap.parse_args()

    # Need Athena to be present
    if not Path(args.athena_dir).exists():
        print(f"Error: Athena directory '{args.athena_dir}' not found.")
        sys.exit(1)

    print("Loading Athena vocabulary...")
    concept, crel, cans = load_athena(Path(args.athena_dir))

    print(f"Loading custom labels from {args.label_map}...")
    CUSTOM_LABELS = load_custom_labels(args.label_map)
    if not CUSTOM_LABELS:
        print("Error: No valid labels found in map.")
        sys.exit(1)
        
    ADE_CAT_ORDER = list(CUSTOM_LABELS.keys())
    ADE_PREFIXES = _norm_prefixes(CUSTOM_LABELS)

    # --- Load MIMIC CSVs ---
    print("Loading MIMIC data...")
    patients = pd.read_csv(args.patients, usecols=["subject_id","gender","anchor_age"])
    patients = patients.rename(columns={"anchor_age":"ageYear"})
    diag = pd.read_csv(args.diagnoses, usecols=["subject_id","hadm_id","icd_code","icd_version"])
    try:
        emar = pd.read_csv(args.emar, usecols=["subject_id","medication"])
    except Exception:
        print("Warning: Could not load emar.csv.gz, proceeding without drugs.")
        emar = pd.DataFrame(columns=["subject_id","medication"])

    # --- Expand Roots ---
    print("Expanding ICD-10 roots via Athena...")
    ade_sets = expand_icd10_descendants(concept, CUSTOM_LABELS)
    
    # --- Build per-patient aggregates ---
    print("Building drug lists...")
    drugs = build_drug_lists(emar)
    
    print("Tagging patients with custom labels...")
    ae = tag_adverse_events_icd10_desc(diag, ade_sets, ADE_CAT_ORDER)
    
    ADE_EXCLUDE_SET = set().union(*[ade_sets[c] for c in ADE_CAT_ORDER])

    print("Deriving complications (excluding label codes)...")
    comp = derive_complications(diag, concept, crel, cans, exclude_icd10_set=ADE_EXCLUDE_SET)

    # --- Merge ---
    print("Merging data...")
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

    # --- Build and apply ID maps ---
    print("Encoding features...")
    # Gender
    gender_values = sorted(out["gender"].astype(str).unique().tolist())
    gender_map = {g: i for i, g in enumerate(gender_values, start=1)}
    out["gender"] = out["gender"].map(gender_map).astype("int64")

    # Drugs
    all_drugs = [d for lst in out["drugs"] for d in lst]
    drug_map = make_id_map(all_drugs, start=1)
    out["administered_drugs"] = out["drugs"].apply(lambda lst: "|".join(str(x) for x in apply_list_id_map(lst, drug_map)))

    # Complications
    all_comps = [c for lst in out["complications"] for c in lst]
    comp_map = make_id_map(all_comps, start=1)
    out["complications_ids"] = out["complications"].apply(lambda lst: "|".join(str(x) for x in apply_list_id_map(lst, comp_map)))

    # Labels
    out["adverse_events"] = out["adverse_events"].apply(lambda xs: "|".join(xs) if isinstance(xs, list) else str(xs))

    final = out[["rowId", "ageYear", "gender", "administered_drugs", "complications_ids", "adverse_events"]].rename(
        columns={"complications_ids": "complications"}
    ).sort_values("rowId")

    # --- Save outputs ---
    print(f"Saving to {args.out_csv}...")
    final.to_csv(args.out_csv, index=False)

    pd.DataFrame([{"gender": k, "gender_id": v} for k, v in gender_map.items()]) \
      .sort_values("gender_id").to_csv(args.gender_map_csv, index=False)

    pd.DataFrame([{"drug_name": k, "drug_id": v} for k, v in drug_map.items()]) \
      .sort_values("drug_id").to_csv(args.drug_map_csv, index=False)

    pd.DataFrame([{"complication_name": k, "complication_id": v} for k, v in comp_map.items()]) \
      .sort_values("complication_id").to_csv(args.comp_map_csv, index=False)

    print(f"Done! {len(final)} rows generated.")

if __name__ == "__main__":
    main()
