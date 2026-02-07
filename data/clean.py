#!/usr/bin/env python3
"""
Convert pipe-delimited columns to comma-delimited format.

This script reformats the output from mimic_all_ade.py to use comma-separated
values instead of pipe-separated, which is the format expected by the 
augmentation and training scripts.

Usage:
    python3 clean.py --in_csv mimic4_patient_level_full_all.csv --out_csv mimic4_label.csv
"""

import pandas as pd
import csv
import argparse

def pipe_to_comma(val: str) -> str:
    if pd.isna(val) or val == "":
        return ""
    parts = [p for p in str(val).split("|") if p != ""]
    return ",".join(parts)

def main():
    ap = argparse.ArgumentParser(description="Reformat list columns from pipe-delimited to comma-delimited (quoted).")
    ap.add_argument("--in_csv",  required=True, help="Input CSV (current format).")
    ap.add_argument("--out_csv", required=True, help="Output CSV (quoted comma-separated lists).")
    args = ap.parse_args()

    # Read with expected dtypes and keep column order
    cols = ["ageYear", "gender", "administered_drugs", "complications", "adverse_events"]
    df = pd.read_csv(args.in_csv, usecols=cols, dtype={
        "ageYear": "int64",
        "gender": "int64",
        "administered_drugs": "string",
        "complications": "string",
        "adverse_events": "string",
    })

    # Convert list-like columns from '|' -> ',' (CSV writer will add quotes automatically)
    df["administered_drugs"] = df["administered_drugs"].map(pipe_to_comma)
    df["complications"]      = df["complications"].map(pipe_to_comma)
    df["adverse_events"]      = df["adverse_events"].map(pipe_to_comma)
    # Write out; QUOTE_MINIMAL will put quotes around fields containing commas
    df.to_csv(args.out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(df)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
