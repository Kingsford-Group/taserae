#!/usr/bin/env python3
"""Shared paths and helpers for the ontology-validation pipeline.

The validation set replicates exactly what the real augmentation builds
(reproduction/augment_composite_insertion.py): for each clinical column,
take that column's unique ID tokens from the training split and keep all
unordered pairs with cosine similarity > THRESHOLD on L2-normalized
Word2Vec embeddings (reproduction/augment_utils.compute_similarity_dict).

IDs are column-scoped for naming (drug id 47 != complication id 47) but
share a single embedding vector keyed by the bare ID string -- a property
of the underlying TASER-AE method, noted as a caveat in the writeup.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                        # repository root
MAPS_DIR = REPO / "data"                      # written by data/mimic_all_ade.py

EMBEDDINGS = REPO / "data" / "processed" / "embeddings.npy"
TRAIN = REPO / "data" / "processed" / "train_pure.csv"
COMP_MAP = MAPS_DIR / "complication_id_map.csv"
DRUG_MAP = MAPS_DIR / "drug_id_map.csv"

CACHE = HERE / "cache"
OUT = HERE / "outputs"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

THRESHOLD = 0.8   # matches augment_utils default (cosine > threshold)

# Cosine bins for reporting (lower-inclusive, upper-exclusive except last)
BINS = [(0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.0001)]
BIN_LABELS = ["[0.80,0.85)", "[0.85,0.90)", "[0.90,0.95)", "[0.95,1.00]"]


def bin_label(cos: float) -> str:
    for (lo, hi), lab in zip(BINS, BIN_LABELS):
        if lo <= cos < hi:
            return lab
    return BIN_LABELS[-1] if cos >= 0.80 else "below"
