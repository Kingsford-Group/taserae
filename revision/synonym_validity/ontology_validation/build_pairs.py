#!/usr/bin/env python3
"""Step 1: build the Word2Vec synonym-thesaurus pair set to validate.

For drugs and complications separately:
  - collect that column's unique ID tokens from the training split
  - compute pairwise cosine on L2-normalized embeddings
  - keep all unordered pairs with cosine > THRESHOLD
  - resolve each token id to its name via the id->name map

Outputs:
  outputs/thesaurus_pairs.csv  (token_type, id_a, id_b, name_a, name_b, cosine)
  also writes the per-type unique token vocab for the baseline step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C


def unique_column_tokens(train_df: pd.DataFrame, col: str) -> list[str]:
    # mirrors augment_composite_insertion.py:73 (explode + unique), then sort
    vals = (
        train_df[col]
        .dropna()
        .apply(lambda x: [t.strip() for t in str(x).split(",")])
        .explode()
        .unique()
    )
    toks = [t for t in vals if t]
    try:
        return sorted(toks, key=lambda x: (0, int(x)))
    except ValueError:
        return sorted(toks)


def pairs_above_threshold(tokens, embeddings, threshold):
    valid = [t for t in tokens if t in embeddings]
    vecs = np.asarray([embeddings[t] for t in valid], dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    nrm = vecs / norms
    sim = nrm @ nrm.T
    iu = np.triu_indices(len(valid), k=1)
    s = sim[iu]
    mask = s > threshold          # strict >, matching compute_similarity_dict
    ia, ib = iu[0][mask], iu[1][mask]
    return (
        [valid[i] for i in ia],
        [valid[i] for i in ib],
        s[mask],
        valid,
    )


def load_name_map(path, id_col, name_col):
    df = pd.read_csv(path)
    return dict(zip(df[id_col].astype(str), df[name_col].astype(str)))


def main():
    embeddings = np.load(C.EMBEDDINGS, allow_pickle=True).item()
    train_df = pd.read_csv(C.TRAIN)
    comp_names = load_name_map(C.COMP_MAP, "complication_id", "complication_name")
    drug_names = load_name_map(C.DRUG_MAP, "drug_id", "drug_name")

    specs = [
        ("drug", "administered_drugs", drug_names),
        ("complication", "complications", comp_names),
    ]

    rows = []
    vocab = {}
    for ttype, col, name_map in specs:
        tokens = unique_column_tokens(train_df, col)
        ida, idb, scores, valid = pairs_above_threshold(
            tokens, embeddings, C.THRESHOLD
        )
        vocab[ttype] = valid
        for a, b, c in zip(ida, idb, scores):
            rows.append(
                {
                    "token_type": ttype,
                    "id_a": a,
                    "id_b": b,
                    "name_a": name_map.get(a, a),
                    "name_b": name_map.get(b, b),
                    "cosine": round(float(c), 6),
                }
            )
        n_unique_in_pairs = len({*ida, *idb})
        print(
            f"[{ttype:12s}] column tokens={len(tokens):5d}  "
            f"in-embeddings={len(valid):5d}  "
            f"pairs(cos>{C.THRESHOLD})={len(scores):6d}  "
            f"unique tokens in pairs={n_unique_in_pairs:5d}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "thesaurus_pairs.csv", index=False)
    # save per-type full vocab (ids in embeddings) for the baseline
    for ttype, valid in vocab.items():
        pd.Series(valid, name="id").to_csv(
            C.OUT / f"vocab_{ttype}.csv", index=False
        )
    print(f"\nWrote {len(out)} pairs -> {C.OUT / 'thesaurus_pairs.csv'}")

    # cosine distribution by bin
    print("\nCosine bin distribution:")
    out["bin"] = out["cosine"].apply(C.bin_label)
    print(out.groupby(["token_type", "bin"]).size().to_string())


if __name__ == "__main__":
    main()
