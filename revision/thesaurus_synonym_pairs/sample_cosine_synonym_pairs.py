#!/usr/bin/env python3
"""
Sample random synonym-like pairs from the TASER-AE Word2Vec thesaurus.

For each clinical column (complications, administered_drugs), take all ID tokens
that appear in the training split, compute pairwise cosine similarity on
L2-normalized vectors, keep pairs in [cosine_low, cosine_high], then sample
n_per_category disjoint *pair indices* (IDs may repeat across rows).

Matches reproduction/augment_utils.compute_similarity_dict neighbor logic
(threshold is applied as cosine > threshold there; here we use an inclusive band).

Name columns: optional complication/drug map CSVs resolve IDs to strings. If maps are
disabled or missing, name_a/name_b fall back to the same token string as id_a/id_b
(so real-name training CSVs still get readable output without separate map files).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _token_sort_key(x: str) -> tuple:
    try:
        return (0, int(x))
    except ValueError:
        return (1, x)


def unique_column_tokens(train_df: pd.DataFrame, col: str) -> list[str]:
    ids: set[str] = set()
    for x in train_df[col].dropna().astype(str):
        for t in x.split(","):
            t = t.strip()
            if t:
                ids.add(t)
    return sorted(ids, key=_token_sort_key)


def pairs_in_cosine_band(
    tokens: list[str],
    embeddings: dict,
    cosine_low: float,
    cosine_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = [t for t in tokens if t in embeddings]
    if len(valid) < 2:
        return np.array([]), np.array([]), np.array([])

    vecs = np.array([embeddings[t] for t in valid], dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = vecs / norms
    sim = normalized @ normalized.T

    iu = np.triu_indices(len(valid), k=1)
    scores = sim[iu]
    mask = (scores >= cosine_low) & (scores <= cosine_high)
    ia, ib = iu[0][mask], iu[1][mask]
    scores = scores[mask]
    id_a = np.array([valid[i] for i in ia])
    id_b = np.array([valid[i] for i in ib])
    return id_a, id_b, scores


def sample_pair_indices(
    n_pairs: int,
    rng: np.random.Generator,
    total: int,
) -> np.ndarray:
    k = min(n_pairs, total)
    return rng.choice(total, size=k, replace=False)


def _load_map_csv(path: Path, id_col: str, name_col: str) -> dict[str, str]:
    df = pd.read_csv(path)
    if id_col not in df.columns or name_col not in df.columns:
        raise SystemExit(f"{path}: expected columns {id_col!r} and {name_col!r}")
    return dict(zip(df[id_col].astype(str), df[name_col].astype(str)))


def resolve_name_maps(
    comp_path: Path,
    drug_path: Path,
    no_maps: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    if no_maps:
        return {}, {}

    comp_names: dict[str, str] = {}
    drug_names: dict[str, str] = {}

    if comp_path.is_file():
        comp_names = _load_map_csv(comp_path, "complication_id", "complication_name")
    else:
        print(f"Warning: complication map not found ({comp_path}); using raw tokens for name_a/name_b.", file=sys.stderr)

    if drug_path.is_file():
        drug_names = _load_map_csv(drug_path, "drug_id", "drug_name")
    else:
        print(f"Warning: drug map not found ({drug_path}); using raw tokens for name_a/name_b.", file=sys.stderr)

    return comp_names, drug_names


def display_name(name_map: dict[str, str], token: str) -> str:
    return name_map[token] if token in name_map else token


def main() -> None:
    root = _repo_root()
    default_maps_dir = root / "data"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=root / "data" / "processed" / "embeddings.npy",
        help="Word2Vec embeddings dict (.npy)",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=root / "data" / "processed" / "train_pure.csv",
        help="Training CSV (complications / administered_drugs columns)",
    )
    parser.add_argument(
        "--complication-map",
        type=Path,
        default=default_maps_dir / "complication_id_map.csv",
        help="CSV with complication_id, complication_name (optional if file missing or --no-maps)",
    )
    parser.add_argument(
        "--drug-map",
        type=Path,
        default=default_maps_dir / "drug_id_map.csv",
        help="CSV with drug_id, drug_name (optional if file missing or --no-maps)",
    )
    parser.add_argument(
        "--no-maps",
        action="store_true",
        help="Do not load map CSVs; name_a/name_b are the same as id tokens (use with realname.csv or ID-only data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "thesaurus_synonym_pairs_cosine_random.csv",
        help="Output CSV path",
    )
    parser.add_argument("--cosine-low", type=float, default=0.8, help="Inclusive lower cosine bound")
    parser.add_argument("--cosine-high", type=float, default=1.0, help="Inclusive upper cosine bound")
    parser.add_argument(
        "--n-per-category",
        type=int,
        default=50,
        help="Random pairs to sample per category (complications, administered_drugs)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    args = parser.parse_args()

    embeddings = np.load(args.embeddings, allow_pickle=True).item()
    train_df = pd.read_csv(args.train)
    comp_names, drug_names = resolve_name_maps(
        args.complication_map, args.drug_map, args.no_maps
    )

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    for category, col, name_map in [
        ("complications", "complications", comp_names),
        ("administered_drugs", "administered_drugs", drug_names),
        ("age", "ageYear", {}),
        ("sex", "gender", {}),
    ]:
        if col not in train_df.columns:
            raise SystemExit(f"Column {col!r} not in {args.train}")

        tokens = unique_column_tokens(train_df, col)
        id_a, id_b, scores = pairs_in_cosine_band(
            tokens, embeddings, args.cosine_low, args.cosine_high
        )
        n = len(scores)
        if n == 0:
            continue
        pick = sample_pair_indices(args.n_per_category, rng, n)
        for j in pick:
            a, b, c = id_a[j], id_b[j], scores[j]
            rows.append(
                {
                    "category": category,
                    "id_a": a,
                    "id_b": b,
                    "name_a": display_name(name_map, a),
                    "name_b": display_name(name_map, b),
                    "cosine": round(float(c), 6),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
