#!/usr/bin/env python3
"""
Quantitative Augmentation Quality Analysis.

Three analyses on mean-pooled Word2Vec patient embeddings:
1) Nearest-neighbor distance: real↔real vs augmented→real (+ Mann-Whitney U)
2) Local Outlier Factor (novelty) on augmented vs real test
3) Coverage: test→train NN distance before vs after adding augmented (+ Wilcoxon)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

LABEL_COL = "labels"


def tokenize_row(row: np.ndarray) -> List[str]:
    """Match reproduction/train_classifier.py tokenize_sample (no labels in row)."""
    tokens = []
    for field in row:
        for part in str(field).split(","):
            part = part.strip()
            if part:
                tokens.extend(part.split())
    return tokens


def row_to_embedding(
    row: np.ndarray,
    emb_dict: dict,
    dim: int,
) -> Tuple[np.ndarray, bool]:
    """Mean-pool token embeddings; zero vector if no hits."""
    toks = tokenize_row(row)
    vecs = [emb_dict[t] for t in toks if t in emb_dict]
    if not vecs:
        return np.zeros(dim, dtype=np.float64), True
    return np.mean(np.stack(vecs, axis=0), axis=0), False


def compute_embeddings_matrix(
    df: pd.DataFrame,
    emb_dict: dict,
    dim: int,
) -> Tuple[np.ndarray, int]:
    """Drop labels, return (n, dim) and count of zero-token rows."""
    feat = df.drop(columns=[LABEL_COL], errors="ignore")
    arr = feat.astype(str).to_numpy()
    out = np.zeros((len(arr), dim), dtype=np.float64)
    n_empty = 0
    for i in range(len(arr)):
        vec, empty = row_to_embedding(arr[i], emb_dict, dim)
        out[i] = vec
        if empty:
            n_empty += 1
    return out, n_empty


def extract_synthetic_only(
    augmented_df: pd.DataFrame,
    real_train_df: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    if mode == "append_tail":
        if len(augmented_df) < len(real_train_df):
            raise ValueError(
                "Augmented CSV has fewer rows than real train; cannot use append_tail."
            )
        return augmented_df.iloc[len(real_train_df) :].copy().reset_index(drop=True)
    if mode == "anti_join":
        feat_cols = [c for c in augmented_df.columns if c != LABEL_COL]
        real_key = real_train_df[feat_cols].astype(str).apply(
            lambda r: "|".join(r.values), axis=1
        )
        aug_key = augmented_df[feat_cols].astype(str).apply(
            lambda r: "|".join(r.values), axis=1
        )
        real_set = set(real_key)
        mask = ~aug_key.isin(real_set)
        return augmented_df.loc[mask].copy().reset_index(drop=True)
    raise ValueError(f"Unknown extract mode: {mode}")


def subsample_rows(X: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if len(X) <= max_n:
        return X
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx]


def analysis_nn_distances(
    emb_real_train: np.ndarray,
    emb_aug: np.ndarray,
    analysis1_max_real: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Nearest-neighbour distance of synthetic and real records to the real training set."""
    X_real_sub = subsample_rows(emb_real_train, analysis1_max_real, rng)
    n_real = len(X_real_sub)

    nn_self = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute")
    nn_self.fit(X_real_sub)
    dists, _ = nn_self.kneighbors(X_real_sub)
    d_real = dists[:, 1].astype(np.float64)

    nn_real = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn_real.fit(emb_real_train)
    d_aug, _ = nn_real.kneighbors(emb_aug)
    d_aug = d_aug.ravel().astype(np.float64)

    p95 = float(np.percentile(d_real, 95))
    pct_within = float(np.mean(d_aug <= p95) * 100.0)

    try:
        mw = stats.mannwhitneyu(
            d_aug,
            d_real,
            alternative="less",
            method="auto",
        )
    except TypeError:
        mw = stats.mannwhitneyu(d_aug, d_real, alternative="less")

    return {
        "n_real_used_analysis1": float(n_real),
        "mean_d_real": float(np.mean(d_real)),
        "median_d_real": float(np.median(d_real)),
        "std_d_real": float(np.std(d_real)),
        "mean_d_aug": float(np.mean(d_aug)),
        "median_d_aug": float(np.median(d_aug)),
        "std_d_aug": float(np.std(d_aug)),
        "pct_aug_within_real_p95": pct_within,
        "mannwhitney_U": float(mw.statistic),
        "mannwhitney_p": float(mw.pvalue),
    }


def analysis_lof(
    emb_real_train: np.ndarray,
    emb_aug: np.ndarray,
    emb_test: np.ndarray,
    n_neighbors: int,
) -> Dict[str, float]:
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        metric="cosine",
        novelty=True,
        contamination="auto",
    )
    lof.fit(emb_real_train)

    pred_aug = lof.predict(emb_aug)
    pred_test = lof.predict(emb_test)
    inlier_rate_aug = float(np.mean(pred_aug == 1) * 100.0)
    inlier_rate_test = float(np.mean(pred_test == 1) * 100.0)

    score_aug = lof.decision_function(emb_aug)
    score_test = lof.decision_function(emb_test)

    return {
        "inlier_rate_aug": inlier_rate_aug,
        "inlier_rate_test": inlier_rate_test,
        "mean_lof_aug": float(np.mean(score_aug)),
        "mean_lof_test": float(np.mean(score_test)),
        "std_lof_aug": float(np.std(score_aug)),
        "std_lof_test": float(np.std(score_test)),
    }


def analysis_coverage(
    emb_real_train: np.ndarray,
    emb_aug: np.ndarray,
    emb_test: np.ndarray,
) -> Dict[str, float]:
    nn_before = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn_before.fit(emb_real_train)
    d_before, _ = nn_before.kneighbors(emb_test)
    d_before = d_before.ravel().astype(np.float64)

    combined = np.vstack([emb_real_train, emb_aug])
    nn_after = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
    nn_after.fit(combined)
    d_after, _ = nn_after.kneighbors(emb_test)
    d_after = d_after.ravel().astype(np.float64)

    pct_improved = float(np.mean(d_after < d_before) * 100.0)
    mean_reduction = float(np.mean(d_before - d_after))

    try:
        try:
            w = stats.wilcoxon(
                d_before,
                d_after,
                alternative="greater",
                method="auto",
            )
        except TypeError:
            try:
                w = stats.wilcoxon(d_before, d_after, alternative="greater")
            except TypeError:
                w = stats.wilcoxon(d_before, d_after)
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    return {
        "mean_d_before": float(np.mean(d_before)),
        "mean_d_after": float(np.mean(d_after)),
        "median_d_before": float(np.median(d_before)),
        "median_d_after": float(np.median(d_after)),
        "pct_improved": pct_improved,
        "mean_reduction": mean_reduction,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
    }


def aggregate_seeds(rows: List[Dict], numeric_keys: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in numeric_keys:
        vals = [r[k] for r in rows if k in r and r[k] == r[k]]  # skip nan
        if not vals:
            out[f"{k}_mean"] = float("nan")
            out[f"{k}_std"] = float("nan")
        else:
            a = np.array(vals, dtype=np.float64)
            out[f"{k}_mean"] = float(np.mean(a))
            out[f"{k}_std"] = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantitative augmentation quality (NN distance, LOF, coverage)."
    )
    parser.add_argument("--real_train_file", type=str, required=True)
    parser.add_argument(
        "--augmented_file",
        type=str,
        required=True,
        help="Path to augmented CSV or template with {seed} (e.g. ...seed{seed}.csv)",
    )
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="6,42,81")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MIMIC",
        help="Dataset label used in output files",
    )
    parser.add_argument(
        "--extract_mode",
        type=str,
        default="append_tail",
        choices=["append_tail", "anti_join"],
        help="How to isolate synthetic rows from the augmented CSV.",
    )
    parser.add_argument(
        "--analysis1_max_real",
        type=int,
        default=50000,
        help="Max real-train rows for Analysis 1 (subsample if larger).",
    )
    parser.add_argument(
        "--lof_n_neighbors",
        type=int,
        default=20,
        help="LOF n_neighbors (sklearn default 20).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    emb_obj = np.load(args.embeddings, allow_pickle=True).item()
    sample_key = next(iter(emb_obj))
    emb_dim = len(np.asarray(emb_obj[sample_key]).ravel())

    real_train = pd.read_csv(args.real_train_file).fillna("")
    test_df = pd.read_csv(args.test_file).fillna("")

    if LABEL_COL not in real_train.columns:
        raise ValueError(f"{args.real_train_file} must have column '{LABEL_COL}'")
    if LABEL_COL not in test_df.columns:
        raise ValueError(f"{args.test_file} must have column '{LABEL_COL}'")

    emb_real_train, n_empty_train = compute_embeddings_matrix(real_train, emb_obj, emb_dim)
    emb_test, n_empty_test = compute_embeddings_matrix(test_df, emb_obj, emb_dim)

    nn_rows: List[Dict] = []
    lof_rows: List[Dict] = []
    cov_rows: List[Dict] = []

    run_config = {
        "dataset_name": args.dataset_name,
        "real_train_file": args.real_train_file,
        "augmented_file_template": args.augmented_file,
        "test_file": args.test_file,
        "embeddings": args.embeddings,
        "emb_dim": emb_dim,
        "seeds": seeds,
        "extract_mode": args.extract_mode,
        "analysis1_max_real": args.analysis1_max_real,
        "n_empty_token_rows_train": n_empty_train,
        "n_empty_token_rows_test": n_empty_test,
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    for seed in seeds:
        aug_path = (
            args.augmented_file.format(seed=seed)
            if "{seed}" in args.augmented_file
            else args.augmented_file
        )
        if not os.path.isfile(aug_path):
            print(f"Warning: missing augmented file for seed {seed}: {aug_path}", file=sys.stderr)
            continue

        aug_full = pd.read_csv(aug_path).fillna("")
        synthetic_df = extract_synthetic_only(
            aug_full, real_train, args.extract_mode
        )
        if len(synthetic_df) == 0:
            print(f"Warning: zero synthetic rows for seed {seed}", file=sys.stderr)
            continue

        emb_aug, n_empty_aug = compute_embeddings_matrix(synthetic_df, emb_obj, emb_dim)

        rng = np.random.default_rng(seed + 10_000)
        nn_stats = analysis_nn_distances(
            emb_real_train,
            emb_aug,
            args.analysis1_max_real,
            rng,
        )
        nn_stats["seed"] = seed
        nn_rows.append(nn_stats)

        lof_stats = analysis_lof(
            emb_real_train,
            emb_aug,
            emb_test,
            args.lof_n_neighbors,
        )
        lof_stats["seed"] = seed
        lof_rows.append(lof_stats)

        cov_stats = analysis_coverage(emb_real_train, emb_aug, emb_test)
        cov_stats["seed"] = seed
        cov_rows.append(cov_stats)

        print(
            f"seed={seed} synthetic_n={len(synthetic_df)} "
            f"mean_d_aug={nn_stats['mean_d_aug']:.4f} "
            f"inlier_aug={lof_stats['inlier_rate_aug']:.1f}% "
            f"mean_reduction={cov_stats['mean_reduction']:.6f}"
        )

    if not nn_rows:
        print("No results to save.", file=sys.stderr)
        sys.exit(1)

    pd.DataFrame(nn_rows).to_csv(
        os.path.join(args.output_dir, "nn_distance_by_seed.csv"), index=False
    )
    pd.DataFrame(lof_rows).to_csv(
        os.path.join(args.output_dir, "lof_by_seed.csv"), index=False
    )
    pd.DataFrame(cov_rows).to_csv(
        os.path.join(args.output_dir, "coverage_by_seed.csv"), index=False
    )

    nn_keys = [
        "mean_d_real",
        "median_d_real",
        "std_d_real",
        "mean_d_aug",
        "median_d_aug",
        "std_d_aug",
        "pct_aug_within_real_p95",
        "mannwhitney_U",
        "mannwhitney_p",
        "n_real_used_analysis1",
    ]
    lof_keys = [
        "inlier_rate_aug",
        "inlier_rate_test",
        "mean_lof_aug",
        "mean_lof_test",
        "std_lof_aug",
        "std_lof_test",
    ]
    cov_keys = [
        "mean_d_before",
        "mean_d_after",
        "median_d_before",
        "median_d_after",
        "pct_improved",
        "mean_reduction",
        "wilcoxon_stat",
        "wilcoxon_p",
    ]

    nn_agg = aggregate_seeds(nn_rows, nn_keys)
    lof_agg = aggregate_seeds(lof_rows, lof_keys)
    cov_agg = aggregate_seeds(cov_rows, cov_keys)

    pd.DataFrame([nn_agg]).to_csv(
        os.path.join(args.output_dir, "nn_distance_aggregated.csv"), index=False
    )
    pd.DataFrame([lof_agg]).to_csv(
        os.path.join(args.output_dir, "lof_aggregated.csv"), index=False
    )
    pd.DataFrame([cov_agg]).to_csv(
        os.path.join(args.output_dir, "coverage_aggregated.csv"), index=False
    )

    summary = {"dataset_name": args.dataset_name, **nn_agg, **lof_agg, **cov_agg}
    pd.DataFrame([summary]).to_csv(
        os.path.join(args.output_dir, "augmentation_quality_summary.csv"), index=False
    )

    print(f"Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
