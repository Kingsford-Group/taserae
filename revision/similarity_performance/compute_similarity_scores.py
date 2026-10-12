#!/usr/bin/env python3
"""
Compute per-generated-sample similarity scores to real training samples.
"""

import argparse
import os
import random
import re
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _parse_list_field(value: str) -> List[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _encode_gender(value: str) -> np.ndarray:
    v = str(value).strip().lower()
    if v in {"m", "male"}:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if v in {"f", "female"}:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _safe_mean_embedding(tokens: List[str], embeddings: dict, dim: int) -> np.ndarray:
    vecs = [embeddings[t] for t in tokens if t in embeddings]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.array(vecs, dtype=np.float32), axis=0)


def _build_feature_matrix(df: pd.DataFrame, embeddings: dict, age_mean: float, age_std: float) -> np.ndarray:
    sample_key = next(iter(embeddings))
    emb_dim = len(embeddings[sample_key])
    feature_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding feature vectors"):
        drugs = _parse_list_field(row.get("administered_drugs", ""))
        complications = _parse_list_field(row.get("complications", ""))
        drug_vec = _safe_mean_embedding(drugs, embeddings, emb_dim)
        comp_vec = _safe_mean_embedding(complications, embeddings, emb_dim)

        age_raw = row.get("ageYear", np.nan)
        if pd.isna(age_raw):
            age_z = 0.0
        else:
            # Handle multi-value age fields from augmentation (e.g. '75, 68')
            age_str = str(age_raw)
            if "," in age_str:
                age_vals_row = [float(x.strip()) for x in age_str.split(",") if x.strip()]
                age_val = np.mean(age_vals_row) if age_vals_row else 0.0
            else:
                age_val = float(age_str)
            age_z = (age_val - age_mean) / age_std

        gender_vec = _encode_gender(row.get("gender", ""))
        feature_vec = np.concatenate(
            [drug_vec, comp_vec, np.array([age_z], dtype=np.float32), gender_vec],
            axis=0,
        )
        feature_rows.append(feature_vec)

    return np.array(feature_rows, dtype=np.float32)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return matrix / norms


def _max_pairwise_similarity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    sim = np.dot(query, reference.T)
    return np.max(sim, axis=1)


def _topk_mean_similarity(query: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    sim = np.dot(query, reference.T)
    k_eff = min(k, sim.shape[1])
    if k_eff <= 0:
        return np.zeros(sim.shape[0], dtype=np.float32)
    topk = np.partition(sim, -k_eff, axis=1)[:, -k_eff:]
    return np.mean(topk, axis=1)


def _labels_overlap(label_a: str, label_b: str) -> bool:
    la = {x.strip() for x in str(label_a).split(",") if x.strip()}
    lb = {x.strip() for x in str(label_b).split(",") if x.strip()}
    return len(la & lb) > 0


def _label_aware_max_similarity(
    generated_df: pd.DataFrame,
    generated_norm: np.ndarray,
    train_df: pd.DataFrame,
    train_norm: np.ndarray,
) -> np.ndarray:
    """Vectorized label-aware similarity: group by unique label, batch dot product."""
    train_label_sets = [
        frozenset(x.strip() for x in str(t).split(",") if x.strip())
        for t in train_df["labels"].tolist()
    ]
    gen_label_strs = generated_df["labels"].astype(str).tolist()
    n = len(generated_df)
    out = np.zeros(n, dtype=np.float32)

    # Group generated indices by their label string for batch processing
    from collections import defaultdict
    label_groups = defaultdict(list)
    for i, lbl in enumerate(gen_label_strs):
        label_groups[lbl].append(i)

    # Pre-compute full similarity as fallback
    full_sim = np.dot(generated_norm, train_norm.T)  # (n_gen, n_train)

    print(f"  Processing {len(label_groups)} unique label groups...")
    for lbl_str, indices in tqdm(label_groups.items(), desc="Label-aware NN similarity"):
        gen_label_set = frozenset(x.strip() for x in str(lbl_str).split(",") if x.strip())
        mask = np.array([len(gen_label_set & t_set) > 0 for t_set in train_label_sets], dtype=bool)
        idx_arr = np.array(indices)
        if np.any(mask):
            out[idx_arr] = np.max(full_sim[np.ix_(idx_arr, mask)], axis=1)
        else:
            out[idx_arr] = np.max(full_sim[idx_arr], axis=1)

    return out


def main():
    parser = argparse.ArgumentParser(description="Compute similarity scores for TASER-AE generated samples.")
    parser.add_argument(
        "--train_file",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "processed", "train_pure.csv"),
        help="Path to original real train file.",
    )
    parser.add_argument(
        "--augmented_file",
        type=str,
        required=True,
        help="Path to TASER-AE augmented CSV (contains original+generated rows).",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "processed", "embeddings.npy"),
        help="Path to embeddings.npy",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.path.join(SCRIPT_DIR, "outputs", "generated_similarity_scores.csv"),
        help="Output CSV path for per-sample scores.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Top-k for robust top-k average similarity.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. If omitted, inferred from augmented filename suffix seedX.",
    )
    args = parser.parse_args()

    inferred_seed = None
    match = re.search(r"seed(\d+)", os.path.basename(args.augmented_file))
    if match:
        inferred_seed = int(match.group(1))
    run_seed = args.seed if args.seed is not None else (inferred_seed if inferred_seed is not None else 42)
    _seed_everything(run_seed)
    print(f"Using seed: {run_seed}")

    if "{seed}" in args.output_csv:
        args.output_csv = args.output_csv.format(seed=run_seed)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    print(f"Loading train: {args.train_file}")
    train_df = pd.read_csv(args.train_file).fillna("")
    print(f"Loading augmented: {args.augmented_file}")
    aug_df = pd.read_csv(args.augmented_file).fillna("")

    if len(aug_df) < len(train_df):
        raise ValueError("Augmented file has fewer rows than train; expected original+generated concatenation.")

    generated_df = aug_df.iloc[len(train_df) :].copy().reset_index(drop=True)
    if generated_df.empty:
        raise ValueError("No generated rows detected. Check augmented_file content.")

    embeddings = np.load(args.embeddings, allow_pickle=True).item()
    if not embeddings:
        raise ValueError("Embeddings dictionary is empty.")

    age_vals = pd.to_numeric(train_df["ageYear"], errors="coerce")
    age_mean = float(np.nanmean(age_vals))
    age_std = float(np.nanstd(age_vals))
    if age_std == 0:
        age_std = 1.0

    train_mat = _build_feature_matrix(train_df, embeddings, age_mean, age_std)
    generated_mat = _build_feature_matrix(generated_df, embeddings, age_mean, age_std)

    train_norm = _normalize_rows(train_mat)
    generated_norm = _normalize_rows(generated_mat)

    sim_max = _max_pairwise_similarity(generated_norm, train_norm)
    sim_topk = _topk_mean_similarity(generated_norm, train_norm, args.topk)
    sim_labelaware = _label_aware_max_similarity(generated_df, generated_norm, train_df, train_norm)

    out = generated_df.copy()
    out.insert(0, "generated_index", np.arange(len(generated_df)))
    out["similarity_max"] = sim_max
    out["similarity_topk_mean"] = sim_topk
    out["similarity_labelaware_max"] = sim_labelaware
    out["nn_distance"] = 1.0 - sim_max
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out)} generated samples with similarity scores to: {args.output_csv}")


if __name__ == "__main__":
    main()
