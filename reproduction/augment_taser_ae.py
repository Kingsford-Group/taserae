import pandas as pd
import numpy as np
import argparse
import os
import sys
import random
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import augment_utils

# --- Component Functions ---

def augment_drop(items):
    """Randomly drop one item from a list."""
    if len(items) < 2:
        return None
    drop_idx = random.randint(0, len(items) - 1)
    return items[:drop_idx] + items[drop_idx+1:]

def augment_shuffle(items):
    """Shuffle items."""
    if len(items) < 2:
        return None
    new_items = items[:]
    random.shuffle(new_items)
    return new_items

def augment_insertion(items, sim_dict):
    """Insert a synonym of a random token at a random position."""
    candidate_ids = [i for i, item in enumerate(items) if item in sim_dict and sim_dict[item]]
    if not candidate_ids:
        return None
    target_idx = random.choice(candidate_ids)
    token = items[target_idx]
    synonyms = [s for s in sim_dict[token] if s not in items]
    if not synonyms:
        return None
    synonym = random.choice(synonyms)
    insert_pos = random.randint(0, len(items))
    new_items = items[:]
    new_items.insert(insert_pos, synonym)
    return new_items

def augment_row_insertion(row, feature, sim_dict):
    """
    Synonym insertion on any feature (age, gender, drugs, complications).
    For multi-value fields (comma-separated): insert a synonym into the list.
    For single-value fields: append a synonym to create a pair.
    """
    generated = []
    original_val = row[feature]

    if isinstance(original_val, str) and "," in original_val:
        items = [tok.strip() for tok in original_val.split(",") if tok.strip()]
        inserted = augment_insertion(items, sim_dict)
        if inserted:
            new = row.copy()
            new[feature] = ', '.join(inserted)
            generated.append(new)
    elif not pd.isna(original_val):
        key = str(original_val)
        if key in sim_dict and sim_dict[key]:
            synonym = random.choice(sim_dict[key])
            new = row.copy()
            new[feature] = f"{key}, {synonym}"
            generated.append(new)

    return generated

def augment_row_shuffle(row, feature):
    """Shuffle augmentation — only for multi-value (list) features."""
    generated = []
    val = row[feature]
    if isinstance(val, str) and "," in val:
        items = [tok.strip() for tok in val.split(",") if tok.strip()]
        shuffled = augment_shuffle(items)
        if shuffled:
            new = row.copy()
            new[feature] = ', '.join(shuffled)
            generated.append(new)
    return generated

def augment_row_drop(row, feature):
    """Random drop augmentation — only for multi-value (list) features."""
    generated = []
    val = row[feature]
    if isinstance(val, str) and "," in val:
        items = [tok.strip() for tok in val.split(",") if tok.strip()]
        dropped = augment_drop(items)
        if dropped:
            new = row.copy()
            new[feature] = ', '.join(dropped)
            generated.append(new)
    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Augment data using TASER-AE: synonym insertion on all 4 features "
                    "(age, sex, drugs, complications) + shuffle & random drop on drugs/complications"
    )
    parser.add_argument("--input", type=str, default="../mimic4.csv", help="Path to input CSV")
    parser.add_argument("--embeddings", type=str, default="../ishikawa/embeddings.npy", help="Path to embeddings.npy")
    parser.add_argument("--strategy", type=str, default="1-1X", choices=None, help="Augmentation strategy")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold")
    parser.add_argument("--multiplier", type=float, default=3, help="Multiplier for Target_Multiplier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    augment_utils.seed_everything(args.seed)

    train_df = augment_utils.load_and_split_data(args.input)
    targets = augment_utils.get_augment_targets(train_df, strategy=args.strategy, multiplier=args.multiplier)
    print(f"Augmentation Targets ({args.strategy}): {sum(targets.values())} samples.")

    print(f"Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings, allow_pickle=True).item()

    # Build similarity dicts for ALL 4 features (needed for synonym insertion on all)
    all_features = ['ageYear', 'gender', 'administered_drugs', 'complications']
    similarity_dicts = {}

    for feature in all_features:
        if feature in ['administered_drugs', 'complications']:
            unique_vals = train_df[feature].dropna().apply(
                lambda x: [item.strip() for item in x.split(',')]
            ).explode().unique()
        else:
            unique_vals = train_df[feature].unique().astype(str)

        sim_dict = augment_utils.compute_similarity_dict(unique_vals, embeddings, args.threshold)
        similarity_dicts[feature] = sim_dict

    # Features for each operation
    insertion_features = ['ageYear', 'gender', 'administered_drugs', 'complications']
    list_only_features = ['administered_drugs', 'complications']  # shuffle & drop

    augmented_rows = []
    total_generated = 0

    for label, target in targets.items():
        if target <= 0:
            continue

        group_df = augment_utils.get_samples_for_label(train_df, label)
        generated_for_label = 0
        pbar = tqdm(total=target, desc=f"Augmenting {label}")

        while generated_for_label < target:
            sample_df = group_df.sample(frac=1, random_state=args.seed)
            for _, row in sample_df.iterrows():
                if generated_for_label >= target:
                    break

                candidates = []

                # 1. Synonym insertion on ALL 4 features
                for feature in insertion_features:
                    if feature in similarity_dicts:
                        cands = augment_row_insertion(row, feature, similarity_dicts[feature])
                        candidates.extend(cands)

                # 2. Shuffle on drugs & complications only
                for feature in list_only_features:
                    cands = augment_row_shuffle(row, feature)
                    candidates.extend(cands)

                # 3. Random drop on drugs & complications only
                for feature in list_only_features:
                    cands = augment_row_drop(row, feature)
                    candidates.extend(cands)

                if candidates:
                    chosen = random.choice(candidates)
                    augmented_rows.append(chosen)
                    generated_for_label += 1
                    pbar.update(1)
        pbar.close()
        total_generated += generated_for_label

    augment_utils.save_augmented_data(train_df, augmented_rows, "taser_ae", args.strategy)

if __name__ == "__main__":
    main()
