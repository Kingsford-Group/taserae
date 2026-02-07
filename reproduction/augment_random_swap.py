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

def augment_row(row, feature):
    val = row[feature]
    if isinstance(val, str) and "," in val:
        items = [tok.strip() for tok in val.split(",") if tok.strip()]
        if len(items) < 2:
            return []
        
        # Swap two random items
        i, j = random.sample(range(len(items)), 2)
        items[i], items[j] = items[j], items[i]
        
        new = row.copy()
        new[feature] = ', '.join(items)
        return [new]
    return []

def main():
    parser = argparse.ArgumentParser(description="Augment data using Random Swap method")
    parser.add_argument("--input", type=str, default="../mimic4.csv", help="Path to input CSV")
    parser.add_argument("--strategy", type=str, default="1-1X", choices=None, help="Augmentation strategy")
    parser.add_argument("--multiplier", type=int, default=3, help="Multiplier for Target_Multiplier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    label_col = 'labels'
    args = parser.parse_args()
    augment_utils.seed_everything(args.seed)
    
    train_df = augment_utils.load_and_split_data(args.input)
    targets = augment_utils.get_augment_targets(train_df, strategy=args.strategy, multiplier=args.multiplier)
    print(f"Augmentation Targets ({args.strategy}): {sum(targets.values())} samples.")
    
    features_to_augment = ['administered_drugs', 'complications']
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
                for feature in features_to_augment:
                    cands = augment_row(row, feature)
                    candidates.extend(cands)
                    
                if candidates:
                    chosen = random.choice(candidates)
                    chosen['labels'] = label
                    augmented_rows.append(chosen)
                    generated_for_label += 1
                    pbar.update(1)
        pbar.close()
        total_generated += generated_for_label
        
    augment_utils.save_augmented_data(train_df, augmented_rows, "random_swap", args.strategy)

if __name__ == "__main__":
    main()
