import torch # Import torch first to avoid MKL deadlocks
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_similarity_dict(keys, embeddings, threshold=0.8):
    """
    Computes similarity dictionary for a list of keys (strings) given embeddings and threshold.
    Returns: dict {key: [list of similar keys]}
    """
    similarity_dict = {}
    
    # --- Semantic Dictionary Override ---
    if os.environ.get('MIMIC_USE_SEMANTIC_DICT') == '1':
        print(" [Using Semantic Similarity Dictionary for Augmentation] ")
        import pickle
        try:
            # Load Drugs
            with open("semantic_drug_similarity.pkl", 'rb') as f:
                drug_sim = pickle.load(f)
                
            # Load Complications
            try:
                with open("semantic_complication_similarity.pkl", 'rb') as f:
                    comp_sim = pickle.load(f)
            except FileNotFoundError:
                comp_sim = {}
                
            # Merge
            combined_sim = {**drug_sim, **comp_sim}
            
            count = 0
            for k in keys:
                if k in combined_sim:
                    similarity_dict[k] = combined_sim[k]
                    count += 1
            
            print(f"Loaded {count} semantic entries matching input keys.")
            return similarity_dict
            
        except Exception as e:
            print(f"Error loading Semantic Dictionaries: {e}")
            print("Falling back to Vector Similarity...")
            
    
    valid_keys = [k for k in keys if k in embeddings]
    if not valid_keys:
        return {}
        
    # Stack embeddings
    vectors = np.array([embeddings[k] for k in valid_keys])
    
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = vectors / norms
    
    # Compute Sim Matrix
    sim_matrix = np.dot(normalized, normalized.T)
    
    # Extract results
    for i, key in enumerate(tqdm(valid_keys, desc="Extracting similarities")):
        row = sim_matrix[i]
        row[i] = -1
        
        matches = np.where(row > threshold)[0]
        similarity_dict[key] = [valid_keys[m] for m in matches]
        
    return similarity_dict


def load_and_split_data(filepath, label_col='labels', test_size=0.2, random_state=42):
    """
    Loads data from filepath.
    Checks if train_split.csv and test_split.csv exist in the same directory as the script/output.
    If not, performs stratified split and saves them.
    Returns: train_df
    """
    base_dir = os.path.dirname(filepath)

    # If input file is "train_pure.csv", assume it's already pre-split and load directly
    if "train_pure.csv" in filepath:
        print(f"Loading pre-split training data from {filepath}...")
        return pd.read_csv(filepath)

    if os.path.exists("train_split.csv") and os.path.exists("test_split.csv"):
        print("Loading existing train/test splits...")
        train_df = pd.read_csv("train_split.csv")
        # Validate if needed
        return train_df

    print(f"Loading original data from {filepath}...")
    df = pd.read_csv(filepath)
    

    counts = df[label_col].value_counts()
    filtered_labels = counts[counts > 1].index.tolist()
    df = df[df[label_col].isin(filtered_labels)].copy()
    
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    print("Performing stratified split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    train_df = X_train.copy()
    train_df[label_col] = y_train
    
    test_df = X_test.copy()
    test_df[label_col] = y_test
    
    print("Saving train_split.csv and test_split.csv...")
    train_df.to_csv("train_split.csv", index=False)
    test_df.to_csv("test_split.csv", index=False)
    
    return train_df

def get_augment_targets(df, label_col='labels', strategy='1-1X', multiplier=3):
    """
    Calculates how many NEW samples to generate for each class.
    Returns: dict {label: num_samples_to_generate}
    """
    all_labels = []
    for val in df[label_col].dropna().astype(str):
        labels = [l.strip() for l in val.split(',') if l.strip()]
        all_labels.extend(labels)
        
    counts = pd.Series(all_labels).value_counts()
    majority_class = counts.idxmax()
    majority_count = counts.max()
    
    print("Label Counts (Individual):")
    print(counts)
    
    targets = {}
    
    for label, count in counts.items():
        if label == 'no' or label == majority_class: 
             continue
             
        if strategy == '1-1X':
            targets[label] = count
            
        elif strategy.startswith('Fixed_'):
            # Parse target from name, e.g., Fixed_5000
            try:
                target_val = int(strategy.split('_')[1])
            except:
                target_val = 10000 # default fallback
                
            needed = target_val - count
            if needed > 0:
                targets[label] = needed
                
        elif 'Target_Multiplier' in strategy:
            needed = int(count * multiplier)
            if needed > 0:
                targets[label] = needed
                
    return targets

def get_samples_for_label(df, label, label_col=None):
    """
    Returns rows where label is present in the label_col (comma-separated).
    """
    if label_col is None:
        label_col = 'labels' if 'labels' in df.columns else 'adverse_events'
    mask = df[label_col].astype(str).apply(lambda x: label in [l.strip() for l in x.split(',')])
    return df[mask]

def save_augmented_data(original_train_df, augmented_rows, method_name, strategy_name):
    """
    Concatenates original train data with augmented rows and saves to CSV.
    """
    # If augmented_rows is empty, we still want to save the original df (for 0 augmentation strat)
    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        combined_df = pd.concat([original_train_df, aug_df], ignore_index=True)
    else:
        print("No augmented rows generated. Saving original data only.")
        combined_df = original_train_df.copy()
    
    # Save to augmented_ehr_data/ subfolder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "augmented_ehr_data")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"augmented_ehr_data_{method_name}_{strategy_name}.csv"
    output_path = os.path.join(output_dir, filename)
    print(f"Saving {len(combined_df)} rows to {output_path}...")
    combined_df.to_csv(output_path, index=False)
