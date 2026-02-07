#!/usr/bin/env python3
"""
Decision Tree Baseline for TASER-AE Benchmark

This script trains a Decision Tree classifier on the non-augmented training data
and evaluates on the test set. Uses the same preprocessing as the Transformer.

Usage:
    python3 baseline_decision_tree.py --train_file train_pure.csv --test_file test_split.csv \
        --seed 42 --save_metrics metrics_decision_tree_baseline.json
"""

import pandas as pd
import numpy as np
import argparse
import json
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import augment_utils


def tokenize_sample(sample):
    """Tokenize a sample (same as Transformer)."""
    tokens = []
    for field in sample:
        for token in field.split(","):
            token = token.strip()
            if token:
                tokens.extend(token.split())
    return tokens


def main():
    parser = argparse.ArgumentParser(description="Decision Tree Baseline")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_metrics", type=str, default="metrics_decision_tree_baseline.json")
    parser.add_argument("--use_gridsearch", action="store_true", help="Use GridSearchCV (slower)")
    args = parser.parse_args()
    
    augment_utils.seed_everything(args.seed)
    
    print("=" * 50)
    print("Decision Tree Baseline")
    print("=" * 50)
    
    # Load data
    label_col = 'labels'
    train_df = pd.read_csv(args.train_file).fillna("")
    test_df = pd.read_csv(args.test_file).fillna("")
    
    y_train_raw = train_df[label_col].astype(str).tolist()
    y_test_raw = test_df[label_col].astype(str).tolist()
    
    X_train_df = train_df.drop(columns=[label_col])
    X_test_df = test_df.drop(columns=[label_col])
    
    X_train_raw = X_train_df.astype(str).to_numpy()
    X_test_raw = X_test_df.astype(str).to_numpy()
    
    # Filter to single-label samples (for multiclass classification)
    def is_single_label(label_str):
        labels = [l.strip() for l in label_str.split(",")]
        return len(labels) == 1
    
    train_mask = [is_single_label(l) for l in y_train_raw]
    test_mask = [is_single_label(l) for l in y_test_raw]
    
    X_train_raw = X_train_raw[train_mask]
    y_train_str = [y_train_raw[i].strip() for i in range(len(y_train_raw)) if train_mask[i]]
    
    X_test_raw = X_test_raw[test_mask]
    y_test_str = [y_test_raw[i].strip() for i in range(len(y_test_raw)) if test_mask[i]]
    
    print(f"Train samples (single-label): {len(y_train_str)}")
    print(f"Test samples (single-label): {len(y_test_str)}")
    
    # Tokenize and encode
    tokenized_train = [tokenize_sample(sample) for sample in X_train_raw]
    tokenized_test = [tokenize_sample(sample) for sample in X_test_raw]
    
    vocab = set()
    for tokens in tokenized_train:
        vocab.update(tokens)
    vocab = sorted(vocab)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    
    def encode_sequences(tokenized_samples):
        sequences = []
        for tokens in tokenized_samples:
            seq = [word2idx.get(t, 0) for t in tokens]
            sequences.append(seq)
        return sequences
    
    train_seqs = encode_sequences(tokenized_train)
    test_seqs = encode_sequences(tokenized_test)
    
    max_len = max(max(len(s) for s in train_seqs), max(len(s) for s in test_seqs) if test_seqs else 0)
    
    def pad_sequence(seq, max_len):
        return seq + [0] * (max_len - len(seq))
    
    X_train = np.array([pad_sequence(s, max_len) for s in train_seqs])
    X_test = np.array([pad_sequence(s, max_len) for s in test_seqs])
    
    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_test = le.transform(y_test_str)
    
    print(f"Classes: {le.classes_}")
    
    # Train Decision Tree
    if args.use_gridsearch:
        print("Training with GridSearchCV...")
        params_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }
        dt = DecisionTreeClassifier(random_state=args.seed)
        grid_search = GridSearchCV(dt, params_grid, scoring='f1_macro', cv=5, verbose=1, n_jobs=4)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
    else:
        print("Training Decision Tree (default params)...")
        model = DecisionTreeClassifier(random_state=args.seed)
        model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    metrics = {}
    metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_test, y_pred, average='micro', zero_division=0)
    metrics['prec_macro'] = precision_score(y_test, y_pred, average='macro', zero_division=0)
    metrics['rec_macro'] = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    for idx, label in enumerate(le.classes_):
        y_true_bin = (y_test == idx).astype(int)
        y_pred_bin = (y_pred == idx).astype(int)
        metrics[f'f1_{label}'] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        metrics[f'prec_{label}'] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        metrics[f'rec_{label}'] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    
    print("-" * 50)
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Micro F1: {metrics['f1_micro']:.4f}")
    print("-" * 50)
    
    # Save metrics
    with open(args.save_metrics, 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
    print(f"Saved metrics to {args.save_metrics}")


if __name__ == "__main__":
    main()
