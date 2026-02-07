#!/usr/bin/env python3
"""
Conditional GAN Augmentation for TASER-AE Benchmark

This script trains a Conditional GAN on minority class samples and generates
synthetic records to augment the training data.

Usage:
    python3 augment_gan.py --input train_pure.csv --strategy Target_Multiplier_1X_seed6 \
        --multiplier 2 --seed 6 --epochs 5000 --device cuda:0
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import augment_utils


def _parse_list_cell(val):
    if pd.isna(val) or not isinstance(val, str) or not val.strip():
        return []
    return [tok.strip() for tok in val.split(',') if tok.strip()]


def _parse_label_set(s):
    if pd.isna(s) or str(s).strip() == 'no':
        return set()
    return {tok.strip() for tok in str(s).split(",") if tok.strip() and tok.strip() != 'no'}


class Generator(nn.Module):
    def __init__(self, z_dim, cond_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, cond_dim, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, c):
        xc = torch.cat([x, c], dim=1)
        return self.net(xc)


def main():
    parser = argparse.ArgumentParser(description="GAN Augmentation for EHR data")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--strategy", type=str, default="Target_Multiplier_1X_seed42")
    parser.add_argument("--multiplier", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--z_dim", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    augment_utils.seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print("=" * 50)
    print("GAN Augmentation")
    print("=" * 50)
    
    # Load data
    label_col = 'labels'
    df = pd.read_csv(args.input).fillna("")
    labels = df[label_col].astype(str)
    
    # Filter to AE rows (minority classes)
    X_train = df.drop(columns=[label_col])
    ae_mask = labels != 'no'
    X_train_ae = X_train[ae_mask].copy()
    y_train_ae = labels[ae_mask].copy()
    
    print(f"Total samples: {len(df)}")
    print(f"AE samples for training GAN: {len(X_train_ae)}")
    
    # Build vocabularies
    drug_vocab = sorted(set(
        X_train_ae['administered_drugs'].dropna().apply(_parse_list_cell).explode().dropna().unique().tolist()
    ))
    comp_vocab = sorted(set(
        X_train_ae['complications'].dropna().apply(_parse_list_cell).explode().dropna().unique().tolist()
    ))
    gender_vocab = sorted(X_train_ae['gender'].dropna().astype(str).unique().tolist())
    
    label_sets = y_train_ae.apply(_parse_label_set)
    label_vocab = sorted(set().union(*label_sets) if len(label_sets) else set())
    label_idx = {lab: i for i, lab in enumerate(label_vocab)}
    
    # Age scaling
    age_min = float(X_train_ae['ageYear'].min()) if 'ageYear' in X_train_ae.columns else 0.0
    age_max = float(X_train_ae['ageYear'].max()) if 'ageYear' in X_train_ae.columns else 1.0
    age_rng = max(1e-8, age_max - age_min)
    
    # Empirical k samplers for decoding
    def _count_from_col(df_col):
        return df_col.fillna("").apply(lambda x: len(_parse_list_cell(x))).clip(lower=1).values
    
    drug_counts = _count_from_col(X_train_ae['administered_drugs'])
    comp_counts = _count_from_col(X_train_ae['complications'])
    
    def make_k_sampler(counts, k_max=None, k_min=1):
        vals, freqs = np.unique(counts, return_counts=True)
        probs = freqs / freqs.sum()
        def sample():
            k = int(np.random.choice(vals, p=probs))
            k = max(k_min, k)
            if k_max is not None:
                k = min(k, k_max)
            return k
        return sample
    
    MAX_DRUGS = 5
    MAX_COMPS = 3
    TAU = 0.55
    TEMP = 1.0
    
    drug_k_sampler = make_k_sampler(drug_counts, k_max=MAX_DRUGS)
    comp_k_sampler = make_k_sampler(comp_counts, k_max=MAX_COMPS)
    
    def encode_row(row):
        vec = []
        # age -> [0,1]
        if 'ageYear' in row:
            a = float(row['ageYear'])
            vec.append((a - age_min) / age_rng)
        else:
            vec.append(0.0)
        
        # gender one-hot
        g = str(row['gender']) if 'gender' in row and not pd.isna(row['gender']) else ''
        gender_oh = [1.0 if g == cat else 0.0 for cat in gender_vocab]
        vec.extend(gender_oh)
        
        # drugs multihot
        dlist = _parse_list_cell(row.get('administered_drugs', ''))
        dset = set(dlist)
        dmh = [1.0 if tok in dset else 0.0 for tok in drug_vocab]
        vec.extend(dmh)
        
        # complications multihot
        clist = _parse_list_cell(row.get('complications', ''))
        cset = set(clist)
        cmh = [1.0 if tok in cset else 0.0 for tok in comp_vocab]
        vec.extend(cmh)
        
        return np.array(vec, dtype=np.float32)
    
    def encode_cond(label_str):
        labs = _parse_label_set(label_str)
        return np.array([1.0 if lab in labs else 0.0 for lab in label_vocab], dtype=np.float32)
    
    def decode_vector(vec, label_str):
        out = {}
        i = 0
        
        # age
        age_norm = float(vec[i])
        i += 1
        age_val = age_min + np.clip(age_norm, 0.0, 1.0) * age_rng
        out['ageYear'] = int(round(age_val))
        
        # gender
        if len(gender_vocab):
            g_slice = vec[i:i+len(gender_vocab)]
            i += len(gender_vocab)
            g_idx = int(np.argmax(g_slice))
            out['gender'] = gender_vocab[g_idx] if g_idx < len(gender_vocab) else gender_vocab[0]
        else:
            out['gender'] = ""
        
        # drugs (top-k with threshold)
        if len(drug_vocab):
            d_slice = vec[i:i+len(drug_vocab)]
            i += len(drug_vocab)
            d_probs = 1.0 / (1.0 + np.exp(-(d_slice / max(1e-8, TEMP))))
            k_d = drug_k_sampler()
            order = np.argsort(d_probs)[::-1]
            topk = order[:k_d]
            chosen_idx = [j for j in topk if d_probs[j] >= TAU]
            if len(chosen_idx) == 0:
                chosen_idx = [order[0]]
            out['administered_drugs'] = ','.join(sorted({drug_vocab[j] for j in chosen_idx}))
        else:
            out['administered_drugs'] = ""
        
        # complications (top-k with threshold)
        if len(comp_vocab):
            c_slice = vec[i:i+len(comp_vocab)]
            i += len(comp_vocab)
            c_probs = 1.0 / (1.0 + np.exp(-(c_slice / max(1e-8, TEMP))))
            k_c = comp_k_sampler()
            order = np.argsort(c_probs)[::-1]
            topk = order[:k_c]
            chosen_idx = [j for j in topk if c_probs[j] >= TAU]
            if len(chosen_idx) == 0:
                chosen_idx = [order[0]]
            out['complications'] = ','.join(sorted({comp_vocab[j] for j in chosen_idx}))
        else:
            out['complications'] = ""
        
        out[label_col] = label_str
        return out
    
    # Build training arrays
    feat_mat = np.stack([encode_row(X_train_ae.loc[idx]) for idx in X_train_ae.index], axis=0)
    cond_mat = np.stack([encode_cond(y_train_ae.loc[idx]) for idx in X_train_ae.index], axis=0)
    
    feat_dim = feat_mat.shape[1]
    cond_dim = cond_mat.shape[1]
    n_samples = feat_mat.shape[0]
    
    print(f"Feature dim: {feat_dim}, Condition dim: {cond_dim}")
    
    # Create models
    G = Generator(args.z_dim, cond_dim, feat_dim, hidden=args.hidden).to(device)
    D = Discriminator(cond_dim, feat_dim, hidden=args.hidden).to(device)
    
    g_opt = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    d_opt = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    
    real_feats = torch.tensor(feat_mat, dtype=torch.float32, device=device)
    real_conds = torch.tensor(cond_mat, dtype=torch.float32, device=device)
    
    def batch_indices(n, bs):
        idx = np.random.permutation(n)
        for i in range(0, n, bs):
            yield idx[i:i+bs]
    
    print(f"\nTraining GAN for {args.epochs} epochs...")
    for ep in range(1, args.epochs + 1):
        for idxs in batch_indices(n_samples, args.batch_size):
            x_real = real_feats[idxs]
            c_real = real_conds[idxs]
            bs = x_real.size(0)
            
            # Train D
            D.train()
            G.train()
            d_opt.zero_grad()
            
            logits_real = D(x_real, c_real)
            y_real = torch.ones((bs, 1), device=device)
            loss_real = bce(logits_real, y_real)
            
            z = torch.randn(bs, args.z_dim, device=device)
            x_fake = G(z, c_real)
            logits_fake = D(x_fake.detach(), c_real)
            y_fake = torch.zeros((bs, 1), device=device)
            loss_fake = bce(logits_fake, y_fake)
            
            d_loss = loss_real + loss_fake
            d_loss.backward()
            d_opt.step()
            
            # Train G
            g_opt.zero_grad()
            logits_fake_g = D(x_fake, c_real)
            g_loss = bce(logits_fake_g, y_real)
            g_loss.backward()
            g_opt.step()
        
        if ep % 500 == 0 or ep == 1:
            print(f"  Epoch {ep}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
    
    # Calculate how many to generate (match 1X strategy)
    augment_targets = augment_utils.get_augment_targets(df, label_col, args.strategy, args.multiplier)
    n_gen = sum(augment_targets.values())
    print(f"\nGenerating {n_gen} synthetic samples...")
    
    # Generate samples
    G.eval()
    generated_rows = []
    
    with torch.no_grad():
        for label, count in tqdm(augment_targets.items(), desc="Generating"):
            if count <= 0:
                continue
            cond = encode_cond(label)
            cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0).repeat(count, 1)
            z = torch.randn(count, args.z_dim, device=device)
            x_gen = G(z, cond_tensor).cpu().numpy()
            
            for i in range(count):
                row = decode_vector(x_gen[i], label)
                generated_rows.append(row)
    
    augment_utils.save_augmented_data(df, generated_rows, "gan", args.strategy)


if __name__ == "__main__":
    main()
