#!/usr/bin/env python3
"""
Conditional VAE Augmentation for TASER-AE Benchmark

This script trains a Conditional VAE on minority class samples and generates
synthetic records to augment the training data.

Usage:
    python3 augment_vae.py --input train_pure.csv --strategy Target_Multiplier_1X_seed6 \
        --multiplier 2 --seed 6 --epochs 300 --device cuda:0
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
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import augment_utils


def _split_list_cell(val):
    if pd.isna(val) or not isinstance(val, str) or not val.strip():
        return []
    return [tok.strip() for tok in val.split(',') if tok.strip()]


def _parse_label_set(s):
    if pd.isna(s) or str(s).strip() == 'no':
        return set()
    return {x.strip() for x in str(s).split(',') if x.strip() and x.strip() != 'no'}


class AEVAEDataset(Dataset):
    def __init__(self, df, y_series, encode_row_fn, encode_labels_fn):
        self.df = df
        self.y = y_series.astype(str)
        self.idxs = list(df.index)
        self.encode_row = encode_row_fn
        self.encode_labels = encode_labels_fn

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        idx = self.idxs[i]
        row = self.df.loc[idx]
        x = self.encode_row(row)
        cond = self.encode_labels(self.y.loc[idx])
        return torch.from_numpy(x), torch.from_numpy(cond)


class CVAE(nn.Module):
    def __init__(self, x_dim, cond_dim, latent_dim=64, hidden=512):
        super().__init__()
        in_dim = x_dim + cond_dim
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim),
            nn.Sigmoid()
        )

    def encode(self, x, cond):
        h = self.enc(torch.cat([x, cond], dim=1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        return self.dec(torch.cat([z, cond], dim=1))

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, cond)
        return x_hat, mu, logvar


def main():
    parser = argparse.ArgumentParser(description="VAE Augmentation for EHR data")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--strategy", type=str, default="Target_Multiplier_1X_seed42")
    parser.add_argument("--multiplier", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    augment_utils.seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print("=" * 50)
    print("VAE Augmentation")
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
    print(f"AE samples for training VAE: {len(X_train_ae)}")
    
    # Build vocabularies
    unique_drugs = sorted(set(
        X_train_ae['administered_drugs'].dropna().apply(_split_list_cell).explode().dropna().unique().tolist()
    ))
    unique_comps = sorted(set(
        X_train_ae['complications'].dropna().apply(_split_list_cell).explode().dropna().unique().tolist()
    ))
    unique_gender = sorted(X_train_ae['gender'].astype(str).fillna("Unknown").unique().tolist())
    
    label_sets = y_train_ae.apply(_parse_label_set)
    atomic_labels = sorted(set.union(*label_sets) if len(label_sets) else set())
    
    # Index maps
    sex_to_idx = {s: i for i, s in enumerate(unique_gender)}
    drug_to_idx = {d: i for i, d in enumerate(unique_drugs)}
    comp_to_idx = {c: i for i, c in enumerate(unique_comps)}
    label_to_idx = {l: i for i, l in enumerate(atomic_labels)}
    
    # Age normalization
    age_series = pd.to_numeric(X_train_ae['ageYear'], errors='coerce')
    age_min = float(np.nanmin(age_series.values)) if np.sum(~np.isnan(age_series.values)) else 0.0
    age_max = float(np.nanmax(age_series.values)) if np.sum(~np.isnan(age_series.values)) else 1.0
    if age_max <= age_min:
        age_max = age_min + 1.0
    
    # Dimensions
    AGE_DIM = 1
    GENDER_DIM = len(unique_gender)
    DRUG_DIM = len(unique_drugs)
    COMP_DIM = len(unique_comps)
    X_DIM = AGE_DIM + GENDER_DIM + DRUG_DIM + COMP_DIM
    COND_DIM = len(atomic_labels)
    
    print(f"Feature dim: {X_DIM}, Condition dim: {COND_DIM}")
    
    def encode_row_to_vector(row):
        x = np.zeros(X_DIM, dtype=np.float32)
        age = pd.to_numeric(row.get('ageYear', np.nan), errors='coerce')
        if np.isnan(age):
            age_norm = 0.0
        else:
            age_norm = (float(age) - age_min) / (age_max - age_min)
            age_norm = float(np.clip(age_norm, 0.0, 1.0))
        x[0] = age_norm
        
        gender = str(row.get('gender', 'Unknown'))
        g_idx = sex_to_idx.get(gender, None)
        if g_idx is not None:
            x[1 + g_idx] = 1.0
        
        start = 1 + GENDER_DIM
        for d in _split_list_cell(row.get('administered_drugs', '')):
            j = drug_to_idx.get(d, None)
            if j is not None:
                x[start + j] = 1.0
        
        start2 = 1 + GENDER_DIM + DRUG_DIM
        for c in _split_list_cell(row.get('complications', '')):
            j = comp_to_idx.get(c, None)
            if j is not None:
                x[start2 + j] = 1.0
        
        return x
    
    def encode_labels_to_cond(label_str):
        cond = np.zeros(COND_DIM, dtype=np.float32)
        labs = _parse_label_set(label_str)
        for l in labs:
            idx = label_to_idx.get(l, None)
            if idx is not None:
                cond[idx] = 1.0
        return cond
    
    def decode_vector_to_row(vec, label_str):
        out = {}
        age_norm = float(np.clip(vec[0], 0.0, 1.0))
        out['ageYear'] = int(round(age_min + age_norm * (age_max - age_min)))
        
        if GENDER_DIM > 0:
            g_slice = vec[1:1+GENDER_DIM]
            g_idx = int(np.argmax(g_slice))
            out['gender'] = unique_gender[g_idx] if g_idx < len(unique_gender) else unique_gender[0]
        else:
            out['gender'] = ""
        
        start = 1 + GENDER_DIM
        if DRUG_DIM > 0:
            d_slice = vec[start:start+DRUG_DIM]
            drugs = [unique_drugs[i] for i in range(len(unique_drugs)) if d_slice[i] > 0.5]
            if not drugs:
                drugs = [unique_drugs[int(np.argmax(d_slice))]]
            out['administered_drugs'] = ','.join(drugs)
        else:
            out['administered_drugs'] = ""
        
        start2 = 1 + GENDER_DIM + DRUG_DIM
        if COMP_DIM > 0:
            c_slice = vec[start2:start2+COMP_DIM]
            comps = [unique_comps[i] for i in range(len(unique_comps)) if c_slice[i] > 0.5]
            if not comps:
                comps = [unique_comps[int(np.argmax(c_slice))]]
            out['complications'] = ','.join(comps)
        else:
            out['complications'] = ""
        
        out[label_col] = label_str
        return out
    
    # Create dataset and train
    dataset = AEVAEDataset(X_train_ae, y_train_ae, encode_row_to_vector, encode_labels_to_cond)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    model = CVAE(x_dim=X_DIM, cond_dim=COND_DIM, latent_dim=args.latent_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    bce = nn.BCELoss(reduction='mean')
    mse = nn.MSELoss(reduction='mean')
    
    bin_mask = torch.ones(X_DIM, dtype=torch.bool)
    bin_mask[0] = False  # age uses MSE
    bin_mask = bin_mask.to(device)
    
    beta_kld = 1.0
    age_weight = 10.0
    
    print(f"\nTraining CVAE for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, cond in loader:
            x = x.to(device)
            cond = cond.to(device)
            
            x_hat, mu, logvar = model(x, cond)
            
            bce_loss = bce(x_hat[:, bin_mask], x[:, bin_mask])
            mse_loss = mse(x_hat[:, [0]], x[:, [0]])
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = bce_loss + age_weight * mse_loss + beta_kld * kld
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item() * x.size(0)
        
        epoch_loss /= len(dataset)
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: loss={epoch_loss:.4f}")
    
    # Calculate how many to generate (match 1X strategy)
    augment_targets = augment_utils.get_augment_targets(df, label_col, args.strategy, args.multiplier)
    n_gen = sum(augment_targets.values())
    print(f"\nGenerating {n_gen} synthetic samples...")
    
    # Generate samples
    model.eval()
    generated_rows = []
    
    with torch.no_grad():
        for label, count in tqdm(augment_targets.items(), desc="Generating"):
            if count <= 0:
                continue
            cond = encode_labels_to_cond(label)
            cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0).repeat(count, 1)
            z = torch.randn(count, args.latent_dim, device=device)
            x_gen = model.decode(z, cond_tensor).cpu().numpy()
            
            for i in range(count):
                row = decode_vector_to_row(x_gen[i], label)
                generated_rows.append(row)
    
    augment_utils.save_augmented_data(df, generated_rows, "vae", args.strategy)


if __name__ == "__main__":
    main()
