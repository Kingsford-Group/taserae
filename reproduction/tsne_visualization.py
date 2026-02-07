#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os

# Set working directory to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")
AUG_DIR = os.path.join(SCRIPT_DIR, "augmented_ehr_data")
FIGS_DIR = os.path.join(SCRIPT_DIR, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)
os.chdir(SCRIPT_DIR)

print("Loading data...")
# Load original training data
train_pure = pd.read_csv(os.path.join(DATA_DIR, "train_pure.csv"))
train_pure["set"] = "real"

# Load augmented data
aug_file = os.path.join(AUG_DIR, "augmented_ehr_data_taser_ae_native_Target_Multiplier_1X_seed6.csv")
aug_data = pd.read_csv(aug_file)

# Extract only the new augmented rows (those not in original)
n_original = len(train_pure)
aug_only = aug_data.iloc[n_original:].copy()
aug_only["set"] = "TASER-AE"

# Combine
combined = pd.concat([train_pure, aug_only], ignore_index=True)
print(f"Total samples: {len(combined)} (original: {n_original}, augmented: {len(aug_only)})")

# Filter to minority classes only (no 'no') and single-label only (no combinations)
combined = combined[combined['labels'] != 'no']
# Keep only single-label samples (no comma in label = not multi-label)
combined = combined[~combined['labels'].str.contains(',', na=False)]
combined = combined.dropna()
print(f"After filtering to single-label minority classes: {len(combined)}")

# Extract features
y = combined["labels"].to_numpy()
domain = combined["set"].astype(str).to_numpy()
X_raw = combined.drop(columns=['labels', 'set']).astype(str).to_numpy()

# Tokenize
def tokenize_sample(sample):
    tokens = []
    for field in sample:
        for token in field.split(","):
            token = token.strip()
            if token:
                tokens.extend(token.split())
    return tokens

tokenized_samples = [tokenize_sample(sample) for sample in X_raw]

vocab = set()
for tokens in tokenized_samples:
    vocab.update(tokens)
vocab = sorted(vocab)
print(f"Vocabulary size: {len(vocab)}")

word2idx = {word: idx for idx, word in enumerate(vocab)}
sequences = [[word2idx[token] for token in tokens] for tokens in tokenized_samples]

max_seq_length = max(len(seq) for seq in sequences)
def pad_sequence(seq, max_len):
    return seq + [0]*(max_len - len(seq))

padded_sequences = [pad_sequence(seq, max_seq_length) for seq in sequences]
X = np.array(padded_sequences)
print(f"Padded input shape: {X.shape}")

# Downsample for visualization (25% of each, equal numbers)
rng = np.random.default_rng(42)
real_idx = np.flatnonzero(domain == "real")
aug_idx = np.flatnonzero(domain == "TASER-AE")

n_real = max(1, int(round(0.25 * len(real_idx))))
n_aug = max(1, int(round(0.25 * len(aug_idx))))
n_equal = min(n_real, n_aug)  

keep_real = rng.choice(real_idx, size=n_equal, replace=False) if len(real_idx) else np.array([], dtype=int)
keep_aug = rng.choice(aug_idx, size=n_equal, replace=False) if len(aug_idx) else np.array([], dtype=int)

keep_idx = np.concatenate([keep_real, keep_aug])
keep_idx.sort()


X_vis = X[keep_idx]
y_vis = y[keep_idx]
domain_vis = domain[keep_idx]

# Run t-SNE (adjusted for better spread)
print("Running t-SNE (perplexity=7, learning_rate=500)...")
X_emb = TSNE(n_components=2, perplexity=7, learning_rate=500, 
             early_exaggeration=8, metric="cosine", random_state=42).fit_transform(X_vis)
print("t-SNE complete!")

plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 17,
})

classes = np.unique(y_vis)
# Use tab10 colormap like original
cmap = plt.cm.get_cmap("tab10", len(classes))
color_lookup = {c: cmap(i) for i, c in enumerate(classes)}

# Figure 1: Real only (colored by class)
print("Creating Figure 1: Real only...")
fig, ax = plt.subplots(figsize=(6, 6))

mask = (domain_vis == "real")
for c in classes:
    cmask = mask & (y_vis == c)
    if cmask.sum() > 0:
        ax.scatter(
            X_emb[cmask, 0], X_emb[cmask, 1],
            marker="+",
            label=f"{c}",
            alpha=0.85,
            s=55,
            linewidths=1.6,
            c=[color_lookup[c]],
        )

ax.set_xlabel("t-SNE1")
ax.set_ylabel("t-SNE2")
ax.set_aspect("equal", adjustable="box")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.09),
          ncol=min(4, len(labels)), frameon=False, handletextpad=0.4, columnspacing=0.3, borderaxespad=0.0)
fig.tight_layout(pad=0.05)
fig.savefig(os.path.join(FIGS_DIR, "tsne_real_seed6.png"), dpi=300, bbox_inches="tight", pad_inches=0.0)
plt.close(fig)
print("Saved tsne_real_seed6.png")

# Figure 2: TASER-AE only (colored by class)
print("Creating Figure 2: TASER-AE only...")
fig, ax = plt.subplots(figsize=(6, 6))

mask = (domain_vis == "TASER-AE")
for c in classes:
    cmask = mask & (y_vis == c)
    if cmask.sum() > 0:
        ax.scatter(
            X_emb[cmask, 0], X_emb[cmask, 1],
            marker="+",
            label=f"{c}",
            alpha=0.85,
            s=55,
            linewidths=1.6,
            c=[color_lookup[c]],
        )

ax.set_xlabel("t-SNE1")
ax.set_ylabel("t-SNE2")
ax.set_aspect("equal", adjustable="box")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.09),
          ncol=min(4, len(labels)), frameon=False, handletextpad=0.4, columnspacing=0.3, borderaxespad=0.0)
fig.tight_layout(pad=0.05)
fig.savefig(os.path.join(FIGS_DIR, "tsne_taser_seed6.png"), dpi=300, bbox_inches="tight", pad_inches=0.0)
plt.close(fig)
print("Saved tsne_taser_seed6.png")

# Figure 3: Combined (real vs TASER-AE, NOT colored by class)
print("Creating Figure 3: Combined...")
fig, ax = plt.subplots(figsize=(6, 6))
markers = {"real": "o", "TASER-AE": "+"}

for origin, mk in markers.items():
    cmask = (domain_vis == origin)
    if cmask.sum() > 0:
        ax.scatter(
            X_emb[cmask, 0], X_emb[cmask, 1],
            marker=mk,
            label=origin,
            alpha=0.85,
            s=25,
            linewidths=1.6,
        )

ax.set_xlabel("t-SNE1")
ax.set_ylabel("t-SNE2")
ax.set_aspect("equal", adjustable="box")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.09),
          ncol=min(4, len(labels)), frameon=False, handletextpad=0.4, columnspacing=0.3, borderaxespad=0.0)
fig.tight_layout(pad=0.05)
fig.savefig(os.path.join(FIGS_DIR, "tsne_all_seed6.png"), dpi=300, bbox_inches="tight", pad_inches=0.0)
plt.close(fig)
print("Saved tsne_all_seed6.png")

print("\nDone! Generated 3 figures:")
print("  - tsne_real_seed6.png (real samples, colored by class)")
print("  - tsne_taser_seed6.png (augmented samples, colored by class)")
print("  - tsne_all_seed6.png (real vs augmented, domain markers)")
