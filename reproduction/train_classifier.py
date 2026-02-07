import torch 
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import argparse
import os
import json
import copy
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import augment_utils

# Device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model Definition
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, batch_first=True, dropout=0.1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding_layer(x) 
        
        # Self attention
        attn_output, _ = self.self_attention(embedded, embedded, embedded)
        
        # Pooling (mean over sequence dim)
        pooled = attn_output.mean(dim=1)
        pooled = self.dropout(pooled)
        
        logits = self.fc(pooled)
        return logits

class EHRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def tokenize_sample(sample):
    tokens = []
    # iterating over fields in row
    for field in sample:
        # split by comma
        for token in field.split(","):
            token = token.strip()
            if token:
                tokens.extend(token.split()) # split by space too? matching notebook logic
    return tokens

def main():
    parser = argparse.ArgumentParser(description="Train Transformer Classifier")
    parser.add_argument("--train_file", type=str, required=True, help="Path to augmented training CSV")
    parser.add_argument("--val_file", type=str, default=None, help="Path to validation CSV (Optional)")
    parser.add_argument("--test_file", type=str, default="test_split.csv", help="Path to test CSV")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs") 
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--save_metrics", type=str, default="metrics.json", help="Path to save metrics")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:1)")
    
    args = parser.parse_args()
    augment_utils.seed_everything(args.seed)
    
    # Update global device
    global device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 1. Load Data
    print(f"Loading training data from {args.train_file}...")
    train_df = pd.read_csv(args.train_file)
    
    print(f"Loading test data from {args.test_file}...")
    if not os.path.exists(args.test_file):
        print(f"Error: Test file {args.test_file} not found. Please ensure augment_utils has run.")
        sys.exit(1)
    test_df = pd.read_csv(args.test_file)
    
    train_df = train_df.fillna("")
    test_df = test_df.fillna("")
    
    label_col = 'labels'
    y_train_raw = train_df[label_col].astype(str).tolist()
    y_test_raw = test_df[label_col].astype(str).tolist()

    # Prepare features: Drop the label column, keep everything else
    X_train_df = train_df.drop(columns=[label_col])
    X_test_df = test_df.drop(columns=[label_col])
    
    # Convert to strings and numpy
    X_train_raw = X_train_df.astype(str).to_numpy()
    X_test_raw = X_test_df.astype(str).to_numpy()
    
    # Process labels
    def process_labels(label_str):
        return [l.strip() for l in label_str.split(",")]
        
    y_train_processed = [process_labels(l) for l in y_train_raw]
    y_test_processed = [process_labels(l) for l in y_test_raw]
    
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train_processed)
    y_test = mlb.transform(y_test_processed)
    
    num_classes = y_train.shape[1]
    print(f"Num classes: {num_classes}")
    print(f"Classes: {mlb.classes_}")
    
    # Tokenization
    print("Tokenizing...")
    tokenized_train = [tokenize_sample(sample) for sample in X_train_raw]
    tokenized_test = [tokenize_sample(sample) for sample in X_test_raw]
    
    # Build Vocab
    vocab = set()
    for tokens in tokenized_train:
        vocab.update(tokens)
    vocab = sorted(vocab)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Convert to sequences
    def encode_sequences(tokenized_samples, w2i):
        sequences = []
        for tokens in tokenized_samples:
            seq = []
            for token in tokens:
                if token in w2i:
                    seq.append(w2i[token])
                # else skip unknown
            sequences.append(seq)
        return sequences
        
    train_seqs = encode_sequences(tokenized_train, word2idx)
    test_seqs = encode_sequences(tokenized_test, word2idx)
    
    # Padding
    max_len_train = max(len(s) for s in train_seqs) if train_seqs else 0
    max_len_test = max(len(s) for s in test_seqs) if test_seqs else 0
    max_seq_length = max(max_len_train, max_len_test)
    print(f"Max sequence length: {max_seq_length}")
    
    def pad_sequence(seq, max_len):
        return seq + [0]*(max_len - len(seq))
    
    X_train = np.array([pad_sequence(s, max_seq_length) for s in train_seqs])
    X_test = np.array([pad_sequence(s, max_seq_length) for s in test_seqs])
    

    
    if args.val_file and os.path.exists(args.val_file):
        print(f"Loading external validation data from {args.val_file}...")
        val_df = pd.read_csv(args.val_file).fillna("")
        
        y_val_raw = val_df[label_col].astype(str).tolist()
        y_val_processed = [process_labels(l) for l in y_val_raw]
        y_val = mlb.transform(y_val_processed)
        
        X_val_df = val_df.drop(columns=[label_col])
        X_val_raw = X_val_df.astype(str).to_numpy()
        tokenized_val = [tokenize_sample(sample) for sample in X_val_raw]
        val_seqs = encode_sequences(tokenized_val, word2idx)
        X_val = np.array([pad_sequence(s, max_seq_length) for s in val_seqs])
        
        # Train set is fully used
        X_train_chk = X_train
        y_train_chk = y_train
        
    else:
        print("Using random split for validation...")
        from sklearn.model_selection import train_test_split
        # Split augmented train into train/val
        # Stratify on string labels to ensure distribution is preserved
        X_train_chk, X_val, y_train_chk, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=args.seed, stratify=y_train_raw
        )
    
    train_dataset = EHRDataset(X_train_chk, y_train_chk)
    val_dataset = EHRDataset(X_val, y_val)
    test_dataset = EHRDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Model Setup
    embedding_dim = 64
    model = TransformerModel(vocab_size, embedding_dim, num_classes)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training Loop
    best_minority_f1 = -1.0
    patience_counter = 0
    best_model_state = None
    
    minority_indices = [i for i, label in enumerate(mlb.classes_) if label != 'no']
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation Metrics for Early Stopping
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())
        
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_binary = (val_preds >= 0.5).astype(int)
        
        # Compute Minority Macro F1
        val_minority_f1 = 0.0
        if minority_indices:
            val_minority_f1 = f1_score(val_targets[:, minority_indices], val_binary[:, minority_indices], average='macro', zero_division=0)
            
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}, Val Min F1: {val_minority_f1:.4f}")
        
        if val_minority_f1 > best_minority_f1:
            best_minority_f1 = val_minority_f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    if best_model_state:
        print(f"Restoring best model (Val Min F1: {best_minority_f1:.4f})...")
        model.load_state_dict(best_model_state)
            
    # Final Evaluation on Test Set
    print("\nEvaluating on Test Set...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            outputs = torch.sigmoid(logits) # Apply sigmoid for metrics
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_y.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    threshold = 0.5
    binary_preds = (all_preds >= threshold).astype(int)
    
    # Global Metrics (Micro/Macro)
    metrics = {}
    for avg in ['micro', 'macro']:
        metrics[f'f1_{avg}'] = f1_score(all_labels, binary_preds, average=avg, zero_division=0)
        metrics[f'prec_{avg}'] = precision_score(all_labels, binary_preds, average=avg, zero_division=0)
        metrics[f'rec_{avg}'] = recall_score(all_labels, binary_preds, average=avg, zero_division=0)
        try:
            metrics[f'auc_{avg}'] = roc_auc_score(all_labels, all_preds, average=avg)
        except:
            metrics[f'auc_{avg}'] = float('nan')
            
    print("-" * 30)
    print("Aggregate Metrics:")
    print(f"Micro F1: {metrics['f1_micro']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Micro AUC: {metrics['auc_micro']:.4f}")
    print(f"Macro AUC: {metrics['auc_macro']:.4f}")
    
    # Minority Macro F1 (excluding 'no')
    minority_indices = [i for i, label in enumerate(mlb.classes_) if label != 'no']
    if minority_indices:
         minority_f1 = f1_score(all_labels[:, minority_indices], binary_preds[:, minority_indices], average='macro', zero_division=0)
         print(f"Minority Macro F1: {minority_f1:.4f}")
         
    print("-" * 30)
    
    # Per Class Metrics
    print("Per Class Metrics:")
    for idx, label in enumerate(mlb.classes_):
        true_lbl = all_labels[:, idx]
        pred_lbl = binary_preds[:, idx]
        prob_lbl = all_preds[:, idx]
        
        f1 = f1_score(true_lbl, pred_lbl, zero_division=0)
        prec = precision_score(true_lbl, pred_lbl, zero_division=0)
        rec = recall_score(true_lbl, pred_lbl, zero_division=0)
        
        auc = float('nan')
        try:
            auc = roc_auc_score(true_lbl, prob_lbl)
        except:
            pass
            
        print(f"Class: {label} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | AUC: {auc:.4f}")
        metrics[f"f1_{label}"] = f1
        metrics[f"prec_{label}"] = prec
        metrics[f"rec_{label}"] = rec
        metrics[f"auc_{label}"] = auc
        
    if args.save_metrics:
        with open(args.save_metrics, 'w') as f:
            # Convert numpy types to native for json
            clean_metrics = {k: float(v) for k, v in metrics.items()}
            json.dump(clean_metrics, f, indent=4)

if __name__ == "__main__":
    main()
