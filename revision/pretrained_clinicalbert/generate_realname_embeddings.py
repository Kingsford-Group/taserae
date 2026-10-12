#!/usr/bin/env python3
"""
Generate Word2Vec embeddings for real-name EHR vocabulary.
Produces item-level embeddings so multi-word drug names like
"acetaminophen iv" get a single embedding (mean of word vectors).
"""

import os
import re
import sys

import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, "outputs", "splits", "train_pure.csv")
    output_file = os.path.join(script_dir, "outputs", "realname_embeddings.npy")

    if not os.path.exists(train_file):
        print(f"ERROR: {train_file} not found. Run prepare_splits_realname.py first.")
        sys.exit(1)

    df = pd.read_csv(train_file).fillna("")
    print(f"Loaded {len(df)} rows from {train_file}")

    # Tokenize same way as reproduction/generate_embeddings.py
    cols = [c for c in df.columns if c != "labels"]
    texts = df[cols].astype(str).agg(" ".join, axis=1).tolist()

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")

    sentences = []
    for text in texts:
        clean = re.sub(r"\s+", " ", text.replace(",", " ")).strip()
        tokens = nltk.word_tokenize(clean)
        sentences.append(tokens)

    max_length = max(len(s) for s in sentences)
    print(f"Tokenized {len(sentences)} samples, max length: {max_length}")

    # Train Word2Vec
    print("Training Word2Vec...")
    model = Word2Vec(
        sentences=sentences,
        seed=42,
        workers=16,
        sg=0,
        window=max_length,
        vector_size=100,
        min_count=1,
    )
    model.train(sentences, total_examples=model.corpus_count, epochs=10)

    # Word-level embeddings
    word_emb = {w: model.wv[w] for w in model.wv.key_to_index}
    print(f"Word-level embeddings: {len(word_emb)}")

    # Item-level embeddings: for each unique drug/complication item,
    # compute mean of its word vectors so "acetaminophen iv" gets one embedding
    item_emb = dict(word_emb)

    for feature in ["administered_drugs", "complications"]:
        if feature not in df.columns:
            continue
        items = (
            df[feature]
            .dropna()
            .apply(lambda x: [i.strip() for i in str(x).split(",") if i.strip()])
            .explode()
            .dropna()
            .unique()
        )
        added = 0
        for item in items:
            if item in item_emb:
                continue
            words = nltk.word_tokenize(item)
            vecs = [word_emb[w] for w in words if w in word_emb]
            if vecs:
                item_emb[item] = np.mean(vecs, axis=0)
                added += 1
        print(f"  {feature}: added {added} multi-word item embeddings")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, item_emb)
    print(f"Total embeddings: {len(item_emb)}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
