import pandas as pd
import numpy as np
import nltk
from gensim.models import Word2Vec
import re
import os
import sys

def main():
    print("Generating embeddings from train_pure.csv...")
    
    # Path relative to script location
    input_file = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train_pure.csv")
    
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found.")
        sys.exit(1)

    # Load data
    df = pd.read_csv(input_file)
    

    if 'text' in df.columns:
        X_train = df['text'].astype(str).tolist()
    else:
        print("Warning: 'text' column not found. Using all non-label columns.")
        cols = [c for c in df.columns if c not in ['labels', 'adverse_events']]
        X_train = df[cols].astype(str).agg(' '.join, axis=1).tolist()

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

    print("Tokenizing...")

    cleaned_sentences = []
    for sample in X_train:
        # Notebook replacement logic
        clean = re.sub(r'\s+', ' ', sample.replace(',', ' ')).strip()
        tokens = nltk.word_tokenize(clean)
        cleaned_sentences.append(tokens)

    if not cleaned_sentences:
        print("Error: No data to train on.")
        sys.exit(1)

    max_length = max([len(s) for s in cleaned_sentences])
    print(f"Max sentence length: {max_length}")

    # Train Word2Vec
    print("Training Word2Vec...")
    vector_size = 100
    model = Word2Vec(
        sentences=cleaned_sentences, 
        seed=42, 
        workers=16, 
        sg=0, 
        window=max_length, 
        vector_size=vector_size, 
        min_count=1
    )
    
    model.train(cleaned_sentences, total_examples=model.corpus_count, epochs=10)
    
    # Save embeddings dictionary
    output_file = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "embeddings.npy")
    print(f"Saving embeddings to {output_file}...")
    embeddings = {}
    for word in model.wv.key_to_index:
        embeddings[word] = model.wv[word]
        
    np.save(output_file, embeddings)
    print("Done.")

if __name__ == "__main__":
    main()
