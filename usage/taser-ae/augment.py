#!/usr/bin/env python3
"""
TASER-AE: Augment Your EHR Dataset

This script applies TASER-AE augmentation to your dataset and optionally trains
a classifier on the augmented data.

Usage:
    python3 augment.py --input data.csv --output augmented_data.csv --multiplier 1.0

    # With training:
    python3 augment.py --input train.csv --output augmented.csv --multiplier 1.0 \
        --train --test test.csv --device cuda:0
"""

import argparse
import os
import sys
import subprocess
import pandas as pd
import numpy as np

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
REPRO_DIR = os.path.join(REPO_ROOT, "reproduction")

def generate_embeddings(input_file, output_file):
    """Generate embeddings from input data"""
    print(f"Generating embeddings from {input_file}...")
    
    from gensim.models import Word2Vec
    from sklearn.preprocessing import normalize
    
    df = pd.read_csv(input_file)
    
    # Prepare sentences from drugs and complications
    sentences = []
    for _, row in df.iterrows():
        drugs = str(row['administered_drugs']).split(',') if pd.notna(row['administered_drugs']) else []
        comps = str(row['complications']).split(',') if pd.notna(row['complications']) else []
        sentences.append(drugs + comps)
    
    # Train Word2Vec
    model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, workers=4, seed=42)
    
    # Build embedding matrix
    vocab = list(model.wv.key_to_index.keys())
    embeddings = np.array([model.wv[w] for w in vocab])
    embeddings = normalize(embeddings)
    
    np.save(output_file, embeddings)
    print(f"Saved embeddings to {output_file}")
    return output_file

def run_augmentation(input_file, output_file, method, multiplier, target, seed, embeddings_file):
    """Run TASER-AE augmentation"""
    print(f"\nRunning TASER-AE ({method}) augmentation...")
    
    if method == "native":
        script = os.path.join(REPRO_DIR, "augment_taser_ae_native.py")
    else:
        script = os.path.join(REPRO_DIR, "augment_taser_ae_random.py")
    
    # Determine strategy name
    if target is not None:
        strategy = f"Fixed_{target}_user"
    else:
        strategy = f"Target_Multiplier_{multiplier}X_user"
    
    cmd = [
        sys.executable, script,
        "--input", input_file,
        "--strategy", strategy,
        "--multiplier", str(multiplier),
        "--seed", str(seed),
        "--embeddings", embeddings_file
    ]
    
    result = subprocess.run(cmd, cwd=REPRO_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    # Find and move the generated file
    expected_name = f"augmented_ehr_data_taser_ae_{method}_{strategy}.csv"
    generated_file = os.path.join(REPRO_DIR, "augmented_ehr_data", expected_name)
    
    if os.path.exists(generated_file):
        # Copy to output location
        import shutil
        shutil.copy(generated_file, output_file)
        print(f"Augmented data saved to: {output_file}")
    else:
        # Try finding in current directory
        alt_file = os.path.join(REPRO_DIR, expected_name)
        if os.path.exists(alt_file):
            import shutil
            shutil.copy(alt_file, output_file)
            print(f"Augmented data saved to: {output_file}")
        else:
            print(f"Warning: Could not find generated file. Check {REPRO_DIR}")
    
    return output_file

def run_training(train_file, test_file, metrics_file, device, seed):
    """Train classifier on augmented data"""
    print(f"\nTraining classifier...")
    
    script = os.path.join(REPRO_DIR, "train_classifier.py")
    
    cmd = [
        sys.executable, script,
        "--train_file", train_file,
        "--test_file", test_file,
        "--save_metrics", metrics_file,
        "--seed", str(seed),
        "--device", device
    ]
    
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    
    result = subprocess.run(cmd, cwd=REPRO_DIR, capture_output=True, text=True, env=env)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    print(f"Metrics saved to: {metrics_file}")

def main():
    parser = argparse.ArgumentParser(
        description="TASER-AE: Augment EHR datasets for adverse event detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic augmentation (1X = double minority samples)
    python3 augment.py --input train.csv --output augmented.csv --multiplier 1.0

    # Target specific count per minority class
    python3 augment.py --input train.csv --output augmented.csv --target 5000

    # Use random selection method
    python3 augment.py --input train.csv --output augmented.csv --multiplier 2.0 --method random

    # Augment and train classifier
    python3 augment.py --input train.csv --output augmented.csv --multiplier 1.0 \\
        --train --test test.csv --device cuda:0
        """
    )
    
    # Required arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV file (EHR data with labels)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV file for augmented data")
    
    # Augmentation options
    parser.add_argument("--multiplier", type=float, default=1.0,
                        help="Multiplier for minority samples (1.0 = add 100%% more)")
    parser.add_argument("--target", type=int, default=None,
                        help="Target count per minority class (overrides multiplier)")
    parser.add_argument("--method", type=str, choices=["native", "random"], default="native",
                        help="TASER-AE method: native (parallel) or random (selection)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Embeddings
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Path to embeddings.npy (auto-generated if not provided)")
    
    # Training options
    parser.add_argument("--train", action="store_true",
                        help="Train classifier after augmentation")
    parser.add_argument("--test", type=str, default=None,
                        help="Test CSV file (required if --train is set)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for training (cuda:0 or cpu)")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Output file for metrics JSON")
    
    args = parser.parse_args()
    
    # Validate
    if args.train and args.test is None:
        parser.error("--test is required when --train is set")
    
    input_file = os.path.abspath(args.input)
    output_file = os.path.abspath(args.output)
    
    print("=" * 60)
    print("TASER-AE: Augmenting Electronic Health Records")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Method: TASER-AE ({args.method})")
    if args.target:
        print(f"Target: {args.target} samples per minority class")
    else:
        print(f"Multiplier: {args.multiplier}X")
    print("=" * 60)
    
    # Generate or use provided embeddings
    if args.embeddings:
        embeddings_file = os.path.abspath(args.embeddings)
    else:
        embeddings_file = os.path.join(os.path.dirname(output_file), "embeddings.npy")
        generate_embeddings(input_file, embeddings_file)
    
    # Run augmentation
    run_augmentation(
        input_file, output_file, args.method,
        args.multiplier, args.target, args.seed, embeddings_file
    )
    
    # Optional training
    if args.train:
        test_file = os.path.abspath(args.test)
        metrics_file = args.metrics or output_file.replace(".csv", "_metrics.json")
        run_training(output_file, test_file, metrics_file, args.device, args.seed)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
