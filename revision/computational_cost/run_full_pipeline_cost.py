#!/usr/bin/env python3
"""
End-to-end cost report: optional Word2Vec + TASER-AE augmentation + transformer train + test inference.
- Embeddings: reproduction/generate_embeddings.py (optional, --run_embeddings)
- Augmentation: augment_taser_ae.py
- Training/inference: train_classifier.py --save_timing
"""

import argparse
import json
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Full pipeline computational cost (augment + train + infer).")
    parser.add_argument("--train_pure", type=str, default="data/processed/train_pure.csv")
    parser.add_argument("--test_file", type=str, default="data/processed/test_split.csv")
    parser.add_argument("--embeddings", type=str, default="data/processed/embeddings.npy")
    parser.add_argument("--strategy", type=str, default="Target_Multiplier_1X_seed6")
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--skip_augment", action="store_true", help="Skip augmentation subprocess; still record aug file path")
    parser.add_argument(
        "--run_embeddings",
        action="store_true",
        help="Run Word2Vec (generate_embeddings.py) first and record wall time (overwrites embeddings.npy)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for embeddings_timing.json, augment_timing.json, transformer_timing.json, full_report.json",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    out_dir = args.output_dir or os.path.join(script_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    emb_json = os.path.join(out_dir, "embeddings_timing.json")
    aug_json = os.path.join(out_dir, "augment_timing.json")
    xf_json = os.path.join(out_dir, "transformer_timing.json")
    full_json = os.path.join(out_dir, "full_report.json")

    measure_emb = os.path.join(script_dir, "measure_generate_embeddings.py")
    measure_aug = os.path.join(script_dir, "measure_taser_ae_augment.py")
    measure_xf = os.path.join(script_dir, "measure_transformer_train_infer.py")

    train_pure = os.path.join(repo_root, args.train_pure)
    test_file = os.path.join(repo_root, args.test_file)
    embeddings = os.path.join(repo_root, args.embeddings)

    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")

    emb_record = None
    if args.run_embeddings:
        t_emb0 = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                measure_emb,
                "--output_json",
                emb_json,
            ],
            cwd=repo_root,
            env=env,
            check=True,
        )
        t_emb1 = time.perf_counter()
        with open(emb_json) as f:
            emb_record = json.load(f)
        emb_record["wrapper_wall_sec"] = t_emb1 - t_emb0
        with open(emb_json, "w") as f:
            json.dump(emb_record, f, indent=2)
    else:
        emb_record = {
            "component": "word2vec_generate_embeddings",
            "skipped": True,
            "note": "Pass --run_embeddings to time generate_embeddings.py (uses data/processed/train_pure.csv)",
        }
        with open(emb_json, "w") as f:
            json.dump(emb_record, f, indent=2)

    aug_record = None
    if not args.skip_augment:
        t0 = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                measure_aug,
                "--input",
                train_pure,
                "--strategy",
                args.strategy,
                "--multiplier",
                str(args.multiplier),
                "--seed",
                str(args.seed),
                "--embeddings",
                embeddings,
                "--output_json",
                aug_json,
            ],
            cwd=repo_root,
            env=env,
            check=True,
        )
        t1 = time.perf_counter()
        with open(aug_json) as f:
            aug_record = json.load(f)
        aug_record["wrapper_wall_sec"] = t1 - t0
    else:
        repro_dir = os.path.join(repo_root, "reproduction")
        out_name = f"augmented_ehr_data_taser_ae_{args.strategy}.csv"
        aug_csv = os.path.join(repro_dir, "augmented_ehr_data", out_name)
        aug_record = {
            "component": "taser_ae_augmentation",
            "skipped": True,
            "expected_output_csv": aug_csv,
            "wall_sec": None,
        }
        with open(aug_json, "w") as f:
            json.dump(aug_record, f, indent=2)

    repro_dir = os.path.join(repo_root, "reproduction")
    aug_csv = os.path.join(repro_dir, "augmented_ehr_data", f"augmented_ehr_data_taser_ae_{args.strategy}.csv")
    if not os.path.isfile(aug_csv):
        print(f"Warning: augmented train file not found: {aug_csv}", file=sys.stderr)

    subprocess.run(
        [
            sys.executable,
            measure_xf,
            "--train_file",
            aug_csv,
            "--test_file",
            test_file,
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--save_timing",
            xf_json,
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )

    with open(xf_json) as f:
        xf_record = json.load(f)

    emb_sec = emb_record.get("wall_sec") if isinstance(emb_record.get("wall_sec"), (int, float)) else None
    aug_sec = aug_record.get("wall_sec")
    train_sec = xf_record.get("training_wall_sec")
    infer_sec = xf_record.get("inference_test_wall_sec")

    train_infer = None
    if train_sec is not None and infer_sec is not None:
        train_infer = float(train_sec) + float(infer_sec)
    aug_train_infer = None
    if aug_sec is not None and train_infer is not None:
        aug_train_infer = float(aug_sec) + train_infer
    full_stack = None
    if emb_sec is not None and aug_train_infer is not None:
        full_stack = float(emb_sec) + aug_train_infer
    elif emb_sec is not None and train_infer is not None and aug_sec is None:
        full_stack = float(emb_sec) + train_infer

    full = {
        "description": "Optional Word2Vec + TASER-AE + transformer: embedding prep, augmentation, training (incl. val), test inference",
        "strategy": args.strategy,
        "seed": args.seed,
        "device": args.device,
        "embeddings": emb_record,
        "augmentation": aug_record,
        "transformer": xf_record,
        "summary": {
            "word2vec_embeddings_wall_sec": emb_sec,
            "augmentation_wall_sec": aug_sec,
            "transformer_training_wall_sec": train_sec,
            "transformer_test_inference_wall_sec": infer_sec,
            "transformer_train_plus_infer_sec": train_infer,
            "augment_plus_train_plus_infer_sec": aug_train_infer,
            "embeddings_plus_augment_plus_train_plus_infer_sec": full_stack,
        },
    }
    with open(full_json, "w") as f:
        json.dump(full, f, indent=2)
    print(json.dumps(full["summary"], indent=2))
    print(f"Full report: {full_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
