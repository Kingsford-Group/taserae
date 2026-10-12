# SMOTE baseline

SMOTE oversampling followed by a random forest, three seeds.

```bash
python3 revision/smote_baseline/run_smote_baseline.py \
    --train_file data/processed/train_pure.csv --test_file data/processed/test_split.csv \
    --seeds 6,42,81 --output_dir revision/smote_baseline/outputs
```
Run from the repository root.
