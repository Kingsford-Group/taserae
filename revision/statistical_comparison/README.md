# Statistical comparison

Per-seed results and paired comparisons between TASER-AE and every other method.
Run from this folder after the benchmark.

```bash
# Per-seed F1 and seed-level paired t-tests (Holm-corrected), from logged metrics
python3 seed_level_tests.py --log_dir ../../reproduction/log --out_dir outputs

# Patient-level paired bootstrap on the fixed test set. First save the test-set
# predictions of each run (from reproduction/):
#   python3 save_test_predictions.py --aug_dir augmented_ehr_data --out_dir preds --verify_dir log
python3 paired_bootstrap.py --preds_dir ../../reproduction/preds --out_dir outputs

# Per-seed TP / FP / predicted-positive counts, written by paired_bootstrap.py
python3 make_counts_table.py --counts outputs/prediction_counts.csv --out outputs/table_counts.tex

# The same counts recovered from logged metrics alone, without predictions
python3 derive_counts_from_metrics.py --log_dir ../../reproduction/log \
    --test_file ../../data/processed/test_split.csv --out outputs/prediction_counts_derived.csv
```
