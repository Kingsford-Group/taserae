# Stratified performance

Test performance by age group and sex, for the non-augmented and TASER-AE models.

```bash
python3 revision/stratified_performance/run_stratified_performance.py \
    --train_file_no_aug data/processed/train_pure.csv \
    --train_file_taser "reproduction/augmented_ehr_data/augmented_ehr_data_taser_ae_Target_Multiplier_1X_seed{seed}.csv" \
    --test_file data/processed/test_split.csv \
    --output_dir revision/stratified_performance/outputs --device cuda:0
```
Run from the repository root.
