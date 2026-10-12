# Quantitative augmentation quality

Nearest-neighbour distance, local outlier factor and coverage of TASER-AE records
relative to real records, in the Word2Vec embedding space.

```bash
python3 revision/quantitative_augmentation_quality/quantitative_augmentation_analysis.py \
    --real_train_file data/processed/train_pure.csv \
    --augmented_file "reproduction/augmented_ehr_data/augmented_ehr_data_taser_ae_Target_Multiplier_1X_seed{seed}.csv" \
    --test_file data/processed/test_split.csv \
    --embeddings data/processed/embeddings.npy \
    --dataset_name MIMIC-IV --output_dir revision/quantitative_augmentation_quality/outputs
```
Run from the repository root.
