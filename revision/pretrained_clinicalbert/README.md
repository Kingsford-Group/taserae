# Bio_ClinicalBERT

Bio_ClinicalBERT with and without TASER-AE. Requires a cohort build that keeps drug
and complication names instead of IDs. Run from the repository root:

```bash
D=revision/pretrained_clinicalbert
python3 $D/build_realname_mimic_csv.py --patients data/mimic/patients.csv.gz \
    --diagnoses data/mimic/diagnoses_icd.csv.gz --emar data/mimic/emar.csv.gz \
    --athena_dir data/athena --out_csv $D/outputs/mimic_realname.csv
python3 $D/prepare_splits_realname.py --input_csv $D/outputs/mimic_realname.csv --output_dir $D/outputs/splits
python3 $D/run_noaug_vs_taser.py --train_file $D/outputs/splits/train_pure.csv \
    --test_file $D/outputs/splits/test_split.csv --device cuda:0
```

`run_noaug_vs_taser.py` trains the real-name Word2Vec embeddings on first use
(`generate_realname_embeddings.py`). Results go to `revision/pretrained_clinicalbert/outputs/noaug_vs_taser/`.
