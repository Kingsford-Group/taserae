# TASER-AE

**Augmenting Electronic Health Records for Adverse Event Detection**

TASER-AE applies targeted, NLP-inspired augmentation (synonym insertion, random
drop and shuffling) to structured EHR data to improve detection of rare adverse
events. This repository is organised into three parts:

1. **[Usage](#part-1-usage)**: apply TASER-AE to your own data or extract custom
   labels from MIMIC-IV.
2. **[Reproduction](#part-2-reproduction)**: replicate the paper's benchmark on
   MIMIC-IV.
3. **[Revision analyses](#part-3-revision-analyses)**: the additional analyses
   reported in the supplementary material.

### Prerequisites
- Python 3.8+
- Packages: `pandas numpy scipy scikit-learn torch gensim nltk tqdm` (see
  `environment.yml`)
- MIMIC-IV data access (PhysioNet credentialed access) and the OHDSI Athena
  vocabulary, for Parts 2 and 3 and for label extraction

MIMIC-IV data, and anything derived from it at the patient level, cannot be
redistributed. This repository contains code only.

---

## Part 1: Usage

### A. Augment an existing CSV
Your CSV must have the columns `ageYear`, `gender`, `administered_drugs`,
`complications` and `labels`.

```bash
cd usage/taser-ae

# Augment (1X = one synthetic record per minority record)
python3 augment.py --input train.csv --output augmented.csv --multiplier 1.0

# Augment and train the classifier
python3 augment.py --input train.csv --output augmented.csv --train --test test.csv --device cuda:0
```

### B. Extract custom labels from MIMIC-IV
1. **Define labels**: create a CSV map (e.g. `my_labels.csv`) linking each label
   to a file of ICD codes:
   ```csv
   Label,Path
   Sepsis,codes/sepsis_icd.csv
   HeartFailure,codes/hf_icd.csv
   ```

2. **Run extraction**:
   ```bash
   cd usage/taser-ae

   python3 mimic_custom_label.py \
       --patients ../../data/mimic/patients.csv.gz \
       --diagnoses ../../data/mimic/diagnoses_icd.csv.gz \
       --emar ../../data/mimic/emar.csv.gz \
       --athena_dir ../../data/athena \
       --label_map my_labels.csv \
       --out_csv my_custom_data.csv

   python3 clean.py --in_csv my_custom_data.csv --out_csv my_custom_final.csv
   ```

3. **Augment**: run `augment.py` on `my_custom_final.csv`.

---

## Part 2: Reproduction

Reproduce the 1X benchmark reported in the paper.

### 1. Data preparation

```bash
# Place MIMIC files (patients.csv.gz, diagnoses_icd.csv.gz, emar.csv.gz) in data/mimic/
# Place Athena files (CONCEPT.csv, etc.) in data/athena/
cd data

# Extract the adverse event labels (Falls, Fractures, Stroke, GI_Bleed)
python3 mimic_all_ade.py --patients mimic/patients.csv.gz --diagnoses mimic/diagnoses_icd.csv.gz \
    --emar mimic/emar.csv.gz --athena_dir athena --out_csv mimic4_patient_level_full_all.csv

# Clean
python3 clean.py --in_csv mimic4_patient_level_full_all.csv --out_csv mimic4_label.csv
```

### 2. Run the benchmark

```bash
cd reproduction

# Split, embed, augment and train every method over seeds 6, 42, 81
python3 run_benchmark.py --data ../data/mimic4_label.csv --device cuda:0

# Mean and standard deviation of every metric across seeds
python3 comprehensive_aggregate.py --log_dir log --out comprehensive_1x.csv
```

`run_benchmark.py` creates the stratified train/test split, trains the Word2Vec
embeddings (`generate_embeddings.py`), generates the augmented training sets and
trains the transformer classifier for each method and seed, together with the
decision tree, random forest and gradient boosting baselines. Each step skips
work that already has output, so an interrupted run can be resumed.

Methods evaluated: no augmentation, random drop, random swap, random synonym
replacement, random synonym insertion, shuffle, composite insertion, CVAE, CGAN,
Ishikawa et al. and TASER-AE.

> **Note on reproducibility.** The Word2Vec embeddings are not deterministic:
> re-running `generate_embeddings.py` on the same input produces different
> vectors and therefore a different synonym thesaurus. The augmentation scripts
> are deterministic given a fixed thesaurus, so `random_drop`, `random_swap` and
> `shuffle` reproduce from code alone, while the thesaurus-dependent methods
> (`synonym_insertion`, `synonym_replacement`, `composite_insertion`, `ishikawa`,
> `taser_ae`) reproduce only against the same `data/processed/embeddings.npy`.
> Generate the embeddings once and keep the file.

### 3. Figures

```bash
# TASER-AE across augmentation multipliers from 0.01X to 10X
python3 run_size_analysis.py
# Output: figs/size_vs_performance.png

# Real and TASER-AE-augmented samples (1X, seed 6) in embedding space
python3 tsne_visualization.py
# Output: figs/tsne_real_seed6.png, figs/tsne_taser_ae_seed6.png, figs/tsne_all_seed6.png
```

---

## Part 3: Revision analyses

The `revision/` folder contains the additional analyses reported in the
supplementary material. They use the split and embeddings produced in Part 2
(`data/processed/`). Each folder has a short README with its command.

| Folder | Analysis |
|---|---|
| [`statistical_comparison/`](revision/statistical_comparison) | Per-seed results, seed-level paired t-tests, patient-level paired bootstrap, per-seed TP/FP counts |
| [`threshold_sensitivity/`](revision/threshold_sensitivity) | Cosine-similarity threshold of the synonym thesaurus |
| [`size_performance_analysis/`](revision/size_performance_analysis) | Per-class F1 as a function of augmentation size |
| [`similarity_performance/`](revision/similarity_performance) | Performance by similarity of synthetic to real records |
| [`quantitative_augmentation_quality/`](revision/quantitative_augmentation_quality) | Nearest-neighbour distance, local outlier factor and coverage |
| [`stratified_performance/`](revision/stratified_performance) | Performance by age group and sex |
| [`imbalance_strategies/`](revision/imbalance_strategies) | Undersampling, oversampling and class-weighted loss |
| [`smote_baseline/`](revision/smote_baseline) | SMOTE baseline |
| [`tree_baselines_on_taser/`](revision/tree_baselines_on_taser) | Tree baselines trained on TASER-AE data |
| [`pretrained_clinicalbert/`](revision/pretrained_clinicalbert) | Bio_ClinicalBERT with and without TASER-AE |
| [`thesaurus_synonym_pairs/`](revision/thesaurus_synonym_pairs) | Sampled synonym pairs for qualitative inspection |
| [`synonym_validity/ontology_validation/`](revision/synonym_validity/ontology_validation) | Synonym pairs checked against ATC and ICD-10-CM |
| [`computational_cost/`](revision/computational_cost) | Wall-clock cost of each pipeline stage |

`revision/generate_mimic_prevalence_table.ipynb` reports adverse event
prevalence in the test split. To run the t-SNE, augmentation quality,
similarity, size, stratified, threshold and tree-baseline analyses in sequence:

```bash
bash revision/run_all_revision_experiments.sh
```

---

## Directory structure

```
TASER-AE/
├── usage/taser-ae/               # Part 1: general-purpose tools
│   ├── augment.py                # Augmentation tool
│   ├── mimic_custom_label.py     # Custom label extraction
│   └── clean.py                  # Data cleaner
│
├── data/                         # Part 2: label extraction and cleaning
│   ├── mimic_all_ade.py          # Adverse event extraction from MIMIC-IV
│   ├── clean.py                  # Cleaning to the benchmark format
│   └── *_ICD.csv                 # ICD code definitions per adverse event
│
├── reproduction/                 # Part 2: benchmark
│   ├── run_benchmark.py          # End-to-end benchmark
│   ├── augment_taser_ae.py       # TASER-AE augmentation
│   ├── augment_*.py              # Comparison augmentation methods
│   ├── train_classifier.py       # Transformer classifier
│   ├── baseline_*.py             # Tree baselines
│   ├── comprehensive_aggregate.py  # Aggregation across seeds
│   └── save_test_predictions.py  # Test-set predictions for the bootstrap
│
└── revision/                     # Part 3: supplementary analyses
```
