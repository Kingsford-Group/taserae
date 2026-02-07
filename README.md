# TASER-AE

**Augmenting Electronic Health Records for Adverse Event Detection**

TASER-AE applies targeted text augmentation strategies to structured EHR data. This repository is organized into two parts:

1.  **[Reproduction](#part-1-reproduction)**: Replicate the paper's benchmark on MIMIC-IV using adverse event labels.
2.  **[TASER-AE Usage](#part-2-taser-ae-usage-your-own-data)**: Apply TASER-AE to your own datasets or extract custom labels from MIMIC.

---

## Part 1: Reproduction

Reproduce the 1X augmentation benchmark results reported in the TASER-AE paper.

### Prerequisites
- Python 3.8+
- Packages: `pandas numpy scikit-learn torch gensim nltk tqdm`
- MIMIC-IV Data Access (PhysioNet)
- OHDSI Athena Vocabulary

### 1. Data Preparation
Follow these steps to generate the benchmark dataset:

```bash
# 1. Setup directories
mkdir -p data/mimic data/athena
# Place MIMIC files (patients.csv.gz, diagnoses_icd.csv.gz, emar.csv.gz) in data/mimic/
# Place Athena files (CONCEPT.csv, etc.) in data/athena/

# 2. Extract standard labels (Falls, Fractures, Stroke, GI_Bleed)
cd data
python3 mimic_all_ade.py --patients mimic/patients.csv.gz --diagnoses mimic/diagnoses_icd.csv.gz --emar mimic/emar.csv.gz --athena_dir athena --out_csv mimic4_patient_level_full_all.csv

# 3. Clean
python3 clean.py --in_csv mimic4_patient_level_full_all.csv --out_csv mimic4_label.csv
```

### 2. Run Benchmark
Execute the reproduction pipeline:

```bash
cd reproduction

# Run 1X Augmentation Benchmark
python3 run_benchmark.py --data ../data/mimic4_label.csv --device cuda:0

# View Results
python3 aggregate_results.py
```

### 3. Analysis & Visualization

Scripts to analyze model behavior and augmentation effectiveness:

#### Size vs Performance Analysis
Evaluates TASER-AE performance across different augmentation multipliers (0.01X to 10X).
```bash
python3 run_size_analysis.py
# Output: figs/size_vs_performance.png
```

#### t-SNE Visualization
Visualizes the feature space distribution of real vs. augmented samples (using TASER-AE Native at 1X).
```bash
python3 tsne_visualization.py
# Output: figs/tsne_real_only.png, figs/tsne_augmented.png, figs/tsne_combined.png
```

---

## Part 2: TASER-AE Usage (Your Own Data)

Use TASER-AE tools to augment your own structured EHR datasets or define custom adverse events.

### A. Apply Augmentation (Existing CSV)
If you already have a patient-level CSV with a `labels` column:

```bash
cd usage/taser-ae

# Augment data (1X = double minority samples)
python3 augment.py --input train.csv --output augmented.csv --multiplier 1.0

# Augment and Train Classifier
python3 augment.py --input train.csv --output augmented.csv --train --test test.csv --device cuda:0
```
*Note: Your CSV must have columns: `ageYear`, `gender`, `administered_drugs`, `complications`, `labels`.*

### B. Extract Custom Labels from MIMIC-IV
Extract your own adverse events (e.g., "Sepsis", "Heart Failure") from MIMIC-IV data.

1.  **Define Labels**: Create a CSV map file (e.g., `my_labels.csv`) linking labels to their ICD code files:
    ```csv
    Label,Path
    Sepsis,codes/sepsis_icd.csv
    HeartFailure,codes/hf_icd.csv
    ```

2.  **Run Extraction** (using the provided tool):
    ```bash
    cd usage/taser-ae

    # Extract custom dataset
    python3 mimic_custom_label.py \
        --patients ../../data/mimic/patients.csv.gz \
        --diagnoses ../../data/mimic/diagnoses_icd.csv.gz \
        --emar ../../data/mimic/emar.csv.gz \
        --athena_dir ../../data/athena \
        --label_map my_labels.csv \
        --out_csv my_custom_data.csv
    
    # Process for training
    python3 clean.py --in_csv my_custom_data.csv --out_csv my_custom_final.csv
    ```

3.  **Augment**: Now use `augment.py` on `my_custom_final.csv`.

---

## Directory Structure

```
TASER-AE/
├── data/                       # Standard reproduction data tools
│   ├── mimic_all_ade.py        # Standard extraction script
│   └── *_ICD.csv               # Standard code definitions
│
├── reproduction/               # Benchmark scripts
│   ├── run_benchmark.py
│   └── (baselines & methods)
│
└── usage/
    └── taser-ae/               # General usage tools
        ├── augment.py          # Augmentation tool
        ├── mimic_custom_label.py # Custom extraction tool
        └── clean.py            # Data cleaner
```
