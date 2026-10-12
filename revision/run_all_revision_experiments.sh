#!/bin/bash
# Re-run all MIMIC-IV revision experiments from scratch
# Skips steps that have already produced output files.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
REPRO_DIR="${REPO_ROOT}/reproduction"
DATA_DIR="${REPO_ROOT}/data/processed"
REVISION_DIR="${REPO_ROOT}/revision"
TRAIN_FILE="${DATA_DIR}/train_pure.csv"
TEST_FILE="${DATA_DIR}/test_split.csv"
EMBEDDINGS="${DATA_DIR}/embeddings.npy"
DEVICE="cuda:0"
SEEDS="6,42,81"

echo "=============================================="
echo " MIMIC-IV Revision Experiments"
echo " Seeds: $SEEDS | Device: $DEVICE"
echo " Revision dir: $REVISION_DIR"
echo "=============================================="

# -----------------------------------------------
# 1. t-SNE Visualization
# -----------------------------------------------
TSNE_OUT="${REPRO_DIR}/figs/tsne_all_seed6.png"
if [ -f "$TSNE_OUT" ]; then
    echo ""
    echo ">>> [1/7] t-SNE Visualization — SKIPPING (output exists)"
else
    echo ""
    echo ">>> [1/7] t-SNE Visualization"
    cd "$REPRO_DIR"
    python3 tsne_visualization.py
fi

# -----------------------------------------------
# 2. Quantitative Augmentation Quality
# -----------------------------------------------
OUT_QAQ="${REVISION_DIR}/quantitative_augmentation_quality/outputs"
QAQ_DONE="${OUT_QAQ}/augmentation_quality_summary.csv"
if [ -f "$QAQ_DONE" ]; then
    echo ""
    echo ">>> [2/7] Quantitative Augmentation Quality — SKIPPING (output exists)"
else
    echo ""
    echo ">>> [2/7] Quantitative Augmentation Quality"
    mkdir -p "$OUT_QAQ"
    cd "$REPO_ROOT"
    python3 revision/quantitative_augmentation_quality/quantitative_augmentation_analysis.py \
        --real_train_file "$TRAIN_FILE" \
        --augmented_file "${REPRO_DIR}/augmented_ehr_data/augmented_ehr_data_taser_ae_Target_Multiplier_1X_seed{seed}.csv" \
        --test_file "$TEST_FILE" \
        --embeddings "$EMBEDDINGS" \
        --output_dir "$OUT_QAQ" \
        --dataset_name "MIMIC-IV" \
        --seeds "$SEEDS"
fi

# -----------------------------------------------
# 3. Similarity Performance
# -----------------------------------------------
OUT_SIM="${REVISION_DIR}/similarity_performance/outputs"
SIM_DONE="${OUT_SIM}/analysis/similarity_performance_summary.csv"
if [ -f "$SIM_DONE" ]; then
    echo ""
    echo ">>> [3/7] Similarity Performance — SKIPPING (output exists)"
else
    echo ""
    echo ">>> [3/7] Similarity Performance"
    mkdir -p "$OUT_SIM"
    cd "$REPO_ROOT"
    python3 revision/similarity_performance/run_full_pipeline.py \
        --seeds "$SEEDS" \
        --output_dir "$OUT_SIM" \
        --device "$DEVICE"
fi

# -----------------------------------------------
# 4. Size Performance Analysis
# -----------------------------------------------
OUT_SIZE="${REVISION_DIR}/size_performance_analysis/outputs"
SIZE_DONE="${OUT_SIZE}/size_vs_performance_per_class_aggregated.csv"
if [ -f "$SIZE_DONE" ]; then
    echo ""
    echo ">>> [4/7] Size Performance Analysis — SKIPPING (output exists)"
else
    echo ""
    echo ">>> [4/7] Size Performance Analysis"
    mkdir -p "$OUT_SIZE"
    cd "$REPO_ROOT"
    python3 revision/size_performance_analysis/run_size_analysis_per_class.py
fi

# -----------------------------------------------
# 5. Stratified Performance
# -----------------------------------------------
OUT_STRAT="${REVISION_DIR}/stratified_performance/outputs"
STRAT_DONE="${OUT_STRAT}/stratified_metrics_aggregated.csv"
if [ -f "$STRAT_DONE" ]; then
    echo ""
    echo ">>> [5/7] Stratified Performance — SKIPPING (output exists)"
else
    echo ""
    echo ">>> [5/7] Stratified Performance"
    mkdir -p "$OUT_STRAT"
    cd "$REPO_ROOT"
    python3 revision/stratified_performance/run_stratified_performance.py \
        --train_file_no_aug "$TRAIN_FILE" \
        --train_file_taser "${REPRO_DIR}/augmented_ehr_data/augmented_ehr_data_taser_ae_Target_Multiplier_1X_seed{seed}.csv" \
        --test_file "$TEST_FILE" \
        --output_dir "${REVISION_DIR}/stratified_performance/outputs" \
        --seeds "$SEEDS" \
        --device "$DEVICE"
fi

# -----------------------------------------------
# 6. Threshold Sensitivity
# -----------------------------------------------
OUT_THR="${REVISION_DIR}/threshold_sensitivity/outputs"
THR_DONE="${OUT_THR}/per_label_metrics_by_threshold_aggregated.csv"
if [ -f "$THR_DONE" ]; then
    echo ""
    echo ">>> [6/7] Threshold Sensitivity — SKIPPING (output exists)"
else
    echo ""
    echo ">>> [6/7] Threshold Sensitivity"
    mkdir -p "$OUT_THR"
    cd "$REPO_ROOT"
    python3 revision/threshold_sensitivity/run_threshold_sensitivity.py \
        --seeds "$SEEDS" \
        --device "$DEVICE"
fi

# -----------------------------------------------
# 7. Tree Baselines on TASER
# -----------------------------------------------
OUT_TREE="${REVISION_DIR}/tree_baselines_on_taser/outputs"
TREE_DONE="${OUT_TREE}/tree_baselines_taser_metrics_aggregated.csv"
if [ -f "$TREE_DONE" ]; then
    echo ""
    echo ">>> [7/7] Tree Baselines on TASER — SKIPPING (output exists)"
else
    echo ""
    echo ">>> [7/7] Tree Baselines on TASER"
    mkdir -p "$OUT_TREE"
    cd "$REPO_ROOT"
    python3 revision/tree_baselines_on_taser/run_tree_baselines_on_taser.py \
        --seeds "$SEEDS"
fi

echo ""
echo "=============================================="
echo " ALL DONE!"
echo "=============================================="
