#!/bin/bash
# Full pipeline runner for BiSLU + PABEE

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parameters
HF_DATASET="chirunder/MixAtis_for_DecoderOnly"
MODEL_NAME="meta-llama/Llama-3.2-1B"
OUTPUT_DIR="./outputs/full_pipeline_freq_$(date +%Y%m%d_%H%M%S)"
GPU=0
MAX_SEQ=100
TRAIN_BATCH=2
EVAL_BATCH=2
GRAD_ACCUM=8
NUM_EPOCHS=12

log_info "================================================"
log_info "Starting Full Pipeline for BiSLU + PABEE"
log_info "================================================"
log_info "Output directory: $OUTPUT_DIR"
log_info ""

# Step 1: HPO
log_info "Step 1: Hyperparameter Optimization (30 trials, 3-fold CV)"
python Intent-ablations.py hpo \
    --hf_dataset "$HF_DATASET" \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR/hpo" \
    --n_trials 30 \
    --n_folds 3 \
    --epochs_per_fold 3 \
    --gpu "$GPU" \
    --max_seq_length "$MAX_SEQ" \
    --eval_batch_size "$EVAL_BATCH" \
    --use_intent_context_attention

log_success "HPO completed."

# Step 2: Training
log_info "Step 2: Training with Best HPO Config"
python Intent-ablations.py train \
    --hf_dataset "$HF_DATASET" \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR/train" \
    --do_train \
    --do_eval \
    --gpu "$GPU" \
    --max_seq_length "$MAX_SEQ" \
    --train_batch_size "$TRAIN_BATCH" \
    --eval_batch_size "$EVAL_BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_train_epochs "$NUM_EPOCHS" \
    --use_gc \
    --use_amp \
    --use_freq_exit \
    --logging_steps 100 \
    --early_stopping 5

log_success "Training completed."

# Step 3: Ablation
log_info "Step 3: Ablation Study (8 experiments × 3 seeds)"
python Intent-ablations.py ablation \
    --hpo_config "$OUTPUT_DIR/hpo/best_hpo_config.json" \
    --hf_dataset "$HF_DATASET" \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR/ablation" \
    --gpu "$GPU"

log_success "Ablation study completed."

# Step 4: Stats
log_info "Step 4: Statistical Significance Testing"
python Intent-ablations.py stats \
    --ablation_raw_results "$OUTPUT_DIR/ablation/ablation_raw_results.json" \
    --output_dir "$OUTPUT_DIR/stats" \
    --baseline_experiment exp1_baseline \
    --metrics intent_acc slot_f1 mean_intent_slot semantic_acc

log_success "Statistical testing completed."

# Step 5: Sensitivity
log_info "Step 5: Sensitivity Analysis"
python Intent-ablations.py sensitivity \
    --hpo_config "$OUTPUT_DIR/hpo/best_hpo_config.json" \
    --hf_dataset "$HF_DATASET" \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR/sensitivity" \
    --gpu "$GPU"

log_success "Sensitivity analysis completed."

# Step 6: Cross-dataset
log_info "Step 6: Cross-Dataset Generalization Study"
python Intent-ablations.py cross_dataset \
    --hpo_config "$OUTPUT_DIR/hpo/best_hpo_config.json" \
    --mixatis_dataset "chirunder/MixAtis_for_DecoderOnly" \
    --mixsnips_dataset "chirunder/MixSnips_for_DecoderOnly" \
    --atis_dataset "chirunder/ATIS_for_DecoderOnly" \
    --snips_dataset "chirunder/SNIPS_for_DecoderOnly" \
    --output_dir "$OUTPUT_DIR/cross_dataset" \
    --gpu "$GPU"

log_success "Cross-dataset study completed."

# Step 7: Visualize
log_info "Step 7: Visualization"
python Intent-ablations.py visualize \
    --ablation_raw_results "$OUTPUT_DIR/ablation/ablation_raw_results.json" \
    --ablation_aggregated_results "$OUTPUT_DIR/ablation/ablation_aggregated_results.json" \
    --sensitivity_results "$OUTPUT_DIR/sensitivity/sensitivity_results.json" \
    --stats_vs_baseline "$OUTPUT_DIR/stats/stats_vs_baseline.json" \
    --best_hpo_config "$OUTPUT_DIR/hpo/best_hpo_config.json" \
    --output_dir "$OUTPUT_DIR/visualizations" \
    --instrumentation_dir "$OUTPUT_DIR/ablation/instrumentation"

log_success "Visualization completed."

log_info "================================================"
log_success "PIPELINE COMPLETED SUCCESSFULLY!"
log_info "================================================"
log_info "All results saved to: $OUTPUT_DIR"
