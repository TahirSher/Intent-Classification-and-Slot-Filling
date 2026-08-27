#!/bin/bash

BASE_DIR="/home/tahir/AAAI-26-INTENT/output-base-hpo"
HPO_JSON="$BASE_DIR/hpo/best_hyperparameters.json"

# Check if HPO JSON exists
if [ ! -f "$HPO_JSON" ]; then
    echo "ERROR: HPO JSON not found at $HPO_JSON"
    exit 1
fi

echo "========================================="
echo "Starting Remaining Analyses"
echo "Base Directory: $BASE_DIR"
echo "HPO JSON: $HPO_JSON"
echo "========================================="

echo "========================================="
echo "1. Running Ablation Study"
echo "========================================="
python Intent-base-hpo.py \
    --mode ablation \
    --hf_dataset "chirunder/MixAtis_for_DecoderOnly" \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --output_dir "$BASE_DIR" \
    --gpu 0 \
    --best_hp_json "$HPO_JSON" \
    --tuning_metric "mean_f1" \
    --max_seq_length 100 \
    --train_batch_size 8 \
    --eval_batch_size 4 \
    --seeds "42,43,44" \
    --num_train_epochs 10 \
    --early_stopping 3 \
    --use_amp

# Check if ablation completed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Ablation study failed"
    exit 1
fi

echo "========================================="
echo "2. Running Statistical Analysis"
echo "========================================="
# Check if ablation results exist
if [ ! -f "$BASE_DIR/ablation/ablation_results.csv" ]; then
    echo "WARNING: ablation_results.csv not found at $BASE_DIR/ablation/"
    echo "Skipping statistical analysis"
else
    python Intent-base-hpo.py \
        --mode stats \
        --results_csv "$BASE_DIR/ablation/ablation_results.csv" \
        --output_dir "$BASE_DIR" \
        --group_col "experiment" \
        --baseline_exp "E0_full_model" \
        --tuning_metric "mean_f1"
fi

echo "========================================="
echo "3. Running Sensitivity Analysis"
echo "========================================="
python Intent-base-hpo.py \
    --mode sensitivity \
    --hf_dataset "chirunder/MixAtis_for_DecoderOnly" \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --output_dir "$BASE_DIR" \
    --gpu 0 \
    --best_hp_json "$HPO_JSON" \
    --tuning_metric "mean_f1" \
    --max_seq_length 100 \
    --train_batch_size 8 \
    --eval_batch_size 4 \
    --sensitivity_epochs 3 \
    --sensitivity_early_stopping 2 \
    --use_amp

echo "========================================="
echo "4. Running Layer Contribution Analysis"
echo "========================================="
python Intent-base-hpo.py \
    --mode layer_contribution \
    --hf_dataset "chirunder/MixAtis_for_DecoderOnly" \
    --model_name_or_path "meta-llama/Llama-3.2-1B" \
    --output_dir "$BASE_DIR" \
    --gpu 0 \
    --best_hp_json "$HPO_JSON" \
    --max_seq_length 100 \
    --train_batch_size 8 \
    --eval_batch_size 4 \
    --unfrozen_ratio 0.5 \
    --unfreeze_position "front"

echo "========================================="
echo "5. Generating Figures"
echo "========================================="
python -c "
import sys
sys.path.insert(0, '.')
import pandas as pd
from Intent_base_hpo import build_argparser, generate_all_figures

BASE_DIR = '$BASE_DIR'

args = build_argparser().parse_args([
    '--mode', 'train',
    '--hf_dataset', 'chirunder/MixAtis_for_DecoderOnly',
    '--model_name_or_path', 'meta-llama/Llama-3.2-1B',
    '--output_dir', BASE_DIR,
    '--tuning_metric', 'mean_f1'
])

# Load existing results
try:
    ratio_df = pd.read_csv(f'{BASE_DIR}/unfreeze/layer_ratio_results.csv')
    print('? Loaded layer_ratio_results.csv')
except:
    ratio_df = None
    print('? layer_ratio_results.csv not found')

try:
    ablation_df = pd.read_csv(f'{BASE_DIR}/ablation/ablation_results.csv')
    print('? Loaded ablation_results.csv')
except:
    ablation_df = None
    print('? ablation_results.csv not found')

try:
    sens_df = pd.read_csv(f'{BASE_DIR}/sensitivity/sensitivity_results.csv')
    print('? Loaded sensitivity_results.csv')
except:
    sens_df = None
    print('? sensitivity_results.csv not found')

try:
    contrib_df = pd.read_csv(f'{BASE_DIR}/unfreeze/layer_contribution.csv')
    print('? Loaded layer_contribution.csv')
except:
    contrib_df = None
    print('? layer_contribution.csv not found')

# Generate figures
fig_paths = generate_all_figures(
    output_dir=args.output_dir,
    ratio_df=ratio_df,
    ablation_df=ablation_df,
    sens_df=sens_df,
    contrib_df=contrib_df,
    metric=args.tuning_metric
)

print('\n? Figures generated successfully!')
print('Figure paths:')
for name, path in fig_paths.items():
    if path:
        print(f'  {name}: {path}')
"

echo "========================================="
echo "? All Remaining Analyses Complete!"
echo "Results saved in: $BASE_DIR"
echo "========================================="

# Show summary of generated files
echo ""
echo "Generated files summary:"
echo "  Ablation: $BASE_DIR/ablation/"
echo "  Stats: $BASE_DIR/stats/"
echo "  Sensitivity: $BASE_DIR/sensitivity/"
echo "  Layer Contribution: $BASE_DIR/unfreeze/layer_contribution.csv"
echo "  Figures: $BASE_DIR/figures/"
echo ""