#!/bin/bash

BASE_DIR="/home/tahir/AAAI-26-INTENT/output-base-hpo-mixsnips"
HPO_JSON="$BASE_DIR/hpo/best_hyperparameters.json"

# Check if HPO JSON exists
if [ ! -f "$HPO_JSON" ]; then
    echo "ERROR: HPO JSON not found at $HPO_JSON"
    exit 1
fi

echo "========================================="
echo "Starting Remaining Analyses (Continued)"
echo "Base Directory: $BASE_DIR"
echo "HPO JSON: $HPO_JSON"
echo "========================================="

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

# Check if layer contribution completed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Layer contribution analysis failed"
    exit 1
fi

echo "========================================="
echo "5. Generating Figures"
echo "========================================="
python -c "
import sys
import os
sys.path.insert(0, '.')
import pandas as pd

# Import from the correct module name (with hyphen, not underscore)
# Using __import__ since the module name has a hyphen
IntentBaseHPO = __import__('Intent-base-hpo')
build_argparser = IntentBaseHPO.build_argparser
generate_all_figures = IntentBaseHPO.generate_all_figures

BASE_DIR = '$BASE_DIR'

args = build_argparser().parse_args([
    '--mode', 'train',
    '--hf_dataset', 'chirunder/MixAtis_for_DecoderOnly',
    '--model_name_or_path', 'meta-llama/Llama-3.2-1B',
    '--output_dir', BASE_DIR,
    '--tuning_metric', 'mean_f1'
])

# Helper function to safely load CSV
def safe_load_csv(filepath):
    try:
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(f'? File not found: {filepath}')
            return None
    except Exception as e:
        print(f'? Error loading {filepath}: {e}')
        return None

# Load existing results
print('Loading existing results...')
ratio_df = safe_load_csv(f'{BASE_DIR}/unfreeze/layer_ratio_results.csv')
position_df = safe_load_csv(f'{BASE_DIR}/unfreeze/layer_position_results.csv')
ablation_df = safe_load_csv(f'{BASE_DIR}/ablation/ablation_results.csv')
sens_df = safe_load_csv(f'{BASE_DIR}/sensitivity/sensitivity_results.csv')
contrib_df = safe_load_csv(f'{BASE_DIR}/unfreeze/layer_contribution.csv')
sig_df = safe_load_csv(f'{BASE_DIR}/stats/statistical_significance.csv')

# Print what was loaded
print('\nLoaded files:')
print(f'  ratio_df: {ratio_df is not None}')
print(f'  position_df: {position_df is not None}')
print(f'  ablation_df: {ablation_df is not None}')
print(f'  sens_df: {sens_df is not None}')
print(f'  contrib_df: {contrib_df is not None}')
print(f'  sig_df: {sig_df is not None}')

# Generate figures
print('\nGenerating figures...')
fig_paths = generate_all_figures(
    output_dir=args.output_dir,
    ratio_df=ratio_df,
    position_df=position_df,
    contrib_df=contrib_df,
    ablation_df=ablation_df,
    ablation_sig_df=sig_df,
    sens_df=sens_df,
    metric=args.tuning_metric
)

print('\n? Figures generated successfully!')
print('Figure paths:')
for name, path in fig_paths.items():
    if path:
        print(f'  {name}: {path}')
    else:
        print(f'  {name}: (not generated - missing required data)')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Figure generation failed"
    exit 1
fi

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

# List all figure files if they exist
if [ -d "$BASE_DIR/figures" ]; then
    echo "Figures generated:"
    ls -la "$BASE_DIR/figures/" | grep -E "\.(png|jpg|pdf)$" | awk '{print "  " $9}'
    FIGURE_COUNT=$(ls -1 "$BASE_DIR/figures/" | grep -E "\.(png|jpg|pdf)$" | wc -l)
    echo ""
    echo "Total figures: $FIGURE_COUNT"
fi

# Check for final model results
if [ -f "$BASE_DIR/final_model/test_results.json" ]; then
    echo ""
    echo "Final model test results:"
    cat "$BASE_DIR/final_model/test_results.json" | python -m json.tool
fi

# Check for pipeline summary
if [ -f "$BASE_DIR/pipeline_summary.json" ]; then
    echo ""
    echo "Pipeline summary:"
    cat "$BASE_DIR/pipeline_summary.json" | python -m json.tool
fi

echo ""
echo "========================================="
echo "To view the results, check:"
echo "  - $BASE_DIR/pipeline_summary.json (if full pipeline was run)"
echo "  - $BASE_DIR/figures/ for all visualizations"
echo "  - $BASE_DIR/final_model/test_results.json for final model performance"
echo "  - $BASE_DIR/ablation/ablation_results.csv for ablation study results"
echo "  - $BASE_DIR/sensitivity/sensitivity_results.csv for sensitivity analysis"
echo "========================================="