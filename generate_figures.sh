#!/bin/bash

BASE_DIR="/home/tahir/AAAI-26-INTENT/output-base-hpo-mixatis"
HPO_JSON="$BASE_DIR/hpo/best_hyperparameters.json"

# Check if HPO JSON exists
if [ ! -f "$HPO_JSON" ]; then
    echo "ERROR: HPO JSON not found at $HPO_JSON"
    exit 1
fi

echo "========================================="
echo "Generating Figures"
echo "Base Directory: $BASE_DIR"
echo "HPO JSON: $HPO_JSON"
echo "========================================="

# Set environment variable to avoid tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

python -c "
import sys
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import the module with hyphen in name
IntentBaseHPO = __import__('Intent-base-hpo')
generate_all_figures = IntentBaseHPO.generate_all_figures

BASE_DIR = '$BASE_DIR'

print('=' * 50)
print('Loading results from:', BASE_DIR)
print('=' * 50)

# Helper function to safely load CSV
def safe_load_csv(filepath, description):
    try:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f'? Loaded {description}: {filepath}')
            print(f'  - Shape: {df.shape}')
            print(f'  - Columns: {list(df.columns)}')
            return df
        else:
            print(f'? File not found: {filepath}')
            return None
    except Exception as e:
        print(f'? Error loading {description}: {e}')
        return None

# Load all existing results
print('\nLoading existing results...')
print('-' * 50)

ratio_df = safe_load_csv(f'{BASE_DIR}/unfreeze/layer_ratio_results.csv', 'layer_ratio_results')
position_df = safe_load_csv(f'{BASE_DIR}/unfreeze/layer_position_results.csv', 'layer_position_results')
ablation_df = safe_load_csv(f'{BASE_DIR}/ablation/ablation_results.csv', 'ablation_results')
sens_df = safe_load_csv(f'{BASE_DIR}/sensitivity/sensitivity_results.csv', 'sensitivity_results')
contrib_df = safe_load_csv(f'{BASE_DIR}/unfreeze/layer_contribution.csv', 'layer_contribution')
sig_df = safe_load_csv(f'{BASE_DIR}/stats/statistical_significance.csv', 'statistical_significance')

# Check for history (learning curves) - try to load from final model
history = None
try:
    history_path = f'{BASE_DIR}/final_model/trainer_state.json'
    if os.path.exists(history_path):
        import json
        with open(history_path, 'r') as f:
            state = json.load(f)
        # Convert to history format expected by generate_all_figures
        # This is a best-effort conversion
        if 'history' in state:
            history = state['history']
            print(f'? Loaded training history from: {history_path}')
        else:
            print('? No history found in trainer_state.json')
except Exception as e:
    print(f'? Could not load training history: {e}')

# Summary of loaded data
print('\n' + '=' * 50)
print('Loaded Data Summary:')
print('-' * 50)
print(f'  ratio_df:      {ratio_df is not None}')
print(f'  position_df:   {position_df is not None}')
print(f'  ablation_df:   {ablation_df is not None}')
print(f'  sens_df:       {sens_df is not None}')
print(f'  contrib_df:    {contrib_df is not None}')
print(f'  sig_df:        {sig_df is not None}')
print(f'  history:       {history is not None}')
print('=' * 50)

# Generate figures
print('\nGenerating figures...')
print('-' * 50)

fig_paths = generate_all_figures(
    output_dir=BASE_DIR,
    ratio_df=ratio_df,
    position_df=position_df,
    contrib_df=contrib_df,
    ablation_df=ablation_df,
    ablation_sig_df=sig_df,
    sens_df=sens_df,
    history=history,
    metric='mean_f1'
)

print('\n' + '=' * 50)
print('? Figures Generation Complete!')
print('=' * 50)
print('\nGenerated Figures:')
print('-' * 50)

generated_count = 0
for name, path in fig_paths.items():
    if path and os.path.exists(path):
        size = os.path.getsize(path) / 1024  # size in KB
        print(f'  ? {name:30s} -> {os.path.basename(path)} ({size:.1f} KB)')
        generated_count += 1
    else:
        print(f'  ? {name:30s} -> (not generated - missing required data)')

print('-' * 50)
print(f'Total figures generated: {generated_count}/10')
print('=' * 50)

# List all figures in the directory
fig_dir = f'{BASE_DIR}/figures'
if os.path.exists(fig_dir):
    print('\nAll figure files in directory:')
    print('-' * 50)
    for f in sorted(os.listdir(fig_dir)):
        if f.endswith(('.png', '.jpg', '.pdf')):
            size = os.path.getsize(os.path.join(fig_dir, f)) / 1024
            print(f'  {f} ({size:.1f} KB)')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Figure generation failed"
    exit 1
fi

echo ""
echo "========================================="
echo "? Figure Generation Complete!"
echo "Figures saved in: $BASE_DIR/figures/"
echo "========================================="

# Check what figures were generated
if [ -d "$BASE_DIR/figures" ]; then
    echo ""
    echo "Generated figures:"
    ls -lh "$BASE_DIR/figures/" | grep -E "\.(png|jpg|pdf)$" | awk '{print "  " $9 " (" $5 ")"}'
    FIGURE_COUNT=$(ls -1 "$BASE_DIR/figures/" | grep -E "\.(png|jpg|pdf)$" | wc -l)
    echo ""
    echo "Total figures: $FIGURE_COUNT / 10"
    echo ""
    
    if [ $FIGURE_COUNT -eq 0 ]; then
        echo "WARNING: No figures were generated. This might be because:"
        echo "  1. The required data files don't exist yet"
        echo "  2. The generate_all_figures function expects different data"
        echo ""
        echo "Check that the following files exist:"
        echo "  - $BASE_DIR/unfreeze/layer_ratio_results.csv"
        echo "  - $BASE_DIR/unfreeze/layer_position_results.csv"
        echo "  - $BASE_DIR/ablation/ablation_results.csv"
        echo "  - $BASE_DIR/sensitivity/sensitivity_results.csv"
        echo "  - $BASE_DIR/unfreeze/layer_contribution.csv"
        echo "  - $BASE_DIR/stats/statistical_significance.csv"
    fi
fi

echo ""
echo "========================================="
echo "To view the figures:"
echo "  cd $BASE_DIR/figures/"
echo "  ls -la"
echo "========================================="