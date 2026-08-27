#!/bin/bash

BASE_DIR="/home/tahir/AAAI-26-INTENT/output-base-hpo"

echo "========================================="
echo "Generating Figures from Existing Results"
echo "Base Directory: $BASE_DIR"
echo "========================================="

# Create figures directory if it doesn't exist
mkdir -p "$BASE_DIR/figures"

python3 << 'EOF'
import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Import the functions from Intent-base-hpo.py
import importlib.util
spec = importlib.util.spec_from_file_location('Intent_base_hpo', 'Intent-base-hpo.py')
Intent_base_hpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Intent_base_hpo)

BASE_DIR = '/home/tahir/AAAI-26-INTENT/output-base-hpo'
FIGS_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

print(f'Generating figures in {FIGS_DIR}...')
print('=' * 60)

# Load all available results
print('\nLoading results files...')

# 1. Load ablation results
try:
    ablation_df = pd.read_csv(f'{BASE_DIR}/ablation/ablation_results.csv')
    print('Loaded ablation_results.csv')
except Exception as e:
    ablation_df = None
    print(f'Could not load ablation_results.csv: {e}')

# 2. Load unfreeze ratio results
try:
    ratio_df = pd.read_csv(f'{BASE_DIR}/unfreeze/layer_ratio_results.csv')
    print('Loaded layer_ratio_results.csv')
except Exception as e:
    ratio_df = None
    print(f'Could not load layer_ratio_results.csv: {e}')

# 3. Load unfreeze position results
try:
    position_df = pd.read_csv(f'{BASE_DIR}/unfreeze/layer_position_results.csv')
    print('Loaded layer_position_results.csv')
except Exception as e:
    position_df = None
    print(f'Could not load layer_position_results.csv: {e}')

# 4. Load layer contribution results
try:
    contrib_df = pd.read_csv(f'{BASE_DIR}/unfreeze/layer_contribution.csv')
    print('Loaded layer_contribution.csv')
except Exception as e:
    contrib_df = None
    print(f'Could not load layer_contribution.csv: {e}')

# 5. Load sensitivity results
try:
    sens_df = pd.read_csv(f'{BASE_DIR}/sensitivity/sensitivity_results.csv')
    print('Loaded sensitivity_results.csv')
except Exception as e:
    sens_df = None
    print(f'Could not load sensitivity_results.csv: {e}')

# 6. Load significance results
try:
    sig_df = pd.read_csv(f'{BASE_DIR}/stats/statistical_significance.csv')
    print('Loaded statistical_significance.csv')
except Exception as e:
    sig_df = None
    print(f'Could not load statistical_significance.csv: {e}')

print('\n' + '=' * 60)
print('Generating figures...\n')

# Figure 1: Ablation Bar Chart
if ablation_df is not None and not ablation_df.empty:
    print('1. Generating ablation bar chart...')
    try:
        metric = 'mean_f1'
        clean_df = ablation_df[~ablation_df.get('failed', False).astype(bool)]
        agg = clean_df.groupby('experiment')[metric].agg(['mean', 'std']).reset_index()
        agg = agg.sort_values('mean', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(agg['experiment'], agg['mean'], yerr=agg['std'].fillna(0.0), 
                     capsize=4, color='steelblue', edgecolor='black', alpha=0.8)
        
        for bar, val in zip(bars, agg['mean']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel('Experiment Configuration', fontsize=12)
        ax.set_title('Component Ablation Results (mean +/- std across seeds)', fontsize=14, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS_DIR, '01_ablation_barchart.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved: 01_ablation_barchart.png')
    except Exception as e:
        print(f'  Failed: {e}')

# Figure 2: Performance vs Unfreeze Ratio
if ratio_df is not None and not ratio_df.empty:
    print('2. Generating unfreeze ratio performance plot...')
    try:
        metric = 'mean_f1'
        clean_df = ratio_df[~ratio_df.get('failed', False).astype(bool)]
        agg = ratio_df.groupby('unfrozen_ratio')[metric].agg(['mean', 'std', 'count']).reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(agg['unfrozen_ratio'], agg['mean'], yerr=agg['std'].fillna(0.0),
                   fmt='o-', capsize=4, color='steelblue', markersize=8, linewidth=2)
        
        if len(agg) >= 4:
            coeffs = np.polyfit(agg['unfrozen_ratio'], agg['mean'], deg=2)
            xs = np.linspace(agg['unfrozen_ratio'].min(), agg['unfrozen_ratio'].max(), 100)
            ax.plot(xs, np.polyval(coeffs, xs), '--', color='crimson', alpha=0.7, label='Quadratic fit')
            ax.legend()
        
        ax.set_xlabel('Unfrozen Ratio (fraction of backbone layers trainable)', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title('Performance vs. Unfrozen Backbone Ratio', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS_DIR, '02_performance_vs_unfreeze.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved: 02_performance_vs_unfreeze.png')
    except Exception as e:
        print(f'  Failed: {e}')

# Figure 3: Layer Contribution Heatmap
if contrib_df is not None and not contrib_df.empty:
    print('3. Generating layer contribution heatmap...')
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(6, 0.4 * len(contrib_df))))
        
        vals1 = contrib_df['cka_vs_frozen'].to_numpy().reshape(-1, 1)
        im1 = ax1.imshow(vals1, aspect='auto', cmap='viridis')
        ax1.set_yticks(range(len(contrib_df)))
        ax1.set_yticklabels(contrib_df['layer'])
        ax1.set_xticks([])
        ax1.set_ylabel('Backbone Layer Index', fontsize=11)
        ax1.set_title('Linear CKA vs. Fully-Frozen Model', fontsize=11)
        plt.colorbar(im1, ax=ax1, fraction=0.05, label='CKA Similarity')
        
        vals2 = contrib_df['grad_norm_mean'].fillna(0).to_numpy().reshape(-1, 1)
        im2 = ax2.imshow(vals2, aspect='auto', cmap='magma')
        ax2.set_yticks(range(len(contrib_df)))
        ax2.set_yticklabels(contrib_df['layer'])
        ax2.set_xticks([])
        ax2.set_ylabel('Backbone Layer Index', fontsize=11)
        ax2.set_title('Mean Gradient Norm per Layer', fontsize=11)
        plt.colorbar(im2, ax=ax2, fraction=0.05, label='Gradient Norm')
        
        fig.suptitle('Layer-wise Contribution Analysis', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS_DIR, '03_layer_contribution_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved: 03_layer_contribution_heatmap.png')
    except Exception as e:
        print(f'  Failed: {e}')

# Figure 4: Sensitivity Heatmap
if sens_df is not None and not sens_df.empty:
    print('4. Generating sensitivity heatmap...')
    try:
        params = list(sens_df['param'].unique())
        n_params = len(params)
        n_values = max(len(sens_df[sens_df['param'] == p]) for p in params)
        
        grid = np.full((n_params, n_values), np.nan)
        param_labels = []
        
        for i, p in enumerate(params):
            param_labels.append(p)
            subset = sens_df[sens_df['param'] == p].sort_values('value')
            scores = subset['score'].to_numpy()
            finite = scores[np.isfinite(scores)]
            if finite.size > 0:
                lo, hi = finite.min(), finite.max()
                if hi > lo:
                    norm = (scores - lo) / (hi - lo)
                else:
                    norm = np.zeros_like(scores)
                grid[i, :len(norm)] = norm
        
        fig, ax = plt.subplots(figsize=(max(10, n_values * 0.6), max(6, n_params * 0.5)))
        im = ax.imshow(grid, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        
        ax.set_yticks(range(n_params))
        ax.set_yticklabels(param_labels, fontsize=10)
        ax.set_xticks(range(n_values))
        ax.set_xticklabels([str(i+1) for i in range(n_values)])
        ax.set_xlabel('Grid Position (low -> high value)', fontsize=11)
        ax.set_title('Hyperparameter Sensitivity (normalized dev metric)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Normalized Score')
        
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS_DIR, '04_sensitivity_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved: 04_sensitivity_heatmap.png')
    except Exception as e:
        print(f'  Failed: {e}')

# Figure 5: Significance Table Figure
if sig_df is not None and not sig_df.empty:
    print('5. Generating significance table figure...')
    try:
        cols = ['experiment', 'n', 'mean', 'std', 'delta', 'cohens_d']
        if 't_pvalue_fdr_reject_at_0.05' in sig_df.columns:
            cols.append('t_pvalue_fdr_reject_at_0.05')
        table_cols = [c for c in cols if c in sig_df.columns]
        
        table_data = sig_df[table_cols].copy()
        for c in table_data.columns:
            if table_data[c].dtype in ['float64', 'float32']:
                table_data[c] = table_data[c].map(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
            elif c == 't_pvalue_fdr_reject_at_0.05':
                table_data[c] = table_data[c].map(lambda x: 'YES' if x else '')
            elif c == 'n':
                table_data[c] = table_data[c].astype(int)
        
        fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(table_cols)), max(4, 0.4 * len(table_data) + 1)))
        ax.axis('off')
        
        table = ax.table(cellText=table_data.values.tolist(), 
                        colLabels=table_cols, 
                        loc='center',
                        cellLoc='center',
                        colWidths=[0.15] * len(table_cols))
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        ax.set_title('Statistical Significance vs. Baseline', 
                    fontsize=14, fontweight='bold', pad=20)
        
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS_DIR, '05_significance_table.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved: 05_significance_table.png')
    except Exception as e:
        print(f'  Failed: {e}')

# Figure 6: Sensitivity Correlations
try:
    corr_df = pd.read_csv(f'{BASE_DIR}/sensitivity/sensitivity_correlations.csv')
    if corr_df is not None and not corr_df.empty:
        print('6. Generating sensitivity correlation bar chart...')
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            corr_df_sorted = corr_df.sort_values('spearman_rho', ascending=True)
            colors = ['red' if x < 0 else 'steelblue' for x in corr_df_sorted['spearman_rho']]
            bars = ax.barh(corr_df_sorted['param'], corr_df_sorted['spearman_rho'], color=colors, alpha=0.7)
            
            for i, (idx, row) in enumerate(corr_df_sorted.iterrows()):
                p_val = row['spearman_p']
                p_label = f'p={p_val:.3f}' if pd.notna(p_val) else 'N/A'
                ax.text(row['spearman_rho'] + 0.01 * (1 if row['spearman_rho'] >= 0 else -1), 
                       i, f'  {p_label}', va='center', fontsize=8)
            
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Spearman Correlation (positive = higher value -> better performance)', fontsize=11)
            ax.set_ylabel('Hyperparameter', fontsize=11)
            ax.set_title('Sensitivity Analysis: Parameter vs. Performance Correlation', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            fig.tight_layout()
            fig.savefig(os.path.join(FIGS_DIR, '06_sensitivity_correlations.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print('  Saved: 06_sensitivity_correlations.png')
        except Exception as e:
            print(f'  Failed: {e}')
except Exception as e:
    print('  Could not load sensitivity_correlations.csv')

# Figure 7: Unfreeze Position Results
if position_df is not None and not position_df.empty:
    print('7. Generating unfreeze position comparison plot...')
    try:
        metric = 'mean_f1'
        clean_df = position_df[~position_df.get('failed', False).astype(bool)]
        agg = clean_df.groupby('unfreeze_position')[metric].agg(['mean', 'std']).reset_index()
        agg = agg.sort_values('mean', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(agg['unfreeze_position'], agg['mean'], yerr=agg['std'].fillna(0.0),
                     capsize=4, color='teal', edgecolor='black', alpha=0.8)
        
        for bar, val in zip(bars, agg['mean']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel('Unfreeze Position', fontsize=12)
        ax.set_title('Performance vs. Unfreeze Position (all layers trainable)', fontsize=14, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS_DIR, '07_unfreeze_position.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved: 07_unfreeze_position.png')
    except Exception as e:
        print(f'  Failed: {e}')

# Figure 8: Pareto Frontier
if (ablation_df is not None and not ablation_df.empty) or (ratio_df is not None and not ratio_df.empty):
    print('8. Generating Pareto frontier plot...')
    try:
        if ablation_df is not None and not ablation_df.empty:
            df_to_use = ablation_df[~ablation_df.get('failed', False).astype(bool)]
            group_col = 'experiment'
        elif ratio_df is not None and not ratio_df.empty:
            df_to_use = ratio_df[~ratio_df.get('failed', False).astype(bool)]
            group_col = 'unfrozen_ratio'
        else:
            raise ValueError('No data available')
        
        metric = 'mean_f1'
        cost_col = 'peak_gpu_mem_MB'
        
        if cost_col in df_to_use.columns:
            agg = df_to_use.groupby(group_col).agg(
                perf=(metric, 'mean'),
                cost=(cost_col, 'mean')
            ).reset_index()
            
            pts = agg[['cost', 'perf']].to_numpy()
            order = np.argsort(-pts[:, 1])
            pareto_mask = np.zeros(len(pts), dtype=bool)
            best_cost_so_far = np.inf
            for idx in order:
                if pts[idx, 0] <= best_cost_so_far:
                    pareto_mask[idx] = True
                    best_cost_so_far = pts[idx, 0]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(agg.loc[~pareto_mask, 'cost'], agg.loc[~pareto_mask, 'perf'], 
                      c='gray', alpha=0.5, s=50, label='Dominated')
            ax.scatter(agg.loc[pareto_mask, 'cost'], agg.loc[pareto_mask, 'perf'], 
                      c='crimson', s=100, label='Pareto-optimal', zorder=3)
            
            pf = agg.loc[pareto_mask].sort_values('cost')
            if len(pf) > 1:
                ax.plot(pf['cost'], pf['perf'], '--', c='crimson', alpha=0.5)
            
            for _, row in agg.iterrows():
                label = str(row[group_col])
                if len(label) > 20:
                    label = label[:18] + '...'
                ax.annotate(label, (row['cost'], row['perf']), fontsize=7,
                           xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('Peak GPU Memory (MB)', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title('Pareto Frontier: Performance vs. Memory Cost', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(FIGS_DIR, '08_pareto_frontier.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print('  Saved: 08_pareto_frontier.png')
        else:
            print('  Memory data not available for Pareto plot')
    except Exception as e:
        print(f'  Failed: {e}')

print('\n' + '=' * 60)
print('Figure generation complete!')
print(f'Figures saved in: {FIGS_DIR}')
print('\nGenerated files:')
for f in sorted(os.listdir(FIGS_DIR)):
    if f.endswith('.png'):
        size = os.path.getsize(os.path.join(FIGS_DIR, f)) / 1024
        print(f'  {f} ({size:.1f} KB)')
EOF

echo ""
echo "========================================="
echo "Figure generation complete!"
echo "Figures saved in: $BASE_DIR/figures/"
echo "========================================="