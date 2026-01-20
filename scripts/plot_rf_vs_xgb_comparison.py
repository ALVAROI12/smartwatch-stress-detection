import json
import matplotlib.pyplot as plt
import numpy as np

# Load aggregated comparison metrics for Random Forest vs XGBoost
with open('results/wesad_ml_results.json', 'r') as f:
    results = json.load(f)

comparison_rows = results['comparison']
metrics = [row['Metric'].capitalize() for row in comparison_rows]
rf_scores = [float(row['Random Forest']) for row in comparison_rows]
xgb_scores = [float(row['XGBoost']) for row in comparison_rows]
rf_std = [float(row['RF_std']) for row in comparison_rows]
xgb_std = [float(row['XGB_std']) for row in comparison_rows]

positions = np.arange(len(metrics))
bar_height = 0.35
colors = {'rf': '#1f77b4', 'xgb': '#ff7f0e'}

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(positions - bar_height / 2, rf_scores, bar_height, xerr=rf_std,
        color=colors['rf'], label='Random Forest', capsize=5)
ax.barh(positions + bar_height / 2, xgb_scores, bar_height, xerr=xgb_std,
        color=colors['xgb'], label='XGBoost', capsize=5)

ax.set_xlabel('Mean CV Score')
ax.set_xlim(0, 1.1)
ax.set_yticks(positions)
ax.set_yticklabels(metrics)
ax.set_title('Random Forest vs XGBoost Performance Comparison')
ax.legend(loc='lower right')
ax.xaxis.grid(True, linestyle='--', alpha=0.3)

for idx, metric in enumerate(metrics):
    diff = xgb_scores[idx] - rf_scores[idx]
    max_score = max(rf_scores[idx], xgb_scores[idx])
    ax.text(max_score + 0.03, positions[idx], f'{diff:+.2f}',
            va='center', fontsize=9, color='#333333')

plt.tight_layout()
output_path = 'results/advanced_figures/rf_vs_xgb_comparison.png'
plt.savefig(output_path, dpi=300)
print(f'Random Forest vs XGBoost comparison plot saved to {output_path}')
