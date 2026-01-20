import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RESULTS_PATH = Path('results/wesad_ml_results.json')
if not RESULTS_PATH.exists():
    raise FileNotFoundError('results/wesad_ml_results.json not found. Run the evaluation pipeline first.')

with RESULTS_PATH.open('r') as f:
    results = json.load(f)

comparison_rows = results['comparison']
comparison_map = {row['Metric']: row for row in comparison_rows}

def get_metric(metric_key, model_key, std_key):
    row = comparison_map[metric_key]
    mean_val = float(row[model_key])
    std_val = float(row[std_key])
    return mean_val, std_val

metrics = [
    ('accuracy', 'Accuracy'),
    ('precision', 'Precision'),
    ('recall', 'Recall'),
    ('f1', 'F1 Score'),
    ('roc_auc', 'ROC AUC')
]

rf_color = '#1f77b4'
xgb_color = '#ff7f0e'
diff_color = '#39465b'

card_specs = {
    'rf': {'label': 'Random Forest', 'color': rf_color, 'pos': (0.05, 0.2)},
    'xgb': {'label': 'XGBoost', 'color': xgb_color, 'pos': (0.55, 0.2)}
}

card_size = (0.38, 0.6)
row_count = len(metrics)
row_spacing = card_size[1] / (row_count + 1)

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

ax.text(0.05, 0.9, 'Model Performance Summary', fontsize=20, fontweight='bold', color=diff_color)
ax.text(0.05, 0.84, 'Cross-validated results on WESAD (leave-one-subject-out)', fontsize=12, color='#607089')

for key, spec in card_specs.items():
    box = FancyBboxPatch(
        spec['pos'],
        card_size[0],
        card_size[1],
        boxstyle='round,pad=0.02',
        facecolor=spec['color'],
        alpha=0.18,
        edgecolor=spec['color'],
        linewidth=1.6
    )
    ax.add_patch(box)
    ax.text(spec['pos'][0] + 0.02, spec['pos'][1] + card_size[1] - 0.04, spec['label'], fontsize=16, fontweight='bold', color=diff_color)

    for idx, (metric_key, metric_label) in enumerate(metrics, start=1):
        mean_val, std_val = get_metric(metric_key, 'Random Forest' if key == 'rf' else 'XGBoost', 'RF_std' if key == 'rf' else 'XGB_std')
        y = spec['pos'][1] + card_size[1] - idx * row_spacing
        ax.text(spec['pos'][0] + 0.02, y, metric_label, fontsize=12, color=diff_color, fontweight='medium')
        ax.text(spec['pos'][0] + card_size[0] - 0.02, y, f"{mean_val:.3f} +- {std_val:.3f}", fontsize=12, color=diff_color, ha='right')

center_x = 0.47
ax.text(center_x, 0.74, 'Delta (XGBoost - Random Forest)', fontsize=13, color=diff_color, ha='center', fontweight='bold')

for idx, (metric_key, metric_label) in enumerate(metrics, start=1):
    rf_mean, _ = get_metric(metric_key, 'Random Forest', 'RF_std')
    xgb_mean, _ = get_metric(metric_key, 'XGBoost', 'XGB_std')
    delta = xgb_mean - rf_mean
    y = 0.74 - idx * (row_spacing * 0.9)
    ax.text(center_x, y, f"{metric_label}: {delta:+.3f}", fontsize=12, color=diff_color, ha='center')

info = results['dataset_info']
ax.text(0.05, 0.12, f"Windows: {info['total_windows']} (stress {info['stress_windows']}, baseline {info['baseline_windows']}) | Subjects: {info['subjects']} | Features: {info['features']}", fontsize=10, color='#6c7a92')
ax.text(0.05, 0.07, 'Error bars denote standard deviation across LOSO folds.', fontsize=10, color='#6c7a92')

output_path = Path('results/advanced_figures/performance_summary.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f'Updated performance summary saved to {output_path}')
