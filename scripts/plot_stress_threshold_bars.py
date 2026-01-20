import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Load feature dataset used for threshold derivation
features_path = Path('data/processed/wesad_features.json')
if not features_path.exists():
    raise FileNotFoundError('Expected data/processed/wesad_features.json; run preprocessing first.')

with features_path.open('r') as f:
    rows = json.load(f)

baseline_rows = [row for row in rows if row.get('label') == 0]
stress_rows = [row for row in rows if row.get('label') == 1]

if not baseline_rows or not stress_rows:
    raise ValueError('Dataset must include baseline (label=0) and stress (label=1) windows.')


def valid_values(source, key):
    values = []
    for item in source:
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        values.append(float(value))
    return np.array(values)


def summarize(rows_subset, key):
    arr = valid_values(rows_subset, key)
    if arr.size == 0:
        return {
            'mean': float('nan'),
            'median': float('nan'),
            'q25': float('nan'),
            'q75': float('nan')
        }
    return {
        'mean': float(np.nanmean(arr)),
        'median': float(np.nanmedian(arr)),
        'q25': float(np.nanpercentile(arr, 25)),
        'q75': float(np.nanpercentile(arr, 75))
    }

stats = {
    'baseline': {
        'hrv': summarize(baseline_rows, 'hr_variability'),
        'eda_sd': summarize(baseline_rows, 'eda_std'),
        'scr_amp': summarize(baseline_rows, 'eda_range')
    },
    'stress': {
        'hrv': summarize(stress_rows, 'hr_variability'),
        'eda_sd': summarize(stress_rows, 'eda_std'),
        'scr_amp': summarize(stress_rows, 'eda_range')
    }
}

# Threshold choices (all derived from the summary statistics above)
hrv_low = stats['baseline']['hrv']['q75']
hrv_mod = stats['baseline']['hrv']['mean']
hrv_high = stats['stress']['hrv']['median']

eda_low = stats['baseline']['eda_sd']['median']
eda_mod = stats['stress']['eda_sd']['median']
eda_high = stats['stress']['eda_sd']['q75']

scr_low = stats['baseline']['scr_amp']['median']
scr_mod = stats['stress']['scr_amp']['median']
scr_high = stats['stress']['scr_amp']['q75']

metrics = ['HRV (ms)', 'EDA SD (uS)', 'SCR amplitude (uS)']
thresholds = {
    'Low': [hrv_low, eda_low, scr_low],
    'Moderate': [hrv_mod, eda_mod, scr_mod],
    'High': [hrv_high, eda_high, scr_high]
}

colors = {
    'Low': '#3c9d5d',
    'Moderate': '#f4b942',
    'High': '#d9534f'
}

x = np.arange(len(metrics))
width = 0.22

fig, ax = plt.subplots(figsize=(9, 5))
for idx, level in enumerate(thresholds):
    values = thresholds[level]
    positions = x + (idx - 1) * width
    bars = ax.bar(positions, values, width, label=f'{level} threshold', color=colors[level])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(values, default=1),
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Value')
ax.set_title('Stress index thresholds derived from WESAD dataset')
ax.legend(frameon=False)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_ylim(bottom=0)

counts = (len(baseline_rows), len(stress_rows))
ax.text(0.02, -0.15, f'Baseline windows n={counts[0]} | Stress windows n={counts[1]}', transform=ax.transAxes,
        fontsize=9, color='#4a576c')

plt.tight_layout()
output_path = Path('results/advanced_figures/stress_index_threshold_bars.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f'Stress threshold bar chart saved to {output_path}')
