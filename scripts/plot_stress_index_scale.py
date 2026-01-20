import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

with open('data/processed/wesad_features.json', 'r') as f:
    features = json.load(f)

baseline_rows = [row for row in features if row.get('label') == 0]
stress_rows = [row for row in features if row.get('label') == 1]

def clean_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        values.append(value)
    return np.array(values)

def compute_stats(rows, key):
    arr = clean_values(rows, key)
    if arr.size == 0:
        return {'median': float('nan'), 'q25': float('nan'), 'q75': float('nan'), 'p10': float('nan'), 'p90': float('nan')}
    return {
        'median': float(np.nanmedian(arr)),
        'q25': float(np.nanpercentile(arr, 25)),
        'q75': float(np.nanpercentile(arr, 75)),
        'p10': float(np.nanpercentile(arr, 10)),
        'p90': float(np.nanpercentile(arr, 90)),
    }

stats = {
    'baseline': {
        'hrv': compute_stats(baseline_rows, 'hr_variability'),
        'eda_std': compute_stats(baseline_rows, 'eda_std'),
        'eda_range': compute_stats(baseline_rows, 'eda_range'),
    },
    'stress': {
        'hrv': compute_stats(stress_rows, 'hr_variability'),
        'eda_std': compute_stats(stress_rows, 'eda_std'),
        'eda_range': compute_stats(stress_rows, 'eda_range'),
    }
}

counts = {'baseline': len(baseline_rows), 'stress': len(stress_rows)}

hrv_low = stats['baseline']['hrv']['q75']
hrv_mid_low = stats['baseline']['hrv']['q25']
hrv_high = stats['stress']['hrv']['median']

eda_std_low = stats['baseline']['eda_std']['q75']
eda_std_high = stats['stress']['eda_std']['median']
eda_std_floor = stats['stress']['eda_std']['p10']

scr_low = stats['baseline']['eda_range']['median']
scr_low_floor = stats['baseline']['eda_range']['p10']
scr_mid = stats['stress']['eda_range']['median']
scr_high = stats['stress']['eda_range']['q75']

def format_value(value, decimals=2):
    if math.isnan(value):
        return 'n/a'
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(value)

levels = [
    {
        'name': 'Low Stress',
        'physio': (
            f"HRV >= {format_value(hrv_low, 0)} ms (baseline Q3)\n"
            f"EDA SD <= {format_value(eda_std_low, 3)} uS\n"
            f"SCR amplitude <= {format_value(scr_low, 2)} uS"
        ),
        'mental': (
            'Resilient baseline\n'
            f"Depression screening: SCR <= {format_value(scr_low_floor, 2)} uS\n"
            'Supports restorative recovery'
        ),
        'color': '#3c9d5d',
        'text_color': 'white'
    },
    {
        'name': 'Moderate Stress',
        'physio': (
            f"HRV {format_value(hrv_mid_low, 0)}-{format_value(hrv_low, 0)} ms (baseline IQR)\n"
            f"EDA SD {format_value(eda_std_low, 3)}-{format_value(eda_std_high, 3)} uS\n"
            f"SCR amplitude {format_value(scr_low, 2)}-{format_value(scr_mid, 2)} uS"
        ),
        'mental': (
            'Anxiety screening window\n'
            'Monitor cumulative load\n'
            'Flag sustained autonomic shift'
        ),
        'color': '#f4b942',
        'text_color': '#1f2d3d'
    },
    {
        'name': 'High Stress',
        'physio': (
            f"HRV <= {format_value(hrv_high, 0)} ms (stress median)\n"
            f"EDA SD >= {format_value(eda_std_high, 3)} uS (stress median)\n"
            f"SCR spikes >= {format_value(scr_high, 2)} uS (stress Q3)"
        ),
        'mental': (
            f"Anxiety: SCR >= {format_value(scr_high, 2)} uS\n"
            f"Suicide risk: EDA SD < {format_value(eda_std_floor, 3)} uS (hyporeactivity)"
        ),
        'color': '#d9534f',
        'text_color': 'white'
    }
]

fig, ax = plt.subplots(figsize=(9, 6))
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

for idx, level in enumerate(levels):
    y_pos = 0.68 - idx * 0.3
    rect = Rectangle((0.05, y_pos), 0.4, 0.22, facecolor=level['color'], alpha=0.9)
    ax.add_patch(rect)
    ax.text(0.07, y_pos + 0.11, level['name'], fontsize=14, fontweight='bold', color=level['text_color'], va='center')
    ax.text(0.07, y_pos + 0.04, level['physio'], fontsize=10.5, color=level['text_color'], va='top')

    card = Rectangle((0.52, y_pos), 0.4, 0.22, facecolor='#f8f9fb', edgecolor='#c6ccd8', linewidth=1.2)
    ax.add_patch(card)
    ax.text(0.54, y_pos + 0.11, 'Mental Health Application', fontsize=12, fontweight='bold', color='#1f2d3d', va='center')
    ax.text(0.54, y_pos + 0.04, level['mental'], fontsize=10.5, color='#1f2d3d', va='top')

ax.text(0.05, 0.92, 'Stress Index Thresholds', fontsize=18, fontweight='bold', color='#1f2d3d')
ax.text(0.05, 0.86, 'Quantitative mapping from wearable biomarkers to clinical triage cues', fontsize=12, color='#39465b')

scale_labels = ['Low', 'Moderate', 'High']
ax.plot([0.25, 0.25], [0.8, 0.0], color='#1f2d3d', linewidth=2)
for idx, label in enumerate(scale_labels):
    y = 0.76 - idx * 0.3
    ax.scatter(0.25, y, s=90, color='#1f2d3d')
    ax.text(0.26, y, label, fontsize=11, color='#1f2d3d', va='center', ha='left')

ax.text(0.05, 0.04, (
    f"Derived from WESAD feature set (baseline windows n={counts['baseline']}, "
    f"stress windows n={counts['stress']})."
), fontsize=9.5, color='#4a576c')

plt.tight_layout()
output_path = 'results/advanced_figures/stress_index_threshold_diagram.png'
plt.savefig(output_path, dpi=300)
print(f'Stress index threshold diagram saved to {output_path}')
