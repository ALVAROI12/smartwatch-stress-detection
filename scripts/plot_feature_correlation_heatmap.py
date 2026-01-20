import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Use processed features from Empatica E4 (or WESAD)
features_path = Path('data/processed/empatica_e4_improved_features.json')
if not features_path.exists():
    raise FileNotFoundError('Feature file not found: data/processed/empatica_e4_improved_features.json')

with open(features_path, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Remove metadata columns
drop_cols = ['subject_id', 'condition', 'label', 'window_start', 'window_end', 'purity', 'window_duration_sec']
feature_cols = [c for c in df.columns if c not in drop_cols]

corr = df[feature_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.7})
plt.title('Feature Correlation Heatmap', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('results/advanced_figures/feature_correlation_heatmap.png', dpi=300)
plt.close()
print('Feature correlation heatmap saved to results/advanced_figures/feature_correlation_heatmap.png')
