import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Try to use the most recent feature importance file
rf_path = Path('results/rf_feature_importance.csv')
xgb_path = Path('results/xgb_feature_importance.csv')

if rf_path.exists():
    df = pd.read_csv(rf_path)
    model = 'Random Forest'
elif xgb_path.exists():
    df = pd.read_csv(xgb_path)
    model = 'XGBoost'
else:
    raise FileNotFoundError('No feature importance file found.')

# Expect columns: feature, importance
if 'feature' not in df.columns or 'importance' not in df.columns:
    raise ValueError('Feature importance file must have columns: feature, importance')

# Sort and select top 15
df = df.sort_values('importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(df['feature'][::-1], df['importance'][::-1], color='#219ebc')
ax.set_xlabel('Importance')
ax.set_title(f'Top 15 Feature Importances ({model})', fontsize=15, weight='bold')
plt.tight_layout()
plt.savefig('results/advanced_figures/feature_importance.png', dpi=300)
plt.close()
print('Feature importance plot saved to results/advanced_figures/feature_importance.png')
