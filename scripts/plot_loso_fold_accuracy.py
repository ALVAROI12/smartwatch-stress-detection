import json
import numpy as np
import matplotlib.pyplot as plt

# Simple LOSO fold accuracy comparison for Random Forest and XGBoost
subjects = [f"S{i}" for i in range(2, 18) if i != 12]

with open('results/wesad_ml_results.json', 'r') as f:
    results = json.load(f)

rf_scores = results['random_forest']['cv_scores']['test_accuracy']
xgb_scores = results['xgboost']['cv_scores']['accuracy']

positions = np.arange(len(subjects))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(positions - width / 2, rf_scores, width, label='Random Forest', color='#1f77b4')
ax.bar(positions + width / 2, xgb_scores, width, label='XGBoost', color='#ff7f0e')

ax.set_ylabel('Accuracy')
ax.set_xlabel('Held-out Subject')
ax.set_title('LOSO Fold Accuracy by Model')
ax.set_xticks(positions)
ax.set_xticklabels(subjects, rotation=45, ha='right')
ax.set_ylim(0, 1.1)
ax.legend()
ax.yaxis.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
output_path = 'results/advanced_figures/loso_fold_accuracy_bar.png'
plt.savefig(output_path, dpi=300)
print(f'LOSO fold accuracy bar plot saved to {output_path}')
