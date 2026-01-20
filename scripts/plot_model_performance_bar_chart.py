import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('results/wesad_ml_results.json', 'r') as f:
    results = json.load(f)

# Extract metrics for each model
metrics = ['accuracy', 'precision', 'recall', 'f1']
models = ['Random Forest', 'XGBoost']
model_keys = ['Random Forest', 'XGBoost']

# Find the comparison table
comparison = results['comparison']

# Prepare data for plotting
bar_data = {metric: [] for metric in metrics}
error_data = {metric: [] for metric in metrics}
for metric in metrics:
    for model in model_keys:
        row = next((item for item in comparison if item['Metric'] == metric), None)
        if row:
            bar_data[metric].append(float(row[model]))
            # Use correct std key
            if model == 'Random Forest':
                error_data[metric].append(row['RF_std'])
            elif model == 'XGBoost':
                error_data[metric].append(row['XGB_std'])
            else:
                error_data[metric].append(0)
        else:
            bar_data[metric].append(0)
            error_data[metric].append(0)

# Plot
x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(8,6))
rects1 = ax.bar(x - width/2, [bar_data[m][0] for m in metrics], width, yerr=[error_data[m][0] for m in metrics], label='Random Forest', capsize=5)
rects2 = ax.bar(x + width/2, [bar_data[m][1] for m in metrics], width, yerr=[error_data[m][1] for m in metrics], label='XGBoost', capsize=5)

ax.set_ylabel('Score')
ax.set_title('Model Performance Metrics')
ax.set_xticks(x)
ax.set_xticklabels([m.capitalize() for m in metrics])
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig('results/advanced_figures/model_performance_bar_chart.png', dpi=300)
print('Model performance metrics bar chart saved to results/advanced_figures/model_performance_bar_chart.png')
