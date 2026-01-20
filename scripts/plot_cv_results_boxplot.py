import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load results
with open('results/wesad_ml_results.json', 'r') as f:
    results = json.load(f)

# Metrics to plot
metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']
model_keys = ['random_forest', 'xgboost']
model_labels = ['Random Forest', 'XGBoost']

# Prepare data for boxplots
plot_data = []
for model_key, model_label in zip(model_keys, model_labels):
    if model_key in results:
        cv_scores = results[model_key]['cv_scores']
        for metric in metrics:
            if metric in cv_scores:
                for value in cv_scores[metric]:
                    plot_data.append({
                        'Model': model_label,
                        'Metric': metric.replace('test_', '').capitalize(),
                        'Score': value
                    })

# Convert to DataFrame
plot_df = pd.DataFrame(plot_data)

# Plot
plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics):
    plt.subplot(1, len(metrics), i+1)
    subset = plot_df[plot_df['Metric'] == metric.replace('test_', '').capitalize()]
    subset.boxplot(column='Score', by='Model', grid=False)
    plt.title(metric.replace('test_', '').capitalize())
    plt.suptitle('')
    plt.xlabel('')
    plt.ylim(0, 1.1)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle('Cross-Validation Results by Model and Metric')
plt.savefig('results/advanced_figures/cv_results_boxplot.png', dpi=300)
print('Cross-validation results box plot saved to results/advanced_figures/cv_results_boxplot.png')
