import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load model predictions generated earlier
predictions_path = 'results/advanced_figures/model_predictions.csv'
try:
    df = pd.read_csv(predictions_path)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Expected predictions file at {predictions_path}. Run export_model_predictions.py first."
    )

conf_matrix = pd.crosstab(df['y_true'], df['y_pred']).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
conf_values = conf_matrix.values
row_totals = conf_values.sum(axis=1, keepdims=True)
percentages = np.divide(conf_values, row_totals, out=np.zeros_like(conf_values, dtype=float), where=row_totals != 0)

labels = [
    [f"{conf_values[i, j]}\n({percentages[i, j]*100:.1f}%)" for j in range(conf_values.shape[1])]
    for i in range(conf_values.shape[0])
]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(conf_values, cmap='Blues')

for i in range(conf_values.shape[0]):
    for j in range(conf_values.shape[1]):
        text_color = 'white' if conf_values[i, j] > conf_values.max() / 2 else '#0f1a2c'
        ax.text(j, i, labels[i][j], ha='center', va='center', color=text_color, fontsize=11)

ax.set_xticks(np.arange(2))
ax.set_yticks(np.arange(2))
az_labels = ['Baseline', 'Stress']
ax.set_xticklabels([f"Pred {lbl}" for lbl in az_labels])
ax.set_yticklabels([f"True {lbl}" for lbl in az_labels])
ax.set_title('Classification Results (Confusion Matrix)')
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(1.5, -0.5)

plt.tight_layout()
output_path = 'results/advanced_figures/classification_results_confusion_matrix.png'
plt.savefig(output_path, dpi=300)
print(f'Classification results confusion matrix saved to {output_path}')
