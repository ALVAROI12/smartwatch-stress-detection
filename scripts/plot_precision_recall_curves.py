import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Load predictions and probabilities
preds_path = "results/advanced_figures/model_predictions.csv"
df = pd.read_csv(preds_path)

# True and predicted labels
y_true = df['y_true'].values
# Use probability for positive class (label 1)
proba_col = [col for col in df.columns if col.startswith('proba_1')]
if not proba_col:
    raise ValueError('No probability column for positive class found.')
y_score = df[proba_col[0]].values

# Compute Precision-Recall curve and average precision
precision, recall, _ = precision_recall_curve(y_true, y_score)
ap = average_precision_score(y_true, y_score)

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, lw=2, label=f'AP = {ap:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Positive Class)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/advanced_figures/precision_recall_curves.png', dpi=300)
print('Precision-Recall curve figure saved to results/advanced_figures/precision_recall_curves.png')
