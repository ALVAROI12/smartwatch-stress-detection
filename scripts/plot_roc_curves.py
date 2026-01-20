import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from pathlib import Path

# Use predictions from best model (simulate for demo)
wesad_dir = Path("data/wesad")
subject_dirs = sorted([d for d in wesad_dir.iterdir() if d.is_dir() and d.name.startswith('S')])
subject_file = None
for d in subject_dirs:
    pkl_files = list(d.glob("*.pkl"))
    if pkl_files:
        subject_file = pkl_files[0]
        break

if subject_file is None:
    raise FileNotFoundError("No WESAD subject .pkl file found.")

with open(subject_file, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

labels = data['label'].flatten()
classes = [1, 2, 3]
labels_bin = label_binarize(labels, classes=classes)

# Simulate predicted probabilities (for demo)
np.random.seed(42)
preds_bin = labels_bin.copy().astype(float)
preds_bin += np.random.normal(0, 0.1, preds_bin.shape)
preds_bin = np.clip(preds_bin, 0, 1)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i, c in enumerate(classes):
    fpr[c], tpr[c], _ = roc_curve(labels_bin[:, i], preds_bin[:, i])
    roc_auc[c] = auc(fpr[c], tpr[c])

# Plot all ROC curves
plt.figure(figsize=(7, 6))
colors = ['#8ecae6', '#fb8500', '#219ebc']
for i, c in enumerate(classes):
    plt.plot(fpr[c], tpr[c], color=colors[i], lw=2,
             label=f'Class {c} (AUC = {roc_auc[c]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves (Simulated)', fontsize=15, weight='bold')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('results/advanced_figures/roc_curves.png', dpi=300)
plt.close()
print('ROC curves figure saved to results/advanced_figures/roc_curves.png')
