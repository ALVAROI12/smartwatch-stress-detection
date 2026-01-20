import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# WESAD LOSO configuration: 15 subjects (S2-S17 without S12)
subjects = [f"S{i}" for i in range(2, 18) if i != 12]
folds = len(subjects)

# Build matrix storing 0=train, 1=test per fold
split_matrix = np.zeros((folds, folds), dtype=int)
for idx in range(folds):
    split_matrix[idx, idx] = 1

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(split_matrix, cmap=plt.cm.get_cmap('Blues', 2))

ax.set_xticks(np.arange(folds))
ax.set_yticks(np.arange(folds))
ax.set_xticklabels(subjects, rotation=45, ha='right')
ax.set_yticklabels([f"Fold {i+1}" for i in range(folds)])
ax.set_xlabel('Subject ID')
ax.set_ylabel('LOSO Fold (held-out subject)')
ax.set_title('Leave-One-Subject-Out (LOSO) Train/Test Configuration')

train_color = plt.cm.Blues(0.3)
test_color = plt.cm.Blues(0.9)
legend_handles = [
    Patch(facecolor=train_color, edgecolor='none', label='Train set'),
    Patch(facecolor=test_color, edgecolor='none', label='Test subject')
]
ax.legend(handles=legend_handles, loc='upper right', frameon=False)

for row in range(folds):
    for col in range(folds):
        label = 'Test' if split_matrix[row, col] == 1 else 'Train'
        ax.text(col, row, label, ha='center', va='center', fontsize=8, color='#0f1a2c')

plt.tight_layout()
output_path = 'results/advanced_figures/loso_train_test_diagram.png'
plt.savefig(output_path, dpi=300)
print(f'LOSO train/test diagram saved to {output_path}')
