import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Aggregate label counts across all WESAD subjects
data_dir = Path("data/wesad")
subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('S')])
label_counts = {}
label_names = {1: 'Baseline', 2: 'Stress', 3: 'Amusement'}

for d in subject_dirs:
    pkl_files = list(d.glob("*.pkl"))
    if not pkl_files:
        continue
    with open(pkl_files[0], 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    labels = data['label'].flatten()
    for label in np.unique(labels):
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += np.sum(labels == label)

# Only plot known labels
labels_to_plot = [1, 2, 3]
counts = [label_counts.get(l, 0) for l in labels_to_plot]
names = [label_names[l] for l in labels_to_plot]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(names, counts, color=['#8ecae6', '#fb8500', '#219ebc'])

for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{count:,}', ha='center', va='bottom', fontsize=12)

ax.set_ylabel('Number of Samples')
ax.set_title('WESAD Dataset Label Distribution', fontsize=15, weight='bold')
plt.tight_layout()
plt.savefig('results/advanced_figures/wesad_label_distribution.png', dpi=300)
plt.close()
print('WESAD label distribution bar chart saved to results/advanced_figures/wesad_label_distribution.png')
