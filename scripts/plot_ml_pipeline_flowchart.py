import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

# Define box positions and labels
boxes = [
    (0.05, 0.5, 0.18, 0.25, 'Data Collection\n(Wearable Sensors)'),
    (0.28, 0.5, 0.18, 0.25, 'Signal Preprocessing\n(Filter, Interpolate, Normalize)'),
    (0.51, 0.5, 0.18, 0.25, 'Feature Extraction\n(HRV, EDA, ACC, Temp)'),
    (0.74, 0.5, 0.18, 0.25, 'Classification\n(RF, XGBoost, MLP)'),
    (0.90, 0.5, 0.08, 0.25, 'Output\n(Stress/No Stress)'),
]

# Draw boxes
for x, y, w, h, label in boxes:
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", ec="k", fc="#e0eafc", mutation_scale=0.03*fig.dpi*fig.get_figwidth())
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=13, weight='bold', color='#222')

# Draw arrows
arrowprops = dict(arrowstyle=ArrowStyle("-|>", head_length=1.5, head_width=0.8), color="#2196f3", lw=3)
for i in range(len(boxes)-1):
    x1 = boxes[i][0] + boxes[i][2]
    y1 = boxes[i][1] + boxes[i][3]/2
    x2 = boxes[i+1][0]
    y2 = boxes[i+1][1] + boxes[i+1][3]/2
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrowprops)

# Add substeps under preprocessing and feature extraction
ax.text(0.37, 0.35, 'Artifact Removal\nGap Filling', ha='center', va='center', fontsize=10, color='#555')
ax.text(0.60, 0.35, 'Time/Freq Features\nStatistical Features', ha='center', va='center', fontsize=10, color='#555')

plt.title('Smartwatch Stress Detection ML Pipeline', fontsize=16, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/advanced_figures/ml_pipeline_flowchart.png', dpi=300)
plt.close()
print('ML pipeline flowchart saved to results/advanced_figures/ml_pipeline_flowchart.png')
