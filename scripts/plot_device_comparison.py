import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path("results/advanced_figures/device_comparison.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

metrics = [
    "Heart Rate Accuracy (%)",
    "Stress Classification Accuracy (%)",
    "Sampling Rate (Hz)",
    "Battery Life (hrs)",
    "Data Latency (s)"
]

lab_equipment = [98, 92, 1000, 4, 0.5]
smartwatch = [94, 85, 64, 24, 2.0]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
rects1 = ax.bar(x - width/2, lab_equipment, width, label="Lab-Grade Equipment", color="#1f77b4")
rects2 = ax.bar(x + width/2, smartwatch, width, label="Smartwatch", color="#ff7f0e")

ax.set_ylabel("Value")
ax.set_title("Lab Equipment vs. Smartwatch Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=20, ha='right')
ax.legend()

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
print(f"Device comparison chart saved to {OUTPUT_PATH}")
