import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path("results/advanced_figures/device_comparison_gap.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

metrics = [
    "Max Accuracy (3-class) (%)",
    "Data Quality (20% threshold) (%)",
    "Available Sensors (count)",
    "Battery Life (hours)",
    "Approx. Cost (USD)"
]

research = np.array([95.23, 88, 42, 48, 1690])
consumer = np.array([88.60, 63, 2, 44, 275])

gap = research - consumer

x = np.arange(len(metrics)) * 1.25
width = 0.28

fig, ax = plt.subplots(figsize=(10, 5.2))
max_value = max(research.max(), consumer.max())
ax.set_ylim(0, max_value + 80)

bars1 = ax.bar(x - width/2, research, width, label="Research Device", color="#1f78b4")
bars2 = ax.bar(x + width/2, consumer, width, label="Smartwatch", color="#ff7f0e")
ax.set_ylabel("Value")
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=20, ha="right")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2)
ax.grid(axis="y", linestyle="--", alpha=0.25)

for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0f}" if height >= 10 else f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8
        )

for x_pos, value in zip(x, gap):
    ax.annotate(
        f"Gap: {value:.1f}" if abs(value) < 100 else f"Gap: {value:.0f}",
        xy=(x_pos, max_value + 20),
        ha="center",
        fontsize=8,
        color="#444444"
    )

fig.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
print(f"Device comparison chart with gap saved to {OUTPUT_PATH}")
