import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUTPUT_PATH = Path("results/advanced_figures/stress_pipeline_flowchart.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

steps = [
    ("Data Collection", "Raw signals (PPG, EDA, ACC, TEMP)"),
    ("Preprocessing", "Artifact removal, filtering"),
    ("Feature Extraction", "29 HRV / EDA / ACC features"),
    ("Model Training", "Random Forest, XGBoost"),
    ("Validation", "Leave-One-Subject-Out (LOSO)"),
    ("Clinical Integration", "Stress Index mapping"),
]

fig, ax = plt.subplots(figsize=(11, 4))
ax.axis("off")

box_width = 1.6
box_height = 0.8
x_start = 0.15
x_gap = 0.25

for idx, (title, desc) in enumerate(steps):
    x = x_start + idx * (box_width + x_gap)
    y = 0.5

    box = FancyBboxPatch(
        (x, y),
        box_width,
        box_height,
        boxstyle="round,pad=0.12",
        linewidth=1.5,
        edgecolor="#1f3b73",
        facecolor="#e6eefc"
    )
    ax.add_patch(box)
    ax.text(
        x + box_width / 2,
        y + box_height * 0.6,
        title,
        ha="center",
        va="center",
        fontsize=10,
        color="#1f3b73",
        fontweight="bold"
    )
    ax.text(
        x + box_width / 2,
        y + box_height * 0.23,
        desc,
        ha="center",
        va="center",
        fontsize=8.6,
        color="#2b2b2b",
        wrap=True
    )

    if idx < len(steps) - 1:
        arrow_start = (x + box_width, y + box_height / 2)
        arrow_end = (x + box_width + x_gap, y + box_height / 2)
        ax.annotate(
            "",
            xy=arrow_end,
            xytext=arrow_start,
            arrowprops=dict(arrowstyle="-|>", linewidth=1.5, color="#1f3b73")
        )

ax.set_xlim(0, x_start + len(steps) * (box_width + x_gap))
ax.set_ylim(0, 2)
ax.set_title(
    "Stress Detection Pipeline",
    fontsize=14,
    color="#1f3b73",
    pad=20
)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=300)
print(f"Flowchart saved to {OUTPUT_PATH}")
