import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUTPUT_PATH = Path("results/advanced_figures/stress_response_autonomic_diagram.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

box_specs = {
    "stressor": dict(text="Stress Stimulus\n(mental, physical, emotional)", pos=(0.12, 0.75)),
    "cns": dict(text="Central Processing\n(Prefrontal Cortex, Amygdala,\nHypothalamus)", pos=(0.38, 0.75)),
    "ans": dict(text="Autonomic Nervous System\nIntegration", pos=(0.64, 0.75)),
    "symp": dict(text="Sympathetic Branch\n• Adrenal medulla activation\n• Catecholamine release", pos=(0.43, 0.45)),
    "parasymp": dict(text="Parasympathetic Branch\n• Vagal withdrawal\n• HRV reduction", pos=(0.7, 0.45)),
    "symp_effects": dict(text="Physiological Effects\n• ↑ Heart rate\n• ↑ Sweat gland activity\n• ↑ Skin conductance\n• Peripheral vasoconstriction", pos=(0.34, 0.18)),
    "parasymp_effects": dict(text="Physiological Effects\n• ↓ Digestive activity\n• ↓ Restorative processes\n• ↓ Baseline HRV", pos=(0.69, 0.18)),
}

box_width, box_height = 0.22, 0.16
bbox_style = dict(boxstyle="round,pad=0.2", linewidth=1.5, edgecolor="#1f3b73", facecolor="#ecf2ff")

boxes = {}
centers = {}
for key, spec in box_specs.items():
    x, y = spec["pos"]
    box = FancyBboxPatch((x, y), box_width, box_height, **bbox_style)
    ax.add_patch(box)
    ax.text(x + box_width / 2, y + box_height / 2, spec["text"], ha="center", va="center", fontsize=10, color="#1a2a4a")
    centers[key] = (x + box_width / 2, y + box_height / 2)
    boxes[key] = box

def add_arrow(src_key, dst_key):
    sx, sy = centers[src_key]
    dx, dy = centers[dst_key]
    dx_vec, dy_vec = dx - sx, dy - sy
    dist = (dx_vec ** 2 + dy_vec ** 2) ** 0.5
    if dist == 0:
        return
    unit_x, unit_y = dx_vec / dist, dy_vec / dist
    src_box = boxes[src_key]
    dst_box = boxes[dst_key]
    start_x = sx + unit_x * (box_width / 2)
    start_y = sy + unit_y * (box_height / 2)
    end_x = dx - unit_x * (box_width / 2)
    end_y = dy - unit_y * (box_height / 2)
    arrow = FancyArrowPatch(
        (start_x, start_y),
        (end_x, end_y),
        arrowstyle="-|>",
        linewidth=1.5,
        color="#1f3b73",
        connectionstyle="arc3,rad=0.0"
    )
    ax.add_patch(arrow)

flow_pairs = [
    ("stressor", "cns"),
    ("cns", "ans"),
    ("ans", "symp"),
    ("ans", "parasymp"),
    ("symp", "symp_effects"),
    ("parasymp", "parasymp_effects"),
]

for src, dst in flow_pairs:
    add_arrow(src, dst)

ax.text(0.07, 0.92, "Stress Response Pathway", fontsize=14, color="#1f3b73", fontweight="bold")
ax.text(0.07, 0.88, "Autonomic Nervous System Activation", fontsize=11, color="#2c3e6f")

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=300)
print(f"Stress response diagram saved to {OUTPUT_PATH}")
