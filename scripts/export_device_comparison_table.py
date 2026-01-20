from pathlib import Path

OUTPUT_PATH = Path("results/advanced_figures/device_comparison_table.md")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

table_lines = [
    "| Metric | Research Device | Consumer Smartwatch | Gap |",
    "| --- | --- | --- | --- |",
    "| Max Accuracy (3-class) | 95.23% | 88.60% | 6.63% |",
    "| Data Quality (20% threshold) | 88% | 63% | 25% |",
    "| Available Sensors | 42 (stress relevant) | 2 | 40 |",
    "| Battery Life (hours) | 48 | 44 | 4 |",
    "| Approx. Cost (USD) | $1,690 | $275 | $1,415 |",
]

OUTPUT_PATH.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
print(f"Device comparison table saved to {OUTPUT_PATH}")
