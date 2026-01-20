# HRV Features Table (Markdown)

hrv_features = [
    ["Mean RR", "Time", "Mean of RR intervals (ms)"],
    ["SDNN", "Time", "Standard deviation of RR intervals (ms)"],
    ["RMSSD", "Time", "Root mean square of successive RR differences"],
    ["pNN50", "Time", "% of RR intervals differing by >50 ms"],
    ["SDSD", "Time", "Standard deviation of successive RR differences"],
    ["HRV Triangular Index", "Time", "Total RR count / height of RR histogram"],
    ["TINN", "Time", "Baseline width of RR histogram"],
    ["LF Power", "Frequency", "Power in low-frequency band (0.04–0.15 Hz)"],
    ["HF Power", "Frequency", "Power in high-frequency band (0.15–0.4 Hz)"],
    ["VLF Power", "Frequency", "Power in very low-frequency band (0–0.04 Hz)"],
    ["LF/HF Ratio", "Frequency", "Ratio of LF to HF power"],
    ["pLF", "Frequency", "Proportion of LF power"],
    ["pHF", "Frequency", "Proportion of HF power"],
]

with open("results/advanced_figures/hrv_features_table.md", "w") as f:
    f.write("| Feature Name         | Domain         | Description                                      |\n")
    f.write("|----------------------|---------------|--------------------------------------------------|\n")
    for row in hrv_features:
        f.write(f"| {row[0]:<20} | {row[1]:<13} | {row[2]} |\n")
print("HRV features table saved to results/advanced_figures/hrv_features_table.md")
