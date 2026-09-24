"""Nadeau-Bengio corrected 95% CIs for the modality comparison quoted in the paper (Section V-A).

Reads repeated_subject_splits.csv (mean and SD over 20 subject-grouped 80/20 splits) and writes
modality_nb_ci.csv: mean +/- t(0.975, n-1) * SD * sqrt(1/n + 0.2/0.8).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TABLES = Path(__file__).resolve().parents[1] / "outputs" / "tables" / "jbhi_v2"
RATIO = 0.2 / 0.8

d = pd.read_csv(TABLES / "repeated_subject_splits.csv")
d = d[d.task == "shared_Baseline_vs_Stress_pooled"].copy()
half = stats.t.ppf(0.975, d.n - 1) * d.balanced_accuracy_sd * np.sqrt(1 / d.n + RATIO)
d["nb_ci95_low"], d["nb_ci95_high"] = d.balanced_accuracy_mean - half, d.balanced_accuracy_mean + half
out = d[["task", "normalisation", "modality", "n", "balanced_accuracy_mean", "nb_ci95_low", "nb_ci95_high"]].round(4)
out.to_csv(TABLES / "modality_nb_ci.csv", index=False)
print(out.to_string(index=False))
