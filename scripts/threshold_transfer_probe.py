#!/usr/bin/env python3
"""Is the leave-one-dataset-out gap a ranking failure or a threshold shift?

Same task, scaling, splits and model as leave_one_dataset_out.py, external condition only (trained on
all OTHER datasets). For each held-out subject set, balanced accuracy is scored three ways:
  fixed_0.5     the probability cut used in the main table
  prevalence    label-free: cut the target scores at the quantile that reproduces the source stress rate
  oracle        best cut chosen with the target labels (upper bound; not deployable)
AUROC is threshold-free, so oracle minus fixed_0.5 is the part of the gap a better threshold could recover.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

from probe_normalisation import fit_predict
from run_jbhi_experiments import META, REPO_ROOT, grouped_split

NON_STRESS = ["Baseline", "Rest", "Meditation", "Amusement"]


def oracle_balanced_accuracy(truth: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(truth, p)
    return float(((tpr + 1 - fpr) / 2).max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "combined" / "harmonized_windows_v2.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "tables" / "jbhi_v2")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    physiology = [c for c in df.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
    feature_sets = {"physiology": physiology, "hr_hrv_eda": [c for c in physiology if not c.startswith("temp_")]}
    df = df[~df["harmonized_label"].isin(["Aerobic", "Anaerobic"])]
    grouped = df.groupby("subject_uid")[physiology]
    df[physiology] = (df[physiology] - grouped.transform("mean")) / grouped.transform("std").replace(0, 1)
    df = df[df["harmonized_label"].isin(NON_STRESS + ["Stress"])]
    datasets = [d for d, g in df.groupby("dataset") if (g["harmonized_label"] == "Stress").any()]
    df = df[df["dataset"].isin(datasets)].reset_index(drop=True)
    y = (df["harmonized_label"] == "Stress").to_numpy(int)

    rows = []
    for fname, cols in feature_sets.items():
        x = df[cols].to_numpy()
        for target in datasets:
            target_idx, other_idx = np.flatnonzero(df["dataset"] == target), np.flatnonzero(df["dataset"] != target)
            source_rate = y[other_idx].mean()
            for seed in range(20):
                _, test = grouped_split(df.iloc[target_idx], seed)
                truth = y[target_idx[test]]
                if len(set(truth)) < 2:
                    continue
                p = fit_predict(x[other_idx], y[other_idx], x[target_idx[test]])
                cut = np.quantile(p, 1 - source_rate)
                row = {"features": fname, "test_dataset": target, "split": seed,
                       "source_stress_rate": source_rate, "target_stress_rate": truth.mean(),
                       "fixed_0.5": balanced_accuracy_score(truth, p > 0.5),
                       "prevalence": balanced_accuracy_score(truth, p > cut),
                       "oracle": oracle_balanced_accuracy(truth, p), "auroc": roc_auc_score(truth, p),
                       "mean_score": p.mean()}
                assert row["oracle"] >= max(row["fixed_0.5"], row["prevalence"]) - 1e-9
                rows.append(row)

    per_split = pd.DataFrame(rows)
    per_split.to_csv(args.output_dir / "threshold_transfer_probe_per_split.csv", index=False)
    summary = per_split.groupby(["features", "test_dataset"], sort=False).mean(numeric_only=True).drop(columns="split").round(3)
    summary.reset_index().to_csv(args.output_dir / "threshold_transfer_probe_summary.csv", index=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
