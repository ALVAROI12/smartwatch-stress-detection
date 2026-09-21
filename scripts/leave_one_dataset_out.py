#!/usr/bin/env python3
"""Leave-one-dataset-out stress detection across every dataset that has stress and non-stress states.

Task: Stress vs all non-stress states (Baseline, Rest, Meditation, Amusement), per-subject z-scored
features, XGBoost. For 20 held-out subject sets of each target dataset, the SAME test subjects are
scored by models trained on (a) other subjects of the target dataset, (b) all OTHER datasets only,
(c) everything. (a) is the within-dataset ceiling, (b) is true external validation.

Two feature sets: all physiology, and heart rate/HRV + EDA only (UBFC-Phys released no temperature,
and temperature mostly tracks time in session).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from probe_normalisation import fit_predict
from run_jbhi_experiments import META, REPO_ROOT, grouped_split

NON_STRESS = ["Baseline", "Rest", "Meditation", "Amusement"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "combined" / "harmonized_windows_v2.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "tables" / "jbhi_v2")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    physiology = [c for c in df.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
    feature_sets = {"physiology": physiology, "hr_hrv_eda": [c for c in physiology if not c.startswith("temp_")]}
    # Per-subject scaling uses every window of that person's stress-protocol recording (label-free), including
    # states that are not evaluated; separate exercise sessions are left out so they do not shift the reference.
    df = df[~df["harmonized_label"].isin(["Aerobic", "Anaerobic"])]
    grouped = df.groupby("subject_uid")[physiology]
    df[physiology] = (df[physiology] - grouped.transform("mean")) / grouped.transform("std").replace(0, 1)
    df = df[df["harmonized_label"].isin(NON_STRESS + ["Stress"])]
    datasets = [d for d, g in df.groupby("dataset") if (g["harmonized_label"] == "Stress").any()]
    df = df[df["dataset"].isin(datasets)].reset_index(drop=True)
    z = df[physiology]
    y = (df["harmonized_label"] == "Stress").to_numpy(int)
    print(df.groupby("dataset").agg(subjects=("subject_uid", "nunique"), windows=("label", "size"),
                                    stress_windows=("harmonized_label", lambda s: int((s == "Stress").sum()))).to_string())

    rows = []
    for fname, cols in feature_sets.items():
        x = z[cols].to_numpy()
        for target in datasets:
            target_idx, other_idx = np.flatnonzero(df["dataset"] == target), np.flatnonzero(df["dataset"] != target)
            for seed in range(20):
                train, test = grouped_split(df.iloc[target_idx], seed)
                truth = y[target_idx[test]]
                if len(set(truth)) < 2:
                    continue
                for name, idx in (("within_dataset", target_idx[train]), ("other_datasets_only", other_idx),
                                  ("all_datasets", np.concatenate([target_idx[train], other_idx]))):
                    p = fit_predict(x[idx], y[idx], x[target_idx[test]])
                    rows.append({"features": fname, "test_dataset": target, "trained_on": name, "split": seed,
                                 "balanced_accuracy": balanced_accuracy_score(truth, p > 0.5), "auroc": roc_auc_score(truth, p)})
    per_split = pd.DataFrame(rows)
    per_split.to_csv(args.output_dir / "leave_one_dataset_out_per_split.csv", index=False)
    summary = per_split.groupby(["features", "test_dataset", "trained_on"], sort=False)[["balanced_accuracy", "auroc"]].mean().round(3)
    summary.reset_index().to_csv(args.output_dir / "leave_one_dataset_out_summary.csv", index=False)
    for metric in ("balanced_accuracy", "auroc"):
        print(f"\n===== {metric} =====")
        print(summary[metric].unstack("trained_on")[["within_dataset", "other_datasets_only", "all_datasets"]].to_string())


if __name__ == "__main__":
    main()
