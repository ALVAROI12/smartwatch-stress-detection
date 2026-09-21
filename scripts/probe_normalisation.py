#!/usr/bin/env python3
"""Which personal normalisation helps stress detection transfer, and is the gain real?

Two findings this script reproduces (physiology features only, XGBoost, WESAD + PhysioNet):

1. Order confound. Baseline is always recorded first, so "change from personal baseline" also encodes
   "later in the session". A Baseline-vs-Stress model built on baseline-referenced features calls
   post-stressor rest/meditation/amusement windows "stress" far more often than a raw-feature model.
2. On the unconfounded task, Stress vs ALL non-stress states, per-subject z-scoring over the whole
   recording is the most reliable normalisation; baseline- and trailing-window referencing are not.

Writes order_confound_check.csv and normalisation_probe.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from run_jbhi_experiments import META, REPO_ROOT, grouped_split

LATE_NON_STRESS = ["Rest", "Meditation", "Amusement"]  # occur after a stressor in both protocols
CALIBRATION_WINDOWS, BUFFER_WINDOWS = 2, 3  # 90 s of baseline as reference; the next window overlaps it, so drop it too
PAIRS = (("WESAD", "PhysioNet"), ("PhysioNet", "WESAD"))


def model() -> XGBClassifier:
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9, n_jobs=8, verbosity=0)


def fit_predict(x_train, y_train, x_test) -> np.ndarray:
    weight = np.where(y_train == 1, (y_train == 0).sum() / max(1, (y_train == 1).sum()), 1.0)
    return model().fit(x_train, y_train, sample_weight=weight).predict_proba(x_test)[:, 1]


def trailing_delta(df: pd.DataFrame, cols: list[str], minutes: int) -> pd.DataFrame:
    """Window minus the median of the same recording's strictly earlier, non-overlapping windows."""
    parts = []
    for _, g in df.groupby(["subject_uid", "subject_id"]):
        t, x = g["timestamp_start"].to_numpy(), g[cols].to_numpy()
        out = np.full_like(x, np.nan)
        for i in range(len(g)):
            earlier = (t < t[i] - 30) & (t >= t[i] - 60 * minutes)
            if earlier.sum() >= 3:
                out[i] = x[i] - np.nanmedian(x[earlier], axis=0)
        parts.append(pd.DataFrame(out, index=g.index, columns=cols))
    return pd.concat(parts).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "combined" / "harmonized_windows_v2.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "tables" / "jbhi_v2")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    cols = [c for c in df.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
    df = df[df["dataset"].isin(["WESAD", "PhysioNet"]) & ~df["harmonized_label"].isin(["Aerobic", "Anaerobic"])]
    df = df.sort_values(["subject_uid", "subject_id", "timestamp_start"]).reset_index(drop=True)
    label, y = df["harmonized_label"], (df["harmonized_label"] == "Stress").to_numpy(int)

    baseline = df[label == "Baseline"]
    calibration = baseline.groupby("subject_uid").head(CALIBRATION_WINDOWS).index
    excluded = baseline.groupby("subject_uid").head(BUFFER_WINDOWS).index
    reference = df.loc[calibration].groupby("subject_uid")[cols].mean()
    grouped = df.groupby("subject_uid")[cols]
    frames = {"raw": df[cols],
              "whole_recording_zscore": (df[cols] - grouped.transform("mean")) / grouped.transform("std").replace(0, 1),
              "personal_baseline_delta": df[cols] - reference.reindex(df["subject_uid"]).to_numpy(),
              "trailing_10min_delta": trailing_delta(df, cols, 10)}

    # 1. Order confound: train Baseline vs Stress on the source, then score every target state.
    usable = ~df.index.isin(excluded)
    rows = []
    for name in ("raw", "personal_baseline_delta"):
        x = frames[name].to_numpy()
        for source, target in PAIRS:
            train = usable & (df["dataset"] == source).to_numpy() & label.isin(["Baseline", "Stress"]).to_numpy()
            test = usable & (df["dataset"] == target).to_numpy()
            p, state = fit_predict(x[train], y[train], x[test]), label[test]
            vs_base, vs_late = state.isin(["Baseline", "Stress"]).to_numpy(), state.isin(LATE_NON_STRESS + ["Stress"]).to_numpy()
            rows.append({"features": name, "setting": f"{source}->{target}",
                         "auroc_stress_vs_baseline": roc_auc_score(state[vs_base] == "Stress", p[vs_base]),
                         "auroc_stress_vs_later_non_stress": roc_auc_score(state[vs_late] == "Stress", p[vs_late]),
                         "pct_later_non_stress_called_stress": 100 * (p[state.isin(LATE_NON_STRESS).to_numpy()] > 0.5).mean(),
                         "pct_baseline_called_stress": 100 * (p[(state == "Baseline").to_numpy()] > 0.5).mean()})
    confound = pd.DataFrame(rows).round(3)
    confound.to_csv(args.output_dir / "order_confound_check.csv", index=False)
    print(confound.to_string(index=False))

    # 2. Unconfounded task: Stress vs ALL non-stress, identical windows for every normalisation.
    valid = usable & frames["trailing_10min_delta"].notna().any(axis=1).to_numpy()
    data, target_y, rows = df[valid], y[valid], []
    for name, frame in frames.items():
        x = frame[valid].to_numpy()
        for source, target in PAIRS:
            s, t = (data["dataset"] == source).to_numpy(), (data["dataset"] == target).to_numpy()
            p = fit_predict(x[s], target_y[s], x[t])
            rows.append({"features": name, "setting": f"{source}->{target}",
                         "balanced_accuracy": balanced_accuracy_score(target_y[t], p > 0.5), "auroc": roc_auc_score(target_y[t], p)})
        scores = []
        for seed in range(20):
            train, test = grouped_split(data, seed)
            p = fit_predict(x[train], target_y[train], x[test])
            scores.append((balanced_accuracy_score(target_y[test], p > 0.5), roc_auc_score(target_y[test], p)))
        rows.append({"features": name, "setting": "pooled_unseen_subjects",
                     "balanced_accuracy": np.mean([s[0] for s in scores]), "auroc": np.mean([s[1] for s in scores])})
    probe = pd.DataFrame(rows).round(3)
    probe.to_csv(args.output_dir / "normalisation_probe.csv", index=False)
    print(f"\nStress vs all non-stress: {valid.sum()} windows, {int(target_y.sum())} stress")
    print(probe.pivot_table(index="features", columns="setting", values=["balanced_accuracy", "auroc"], sort=False).to_string())


if __name__ == "__main__":
    main()
