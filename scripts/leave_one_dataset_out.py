#!/usr/bin/env python3
"""Leave-one-dataset-out stress detection across every dataset that has stress and non-stress states.

Task: Stress vs all non-stress states (Baseline, Rest, Meditation, Amusement), per-subject z-scored
features, XGBoost. For 20 held-out subject sets of each target dataset, the SAME test subjects are
scored by models trained on (a) other subjects of the target dataset, (b) all OTHER datasets only,
(c) everything. (a) is the within-dataset ceiling, (b) is true external validation.

Two feature sets: all physiology, and heart rate/HRV + EDA only (UBFC-Phys released no temperature,
and temperature mostly tracks time in session).

Options for the novelty-search checks (defaults reproduce the committed tables):
  --normalisation  session (whole stress-protocol recording, transductive), raw (none), baseline (each
                   subject's Baseline windows only), causal (strictly earlier windows of the same recording;
                   the first CAUSAL_MIN windows share the statistics of that opening calibration period)
  --exercise-negatives  keep PhysioNet Aerobic/Anaerobic windows as non-stress, scaled with the subject's
                   stress-protocol statistics; also reports how often exercise windows are called stress
  --datasets       restrict to these datasets (e.g. WESAD Stress-Predict for the Kwon et al. 2026 setting)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from probe_normalisation import model
from run_jbhi_experiments import META, REPO_ROOT, grouped_split

NON_STRESS = ["Baseline", "Rest", "Meditation", "Amusement"]
EXERCISE = ["Aerobic", "Anaerobic"]
CAUSAL_MIN = 4  # 4 windows of 60 s at a 30 s step = the first 2.5 min of a recording


def fit_predict(x_train, y_train, x_test, n_jobs: int) -> np.ndarray:
    weight = np.where(y_train == 1, (y_train == 0).sum() / max(1, (y_train == 1).sum()), 1.0)
    return model().set_params(n_jobs=n_jobs).fit(x_train, y_train, sample_weight=weight).predict_proba(x_test)[:, 1]


def normalise(df: pd.DataFrame, cols: list[str], how: str) -> pd.DataFrame:
    """Per-subject scaling. Exercise sessions never contribute statistics; they borrow the protocol's."""
    if how == "raw":
        return df[cols]
    protocol = ~df["harmonized_label"].isin(EXERCISE)
    if how in ("session", "baseline"):
        ref = df[protocol & (df["harmonized_label"].eq("Baseline") if how == "baseline" else True)]
        stats = ref.groupby("subject_uid")[cols].agg(["mean", "std"])
        mean = stats.xs("mean", axis=1, level=1).reindex(df["subject_uid"]).to_numpy()
        std = stats.xs("std", axis=1, level=1).reindex(df["subject_uid"]).replace(0, 1).to_numpy()
        return pd.DataFrame((df[cols].to_numpy() - mean) / std, index=df.index, columns=cols)
    out = pd.DataFrame(np.nan, index=df.index, columns=cols)
    for _, g in df[protocol].sort_values("timestamp_start").groupby("subject_uid"):
        x, start, end = g[cols].to_numpy(), g["timestamp_start"].to_numpy(), g["timestamp_end"].to_numpy()
        for i in range(len(g)):
            earlier = np.flatnonzero(end <= start[i])
            ref = x[earlier] if len(earlier) >= CAUSAL_MIN else x[:CAUSAL_MIN]
            sd = np.nanstd(ref, axis=0, ddof=1)
            out.loc[g.index[i]] = (x[i] - np.nanmean(ref, axis=0)) / np.where((sd > 0) & np.isfinite(sd), sd, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "combined" / "harmonized_windows_v2.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "tables" / "jbhi_v2")
    parser.add_argument("--normalisation", choices=["session", "raw", "baseline", "causal"], default="session")
    parser.add_argument("--exercise-negatives", action="store_true")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--tag", default="", help="suffix for the output file names")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    if args.exercise_negatives and args.normalisation == "causal":
        parser.error("exercise sessions have no earlier protocol windows; use session, baseline or raw")
    non_stress = NON_STRESS + (EXERCISE if args.exercise_negatives else [])

    df = pd.read_csv(args.input)
    physiology = [c for c in df.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
    feature_sets = {"physiology": physiology, "hr_hrv_eda": [c for c in physiology if not c.startswith("temp_")]}
    # Per-subject scaling uses every window of that person's stress-protocol recording (label-free), including
    # states that are not evaluated; separate exercise sessions are left out so they do not shift the reference.
    if not args.exercise_negatives:
        df = df[~df["harmonized_label"].isin(EXERCISE)]
    df[physiology] = normalise(df, physiology, args.normalisation)
    df = df[df["harmonized_label"].isin(non_stress + ["Stress"])]
    datasets = [d for d, g in df.groupby("dataset") if (g["harmonized_label"] == "Stress").any()]
    if args.datasets:
        datasets = [d for d in datasets if d in args.datasets]
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
                exercise = df["harmonized_label"].iloc[target_idx[test]].isin(EXERCISE).to_numpy()
                for name, idx in (("within_dataset", target_idx[train]), ("other_datasets_only", other_idx),
                                  ("all_datasets", np.concatenate([target_idx[train], other_idx]))):
                    p = fit_predict(x[idx], y[idx], x[target_idx[test]], args.n_jobs)
                    row = {"features": fname, "test_dataset": target, "trained_on": name, "split": seed,
                           "balanced_accuracy": balanced_accuracy_score(truth, p > 0.5), "auroc": roc_auc_score(truth, p)}
                    if args.exercise_negatives:
                        row["exercise_called_stress"] = (p[exercise] > 0.5).mean() if exercise.any() else np.nan
                    rows.append(row)
    per_split = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_split.to_csv(args.output_dir / f"leave_one_dataset_out_per_split{args.tag}.csv", index=False)
    metrics = [c for c in ("balanced_accuracy", "auroc", "exercise_called_stress") if c in per_split]
    summary = per_split.groupby(["features", "test_dataset", "trained_on"], sort=False)[metrics].mean().round(3)
    summary.reset_index().to_csv(args.output_dir / f"leave_one_dataset_out_summary{args.tag}.csv", index=False)
    for metric in metrics:
        print(f"\n===== {metric} =====")
        print(summary[metric].unstack("trained_on")[["within_dataset", "other_datasets_only", "all_datasets"]].to_string())


if __name__ == "__main__":
    main()
