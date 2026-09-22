#!/usr/bin/env python3
"""Does detection fail on mild stress? Pilot for the mild-stress question.

Stress windows are graded by the participant's own rating rise over their Baseline rating
(self_report_stress_delta, from self_report_labels.py): PhysioNet 1-10 per task, WESAD PANAS
"Stressed" 1-5 for the TSST. No other dataset has per-task ratings of the same people, so only
these two are graded; the others are still used for training.

Every window of the graded dataset gets an out-of-sample stress probability from
  within    leave-one-subject-out inside that dataset
  external  one model trained on all other datasets
(same task, features and per-subject z-scoring as leave_one_dataset_out.py, HR/HRV/EDA features).
Stress windows are then grouped by subject and task (one unit = one rated task of one person), and
recall per unit is related to the rating rise: Spearman rho with a subject-level bootstrap CI, and
recall by rise band. The Baseline/Rest false-stress rate is given as the reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from leave_one_dataset_out import EXERCISE, NON_STRESS, fit_predict, normalise
from run_jbhi_experiments import META, REPO_ROOT

GRADED = ["PhysioNet", "WESAD"]
BANDS = {"PhysioNet": [-np.inf, 1, 3, np.inf], "WESAD": [-np.inf, 1, 2, np.inf]}  # rise <=1 / mid / top
BAND_NAMES = ["mild", "moderate", "strong"]


def bootstrap_rho(units: pd.DataFrame, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI for Spearman rho, resampling subjects (a PhysioNet subject can contribute two tasks)."""
    rng, subjects, rhos = np.random.default_rng(seed), units["subject_uid"].unique(), []
    by_subject = {s: g for s, g in units.groupby("subject_uid")}
    for _ in range(n):
        sample = pd.concat([by_subject[s] for s in rng.choice(subjects, len(subjects))])
        if sample["rise"].nunique() > 1 and sample["recall"].nunique() > 1:
            rhos.append(spearmanr(sample["rise"], sample["recall"]).statistic)
    return tuple(np.percentile(rhos, [2.5, 97.5]).round(3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "combined" / "harmonized_windows_v2.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "outputs" / "tables" / "jbhi_v2" / "mild_stress")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df = df[~df["harmonized_label"].isin(EXERCISE)]
    cols = [c for c in df.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_"))]
    df[cols] = normalise(df, cols, "session")
    df = df[df["harmonized_label"].isin(NON_STRESS + ["Stress"])]
    df = df[df["dataset"].isin(df.loc[df["harmonized_label"] == "Stress", "dataset"].unique())].reset_index(drop=True)
    x, y = df[cols].to_numpy(), (df["harmonized_label"] == "Stress").to_numpy(int)

    preds = []
    for target in GRADED:
        t_idx, o_idx = np.flatnonzero(df["dataset"] == target), np.flatnonzero(df["dataset"] != target)
        p_ext = fit_predict(x[o_idx], y[o_idx], x[t_idx], args.n_jobs)
        p_in = np.empty(len(t_idx))
        subjects = df["subject_uid"].to_numpy()[t_idx]
        for s in np.unique(subjects):
            test = subjects == s
            p_in[test] = fit_predict(x[t_idx[~test]], y[t_idx[~test]], x[t_idx[test]], args.n_jobs)
        preds.append(df.iloc[t_idx][["dataset", "subject_uid", "original_label", "harmonized_label",
                                     "self_report_stress", "self_report_stress_delta"]]
                     .assign(p_within=p_in, p_external=p_ext))
    preds = pd.concat(preds, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output_dir / "mild_stress_window_predictions.csv", index=False)

    summary, bands = [], []
    for target, g in preds.groupby("dataset"):
        stress, rest = g[g["harmonized_label"] == "Stress"], g[g["harmonized_label"] != "Stress"]
        for source in ("within", "external"):
            hit = stress[f"p_{source}"] > 0.5
            units = (stress.assign(hit=hit).groupby(["subject_uid", "original_label"])
                     .agg(rise=("self_report_stress_delta", "first"), recall=("hit", "mean"),
                          mean_p=(f"p_{source}", "mean"), windows=("hit", "size")).reset_index())
            rho = spearmanr(units["rise"], units["recall"]).statistic
            units["band"] = pd.cut(units["rise"], BANDS[target], labels=BAND_NAMES)
            summary.append({"dataset": target, "trained_on": source, "units": len(units),
                            "subjects": units["subject_uid"].nunique(),
                            "rho_rise_recall": round(rho, 3), "rho_ci95": bootstrap_rho(units),
                            "rho_rise_mean_p": round(spearmanr(units["rise"], units["mean_p"]).statistic, 3),
                            "unit_recall": round(units["recall"].mean(), 3),
                            "nonstress_false_stress": round((rest[f"p_{source}"] > 0.5).mean(), 3)})
            band = units.groupby("band", observed=False).agg(units=("recall", "size"), recall=("recall", "mean"),
                                                            mean_p=("mean_p", "mean")).round(3).reset_index()
            bands.append(band.assign(dataset=target, trained_on=source))
    summary, bands = pd.DataFrame(summary), pd.concat(bands, ignore_index=True)
    summary.to_csv(args.output_dir / "mild_stress_summary.csv", index=False)
    bands.to_csv(args.output_dir / "mild_stress_by_band.csv", index=False)
    print(summary.to_string(index=False))
    print()
    print(bands[["dataset", "trained_on", "band", "units", "recall", "mean_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
