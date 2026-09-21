#!/usr/bin/env python3
"""Is WESAD -> PhysioNet a transfer failure, or is PhysioNet just harder?

Task: Stress vs all non-stress states, physiology features, per-subject z-scoring, XGBoost.
For 20 held-out subject sets of each target dataset, the SAME test subjects are scored by models
trained on (a) other subjects of the target dataset, (b) the other dataset only, (c) both.
(a) is the within-dataset ceiling; (a) minus (b) is the true cost of transfer.

Also reports, for the WESAD-trained model, which PhysioNet stages it calls "stress", and the stress
effect size per feature in each dataset, which shows why the datasets differ: WESAD's standing speech
task drives EDA level, PhysioNet's seated mental arithmetic drives heart rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from probe_normalisation import fit_predict
from run_jbhi_experiments import META, REPO_ROOT, grouped_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "combined" / "harmonized_windows_v2.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "tables" / "jbhi_v2")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    cols = [c for c in df.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
    df = df[df["dataset"].isin(["WESAD", "PhysioNet"]) & ~df["harmonized_label"].isin(["Aerobic", "Anaerobic"])].reset_index(drop=True)
    grouped = df.groupby("subject_uid")[cols]
    zframe = (df[cols] - grouped.transform("mean")) / grouped.transform("std").replace(0, 1)
    z, y = zframe.to_numpy(), (df["harmonized_label"] == "Stress").to_numpy(int)

    rows = []
    for target, other in (("PhysioNet", "WESAD"), ("WESAD", "PhysioNet")):
        target_idx, other_idx = np.flatnonzero(df["dataset"] == target), np.flatnonzero(df["dataset"] == other)
        for seed in range(20):
            train, test = grouped_split(df.iloc[target_idx], seed)
            for name, train_idx in ((f"{target} (other subjects)", target_idx[train]), (f"{other} only", other_idx),
                                    ("both datasets", np.concatenate([target_idx[train], other_idx]))):
                p = fit_predict(z[train_idx], y[train_idx], z[target_idx[test]])
                truth = y[target_idx[test]]
                rows.append({"test_dataset": target, "trained_on": name, "split": seed,
                             "balanced_accuracy": balanced_accuracy_score(truth, p > 0.5), "auroc": roc_auc_score(truth, p)})
    per_split = pd.DataFrame(rows)
    per_split.to_csv(args.output_dir / "transfer_gap_per_split.csv", index=False)
    summary = per_split.groupby(["test_dataset", "trained_on"], sort=False)[["balanced_accuracy", "auroc"]].agg(["mean", "std"]).round(3)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.reset_index().to_csv(args.output_dir / "transfer_gap_summary.csv", index=False)
    print(summary.to_string())

    wesad, physionet = (df["dataset"] == "WESAD").to_numpy(), (df["dataset"] == "PhysioNet").to_numpy()
    scored = df[physionet].assign(called_stress=fit_predict(z[wesad], y[wesad], z[physionet]) > 0.5)
    stages = scored.groupby("original_label").agg(windows=("called_stress", "size"),
                                                  pct_called_stress=("called_stress", lambda s: 100 * s.mean()),
                                                  self_report_rise=("self_report_stress_delta", "mean")).round(2)
    stages.sort_values("pct_called_stress", ascending=False).to_csv(args.output_dir / "wesad_model_on_physionet_stages.csv")

    means = zframe.assign(y=y, dataset=df["dataset"], subject=df["subject_uid"]).groupby(["dataset", "subject", "y"])[cols].mean().unstack("y")
    effect = (means.xs(1, axis=1, level=1) - means.xs(0, axis=1, level=1)).groupby("dataset").median().T.round(2)
    effect.to_csv(args.output_dir / "stress_effect_size_by_dataset.csv")
    print("\nstress effect (z-units, median over subjects), largest WESAD effects first:")
    print(effect.sort_values("WESAD", key=abs, ascending=False).head(8).to_string())


if __name__ == "__main__":
    main()
