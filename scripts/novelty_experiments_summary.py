#!/usr/bin/env python3
"""Summarise the novelty-search checks run with leave_one_dataset_out.py (outputs/tables/jbhi_v2/novelty_experiments).

1. Normalisation: raw, baseline and causal scaling against whole-session z-scoring (the transductive default).
2. Exercise: PhysioNet Aerobic/Anaerobic windows kept as non-stress negatives.
3. Kwon et al. (2026) setting: WESAD + Stress-Predict only, before vs after the label fixes, with and without
   per-subject z-scoring. Stress-Predict lost S01 in the fixes, so its splits differ and only means are compared;
   WESAD test subjects are identical, so it gets a paired test.

Paired comparisons use the Nadeau-Bengio corrected t-test over the 20 subject splits, Holm-corrected per table.
"""

from __future__ import annotations

import pandas as pd

from compare_models import corrected_ttest, holm
from run_jbhi_experiments import REPO_ROOT

DIR = REPO_ROOT / "outputs" / "tables" / "jbhi_v2" / "novelty_experiments"
KEYS = ["features", "test_dataset", "trained_on"]


def load(tag: str) -> pd.DataFrame:
    return pd.read_csv(DIR / f"leave_one_dataset_out_per_split{tag}.csv")


def paired(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str, datasets=None) -> pd.DataFrame:
    both = a.merge(b, on=KEYS + ["split"], suffixes=("_a", "_b"))
    if datasets:
        both = both[both["test_dataset"].isin(datasets)]
    rows = []
    for key, g in both.groupby(KEYS, sort=False):
        for metric in ("balanced_accuracy", "auroc"):
            diff, t, p = corrected_ttest(g[f"{metric}_a"].to_numpy(), g[f"{metric}_b"].to_numpy())
            rows.append({**dict(zip(KEYS, key)), "metric": metric, name_a: g[f"{metric}_a"].mean(),
                         name_b: g[f"{metric}_b"].mean(), "diff": diff, "t": t, "p": p, "n_splits": len(g)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    return out.round(4)


def main() -> None:
    session = load("_session_check")
    norm = pd.concat([paired(load(f"_{v}"), session, "variant_score", "session").assign(variant=v)
                      for v in ("raw", "baseline", "causal")])
    norm.to_csv(DIR / "normalisation_vs_session.csv", index=False)
    external = norm[(norm["trained_on"] == "other_datasets_only") & (norm["metric"] == "balanced_accuracy")]
    print("External balanced accuracy, variant vs whole-session z-scoring:")
    print(external.pivot_table(index=["features", "test_dataset"], columns="variant", values=["variant_score", "p_holm"]).to_string())
    print(external.groupby(["features"])["session"].agg(["min", "max"]).rename(columns=lambda c: f"session_{c}").to_string())

    exercise = load("_session_exercise")
    summary = exercise.groupby(KEYS, sort=False)[["balanced_accuracy", "auroc", "exercise_called_stress"]].mean().round(3)
    ex = paired(exercise, session, "with_exercise", "without_exercise", datasets=["PhysioNet"])
    ex.to_csv(DIR / "exercise_negatives_physionet.csv", index=False)
    print("\nPhysioNet with exercise windows as negatives:")
    print(summary.loc[(slice(None), "PhysioNet", slice(None))].to_string())
    print(ex.to_string(index=False))

    rows = []
    for tag in ("fixed_session", "fixed_raw", "prefix_session", "prefix_raw"):
        s = load(f"_kwon_{tag}").groupby(KEYS, sort=False)[["balanced_accuracy", "auroc"]].mean()
        rows.append(s.assign(labels=tag.split("_")[0], scaling=tag.split("_")[1]).reset_index())
    kwon = pd.concat(rows).round(3)
    kwon.to_csv(DIR / "kwon_setting_summary.csv", index=False)
    wesad = pd.concat([paired(load(f"_kwon_fixed_{s}"), load(f"_kwon_prefix_{s}"), "after_fixes", "before_fixes",
                              datasets=["WESAD"]).assign(scaling=s) for s in ("session", "raw")])
    wesad.to_csv(DIR / "kwon_setting_wesad_label_fix_test.csv", index=False)
    print("\nKwon setting (WESAD + Stress-Predict), mean over 20 splits:")
    print(kwon.pivot_table(index=["features", "test_dataset", "trained_on"], columns=["labels", "scaling"],
                           values=["balanced_accuracy", "auroc"]).to_string())


if __name__ == "__main__":
    main()
