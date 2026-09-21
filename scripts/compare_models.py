#!/usr/bin/env python3
"""Paired significance tests on the tuned baselines.

Repeated subject-grouped splits reuse the same subjects, so split scores are not independent and a
plain paired t-test or Wilcoxon test is anti-conservative. We use the corrected resampled t-test of
Nadeau & Bengio (2003): variance is inflated by (1/n + n_test/n_train). P-values are Holm-corrected
within each family of comparisons. Cross-dataset rows are excluded: they have one fixed split.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_TRAIN_RATIO = 0.2 / 0.8  # subjects held out / subjects trained on, per outer split
METRIC = "balanced_accuracy"


def corrected_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    d = a - b
    n = len(d)
    if np.allclose(d.var(ddof=1), 0):
        return float(d.mean()), 0.0, 1.0
    t = d.mean() / np.sqrt((1 / n + TEST_TRAIN_RATIO) * d.var(ddof=1))
    return float(d.mean()), float(t), float(2 * stats.t.sf(abs(t), n - 1))


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    adjusted = (p[order] * np.arange(len(p), 0, -1)).cummax().clip(upper=1.0)
    return adjusted.reindex(p.index)


def compare(scores: pd.DataFrame, vary: str, fixed: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in scores.groupby(fixed, sort=False):
        wide = group.pivot(index="split", columns=vary, values=METRIC).dropna()
        for left, right in combinations(wide.columns, 2):
            diff, t, p = corrected_ttest(wide[left].to_numpy(), wide[right].to_numpy())
            rows.append({**dict(zip(fixed, key)), "a": left, "b": right, "mean_a": wide[left].mean(),
                         "mean_b": wide[right].mean(), "mean_diff_a_minus_b": diff, "t": t, "p": p, "n_splits": len(wide)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    out["significant_0.05"] = out["p_holm"] < 0.05
    return out.round(4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "outputs" / "tables" / "jbhi_v2")
    args = parser.parse_args()
    scores = pd.read_csv(args.dir / "tuned_baselines_per_split.csv")
    scores = scores[~scores["task"].str.startswith("cross_dataset_")]

    # Self-check: identical scores must give p = 1, a constant offset with tiny noise must be significant.
    rng = np.random.default_rng(0)
    base = rng.normal(0.8, 0.05, 20)
    assert corrected_ttest(base, base)[2] == 1.0 and corrected_ttest(base + 0.1 + rng.normal(0, 0.005, 20), base)[2] < 0.001

    families = {"models": ("model", ["task", "normalisation", "modality"]),
                "modalities": ("modality", ["task", "normalisation", "model"]),
                "normalisation": ("normalisation", ["task", "modality", "model"])}
    for name, (vary, fixed) in families.items():
        table = compare(scores, vary, fixed)
        table.to_csv(args.dir / f"significance_{name}.csv", index=False)
        print(f"\n=== {name}: {int(table['significant_0.05'].sum())} of {len(table)} comparisons significant after Holm ===")
        print(table.sort_values("p").head(8).drop(columns=["t", "n_splits"]).to_string(index=False))


if __name__ == "__main__":
    main()
