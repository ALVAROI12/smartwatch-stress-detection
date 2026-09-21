#!/usr/bin/env python3
"""Did the label fixes (commit 6070b6b) change leave-one-dataset-out results beyond split noise?

Paired only where the held-out subject sets are identical before and after: WESAD and UBFC-Phys, whose own
windows did not change, so grouped_split draws the same 20 test sets. Their external scores moved only because the
OTHER datasets' labels were fixed. Stress-Predict, Campanella and PhysioNet lost subjects, so their splits differ.
Nadeau-Bengio corrected t-test with Holm correction, as in compare_models.py.
"""

from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path

import pandas as pd

from compare_models import corrected_ttest, holm
from run_jbhi_experiments import REPO_ROOT

TABLE = "outputs/tables/jbhi_v2/leave_one_dataset_out_per_split.csv"
PAIRED = ["WESAD", "UBFC-Phys"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default="53eaf0e", help="Commit holding the pre-fix per-split table.")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "outputs" / "tables" / "jbhi_v2" / "label_fix_sensitivity" / "before_after_significance.csv")
    args = parser.parse_args()
    old = pd.read_csv(io.StringIO(subprocess.run(["git", "show", f"{args.before}:{TABLE}"], cwd=REPO_ROOT, check=True,
                                                 capture_output=True, text=True).stdout))
    new = pd.read_csv(REPO_ROOT / TABLE)
    keys = ["features", "test_dataset", "trained_on", "split"]
    both = old.merge(new, on=keys, suffixes=("_before", "_after"))
    both = both[both["test_dataset"].isin(PAIRED)]

    rows = []
    for (features, dataset, trained_on), g in both.groupby(keys[:3]):
        for metric in ("balanced_accuracy", "auroc"):
            diff, t, p = corrected_ttest(g[f"{metric}_after"].to_numpy(), g[f"{metric}_before"].to_numpy())
            rows.append({"features": features, "test_dataset": dataset, "trained_on": trained_on, "metric": metric,
                         "before": g[f"{metric}_before"].mean(), "after": g[f"{metric}_after"].mean(),
                         "after_minus_before": diff, "t": t, "p": p, "n_splits": len(g)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    out["significant_0.05"] = out["p_holm"] < 0.05
    out = out.round(4)
    out.to_csv(args.output, index=False)
    print(out[out["trained_on"] != "within_dataset"].to_string(index=False))


if __name__ == "__main__":
    main()
