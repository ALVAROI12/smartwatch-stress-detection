#!/usr/bin/env python3
"""Manipulation check from participants' own ratings, without touching the window table.

Reuses the report loaders of self_report_labels.py (which also rewrites the window table; this script does not).
Writes outputs/tables/jbhi_v2/manipulation_check.csv: per dataset and stage, rise of the stress rating over the
subject's own baseline (WESAD PANAS "Stressed" 1-5, PhysioNet 1-10), and SAM arousal (1-9) where rated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_report_labels import REPO_ROOT, epm_reports, physionet_reports, wesad_reports  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "tables" / "jbhi_v2" / "manipulation_check.csv")
    parser.add_argument("--min-delta", type=float, default=1.0)
    args = parser.parse_args()

    reports = pd.concat([wesad_reports(args.data_root), physionet_reports(args.data_root), epm_reports(args.data_root)],
                        ignore_index=True)
    baseline = reports[reports["stage"] == "Baseline"].set_index("subject_uid")["stress"]
    reports["stress_delta"] = reports["stress"] - reports["subject_uid"].map(baseline)
    check = reports.groupby(["dataset", "stage"]).agg(
        n_subjects=("subject_uid", "nunique"),
        stress_mean=("stress", "mean"), stress_delta_mean=("stress_delta", "mean"),
        pct_rose=("stress_delta", lambda d: 100 * (d.dropna() >= args.min_delta).mean() if d.notna().any() else float("nan")),
        sam_arousal_mean=("sam_arousal", "mean")).round(2).reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    check.to_csv(args.output, index=False)
    print(check.to_string(index=False))


if __name__ == "__main__":
    main()
