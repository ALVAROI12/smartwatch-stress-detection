#!/usr/bin/env python3
"""Score the wrist-PPG HR/HRV pipeline against WESAD chest ECG, per 60 s window.

WESAD records wrist BVP and chest ECG on one synchronized clock, so the ECG gives a reference for
exactly the windows used in the experiments. Writes per-window values and a summary table.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd

from extract_features import REPO_ROOT, cardiac_features, clean_beats


def ecg_beats(ecg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        peaks = nk.ecg_peaks(nk.ecg_clean(ecg, sampling_rate=700), sampling_rate=700)[1]["ECG_R_Peaks"] / 700
    ibi = np.diff(peaks) * 1000
    keep = (ibi > 330) & (ibi < 1500)
    return peaks[1:][keep], ibi[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "tables" / "jbhi_v2")
    args = parser.parse_args()
    windows = pd.read_csv(args.data_root / "data" / "processed" / "combined" / "harmonized_windows_v2.csv")
    rows = []
    for pkl in sorted((args.data_root / "WESAD").glob("S*/S*.pkl")):
        # WESAD is only distributed as pickles; these are the files downloaded from the dataset authors.
        with open(pkl, "rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        wrist = clean_beats(data["signal"]["wrist"]["BVP"].ravel().astype(float))
        chest = ecg_beats(data["signal"]["chest"]["ECG"].ravel())
        for w in windows[(windows["dataset"] == "WESAD") & (windows["subject_id"] == pkl.parent.name)].itertuples():
            ppg = cardiac_features(*wrist, w.timestamp_start, w.timestamp_end)
            ecg = cardiac_features(*chest, w.timestamp_start, w.timestamp_end)
            rows.append({"subject": pkl.parent.name, "label": w.harmonized_label, "coverage": ppg["cardiac_coverage"],
                         **{f"ppg_{k}": ppg[k] for k in ("hr_mean", "hrv_rmssd", "hrv_sdnn")},
                         **{f"ecg_{k}": ecg[k] for k in ("hr_mean", "hrv_rmssd", "hrv_sdnn")}})
    per_window = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_window.to_csv(args.output_dir / "wrist_vs_ecg_per_window.csv", index=False)

    summary = []
    for label, group in [("all", per_window), *per_window.groupby("label")]:
        row = {"label": label, "n_windows": len(group), "pct_with_valid_ppg": 100 * group["ppg_hr_mean"].notna().mean()}
        for metric in ("hr_mean", "hrv_rmssd", "hrv_sdnn"):
            pair = group.dropna(subset=[f"ppg_{metric}", f"ecg_{metric}"])
            row |= {f"{metric}_mae": (pair[f"ppg_{metric}"] - pair[f"ecg_{metric}"]).abs().mean(),
                    f"{metric}_bias": (pair[f"ppg_{metric}"] - pair[f"ecg_{metric}"]).mean(),
                    f"{metric}_r": pair[f"ppg_{metric}"].corr(pair[f"ecg_{metric}"])}
        summary.append(row)
    summary = pd.DataFrame(summary).round(2)
    summary.to_csv(args.output_dir / "wrist_vs_ecg_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
