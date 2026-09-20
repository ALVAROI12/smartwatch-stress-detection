#!/usr/bin/env python3
"""Attach protocol-stage labels to the existing 60 s feature windows.

The original pipeline labelled every window by session: a whole PhysioNet STRESS recording
(baseline and rest included) became "Stress", and every EPM-E4 window became "Emotion".
This script re-reads the raw protocol markers and adds, per window:

  subject_uid       dataset-qualified subject with split recordings (_a/_b) merged
  original_label    the stage/clip named by the source dataset
  harmonized_label  the label used for cross-dataset experiments ("Excluded" = transition/unlabelled)
  purity            fraction of the window covered by harmonized_label (filter at analysis time)

Features are not recomputed; windows keep their session-relative timestamps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSIONET_DIR = (
    "wearable-device-dataset/"
    "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1/Wearable_Dataset"
)
EPM_SLICES = "EPM-E4/empatica_wearable_data/preprocessed/unclean-signals/empatica_slices/0.0078125"
EPM_RAW = "EPM-E4/empatica_wearable_data/raw"
EPM_TZ = ZoneInfo("Europe/Madrid")  # slice timestamps are local time; E4 start is unix UTC
BASELINE_SEC = 180  # published protocol: 3-minute baseline in both versions

# Tag indices follow the dataset authors' convention: T[0] = recording start, T[1:] = tags.csv rows.
# Stressor spans are the ones the authors shade in Wearable_Dataset.ipynb; rest spans lie between them.
STAGES_V1 = {(1, 2): "Baseline", (3, 4): "Stroop", (4, 5): "First Rest", (5, 6): "TMCT", (6, 7): "Second Rest",
             (7, 8): "Real Opinion", (9, 10): "Opposite Opinion", (11, 12): "Subtract"}
STAGES_V2 = {(2, 3): "TMCT", (3, 4): "First Rest", (4, 5): "Real Opinion", (6, 7): "Opposite Opinion",
             (7, 8): "Second Rest", (8, 9): "Subtract"}
STAGE_TO_HARMONIZED = {"Baseline": "Baseline", "First Rest": "Rest", "Second Rest": "Rest", "Stroop": "Stress",
                       "TMCT": "Stress", "Real Opinion": "Stress", "Opposite Opinion": "Stress", "Subtract": "Stress"}
EPM_TO_HARMONIZED = {"FEAR": "Fear", "ANGER": "Anger", "SADNESS": "Sadness", "HAPPINESS": "Happiness"}


def physionet_stress_intervals(session_dir: Path) -> list[tuple[float, float, str]]:
    """Return (start_s, end_s, stage) relative to recording start for one STRESS session."""
    with open(session_dir / "EDA.csv") as handle:
        start = pd.to_datetime(handle.readline().strip())
        n_samples = sum(1 for _ in handle) - 1  # minus the sampling-rate row
    duration = n_samples / 4.0
    try:
        tags = pd.to_datetime(pd.read_csv(session_dir / "tags.csv", header=None)[0])
    except pd.errors.EmptyDataError:
        # f14_a: authors note this file holds only the baseline (Bluetooth dropped before task 1).
        return [(max(0.0, duration - BASELINE_SEC), duration, "Baseline")]
    t = [0.0] + [(tag - start).total_seconds() for tag in tags]
    is_v1 = session_dir.name.startswith("S")
    expected = 14 if is_v1 else 10
    if len(t) != expected:
        raise ValueError(f"{session_dir.name}: expected {expected - 1} tags, found {len(t) - 1}")
    intervals = [(t[i], t[j], stage) for (i, j), stage in (STAGES_V1 if is_v1 else STAGES_V2).items()]
    if not is_v1 and not session_dir.name.endswith("_b"):
        # ponytail: v2 has no explicit baseline marks; assume the 3 min before the first tag.
        # Replace with exact marks if the dataset authors can supply them.
        intervals.append((max(0.0, t[1] - BASELINE_SEC), t[1], "Baseline"))
    return intervals


def epm_intervals(root: Path, subject: str) -> list[tuple[float, float, str]]:
    with open(root / EPM_RAW / subject / "empatica" / "EDA.csv") as handle:
        e4_start = float(handle.readline().split(",")[0])
    intervals = []
    for slice_path in sorted((root / EPM_SLICES / subject).glob("*.csv")):
        stamps = pd.to_datetime(pd.read_csv(slice_path, usecols=["TimeStamp"])["TimeStamp"])
        if stamps.empty:  # a few slices ship with a header only
            continue
        bounds = [s.tz_localize(EPM_TZ).timestamp() - e4_start for s in (stamps.iloc[0], stamps.iloc[-1])]
        intervals.append((bounds[0], bounds[1], slice_path.stem))
    return intervals


def label_windows(windows: pd.DataFrame, intervals: list[tuple[float, float, str]], to_harmonized) -> pd.DataFrame:
    """Pick the harmonized class covering most of each window; purity = covered fraction."""
    starts, ends = windows["timestamp_start"].to_numpy(float), windows["timestamp_end"].to_numpy(float)
    length = ends - starts
    by_class: dict[str, np.ndarray] = {}
    by_stage: dict[str, np.ndarray] = {}
    for lo, hi, stage in intervals:
        overlap = np.clip(np.minimum(ends, hi) - np.maximum(starts, lo), 0, None)
        by_stage[stage] = by_stage.get(stage, 0) + overlap
        by_class[to_harmonized(stage)] = by_class.get(to_harmonized(stage), 0) + overlap
    out = pd.DataFrame(index=windows.index, columns=["original_label", "harmonized_label", "purity"])
    out[["original_label", "harmonized_label"]] = "Transition/unlabelled", "Excluded"
    out["purity"] = 0.0
    if by_class:
        cls = pd.DataFrame(by_class, index=windows.index)
        stg = pd.DataFrame(by_stage, index=windows.index)
        covered = cls.max(axis=1) > 0
        out.loc[covered, "harmonized_label"] = cls.idxmax(axis=1)[covered]
        out.loc[covered, "original_label"] = stg.idxmax(axis=1)[covered]
        out.loc[covered, "purity"] = (cls.max(axis=1) / length)[covered]
    return out


def relabel(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    df = df.copy()
    subject = df["subject_id"].astype(str)
    df["subject_uid"] = df["dataset"] + ":" + subject.str.replace(r"_[ab]$", "", regex=True)
    df["original_label"] = df["label"]
    df["harmonized_label"] = df["label"]
    df["purity"] = 1.0

    stress = (df["dataset"] == "PhysioNet") & (df["label"] == "Stress")
    for sid, windows in df[stress].groupby(subject[stress]):
        intervals = physionet_stress_intervals(root / PHYSIONET_DIR / "STRESS" / sid)
        df.loc[windows.index, ["original_label", "harmonized_label", "purity"]] = label_windows(
            windows, intervals, STAGE_TO_HARMONIZED.get)

    exercise = (df["dataset"] == "PhysioNet") & ~stress
    df.loc[exercise, "original_label"] = df.loc[exercise, "label"].str.upper() + " session (whole recording)"

    epm = df["dataset"] == "EPM-E4"
    for sid, windows in df[epm].groupby(subject[epm]):
        df.loc[windows.index, ["original_label", "harmonized_label", "purity"]] = label_windows(
            windows, epm_intervals(root, sid),
            lambda stage: "Neutral" if stage.startswith("NEUTRAL") else EPM_TO_HARMONIZED[stage])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT,
                        help="Directory holding data/processed and the raw dataset folders (both are gitignored).")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    combined = args.data_root / "data" / "processed" / "combined"
    df = relabel(pd.read_csv(combined / "combined_dataset_filled.csv"), args.data_root)

    # Self-check: relabelling must not move windows between subjects or invent coverage.
    assert df["purity"].between(0, 1.0 + 1e-9).all()
    assert not df["subject_uid"].str.contains(r"_[ab]$").any()
    assert (df.loc[df["dataset"] == "WESAD", "harmonized_label"] == df.loc[df["dataset"] == "WESAD", "label"]).all()

    output = args.output or combined / "harmonized_windows.csv"
    df.to_csv(output, index=False)
    print(f"wrote {output} ({len(df)} windows, {df['subject_uid'].nunique()} subjects)")
    print(df.groupby(["dataset", "harmonized_label"]).agg(
        windows=("purity", "size"), pure_windows=("purity", lambda p: int((p >= 0.8).sum())),
        subjects=("subject_uid", "nunique")).to_string())


if __name__ == "__main__":
    main()
