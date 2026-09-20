#!/usr/bin/env python3
"""Rebuild the 60 s feature windows from raw signals, anchored to protocol stages.

Replaces notebooks/03_feature_extraction.ipynb, which had three defects:
  1. WESAD: signals came from the raw E4 CSVs but labels from the synchronized SX.pkl, mapped by a
     length ratio. The raw recording starts 16-26 min earlier, so ~81% of windows got the wrong label.
     Here both signals and labels come from SX.pkl.
  2. HR/HRV: scipy.find_peaks on unfiltered BVP counted noise as beats (RMSSD ~230 ms) and capped
     HR at 120 bpm. Here: NeuroKit2 PPG peak detection + inter-beat-interval artefact rejection,
     validated against WESAD chest ECG (see validate_cardiac_against_ecg.py).
  3. PhysioNet/EPM-E4 windows ignored protocol stages. Here every window lies fully inside one stage.

EDA, temperature, accelerometer and BVP-amplitude features are ported unchanged.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import signal

from relabel_windows import (EPM_RAW, EPM_TO_HARMONIZED, PHYSIONET_DIR, STAGE_TO_HARMONIZED, epm_intervals,
                             physionet_stress_intervals)

REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOW, STEP = 60.0, 30.0
FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
MIN_CARDIAC_COVERAGE = 0.5  # fraction of the window that must be covered by accepted beats
WESAD_LABELS = {1: "Baseline", 2: "Stress", 3: "Amusement", 4: "Meditation"}
# data_constraints.txt: S02 STRESS files contain duplicated samples from these rows on.
S02_STRESS_VALID_ROWS = {"ACC": 49545, "BVP": 99091, "EDA": 6195, "TEMP": 6195}
NO_PPG_TEMP = {("STRESS", "f07")}  # protection dock left on: only EDA and ACC are valid


def clean_beats(bvp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Beat end-times (s) and inter-beat intervals (ms) after artefact rejection."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        peaks = nk.ppg_findpeaks(nk.ppg_clean(bvp, sampling_rate=FS["BVP"]), sampling_rate=FS["BVP"])["PPG_Peaks"]
    times = peaks / FS["BVP"]
    ibi = np.diff(times) * 1000
    local = pd.Series(ibi).rolling(9, center=True, min_periods=3).median().to_numpy()
    keep = (ibi > 330) & (ibi < 1500) & (np.abs(ibi - local) < 0.2 * local)  # 40-180 bpm, <20% off local median
    return times[1:][keep], ibi[keep]


def cardiac_features(beat_t: np.ndarray, ibi: np.ndarray, start: float, end: float) -> dict[str, float]:
    inside = (beat_t >= start) & (beat_t < end)
    x = ibi[inside]
    coverage = x.sum() / ((end - start) * 1000)
    out = dict.fromkeys(("hr_mean", "hr_std", "hrv_rmssd", "hrv_sdnn", "hrv_pnn50"), np.nan)
    out["cardiac_coverage"] = coverage
    if coverage < MIN_CARDIAC_COVERAGE or len(x) < 20:
        return out
    consecutive = np.diff(beat_t[inside]) < 1.6  # successive differences only across truly adjacent beats
    successive = np.diff(x)[consecutive]
    out |= {"hr_mean": 60000 / x.mean(), "hr_std": np.std(60000 / x), "hrv_sdnn": np.std(x)}
    if len(successive) > 5:
        out |= {"hrv_rmssd": np.sqrt(np.mean(successive ** 2)), "hrv_pnn50": 100 * np.mean(np.abs(successive) > 50)}
    return out


def other_features(bvp: np.ndarray, eda: np.ndarray, temp: np.ndarray, acc: np.ndarray) -> dict[str, float]:
    """Ported unchanged from notebook 03 (cells 2-5), minus the peak-based HR/HRV block."""
    f = {"bvp_mean": bvp.mean(), "bvp_std": bvp.std(), "bvp_min": bvp.min(), "bvp_max": bvp.max()}
    f["bvp_range"] = f["bvp_max"] - f["bvp_min"]

    f |= {"eda_mean": eda.mean(), "eda_std": eda.std(), "eda_min": eda.min(), "eda_max": eda.max()}
    f["eda_range"] = f["eda_max"] - f["eda_min"]
    peaks, props = signal.find_peaks(eda, prominence=np.std(np.diff(eda)) * 0.5, distance=FS["EDA"])
    f["eda_scr_count"] = len(peaks)
    f["eda_scr_amp_mean"] = props["prominences"].mean() if len(peaks) else 0.0
    b, a = signal.butter(2, 0.05, btype="low", fs=FS["EDA"])
    tonic = signal.filtfilt(b, a, eda)
    f |= {"eda_tonic_mean": tonic.mean(), "eda_phasic_mean": np.abs(eda - tonic).mean(),
          "eda_slope": (eda[-1] - eda[0]) / len(eda) * FS["EDA"]}

    f |= {"temp_mean": temp.mean(), "temp_std": temp.std(), "temp_min": temp.min(), "temp_max": temp.max()}
    f["temp_range"] = f["temp_max"] - f["temp_min"]
    f["temp_slope"] = (temp[-1] - temp[0]) / len(temp) * FS["TEMP"]

    x, y, z = acc[:, 0], acc[:, 1], acc[:, 2]
    mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    hist, _ = np.histogram(mag, bins=10, density=True)
    hist = hist[hist > 0]
    f |= {"acc_mag_mean": mag.mean(), "acc_mag_std": mag.std(), "acc_mag_min": mag.min(), "acc_mag_max": mag.max(),
          "acc_x_mean": x.mean(), "acc_y_mean": y.mean(), "acc_z_mean": z.mean(),
          "acc_x_std": x.std(), "acc_y_std": y.std(), "acc_z_std": z.std(),
          "acc_sma": np.mean(np.abs(x) + np.abs(y) + np.abs(z)), "acc_energy": np.sum(mag ** 2) / len(mag),
          "acc_entropy": -np.sum(hist * np.log2(hist))}
    return f


def windows_for_recording(signals: dict[str, np.ndarray], segments: list[tuple[float, float, str, str]],
                          meta: dict, ppg_valid: bool = True) -> list[dict]:
    """segments: (start_s, end_s, original_label, harmonized_label). Windows never cross a segment edge."""
    beat_t, ibi = clean_beats(signals["BVP"]) if ppg_valid else (np.array([]), np.array([]))
    duration = min(len(signals[k]) / FS[k] for k in FS)
    rows = []
    for seg_start, seg_end, original, harmonized in segments:
        start = max(0.0, seg_start)
        while start + WINDOW <= min(seg_end, duration):
            cut = {k: signals[k][int(start * FS[k]): int((start + WINDOW) * FS[k])] for k in FS}
            row = other_features(cut["BVP"], cut["EDA"], cut["TEMP"], cut["ACC"])
            row |= cardiac_features(beat_t, ibi, start, start + WINDOW)
            if not ppg_valid:
                row |= {k: np.nan for k in row if k.startswith(("bvp_", "temp_"))}
            rows.append(row | meta | {"original_label": original, "harmonized_label": harmonized, "purity": 1.0,
                                      "timestamp_start": start, "timestamp_end": start + WINDOW})
            start += STEP
    return rows


def load_e4(folder: Path, valid_rows: dict[str, int] | None = None) -> dict[str, np.ndarray]:
    out = {}
    for name in FS:
        data = pd.read_csv(folder / f"{name}.csv", skiprows=2, header=None).to_numpy(float)
        if valid_rows:
            data = data[: valid_rows[name] - 2]
        out[name] = data if name == "ACC" else data.ravel()
    return out


def label_runs(labels: np.ndarray, fs: int) -> list[tuple[float, float, str, str]]:
    edges = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate([[0], edges, [len(labels)]])
    return [(a / fs, b / fs, WESAD_LABELS[labels[a]].lower(), WESAD_LABELS[labels[a]])
            for a, b in zip(bounds[:-1], bounds[1:]) if labels[a] in WESAD_LABELS]


def extract_wesad(root: Path) -> list[dict]:
    rows = []
    for pkl in sorted((root / "WESAD").glob("S*/S*.pkl")):
        # WESAD is only distributed as pickles; these are the files downloaded from the dataset authors.
        with open(pkl, "rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        wrist = data["signal"]["wrist"]
        signals = {k: (wrist[k] if k == "ACC" else wrist[k].ravel()).astype(float) for k in FS}
        sid = pkl.parent.name
        rows += windows_for_recording(signals, label_runs(data["label"], 700),
                                      {"dataset": "WESAD", "subject_id": sid, "subject_uid": f"WESAD:{sid}"})
        print(f"  WESAD {sid}", flush=True)
    return rows


def extract_physionet(root: Path) -> list[dict]:
    rows = []
    for session in ("STRESS", "AEROBIC", "ANAEROBIC"):
        for folder in sorted((root / PHYSIONET_DIR / session).iterdir()):
            if not (folder / "BVP.csv").exists():
                continue
            sid = folder.name
            signals = load_e4(folder, S02_STRESS_VALID_ROWS if (session, sid) == ("STRESS", "S02") else None)
            if session == "STRESS":
                segments = [(a, b, stage, STAGE_TO_HARMONIZED[stage]) for a, b, stage in physionet_stress_intervals(folder)]
            else:  # ponytail: exercise sessions kept whole (warm-up/cool-down included); stage them if Aerobic becomes a headline class
                duration = len(signals["EDA"]) / FS["EDA"]
                segments = [(0.0, duration, f"{session} session (whole recording)", session.capitalize())]
            meta = {"dataset": "PhysioNet", "subject_id": sid,
                    "subject_uid": "PhysioNet:" + sid.removesuffix("_a").removesuffix("_b")}
            rows += windows_for_recording(signals, segments, meta, ppg_valid=(session, sid) not in NO_PPG_TEMP)
        print(f"  PhysioNet {session}", flush=True)
    return rows


def extract_epm(root: Path) -> list[dict]:
    rows = []
    for folder in sorted((root / EPM_RAW).iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not (folder / "empatica" / "BVP.csv").exists():
            continue
        try:
            intervals = epm_intervals(root, folder.name)
        except FileNotFoundError:
            continue  # no clip slices published for this participant
        segments = [(a, b, stage, "Neutral" if stage.startswith("NEUTRAL") else EPM_TO_HARMONIZED[stage])
                    for a, b, stage in intervals]
        rows += windows_for_recording(load_e4(folder / "empatica"), segments,
                                      {"dataset": "EPM-E4", "subject_id": folder.name, "subject_uid": f"EPM-E4:{folder.name}"})
    print("  EPM-E4", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = extract_wesad(args.data_root) + extract_physionet(args.data_root) + extract_epm(args.data_root)
    df = pd.DataFrame(rows)
    df["label"] = df["harmonized_label"]
    df.insert(0, "window_id", df.groupby(["dataset", "subject_id"]).cumcount())
    output = args.output or args.data_root / "data" / "processed" / "combined" / "harmonized_windows_v2.csv"
    df.to_csv(output, index=False)
    print(f"wrote {output}: {len(df)} windows, {df['subject_uid'].nunique()} subjects")
    print(df.groupby(["dataset", "harmonized_label"]).agg(
        windows=("purity", "size"), subjects=("subject_uid", "nunique"), hr_mean=("hr_mean", "mean"),
        rmssd=("hrv_rmssd", "median"), cardiac_valid_pct=("hr_mean", lambda s: 100 * s.notna().mean())).round(1).to_string())


if __name__ == "__main__":
    main()
