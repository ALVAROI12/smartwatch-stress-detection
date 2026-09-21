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
from functools import partial
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import signal

from relabel_windows import (BASELINE_SEC, EPM_RAW, EPM_TO_HARMONIZED, PHYSIONET_DIR, STAGE_TO_HARMONIZED,
                             epm_intervals, physionet_stress_intervals)

REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOW, STEP = 60.0, 30.0
FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
MIN_CARDIAC_COVERAGE = 0.5  # fraction of the window that must be covered by accepted beats
WESAD_LABELS = {1: "Baseline", 2: "Stress", 3: "Amusement", 4: "Meditation"}
# data_constraints.txt: S02 STRESS files contain duplicated samples from these rows on.
S02_STRESS_VALID_ROWS = {"ACC": 49545, "BVP": 99091, "EDA": 6195, "TEMP": 6195}
NO_PPG_TEMP = {("STRESS", "f07")}  # protection dock left on: only EDA and ACC are valid
# Hongn et al. 2025 (Sci Data) discarded this stress record for "bad fit of the wristband"; not in data_constraints.txt.
PHYSIONET_EXCLUDED = {("STRESS", "f13")}


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


TEMP_FEATURES = ("temp_mean", "temp_std", "temp_min", "temp_max", "temp_range", "temp_slope")
ACC_FEATURES = ("acc_mag_mean", "acc_mag_std", "acc_mag_min", "acc_mag_max", "acc_x_mean", "acc_y_mean", "acc_z_mean",
                "acc_x_std", "acc_y_std", "acc_z_std", "acc_sma", "acc_energy", "acc_entropy")


def other_features(bvp: np.ndarray, eda: np.ndarray, temp: np.ndarray | None, acc: np.ndarray | None) -> dict[str, float]:
    """Ported unchanged from notebook 03 (cells 2-5), minus the peak-based HR/HRV block.

    temp / acc may be None for datasets that did not release them; their features are then NaN.
    """
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

    if temp is None:
        f |= dict.fromkeys(TEMP_FEATURES, np.nan)
    else:
        f |= {"temp_mean": temp.mean(), "temp_std": temp.std(), "temp_min": temp.min(), "temp_max": temp.max()}
        f["temp_range"] = f["temp_max"] - f["temp_min"]
        f["temp_slope"] = (temp[-1] - temp[0]) / len(temp) * FS["TEMP"]

    if acc is None:
        return f | dict.fromkeys(ACC_FEATURES, np.nan)
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
    duration = min(len(signals[k]) / FS[k] for k in signals)
    rows = []
    for seg_start, seg_end, original, harmonized in segments:
        start = max(0.0, seg_start)
        while start + WINDOW <= min(seg_end, duration):
            cut = {k: signals[k][int(start * FS[k]): int((start + WINDOW) * FS[k])] for k in signals}
            row = other_features(cut["BVP"], cut["EDA"], cut.get("TEMP"), cut.get("ACC"))
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


def extract_physionet(root: Path, **interval_options) -> list[dict]:
    rows = []
    for session in ("STRESS", "AEROBIC", "ANAEROBIC"):
        for folder in sorted((root / PHYSIONET_DIR / session).iterdir()):
            sid = folder.name
            if not (folder / "BVP.csv").exists() or (session, sid.removesuffix("_a").removesuffix("_b")) in PHYSIONET_EXCLUDED:
                continue
            signals = load_e4(folder, S02_STRESS_VALID_ROWS if (session, sid) == ("STRESS", "S02") else None)
            if session == "STRESS":
                segments = [(a, b, stage, STAGE_TO_HARMONIZED[stage])
                            for a, b, stage in physionet_stress_intervals(folder, **interval_options)]
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


STRESS_PREDICT_STAGES = {"Baseline/Questionniare": ("Baseline (with questionnaire)", "Baseline"), "Stroop Test": ("Stroop", "Stress"),
                         "Interview": ("TSST interview", "Stress"), "Hyperventilation": ("Hyperventilation", "Hyperventilation"),
                         "Relax": ("Relax", "Rest"), "Relax/Baseline": ("Relax", "Rest")}  # Consent and final questionnaire are skipped
STRESS_PREDICT_TZ = "Europe/Dublin"  # time log is local wall-clock at minute resolution; E4 files are unix UTC
# Iqbal et al. 2022 removed one participant because "the data collection protocol was not followed properly";
# S01 is the one missing from their processed labels, and its log has the baseline after the Stroop test.
STRESS_PREDICT_EXCLUDED = {"S01"}
# E4 tags mark task starts and ends (S05-S35: Stroop start/end, Interview start/end, HPT start/end, final). The
# hand-written log is minute-resolution and was 92-163 s off where the tag order confirms the right tag.
STRESS_PREDICT_SNAP_SEC = 180


def stress_predict_segments(log_row: pd.Series, header: pd.Series, date, e4_start: float, tags: np.ndarray):
    """Stage boundaries from Time_logs.xlsx, snapped to the nearest E4 button tag within STRESS_PREDICT_SNAP_SEC.

    A task boundary with no tag that close stays a minute-resolution log time that may be minutes off (S06 Interview
    start: 256 s from its tag; S17 Interview end, S18 and S30 Stroop start: tag missing). That task and any stage
    sharing the boundary are dropped rather than labelled from the log.
    """
    def to_abs(clock) -> float:
        stamp = pd.Timestamp.combine(pd.Timestamp(date).date(), clock).tz_localize(STRESS_PREDICT_TZ).timestamp()
        if stamp < e4_start - 1800:  # the log is a 12-hour clock without AM/PM: "01:11" in an afternoon session is 13:11
            stamp += 12 * 3600
        return stamp

    def snap(stamp: float) -> tuple[float, bool]:
        if len(tags) and np.abs(tags - stamp).min() <= STRESS_PREDICT_SNAP_SEC:
            return tags[np.abs(tags - stamp).argmin()], True
        return stamp, False

    stages, unresolved = [], set()
    for col in range(4, len(header) - 2, 2):
        stage = header.iloc[col]
        if stage not in STRESS_PREDICT_STAGES or pd.isna(log_row.iloc[col]) or pd.isna(log_row.iloc[col + 1]):
            continue
        original, harmonized = STRESS_PREDICT_STAGES[stage]
        logged = to_abs(log_row.iloc[col]), to_abs(log_row.iloc[col + 1])
        (start, start_ok), (end, end_ok) = snap(logged[0]), snap(logged[1])
        if harmonized in ("Stress", "Hyperventilation"):
            unresolved |= {t for t, ok in zip(logged, (start_ok, end_ok)) if not ok}
        stages.append((logged, start, end, original, harmonized))
    return [(start - e4_start + 15, end - e4_start - 15, original, harmonized)  # 15 s guard
            for logged, start, end, original, harmonized in stages if not unresolved & set(logged)]


def extract_stress_predict(root: Path) -> list[dict]:
    base = root / "Stress-Predict"
    log = pd.read_excel(base / "Processed_data" / "Time_logs.xlsx", header=None)
    header = log.iloc[0].ffill()
    rows = []
    for _, log_row in log.iloc[2:].iterrows():
        sid = str(log_row.iloc[0])
        folder = base / "Raw_data" / sid
        if not (folder / "BVP.csv").exists() or sid in STRESS_PREDICT_EXCLUDED:
            continue
        with open(folder / "EDA.csv") as handle:
            e4_start = float(handle.readline().split(",")[0])
        tag_file = folder / f"tags_{sid}.csv"
        tags = pd.read_csv(tag_file, header=None)[0].to_numpy(float) if tag_file.stat().st_size else np.array([])
        segments = stress_predict_segments(log_row, header, log_row.iloc[-1], e4_start, tags)
        rows += windows_for_recording(load_e4(folder), segments, {"dataset": "Stress-Predict", "subject_id": sid,
                                                                   "subject_uid": f"Stress-Predict:{sid}"})
    print("  Stress-Predict", flush=True)
    return rows


def extract_ubfc(root: Path) -> list[dict]:
    """UBFC-Phys: three separate 3-min recordings per subject (T1 rest, T2 speech, T3 arithmetic); BVP + EDA only.

    The control group performed non-evaluative versions of T2/T3, so only the test group's tasks count as Stress.
    """
    rows = []
    for folder in sorted((root / "UBFC-Phys").glob("s*"), key=lambda p: int(p.name[1:])):
        sid = folder.name
        group = (folder / f"info_{sid}.txt").read_text().split()[2]  # "test" or "ctrl"
        for phase, original in (("T1", "T1 rest"), ("T2", "T2 speech"), ("T3", "T3 arithmetic")):
            signals = {"BVP": pd.read_csv(folder / f"bvp_{sid}_{phase}.csv", header=None)[0].to_numpy(float),
                       "EDA": pd.read_csv(folder / f"eda_{sid}_{phase}.csv", header=None)[0].to_numpy(float)}
            harmonized = "Baseline" if phase == "T1" else ("Stress" if group == "test" else "Control task")
            duration = len(signals["EDA"]) / FS["EDA"]
            rows += windows_for_recording(signals, [(0.0, duration, f"{original} ({group} group)", harmonized)],
                                          {"dataset": "UBFC-Phys", "subject_id": f"{sid}_{phase}", "subject_uid": f"UBFC-Phys:{sid}"})
    print("  UBFC-Phys", flush=True)
    return rows


# Campanella et al. 2024 (Data in Brief, CC BY): no per-subject markers exist, so stages follow the published schedule
# from recording start. Wrist movement confirms the 3-min baseline and a still period from minute 27, but shows the
# 2-min breaks starting about a minute late, so breaks are not labelled and task windows keep a 1-min guard.
# The authors label every task as stress. Here only the seated subtraction counts as Stress: it raises EDA in 29/29
# subjects and heart rate by ~5 bpm with still wrists, whereas the Lego tasks show no heart-rate rise and lose
# 50-70% of PPG windows to hand movement. Lego windows are kept as "Manual task" (used for per-subject scaling only);
# "Lego with countdown" also has participants count backwards from 180, so it carries some cognitive load.
# The paper describes no transition time, so the countdown span ends 30 s before its nominal end (1500 s).
CAMPANELLA_SEGMENTS = [(15, 165, "Baseline (3 min rest)", "Baseline"), (240, 720, "Lego without instructions", "Manual task"),
                       (960, 1170, "Lego with instructions", "Manual task"), (1380, 1470, "Lego with countdown", "Manual task")]
# The subtraction has "no time constraint" (Campanella et al. 2024, Sec. 4.3); after it come a 2-min rest, a 1-min
# presentation and a 2-min rest. ponytail: its end is estimated as recording end minus that 300 s tail, which assumes the
# recording stops when the protocol does (implies 134-174 s tasks for subjects 03, 05, 22, 23, 24); an ACC-based
# per-subject end would be more precise if a reviewer asks.
CAMPANELLA_SUBTRACTION = (1650, 1800, 300, "Backward subtraction (up to 2.5 min)")


def load_headerless_e4(path: Path, fs: int) -> np.ndarray:
    """This release strips the E4 header rows from some files and keeps them in others.

    EDA values >= 1 uS were exported through a European-locale spreadsheet, which turned 1.038145 into
    "1.038.145". E4 EDA always has six decimals, so a multi-dot token is its digits / 1e6 (checked: the
    repaired signals are continuous, no 0.25 s jump above 1 uS in any subject).
    """
    text = pd.read_csv(path, header=None, dtype=str)
    mangled = text.apply(lambda col: col.str.count(r"\.") > 1)
    data = text.where(~mangled, text.apply(lambda col: col.str.replace(".", "", regex=False))).astype(float).to_numpy()
    data = np.where(mangled.to_numpy(), data / 1e6, data)
    if data[0, 0] > 1e9:  # unix start time
        data = data[1:]
    if np.all(data[0] == fs):  # sampling-rate row
        data = data[1:]
    return data


def extract_campanella(root: Path) -> list[dict]:
    rows = []
    for folder in sorted((root / "Campanella2024").glob("subject_*")):
        signals = {k: load_headerless_e4(folder / f"{k}.csv", FS[k]) for k in FS}
        signals = {k: (v if k == "ACC" else v.ravel()) for k, v in signals.items()}
        start, latest_end, tail, original = CAMPANELLA_SUBTRACTION
        end = min(latest_end, len(signals["EDA"]) / FS["EDA"] - tail - 15)  # 15 s guard before the post-task rest
        segments = [tuple(map(float, seg[:2])) + seg[2:] for seg in CAMPANELLA_SEGMENTS] + [(start, end, original, "Stress")]
        rows += windows_for_recording(signals, segments,
                                      {"dataset": "Campanella2024", "subject_id": folder.name,
                                       "subject_uid": f"Campanella2024:{folder.name}"})
    print("  Campanella2024", flush=True)
    return rows


EXTRACTORS = {"Campanella2024": extract_campanella, "WESAD": extract_wesad, "PhysioNet": extract_physionet, "EPM-E4": extract_epm,
              "Stress-Predict": extract_stress_predict, "UBFC-Phys": extract_ubfc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", choices=list(EXTRACTORS), default=list(EXTRACTORS),
                        help="Extract only these; rows of other datasets already in the output file are kept.")
    parser.add_argument("--physionet-baseline-sec", type=float, default=BASELINE_SEC,
                        help="Sensitivity run: v2 baseline length before the first tag.")
    parser.add_argument("--physionet-rest-second-half", action="store_true",
                        help="Sensitivity run: keep only the second half of each PhysioNet rest period.")
    args = parser.parse_args()
    EXTRACTORS["PhysioNet"] = partial(extract_physionet, baseline_sec=args.physionet_baseline_sec,
                                      rest_second_half=args.physionet_rest_second_half)
    output = args.output or args.data_root / "data" / "processed" / "combined" / "harmonized_windows_v2.csv"
    df = pd.DataFrame([row for name in args.datasets for row in EXTRACTORS[name](args.data_root)])
    df["label"] = df["harmonized_label"]
    if output.exists() and set(args.datasets) != set(EXTRACTORS):
        kept = pd.read_csv(output)
        df = pd.concat([kept[~kept["dataset"].isin(args.datasets)].drop(columns="window_id"), df], ignore_index=True)
    df.insert(0, "window_id", df.groupby(["dataset", "subject_id"]).cumcount())
    df.to_csv(output, index=False)
    print(f"wrote {output}: {len(df)} windows, {df['subject_uid'].nunique()} subjects")
    print(df.groupby(["dataset", "harmonized_label"]).agg(
        windows=("purity", "size"), subjects=("subject_uid", "nunique"), hr_mean=("hr_mean", "mean"),
        rmssd=("hrv_rmssd", "median"), cardiac_valid_pct=("hr_mean", lambda s: 100 * s.notna().mean())).round(1).to_string())


if __name__ == "__main__":
    main()
