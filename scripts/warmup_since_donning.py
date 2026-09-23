"""Fix 4: sensor warm-up since donning. WESAD from raw E4 CSVs (offset to the synchronised pkl found by exact
match of the wrist EDA stream); PhysioNet, Stress-Predict and Campanella from their raw E4 files, whose clock
already starts at recording start. UBFC-Phys has per-task files (no donning clock) and is excluded."""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
_cli = argparse.ArgumentParser(description=__doc__)
_cli.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", REPO)),
                  help="folder holding the raw datasets and data/processed/combined/harmonized_windows_v2.csv "
                       "(default: $DATA_ROOT, else this repo)")
DATA_ROOT = _cli.parse_args().data_root
sys.path.insert(0, str(REPO / "scripts"))
from extract_features import load_e4, load_headerless_e4  # noqa: E402
from relabel_windows import PHYSIONET_DIR  # noqa: E402

OUT = REPO / "outputs" / "tables" / "jbhi_v2" / "contribution_probes" / "wesad_warmup"
OUT.mkdir(parents=True, exist_ok=True)
FS = 4
HORIZON_MIN = 40
windows = pd.read_csv(DATA_ROOT / "data/processed/combined/harmonized_windows_v2.csv")


def raw_e4_tonic(folder: Path, headerless: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if headerless:
        temp, eda = (load_headerless_e4(folder / f"{k}.csv", FS).ravel() for k in ("TEMP", "EDA"))
    else:
        temp = pd.read_csv(folder / "TEMP.csv", skiprows=2, header=None)[0].to_numpy(float)
        eda = pd.read_csv(folder / "EDA.csv", skiprows=2, header=None)[0].to_numpy(float)
    tonic = pd.Series(eda).rolling(60 * FS, center=True, min_periods=FS).median().to_numpy()
    return temp, tonic


def find_offset(raw_eda: np.ndarray, pkl_eda: np.ndarray, probe: int = 400) -> int:
    """Sample lag at which the pkl wrist EDA starts inside the raw EDA (exact float match after E4 export rounding)."""
    target = pkl_eda[:probe]
    for lag in range(len(raw_eda) - probe):
        if np.allclose(raw_eda[lag:lag + probe], target, atol=1e-5):
            return lag
    raise ValueError("no match")


def trajectories(recs: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Pooled per-minute trajectory after removing each subject's mean over the horizon; slopes by segment."""
    rows = []
    for sid, (temp, tonic) in recs.items():
        n = min(len(temp), len(tonic), HORIZON_MIN * 60 * FS)
        minute = np.arange(n) / (60 * FS)
        d = pd.DataFrame({"minute": minute, "temp": temp[:n] - np.nanmean(temp[:n]), "tonic": tonic[:n] - np.nanmean(tonic[:n])})
        d["bin"] = d["minute"].astype(int)
        g = d.groupby("bin")[["temp", "tonic"]].mean().reset_index()
        g["subject"] = sid
        rows.append(g)
    per_min = pd.concat(rows)
    pooled = per_min.groupby("bin")[["temp", "tonic"]].mean()
    out = {}
    # minute 0 is excluded: the first E4 TEMP minute reads 4-6 C above skin (sensor not yet in contact) in every WESAD subject
    for name, (lo, hi) in {"1_10": (1, 10), "10_20": (10, 20), "20_40": (20, 40)}.items():
        seg = pooled.loc[lo:hi - 1]
        for sig in ("temp", "tonic"):
            out[f"{sig}_slope_{name}"] = np.polyfit(seg.index, seg[sig], 1)[0] if len(seg) > 2 else np.nan
    out["temp_diff_30_40_minus_1_5"] = pooled.loc[30:39, "temp"].mean() - pooled.loc[1:4, "temp"].mean()
    out["tonic_diff_30_40_minus_1_5"] = pooled.loc[30:39, "tonic"].mean() - pooled.loc[1:4, "tonic"].mean()
    out["n_subjects"] = len(recs)
    return out, pooled


def per_subject_temp(recs: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Per recording: first-minute temperature offset and change over the first 30 min (per-minute means)."""
    rows = []
    for sid, (temp, _) in recs.items():
        m = pd.Series(temp[:HORIZON_MIN * 60 * FS]).groupby(np.arange(min(len(temp), HORIZON_MIN * 60 * FS)) // (60 * FS)).mean()
        rows.append({"subject": sid, "temp_min0_minus_min1_4": m[0] - m.loc[1:4].mean(),
                     "temp_min30_minus_min1": m.get(30, np.nan) - m[1],
                     "temp_30_40_minus_1_5": m.loc[30:39].mean() - m.loc[1:4].mean()})
    return pd.DataFrame(rows)


results, curves = [], {}

# --- WESAD: raw E4 vs synchronised pkl -------------------------------------------------------------------
offsets, recs = [], {}
for pkl in sorted((DATA_ROOT / "WESAD").glob("S*/S*.pkl")):
    sid = pkl.parent.name
    with open(pkl, "rb") as h:
        data = pickle.load(h, encoding="latin1")
    pkl_eda = data["signal"]["wrist"]["EDA"].ravel().astype(float)
    raw_eda = pd.read_csv(pkl.parent / f"{sid}_E4_Data" / "EDA.csv", skiprows=2, header=None)[0].to_numpy(float)
    lag = find_offset(raw_eda, pkl_eda)
    offsets.append({"subject": sid, "offset_min": lag / FS / 60, "raw_min": len(raw_eda) / FS / 60,
                    "pkl_min": len(pkl_eda) / FS / 60, "length_ratio": len(raw_eda) / len(pkl_eda)})
    recs[sid] = raw_e4_tonic(pkl.parent / f"{sid}_E4_Data")
offsets = pd.DataFrame(offsets)
offsets.to_csv(OUT / "wesad_offsets.csv", index=False)
wesad_stats, curves["WESAD"] = trajectories(recs)
per_subject = [per_subject_temp(recs).assign(dataset="WESAD")]
w = windows[windows["dataset"] == "WESAD"].merge(offsets, left_on="subject_id", right_on="subject")
w["minutes_since_donning"] = w["timestamp_start"] / 60 + w["offset_min"]
per_class = w.groupby("harmonized_label")["minutes_since_donning"].describe(percentiles=[.25, .5, .75])
per_class.to_csv(OUT / "wesad_minutes_since_donning_by_class.csv")
base = w.loc[w["harmonized_label"] == "Baseline", "minutes_since_donning"]
results.append({"dataset": "WESAD", "clock_origin": "raw E4 recording start (donning), via pkl offset",
                **wesad_stats, "baseline_median_min": base.median(),
                **{f"baseline_within_{m}min": (base <= m).mean() for m in (10, 15, 20)}})

# --- Datasets whose feature-table clock is the raw E4 recording start ----------------------------------------
sources = {
    "PhysioNet": [(p.name, p, False) for p in sorted((DATA_ROOT / PHYSIONET_DIR / "STRESS").iterdir()) if (p / "TEMP.csv").exists()],
    "Stress-Predict": [(p.name, p, False) for p in sorted((DATA_ROOT / "Stress-Predict/Raw_data").glob("S*")) if (p / "TEMP.csv").exists()],
    "Campanella2024": [(p.name, p, True) for p in sorted((DATA_ROOT / "Campanella2024").glob("subject_*"))],
}
for name, items in sources.items():
    recs = {}
    for sid, folder, headerless in items:
        try:
            recs[sid] = raw_e4_tonic(folder, headerless)
        except Exception as e:  # noqa: BLE001
            print(f"skip {name} {sid}: {e}")
    stats, curves[name] = trajectories(recs)
    per_subject.append(per_subject_temp(recs).assign(dataset=name))
    d = windows[(windows["dataset"] == name) & (windows["harmonized_label"] == "Baseline")]
    base = d["timestamp_start"] / 60
    results.append({"dataset": name, "clock_origin": "raw E4 recording start (timestamp_start = 0)", **stats,
                    "baseline_median_min": base.median(), **{f"baseline_within_{m}min": (base <= m).mean() for m in (10, 15, 20)}})

pd.concat(per_subject).round(3).to_csv(OUT / "warmup_per_subject_temp.csv", index=False)
res = pd.DataFrame(results).round(4)
res.to_csv(OUT / "warmup_since_donning.csv", index=False)
pd.concat([c.assign(dataset=k) for k, c in curves.items()]).reset_index().to_csv(OUT / "warmup_curves_1min.csv", index=False)
print(offsets.round(2).to_string(index=False))
print(per_class.round(1).to_string())
print(res.to_string(index=False))
