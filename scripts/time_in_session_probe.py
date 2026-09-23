"""Time-in-session, warm-up drift and wrist sensor ceiling probes. Outputs next to this file."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from leave_one_dataset_out import EXERCISE, NON_STRESS, fit_predict, normalise  # noqa: E402
from run_jbhi_experiments import META, grouped_split  # noqa: E402

OUT = REPO / "outputs" / "tables" / "jbhi_v2" / "contribution_probes" / "timeprobe"
OUT.mkdir(parents=True, exist_ok=True)
_cli = argparse.ArgumentParser(description=__doc__)
_cli.add_argument("--input", type=Path, help="feature table (default: $HARMONIZED_CSV, else data/processed/combined/harmonized_windows_v2.csv)",
                  default=Path(os.environ.get("HARMONIZED_CSV", REPO / "data/processed/combined/harmonized_windows_v2.csv")))
DATA = _cli.parse_args().input
N_JOBS = 8

df = pd.read_csv(DATA)
df = df[~df["harmonized_label"].isin(EXERCISE)].copy()  # stress-protocol recordings only
df["minutes"] = (df["timestamp_start"] - df.groupby("subject_uid")["timestamp_start"].transform("min")) / 60
physiology = [c for c in df.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
HR_EDA = [c for c in physiology if not c.startswith("temp_")]
# UBFC-Phys timestamps restart per task file (max 120 s), so time in session is unknown there.
TIME_OK = [d for d in df["dataset"].unique() if d != "UBFC-Phys"]

# ---------------- Part A: warm-up drift ----------------
rows = []
for d, g in df[df["dataset"].isin(TIME_OK)].groupby("dataset"):
    g = g[g["minutes"] <= 30]
    base = df[(df["dataset"] == d) & (df["harmonized_label"] == "Baseline")]
    for col in ("temp_mean", "eda_tonic_mean", "eda_mean", "hr_mean"):
        y = g[col] - g.groupby("subject_uid")[col].transform("mean")
        ok = y.notna()
        early, late = ok & (g["minutes"] < 10), ok & (g["minutes"] >= 10)
        fit = lambda m: np.polyfit(g.loc[m, "minutes"], y[m], 1)[0] if m.sum() > 10 else np.nan
        rows.append({"dataset": d, "feature": col, "n_windows": int(ok.sum()),
                     "slope_per_min_first10": fit(early), "slope_per_min_after10": fit(late),
                     "mean_min0_5": y[ok & (g["minutes"] < 5)].mean(), "mean_min5_10": y[ok & (g["minutes"] >= 5) & (g["minutes"] < 10)].mean(),
                     "mean_min10_20": y[ok & (g["minutes"] >= 10) & (g["minutes"] < 20)].mean(), "mean_min20_30": y[ok & (g["minutes"] >= 20)].mean(),
                     "baseline_frac_first10min": (base["minutes"] < 10).mean(), "baseline_median_min": base["minutes"].median(),
                     "stress_median_min": df[(df["dataset"] == d) & (df["harmonized_label"] == "Stress")]["minutes"].median()})
warm = pd.DataFrame(rows).round(4)
warm.to_csv(OUT / "warmup_drift.csv", index=False)
# minute-binned trajectory for plotting
traj = []
for d, g in df[df["dataset"].isin(TIME_OK)].groupby("dataset"):
    g = g[g["minutes"] <= 30].copy()
    for col in ("temp_mean", "eda_tonic_mean", "hr_mean"):
        g[col + "_dm"] = g[col] - g.groupby("subject_uid")[col].transform("mean")
    g["bin"] = g["minutes"].floordiv(2) * 2
    t = g.groupby("bin")[["temp_mean_dm", "eda_tonic_mean_dm", "hr_mean_dm"]].mean()
    t["n"] = g.groupby("bin").size()
    t["dataset"] = d
    traj.append(t.reset_index())
pd.concat(traj).round(4).to_csv(OUT / "warmup_trajectory_2min_bins.csv", index=False)

# ---------------- Part B: time in session ----------------
task = df[df["harmonized_label"].isin(NON_STRESS + ["Stress"])].copy()
task[physiology] = normalise(task, physiology, "session")
task = task[task.groupby("dataset")["harmonized_label"].transform(lambda s: (s == "Stress").any())].reset_index(drop=True)
y = (task["harmonized_label"] == "Stress").to_numpy(int)
x_std = task[HR_EDA].to_numpy()
x_time = task[["minutes"]].to_numpy()
x_both = task[HR_EDA + ["minutes"]].to_numpy()
datasets = sorted(task["dataset"].unique())
stress_range = {d: np.percentile(task.loc[(task["dataset"] == d) & (y == 1), "minutes"], [5, 95]) for d in datasets}

rows = []
for d in datasets:
    tgt, oth = np.flatnonzero(task["dataset"] == d), np.flatnonzero(task["dataset"] != d)
    lo, hi = stress_range[d]
    for seed in range(20):
        train, test = grouped_split(task.iloc[tgt], seed)
        te = tgt[test]
        truth = y[te]
        if len(set(truth)) < 2:
            continue
        matched = (y[te] == 1) | ((task["minutes"].to_numpy()[te] >= lo) & (task["minutes"].to_numpy()[te] <= hi))
        preds = {
            "time_only_within": fit_predict(x_time[tgt[train]], y[tgt[train]], x_time[te], N_JOBS),
            "std_within": fit_predict(x_std[tgt[train]], y[tgt[train]], x_std[te], N_JOBS),
            "std_external": fit_predict(x_std[oth], y[oth], x_std[te], N_JOBS),
            "std+time_within": fit_predict(x_both[tgt[train]], y[tgt[train]], x_both[te], N_JOBS),
            "std+time_external": fit_predict(x_both[oth], y[oth], x_both[te], N_JOBS),
        }
        for name, p in preds.items():
            row = {"dataset": d, "model": name, "split": seed, "ba": balanced_accuracy_score(truth, p > 0.5),
                   "auroc": roc_auc_score(truth, p), "n_test": len(te), "n_test_matched": int(matched.sum()),
                   "ba_matched": np.nan, "auroc_matched": np.nan}
            if d != "UBFC-Phys" and len(set(truth[matched])) == 2 and (matched & (y[te] == 0)).sum() >= 5:
                row["ba_matched"] = balanced_accuracy_score(truth[matched], p[matched] > 0.5)
                row["auroc_matched"] = roc_auc_score(truth[matched], p[matched])
            rows.append(row)
per_split = pd.DataFrame(rows)
per_split.to_csv(OUT / "time_probe_per_split.csv", index=False)
summ = per_split.groupby(["dataset", "model"])[["ba", "auroc", "ba_matched", "auroc_matched", "n_test", "n_test_matched"]].mean().round(3)
summ.to_csv(OUT / "time_probe_summary.csv")
print("\n=== Part B summary ===\n", summ.to_string())
print("\nstress minute range (5-95th pct):", {d: v.round(1).tolist() for d, v in stress_range.items()})

# ---------------- Part C: WESAD wrist vs chest ECG ----------------
ecg = pd.read_csv(REPO / "outputs" / "tables" / "jbhi_v2" / "wrist_vs_ecg_per_window.csv")
w = pd.read_csv(DATA)
w = w[w["dataset"] == "WESAD"].reset_index(drop=True)
assert len(w) == len(ecg) and np.allclose(w["cardiac_coverage"], ecg["coverage"]) and (w["subject_id"] == ecg["subject"]).all()
for k in ("hr_mean", "hrv_rmssd", "hrv_sdnn"):
    w["ecg_" + k] = ecg["ecg_" + k].to_numpy()
    assert np.allclose(w[k].fillna(-1), ecg["ppg_" + k].fillna(-1))
w = w[w["harmonized_label"].isin(NON_STRESS + ["Stress"])].reset_index(drop=True)
eda = [c for c in physiology if c.startswith("eda_")]
wrist3, ecg3 = ["hr_mean", "hrv_rmssd", "hrv_sdnn"], ["ecg_hr_mean", "ecg_hrv_rmssd", "ecg_hrv_sdnn"]
wrist5 = [c for c in physiology if c.startswith(("hr_", "hrv_"))]
cols_all = list(dict.fromkeys(physiology + ecg3))
w[cols_all] = normalise(w, cols_all, "session")
yw = (w["harmonized_label"] == "Stress").to_numpy(int)
sets = {"wrist_hr5+eda (current)": wrist5 + eda, "wrist_hr3+eda": wrist3 + eda, "ecg_hr3+eda": ecg3 + eda,
        "ecg_hr3 only": ecg3, "wrist_hr3 only": wrist3, "wrist_hr5 only": wrist5, "eda only": eda, "ecg_hr3+wrist_hr5+eda": ecg3 + wrist5 + eda}
cov = w["cardiac_coverage"].to_numpy()
print("\nWESAD ECG NaN rate:", w[ecg3].isna().mean().round(3).to_dict(), "| wrist HR NaN rate:", w["hr_mean"].isna().mean().round(3),
      "| coverage>=0.5:", (cov >= 0.5).mean().round(3), "| stress windows with wrist HR:", w.loc[yw == 1, "hr_mean"].notna().mean().round(3))
rows = []
for seed in range(20):
    train, test = grouped_split(w, seed)
    tr, te = np.flatnonzero(train), np.flatnonzero(test)
    for name, cols in sets.items():
        p = fit_predict(w[cols].to_numpy()[tr], yw[tr], w[cols].to_numpy()[te], N_JOBS)
        for sub, m in (("all", np.ones(len(te), bool)), ("cov>=0.5", cov[te] >= 0.5), ("cov<0.5", cov[te] < 0.5)):
            if len(set(yw[te][m])) < 2:
                continue
            rows.append({"features": name, "subset": sub, "split": seed, "n": int(m.sum()),
                         "ba": balanced_accuracy_score(yw[te][m], p[m] > 0.5), "auroc": roc_auc_score(yw[te][m], p[m])})
c = pd.DataFrame(rows)
c.to_csv(OUT / "wesad_ecg_ceiling_per_split.csv", index=False)
cs = c.groupby(["features", "subset"])[["ba", "auroc", "n"]].mean().round(3)
cs.to_csv(OUT / "wesad_ecg_ceiling_summary.csv")
print("\n=== Part C WESAD ceiling ===\n", cs.to_string())
print("\n=== Part A ===\n", warm.to_string())
