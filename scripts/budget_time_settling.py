"""Review items REV-8, REV-9, REV-15 (docs/review_simulation/jbhi_review.md).

1. REV-8: subject-bootstrap CIs for the Table V error budget (Shapley gains F1-F4 and residual), reusing the
   committed out-of-sample scores (contribution_probes/arousal/scores.csv). F4 (label-using responder filter)
   is also reported outside the decomposition: Shapley over F1-F3 only, then F4 on top. Removed fractions and
   error rates of removed/kept windows per filter.
2. REV-9: time-only classifier (minutes since recording start) under the position-matched negative rule, for
   every dataset, next to physiology; where matching is impossible, the counts that show why.
3. REV-15: settling control. EPM-E4 (film clips, no stressor, nothing to anticipate) skin temperature and tonic
   EDA from raw E4 recording start, same method as warmup_since_donning.py, against the stress datasets.
Filters, block structure and responder rule are copied from matched_arousal_probe.py; the time-probe task
from time_in_session_probe.py; the raw-E4 trajectory from warmup_since_donning.py (those scripts run at import).
"""
import argparse
import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from extract_features import load_headerless_e4  # noqa: E402
from leave_one_dataset_out import EXERCISE, NON_STRESS, fit_predict, normalise  # noqa: E402
from relabel_windows import EPM_RAW, PHYSIONET_DIR  # noqa: E402
from run_jbhi_experiments import META, grouped_split  # noqa: E402

cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", REPO)),
                 help="folder holding the raw datasets (default: $DATA_ROOT, else this repo)")
cli.add_argument("--input", type=Path, default=None,
                 help="feature table (default: <data-root>/data/processed/combined/harmonized_windows_v2.csv)")
cli.add_argument("--boot", type=int, default=1000, help="subject-bootstrap draws for REV-8")
args = cli.parse_args()
DATA_ROOT = args.data_root
INPUT = args.input or DATA_ROOT / "data/processed/combined/harmonized_windows_v2.csv"
OUT = REPO / "outputs" / "tables" / "jbhi_v2" / "budget_time_settling"
OUT.mkdir(parents=True, exist_ok=True)
SCORES = REPO / "outputs" / "tables" / "jbhi_v2" / "contribution_probes" / "arousal" / "scores.csv"
NJ = 8
rng = np.random.default_rng(0)
raw = pd.read_csv(INPUT)

# =============================== 1. REV-8 error budget CIs ===============================
sc = pd.read_csv(SCORES)
assert len(sc) == len(raw) and (sc.window_id.to_numpy() == raw.window_id.to_numpy()).all(), "scores.csv does not match --input"
lab = raw["harmonized_label"]
ds_arr, subj = raw["dataset"].to_numpy(), raw["subject_uid"].to_numpy()
core = lab.isin(NON_STRESS + ["Stress"]).to_numpy()
y = (lab == "Stress").to_numpy(int)

# block structure (copied from matched_arousal_probe.py)
d = raw[["dataset", "subject_uid", "harmonized_label", "timestamp_start", "timestamp_end"]].copy()
d["i"] = np.arange(len(d))
d = d[~d.harmonized_label.isin(EXERCISE)].sort_values(["subject_uid", "timestamp_start"])
d["blk"] = (d.groupby("subject_uid").apply(lambda g: (g.harmonized_label != g.harmonized_label.shift()) | (g.timestamp_start - g.timestamp_end.shift() > 60)).reset_index(level=0, drop=True)).cumsum()
d["min_since_onset"] = (d.timestamp_start - d.groupby("blk").timestamp_start.transform("min")) / 60
d["min_since_offset"] = np.nan
for s, g in d.groupby("subject_uid"):
    offs = g[g.harmonized_label == "Stress"].groupby("blk").timestamp_end.max().sort_values().to_numpy()
    if len(offs) == 0:
        continue
    ts = g.timestamp_start.to_numpy()
    j = np.searchsorted(offs, ts, side="right") - 1
    d.loc[g.index, "min_since_offset"] = np.where(j >= 0, (ts - offs[np.clip(j, 0, None)]) / 60, np.nan)
d.loc[d.dataset == "UBFC-Phys", "min_since_offset"] = np.nan
mso, msoff = np.full(len(raw), np.nan), np.full(len(raw), np.nan)
mso[d.i], msoff[d.i] = d.min_since_onset.to_numpy(), d.min_since_offset.to_numpy()
resp = {}
for s, g in raw[core].groupby("subject_uid"):
    b, st = g[g.harmonized_label == "Baseline"], g[g.harmonized_label == "Stress"]
    if len(b) < 2 or len(st) == 0:
        resp[s] = np.nan
        continue
    resp[s] = any(np.isfinite(sd := b[c].std()) and sd > 0 and (st[c].mean() - b[c].mean()) > 0.5 * sd for c in ("hr_mean", "eda_tonic_mean"))
is_resp = pd.Series(subj).map(resp).to_numpy()
FILTERS = {
    "F1_recovery": ~((y == 0) & (msoff < 5)),
    "F2_sensor": raw.cardiac_coverage.to_numpy() >= 0.5,
    "F3_onset": ~((y == 1) & (mso < 1)),
    "F4_responder": is_resp == True,  # noqa: E712  (NaN -> excluded)
}
NAMES = list(FILTERS)
FM = np.column_stack([FILTERS[f] for f in NAMES])


def fast_ba(yy, pred):
    if yy.all() or not yy.any():
        return np.nan
    return 0.5 * (pred[yy].mean() + (~pred[~yy]).mean())


def shapley(ba_set, names):
    n, out = len(names), {}
    for f in names:
        others, sh = [o for o in names if o != f], 0.0
        for k in range(n):
            for S in itertools.combinations(others, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                with_f = tuple(x for x in names if x in S + (f,))
                sh += w * (ba_set[with_f] - ba_set[S])
        out[f] = sh
    return out


def budget(yy, pred, fm):
    """All 16 subset BAs -> 4-filter Shapley (Table V), 3-filter Shapley + F4 on top, residuals."""
    ba_set = {}
    for k in range(5):
        for S in itertools.combinations(NAMES, k):
            m = fm[:, [NAMES.index(f) for f in S]].all(axis=1) if S else np.ones(len(yy), bool)
            ba_set[S] = fast_ba(yy[m], pred[m])
    r = {"ba_all": ba_set[()]}
    r.update({f"{f}_shap4": v for f, v in shapley(ba_set, NAMES).items()})
    r["resid4"] = 1 - ba_set[tuple(NAMES)]
    r.update({f"{f}_shap3": v for f, v in shapley(ba_set, NAMES[:3]).items()})
    r["F1_F3_total"] = ba_set[tuple(NAMES[:3])] - ba_set[()]
    r["resid3"] = 1 - ba_set[tuple(NAMES[:3])]
    r["F4_single"] = ba_set[("F4_responder",)] - ba_set[()]
    r["F4_after_F1_F3"] = ba_set[tuple(NAMES)] - ba_set[tuple(NAMES[:3])]
    return r


CASES = [("PhysioNet", "within_LOSO"), ("PhysioNet", "external"), ("Stress-Predict", "within_LOSO"),
         ("Stress-Predict", "external"), ("WESAD", "external")]
ci_rows, rem_rows = [], []
for ds, score in CASES:
    p = sc["p_win" if score == "within_LOSO" else "p_ext"].to_numpy()
    base = np.flatnonzero(core & (ds_arr == ds) & np.isfinite(p))
    yy, pred, fm, sb = y[base].astype(bool), p[base] > 0.5, FM[base], subj[base]
    point = budget(yy, pred, fm)
    groups = [np.flatnonzero(sb == s) for s in np.unique(sb)]
    boots = []
    for _ in range(args.boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boots.append(budget(yy[idx], pred[idx], fm[idx]))
    boots = pd.DataFrame(boots)
    for k, v in point.items():
        ci_rows.append({"dataset": ds, "score": score, "term": k, "estimate": v,
                        "lo": np.nanpercentile(boots[k], 2.5), "hi": np.nanpercentile(boots[k], 97.5),
                        "n_boot_valid": int(boots[k].notna().sum())})
    # removed fractions and error rates
    err = pred != yy
    for fname, keep in [(f, fm[:, i]) for i, f in enumerate(NAMES)] + [("F1-F3", fm[:, :3].all(1)), ("F1-F4", fm.all(1))]:
        rem = ~keep
        rem_rows.append({"dataset": ds, "score": score, "filter": fname, "n_windows": len(yy),
                         "frac_windows_removed": rem.mean(), "frac_stress_removed": rem[yy].mean(), "frac_nonstress_removed": rem[~yy].mean(),
                         "n_subjects": len(groups), "frac_subjects_fully_removed": np.mean([rem[g].all() for g in groups]),
                         "err_removed": err[rem].mean() if rem.any() else np.nan, "err_kept": err[keep].mean(), "err_all": err.mean()})
ci = pd.DataFrame(ci_rows).round(4)
ci.to_csv(OUT / "rev8_budget_bootstrap_ci.csv", index=False)
rem = pd.DataFrame(rem_rows).round(4)
rem.to_csv(OUT / "rev8_filter_removed_fractions.csv", index=False)
print(f"=== REV-8 ({args.boot} subject-bootstrap draws) ===")
print(ci.to_string(index=False))
print(rem.to_string(index=False))

# =============================== 2. REV-9 position-matched rule ===============================
tdf = raw[~raw["harmonized_label"].isin(EXERCISE)].copy()
tdf["minutes"] = (tdf["timestamp_start"] - tdf.groupby("subject_uid")["timestamp_start"].transform("min")) / 60
physiology = [c for c in tdf.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
HR_EDA = [c for c in physiology if not c.startswith("temp_")]
task = tdf[tdf["harmonized_label"].isin(NON_STRESS + ["Stress"])].copy()
task[physiology] = normalise(task, physiology, "session")
task = task[task.groupby("dataset")["harmonized_label"].transform(lambda s: (s == "Stress").any())].reset_index(drop=True)
ty = (task["harmonized_label"] == "Stress").to_numpy(int)
mins, tsub = task["minutes"].to_numpy(), task["subject_uid"].to_numpy()
x_time, x_std = task[["minutes"]].to_numpy(), task[HR_EDA].to_numpy()
# rule A (paper): non-stress kept if inside the dataset's 5th-95th pct of stress minutes
# rule B (stricter): non-stress kept if inside the same subject's 5th-95th pct of stress minutes
match_ds, match_subj = np.zeros(len(task), bool), np.zeros(len(task), bool)
rng_ds = {}
for dname in task["dataset"].unique():
    m = (task["dataset"] == dname).to_numpy()
    rng_ds[dname] = np.percentile(mins[m & (ty == 1)], [5, 95])
    match_ds[m] = (ty[m] == 1) | ((mins[m] >= rng_ds[dname][0]) & (mins[m] <= rng_ds[dname][1]))
for s in np.unique(tsub):
    m = tsub == s
    if (m & (ty == 1)).any():
        lo, hi = np.percentile(mins[m & (ty == 1)], [5, 95])
        match_subj[m] = (ty[m] == 1) | ((mins[m] >= lo) & (mins[m] <= hi))
    else:
        match_subj[m] = False
rows = []
for dname in sorted(task["dataset"].unique()):
    tgt, oth = np.flatnonzero(task["dataset"] == dname), np.flatnonzero(task["dataset"] != dname)
    for seed in range(20):
        train, test = grouped_split(task.iloc[tgt], seed)
        te = tgt[test]
        if len(set(ty[te])) < 2:
            continue
        preds = {"time_only_within": fit_predict(x_time[tgt[train]], ty[tgt[train]], x_time[te], NJ),
                 "phys_within": fit_predict(x_std[tgt[train]], ty[tgt[train]], x_std[te], NJ),
                 "phys_external": fit_predict(x_std[oth], ty[oth], x_std[te], NJ)}
        for name, p in preds.items():
            row = {"dataset": dname, "model": name, "split": seed, "ba_unmatched": balanced_accuracy_score(ty[te], p > 0.5)}
            for rule, mm in (("dataset_range", match_ds), ("subject_range", match_subj)):
                k = mm[te]
                ok = dname != "UBFC-Phys" and len(set(ty[te][k])) == 2 and (k & (ty[te] == 0)).sum() >= 5
                row[f"ba_{rule}"] = balanced_accuracy_score(ty[te][k], p[k] > 0.5) if ok else np.nan
            rows.append(row)
ps = pd.DataFrame(rows)
ps.to_csv(OUT / "rev9_position_matched_per_split.csv", index=False)
summ = ps.groupby(["dataset", "model"]).agg(ba_unmatched=("ba_unmatched", "mean"), ba_dataset_range=("ba_dataset_range", "mean"),
                                            splits_dataset_range=("ba_dataset_range", "count"), ba_subject_range=("ba_subject_range", "mean"),
                                            splits_subject_range=("ba_subject_range", "count")).round(3)
summ.to_csv(OUT / "rev9_position_matched_summary.csv")
feas = []
for dname in sorted(task["dataset"].unique()):
    m = (task["dataset"] == dname).to_numpy()
    ns = m & (ty == 0)
    feas.append({"dataset": dname, "clock": "none (per-task files restart at 0)" if dname == "UBFC-Phys" else "recording start",
                 "stress_min_p5": rng_ds[dname][0], "stress_min_p95": rng_ds[dname][1],
                 "nonstress_min_median": np.median(mins[ns]), "n_nonstress": int(ns.sum()), "n_subjects": len(np.unique(tsub[m])),
                 "n_nonstress_dataset_range": int((ns & match_ds).sum()), "n_subj_with_matched_nonstress_dataset_range": len(np.unique(tsub[ns & match_ds])),
                 "n_nonstress_subject_range": int((ns & match_subj).sum()), "n_subj_with_matched_nonstress_subject_range": len(np.unique(tsub[ns & match_subj]))})
feas = pd.DataFrame(feas).round(2)
feas.to_csv(OUT / "rev9_matching_feasibility.csv", index=False)
print("\n=== REV-9 ===")
print(summ.to_string())
print(feas.to_string(index=False))

# =============================== 3. REV-15 settling control ===============================
FS, HORIZON = 4, 20  # every EPM-E4 recording is >= 23 min; same 20-min horizon for all datasets
TEMP_OK = (20.0, 40.0)  # EPM-E4 has invalid TEMP runs (e.g. 238.6 C, 99-143 C); masked in every dataset alike


def raw_e4_tonic(folder: Path, headerless: bool = False):
    """As in warmup_since_donning.py."""
    if headerless:
        temp, eda = (load_headerless_e4(folder / f"{k}.csv", FS).ravel() for k in ("TEMP", "EDA"))
    else:
        temp = pd.read_csv(folder / "TEMP.csv", skiprows=2, header=None)[0].to_numpy(float)
        eda = pd.read_csv(folder / "EDA.csv", skiprows=2, header=None)[0].to_numpy(float)
    return temp, pd.Series(eda).rolling(60 * FS, center=True, min_periods=FS).median().to_numpy()


def per_minute(temp, tonic):
    n = min(len(temp), len(tonic), HORIZON * 60 * FS)
    temp = np.where((temp >= TEMP_OK[0]) & (temp <= TEMP_OK[1]), temp, np.nan)
    f = pd.DataFrame({"temp": temp[:n], "tonic": tonic[:n], "bin": np.arange(n) // (60 * FS)})
    return f.groupby("bin")[["temp", "tonic"]].mean()


def slope(s, lo, hi):
    seg = s.loc[lo:hi - 1].dropna()
    return np.polyfit(seg.index, seg.to_numpy(), 1)[0] if len(seg) > 2 else np.nan


epm_ids = set(raw.loc[raw.dataset == "EPM-E4", "subject_id"].astype(str))
sources = {
    "EPM-E4": [(p.name, p / "empatica", False) for p in sorted((DATA_ROOT / EPM_RAW).iterdir())
               if p.name in epm_ids and (p / "empatica" / "TEMP.csv").exists()],
    "WESAD": [(p.parent.name, p, False) for p in sorted((DATA_ROOT / "WESAD").glob("S*/S*_E4_Data"))],
    "PhysioNet": [(p.name, p, False) for p in sorted((DATA_ROOT / PHYSIONET_DIR / "STRESS").iterdir()) if (p / "TEMP.csv").exists()],
    "Stress-Predict": [(p.name, p, False) for p in sorted((DATA_ROOT / "Stress-Predict/Raw_data").glob("S*")) if (p / "TEMP.csv").exists()],
    "Campanella2024": [(p.name, p, True) for p in sorted((DATA_ROOT / "Campanella2024").glob("subject_*"))],
}
committed = pd.read_csv(REPO / "outputs/tables/jbhi_v2/contribution_probes/wesad_warmup/warmup_since_donning.csv").set_index("dataset")
set_rows, curves = [], []
for name, items in sources.items():
    pm, first_min, masked = {}, [], []
    for sid, folder, headerless in items:
        try:
            temp, tonic = raw_e4_tonic(folder, headerless)
        except Exception as e:  # noqa: BLE001
            print(f"skip {name} {sid}: {e}")
            continue
        if len(temp) < HORIZON * 60 * FS:
            continue  # full horizon only, so the pooled curve has a fixed subject set
        t20 = temp[:HORIZON * 60 * FS]
        masked.append(np.mean((t20 < TEMP_OK[0]) | (t20 > TEMP_OK[1])))
        m = per_minute(temp, tonic)
        m0 = t20[:60 * FS].mean()  # unmasked: minute-0 readings above 40 C are part of what is being described
        first_min.append((m0 - m.temp.loc[1:4].mean(), m0))
        pm[sid] = m - m.loc[1:].mean()  # subject-centred over minutes 1..HORIZON-1
    pooled = pd.concat(pm.values()).groupby(level=0).mean()
    curves.append(pooled.assign(dataset=name, n=len(pm)).reset_index())
    row = {"dataset": name, "n_subjects": len(pm), "temp_min0_raw_minus_min1_4_median": np.nanmedian([a for a, _ in first_min]),
           "frac_subj_min0_2C_above_min1_4": np.mean([a > 2 for a, _ in first_min]),
           "frac_subj_min0_above_40C": np.mean([b > 40 for _, b in first_min]),
           "frac_temp_samples_masked": np.mean(masked)}
    for sig in ("temp", "tonic"):
        for lo, hi in ((1, 10), (10, 20)):
            per_subj = np.array([slope(v[sig], lo, hi) for v in pm.values()])
            per_subj = per_subj[np.isfinite(per_subj)]
            bs = [rng.choice(per_subj, len(per_subj)).mean() for _ in range(1000)]
            row[f"{sig}_slope_{lo}_{hi}"] = slope(pooled[sig], lo, hi)
            row[f"{sig}_slope_{lo}_{hi}_lo"], row[f"{sig}_slope_{lo}_{hi}_hi"] = np.percentile(bs, [2.5, 97.5])
            row[f"{sig}_frac_subj_rising_{lo}_{hi}"] = (per_subj > 0).mean()
        row[f"{sig}_change_min15_19_minus_1_4"] = pooled[sig].loc[15:19].mean() - pooled[sig].loc[1:4].mean()
    for c in ("temp_slope_1_10", "temp_slope_10_20", "tonic_slope_1_10", "tonic_slope_10_20"):
        row[f"committed40_{c}"] = committed.loc[name, c] if name in committed.index else np.nan
    set_rows.append(row)
st = pd.DataFrame(set_rows).round(4)
st.to_csv(OUT / "rev15_settling_control.csv", index=False)
pd.concat(curves).round(4).to_csv(OUT / "rev15_settling_curves_1min.csv", index=False)
epm = raw[raw.dataset == "EPM-E4"]
print("\n=== REV-15 ===")
print(st.T.to_string())
print("EPM-E4 first labelled clip window, minutes from recording start (median over subjects):",
      round(float(epm.groupby("subject_uid").timestamp_start.min().median() / 60), 2))
