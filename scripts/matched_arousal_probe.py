"""Probe 3: matched-arousal separability (A), ceiling error budget (B), onset/offset curves (C). LODO defaults."""
import argparse
import os, sys, itertools, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from leave_one_dataset_out import normalise, fit_predict, NON_STRESS, EXERCISE
from run_jbhi_experiments import META, grouped_split

OUT = REPO / "outputs" / "tables" / "jbhi_v2" / "contribution_probes" / "arousal"
OUT.mkdir(parents=True, exist_ok=True)
_cli = argparse.ArgumentParser(description=__doc__)
_cli.add_argument("--input", type=Path, help="feature table (default: $HARMONIZED_CSV, else data/processed/combined/harmonized_windows_v2.csv)",
                  default=Path(os.environ.get("HARMONIZED_CSV", REPO / "data/processed/combined/harmonized_windows_v2.csv")))
DATA = _cli.parse_args().input
raw = pd.read_csv(DATA)
phys = [c for c in raw.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
cols = [c for c in phys if not c.startswith("temp_")]
df = raw.copy()
df[phys] = normalise(raw, phys, "session")
X = df[cols].to_numpy()
lab = df["harmonized_label"]
ds_arr, subj = df["dataset"].to_numpy(), df["subject_uid"].to_numpy()
core = lab.isin(NON_STRESS + ["Stress"]).to_numpy()
y = (lab == "Stress").to_numpy(int)
rng = np.random.default_rng(0)
NJ = 8
STRESS_DS = ["WESAD", "PhysioNet", "Stress-Predict", "UBFC-Phys", "Campanella2024"]

# ---------- scores ----------
# external: dataset entirely unseen (EPM-E4: trained on all five)
p_ext = np.full(len(df), np.nan)
for ds in df["dataset"].unique():
    tr = core & (ds_arr != ds); te = ds_arr == ds
    p_ext[te] = fit_predict(X[tr], y[tr], X[te], NJ)
# within: LOSO inside each stress dataset, scoring every class of the held-out subject
p_win = np.full(len(df), np.nan)
for ds in STRESS_DS:
    in_ds = ds_arr == ds
    for s in np.unique(subj[in_ds]):
        me = subj == s
        p_win[me] = fit_predict(X[in_ds & core & ~me], y[in_ds & core & ~me], X[me], NJ)
pd.DataFrame({"window_id": df.window_id, "dataset": ds_arr, "subject": subj, "label": lab, "p_ext": p_ext, "p_win": p_win}).to_csv(OUT / "scores.csv", index=False)

# ---------- Part A ----------
PROBES = {"exercise": ("PhysioNet", ["Aerobic", "Anaerobic"]), "Hyperventilation": ("Stress-Predict", ["Hyperventilation"]),
          "UBFC Control task": ("UBFC-Phys", ["Control task"]), "Campanella Manual task": ("Campanella2024", ["Manual task"]),
          "EPM Fear": ("EPM-E4", ["Fear"]), "EPM Anger": ("EPM-E4", ["Anger"]), "Baseline/Rest (ref)": (None, ["Baseline", "Rest"])}
# cond 3: all five datasets + probed class as negative, held-out subjects over 20 grouped splits, mean per window
p_c3 = {}
for name, (pds, labels) in PROBES.items():
    if pds is None:
        continue
    acc, cnt = np.zeros(len(df)), np.zeros(len(df))
    is_probe = lab.isin(labels).to_numpy()
    for seed in range(20):
        trm, tem = grouped_split(df, seed)
        tr = (trm & core) | (trm & is_probe)
        y3 = y.copy(); y3[is_probe] = 0
        te = tem & (core | is_probe)
        acc[te] += fit_predict(X[tr], y3[tr], X[te], NJ); cnt[te] += 1
    p_c3[name] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)

hr_z, eda_z = df["hr_mean"].to_numpy(), df["eda_tonic_mean"].to_numpy()
IDX = {"hr+eda": np.where(np.isnan(hr_z), eda_z, (hr_z + eda_z) / 2), "eda_only": eda_z, "hr_only": hr_z}
n_hr_nan = int(np.isnan(hr_z).sum())

def binned_auc(a, s, yy, sub, nb=5, n_boot=500):
    """Quintile bins on pooled pair; within-bin AUROC weighted by bin size; subject bootstrap CI (bins fixed)."""
    ok = np.isfinite(a) & np.isfinite(s)
    a, s, yy, sub = a[ok], s[ok], yy[ok], sub[ok]
    if len(np.unique(yy)) < 2 or len(a) < 20:
        return dict(n=len(a), auc_pooled=np.nan, auc_binned=np.nan, lo=np.nan, hi=np.nan, bins="")
    edges = np.unique(np.quantile(a, np.linspace(0, 1, nb + 1)))
    b = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, len(edges) - 2)
    def wauc(m):
        num = den = 0.0; per = []
        for k in range(len(edges) - 1):
            mm = m & (b == k)
            if len(np.unique(yy[mm])) < 2:
                per.append(np.nan); continue
            v = roc_auc_score(yy[mm], s[mm]); per.append(v); num += v * mm.sum(); den += mm.sum()
        return (num / den if den else np.nan), per
    full, per = wauc(np.ones(len(a), bool))
    subs = np.unique(sub); bs = []
    for _ in range(n_boot):
        pick = rng.choice(subs, len(subs), replace=True)
        idx = np.concatenate([np.flatnonzero(sub == p) for p in pick])
        m = np.zeros(len(a), bool)  # weight by multiplicity via index expansion
        aa, ss, yb, bb = a[idx], s[idx], yy[idx], b[idx]
        num = den = 0.0
        for k in range(len(edges) - 1):
            mm = bb == k
            if len(np.unique(yb[mm])) < 2: continue
            v = roc_auc_score(yb[mm], ss[mm]); num += v * mm.sum(); den += mm.sum()
        bs.append(num / den if den else np.nan)
    return dict(n=len(a), auc_pooled=roc_auc_score(yy, s), auc_binned=full, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5),
                bins=" ".join(f"{v:.2f}" if np.isfinite(v) else "na" for v in per))

def class_coef(a, s, yy, sub, n_boot=500):
    """logit(score) ~ arousal + class(stress=1); subject-cluster bootstrap CI on class coefficient."""
    ok = np.isfinite(a) & np.isfinite(s)
    a, s, yy, sub = a[ok], np.clip(s[ok], 1e-4, 1 - 1e-4), yy[ok], sub[ok]
    lg = np.log(s / (1 - s))
    def fit(i):
        A = np.c_[np.ones(len(i)), a[i], yy[i]]
        return np.linalg.lstsq(A, lg[i], rcond=None)[0]
    beta = fit(np.arange(len(a)))
    subs = np.unique(sub); bs = []
    for _ in range(n_boot):
        pick = rng.choice(subs, len(subs), replace=True)
        i = np.concatenate([np.flatnonzero(sub == p) for p in pick])
        if len(np.unique(yy[i])) < 2: continue
        bs.append(fit(i))
    bs = np.array(bs)
    return dict(b_arousal=beta[1], b_class=beta[2], b_class_lo=np.percentile(bs[:, 2], 2.5), b_class_hi=np.percentile(bs[:, 2], 97.5))

A_rows = []
stress_m = y == 1
for name, (pds, labels) in PROBES.items():
    probe_m = lab.isin(labels).to_numpy() & (ds_arr == pds if pds else core)
    for stress_scope in (["same_dataset"] if pds in STRESS_DS else []) + ["pooled_stress"]:
        sm = stress_m & ((ds_arr == pds) if stress_scope == "same_dataset" else True)
        pair = sm | probe_m
        for score_name, sc in [("external", p_ext), ("cond3_added_negative", p_c3.get(name, p_win)), ("within_LOSO", p_win)]:
            if score_name == "cond3_added_negative" and pds is None: continue
            if score_name == "within_LOSO" and pds == "EPM-E4": continue
            for idx_name, a in IDX.items():
                r = binned_auc(a[pair], sc[pair], y[pair], subj[pair])
                r.update(class_coef(a[pair], sc[pair], y[pair], subj[pair]) if np.isfinite(r["auc_binned"]) else {})
                r.update(probe=name, stress_scope=stress_scope, score=score_name, index=idx_name,
                         n_stress=int(sm.sum()), n_probe=int(probe_m.sum()),
                         arousal_stress_med=np.nanmedian(a[sm]), arousal_stress_iqr=np.nanpercentile(a[sm], 75) - np.nanpercentile(a[sm], 25),
                         arousal_probe_med=np.nanmedian(a[probe_m]), arousal_probe_iqr=np.nanpercentile(a[probe_m], 75) - np.nanpercentile(a[probe_m], 25))
                A_rows.append(r)
A = pd.DataFrame(A_rows).round(3)
A.to_csv(OUT / "partA_matched_arousal.csv", index=False)
print(f"hr_mean NaN in {n_hr_nan} windows: index falls back to EDA alone there")
print("=== PART A: within-quintile AUROC (weighted), hr+eda index ===")
print(A[A["index"] == "hr+eda"][["probe", "stress_scope", "score", "n_stress", "n_probe", "arousal_stress_med", "arousal_probe_med",
                                  "auc_pooled", "auc_binned", "lo", "hi", "b_class", "b_class_lo", "b_class_hi", "bins"]].to_string(index=False))
print("\n=== PART A: by index, external score, same-dataset stress where available ===")
sel = A[(A.score == "external") & (A.stress_scope == A.groupby("probe").stress_scope.transform("first"))]
print(sel.pivot_table(index="probe", columns="index", values="auc_binned").round(3).to_string())

# ---------- block structure (per subject, sorted) ----------
d = df[["dataset", "subject_uid", "harmonized_label", "original_label", "timestamp_start", "timestamp_end", "cardiac_coverage"]].copy()
d["i"] = np.arange(len(d))
d = d[~d.harmonized_label.isin(EXERCISE)].sort_values(["subject_uid", "timestamp_start"])
d["blk"] = (d.groupby("subject_uid").apply(lambda g: (g.harmonized_label != g.harmonized_label.shift()) | (g.timestamp_start - g.timestamp_end.shift() > 60)).reset_index(level=0, drop=True)).cumsum()
onset = d.groupby("blk").timestamp_start.transform("min")
d["min_since_onset"] = (d.timestamp_start - onset) / 60
d["blk_half"] = (d.timestamp_start >= d.groupby("blk").timestamp_start.transform("median")).astype(int)  # 1 = later half of block
# minutes since last stressor offset, for non-stress windows
d["min_since_offset"] = np.nan
for s, g in d.groupby("subject_uid"):
    offs = g[g.harmonized_label == "Stress"].groupby("blk").timestamp_end.max().sort_values().to_numpy()
    if len(offs) == 0: continue
    ts = g.timestamp_start.to_numpy()
    j = np.searchsorted(offs, ts, side="right") - 1
    v = np.where(j >= 0, (ts - offs[np.clip(j, 0, None)]) / 60, np.nan)
    d.loc[g.index, "min_since_offset"] = v
d.loc[d.dataset == "UBFC-Phys", "min_since_offset"] = np.nan  # timestamps restart per task file
full = pd.DataFrame(index=np.arange(len(df)))
for c in ["min_since_onset", "min_since_offset", "blk_half"]:
    full[c] = np.nan; full.loc[d.i, c] = d[c].to_numpy()
mso, msoff, half = full.min_since_onset.to_numpy(), full.min_since_offset.to_numpy(), full.blk_half.to_numpy()

# ---------- Part B ----------
# F4 responder: raw stressor-minus-baseline delta in hr_mean or eda_tonic_mean > 0.5 SD of that subject's baseline
resp = {}
for s, g in raw[core].groupby("subject_uid"):
    b, st = g[g.harmonized_label == "Baseline"], g[g.harmonized_label == "Stress"]
    if len(b) < 2 or len(st) == 0: resp[s] = np.nan; continue
    ok = False
    for c in ("hr_mean", "eda_tonic_mean"):
        sd = b[c].std()
        if np.isfinite(sd) and sd > 0 and (st[c].mean() - b[c].mean()) > 0.5 * sd: ok = True
    resp[s] = ok
is_resp = pd.Series(subj).map(resp).to_numpy()
FILTERS = {
    "F1_recovery": ~((y == 0) & (msoff < 5)),
    "F2_sensor": df.cardiac_coverage.to_numpy() >= 0.5,
    "F3_onset": ~((y == 1) & (mso < 1)),
    "F4_responder": is_resp == True,
}
F1b = ~((y == 0) & (lab == "Rest").to_numpy() & (half == 0))  # PhysioNet second-half rests

def ba(m, p):
    m = m & np.isfinite(p)
    return balanced_accuracy_score(y[m], p[m] > 0.5) if len(np.unique(y[m])) == 2 else np.nan

B_rows = []
for ds in STRESS_DS:
    base = core & (ds_arr == ds)
    for score_name, p in (("within_LOSO", p_win), ("external", p_ext)):
        names = list(FILTERS)
        ba_set = {}
        for k in range(5):
            for S in itertools.combinations(names, k):
                m = base.copy()
                for f in S: m &= FILTERS[f]
                ba_set[S] = ba(m, p)
        row = dict(dataset=ds, score=score_name, n_windows=int(base.sum()), n_subjects=len(np.unique(subj[base])),
                   n_responders=int(np.unique(subj[base & (is_resp == True)]).size), ba_all=ba_set[()], ba_all_filters=ba_set[tuple(names)])
        for f in names:
            row[f"{f}_single"] = ba_set[(f,)]
            row[f"{f}_gain"] = ba_set[(f,)] - ba_set[()]
            # bootstrap CI of single-filter gain over subjects
            subs = np.unique(subj[base]); bs = []
            for _ in range(300):
                pick = rng.choice(subs, len(subs), replace=True)
                idx = np.concatenate([np.flatnonzero((subj == p_) & base) for p_ in pick])
                yy, pp = y[idx], p[idx]; fm = FILTERS[f][idx]; okp = np.isfinite(pp)
                if len(np.unique(yy[okp & fm])) < 2 or len(np.unique(yy[okp])) < 2: continue
                bs.append(balanced_accuracy_score(yy[okp & fm], pp[okp & fm] > 0.5) - balanced_accuracy_score(yy[okp], pp[okp] > 0.5))
            row[f"{f}_gain_lo"], row[f"{f}_gain_hi"] = (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if bs else (np.nan, np.nan)
            # Shapley: average marginal over orders = average over subsets not containing f, weighted
            sh, n = 0.0, len(names)
            others = [o for o in names if o != f]
            for k in range(len(others) + 1):
                for S in itertools.combinations(others, k):
                    w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                    a, b_ = ba_set[tuple(x for x in names if x in S + (f,))], ba_set[tuple(x for x in names if x in S)]
                    if np.isfinite(a) and np.isfinite(b_): sh += w * (a - b_)
            row[f"{f}_shapley"] = sh
        if ds == "PhysioNet":
            row["F1b_second_half_rest_single"] = ba(base & F1b, p); row["F1b_gain"] = row["F1b_second_half_rest_single"] - ba_set[()]
        row["n_after_all_filters"] = int((base & np.all([FILTERS[f] for f in names], axis=0)).sum())
        B_rows.append(row)
B = pd.DataFrame(B_rows).round(3)
B.to_csv(OUT / "partB_error_budget.csv", index=False)
print("\n=== PART B: error budget (BA), test-side filters, no retraining. F4 is diagnostic only (uses labels). ===")
print(B[["dataset", "score", "n_subjects", "n_responders", "ba_all"] + [f"{f}_single" for f in FILTERS] + ["ba_all_filters", "n_after_all_filters"]].to_string(index=False))
print(B[["dataset", "score"] + [c for c in B.columns if c.endswith(("_gain", "_gain_lo", "_gain_hi", "_shapley"))]].to_string(index=False))

# ---------- Part C ----------
C_rows = []
on_bins, off_bins = [0, 1, 2, 3, np.inf], [0, 1, 2, 5, 10, np.inf]
for ds in STRESS_DS:
    for score_name, p in (("within_LOSO", p_win), ("external", p_ext)):
        for lo, hi in zip(on_bins[:-1], on_bins[1:]):
            m = core & (ds_arr == ds) & (y == 1) & (mso >= lo) & (mso < hi) & np.isfinite(p)
            C_rows.append(dict(dataset=ds, score=score_name, curve="stress_recall_vs_min_since_onset", bin=f"{lo}-{hi}", n=int(m.sum()), n_subj=len(np.unique(subj[m])), rate=(p[m] > 0.5).mean() if m.any() else np.nan))
        for lo, hi in zip(off_bins[:-1], off_bins[1:]):
            m = core & (ds_arr == ds) & (y == 0) & (msoff >= lo) & (msoff < hi) & np.isfinite(p)
            C_rows.append(dict(dataset=ds, score=score_name, curve="false_stress_vs_min_since_offset", bin=f"{lo}-{hi}", n=int(m.sum()), n_subj=len(np.unique(subj[m])), rate=(p[m] > 0.5).mean() if m.any() else np.nan))
        m = core & (ds_arr == ds) & (y == 0) & np.isnan(msoff) & np.isfinite(p)
        C_rows.append(dict(dataset=ds, score=score_name, curve="false_stress_vs_min_since_offset", bin="no prior stressor", n=int(m.sum()), n_subj=len(np.unique(subj[m])), rate=(p[m] > 0.5).mean() if m.any() else np.nan))
C = pd.DataFrame(C_rows).round(3)
C.to_csv(OUT / "partC_onset_offset_curves.csv", index=False)
print("\n=== PART C ===")
print(C.pivot_table(index=["curve", "dataset", "score"], columns="bin", values="rate", sort=False).to_string())
print(C.pivot_table(index=["curve", "dataset"], columns="bin", values="n", aggfunc="first", sort=False).to_string())
