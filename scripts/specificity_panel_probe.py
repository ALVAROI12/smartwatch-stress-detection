"""Specificity panel (A) and subject-level miss consistency (B). Matches LODO defaults."""
import os
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from leave_one_dataset_out import normalise, fit_predict, NON_STRESS, EXERCISE
from run_jbhi_experiments import META, grouped_split

OUT = REPO / "outputs" / "tables" / "jbhi_v2" / "contribution_probes" / "specificity"
OUT.mkdir(parents=True, exist_ok=True)
# The feature table lives in the thesis worktree (see CLAUDE.md); override with HARMONIZED_CSV.
DATA = Path(os.environ.get("HARMONIZED_CSV", "/Users/octa/Projects/smartwatch-stress-detection/data/processed/combined/harmonized_windows_v2.csv"))
raw = pd.read_csv(DATA)
phys = [c for c in raw.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
cols = [c for c in phys if not c.startswith("temp_")]
df = raw.copy()
df[phys] = normalise(raw, phys, "session")
X = df[cols].to_numpy()
lab = df["harmonized_label"]
core = lab.isin(NON_STRESS + ["Stress"]).to_numpy()
y = (lab == "Stress").to_numpy(int)
rng = np.random.default_rng(0)
NJ = 8

PROBES = {  # name -> (dataset, labels)
    "Campanella Manual task": ("Campanella2024", ["Manual task"]),
    "UBFC ctrl Control task": ("UBFC-Phys", ["Control task"]),
    "Stress-Predict Hyperventilation": ("Stress-Predict", ["Hyperventilation"]),
    "PhysioNet Aerobic": ("PhysioNet", ["Aerobic"]),
    "PhysioNet Anaerobic": ("PhysioNet", ["Anaerobic"]),
    "EPM-E4 emotions": ("EPM-E4", ["Anger", "Fear", "Happiness", "Sadness"]),
}

def boot_rate(sub, hit, n=1000):
    """Pooled window rate, CI by subject bootstrap, per-subject median/IQR."""
    sub, hit = np.asarray(sub), np.asarray(hit, float)
    per = pd.Series(hit).groupby(sub).mean()
    subs = per.index.to_numpy()
    cnt = pd.Series(hit).groupby(sub).agg(["sum", "size"])
    bs = []
    for _ in range(n):
        pick = rng.choice(subs, len(subs), replace=True)
        c = cnt.loc[pick]
        bs.append(c["sum"].sum() / c["size"].sum())
    return dict(rate=hit.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                subj_median=per.median(), subj_q1=per.quantile(.25), subj_q3=per.quantile(.75),
                n_windows=len(hit), n_subjects=len(subs))

# ---------- Part A ----------
scores = []  # rows: condition, probe, dataset, label, subject, p
def add(cond, probe, idx, p):
    scores.append(pd.DataFrame({"condition": cond, "probe": probe, "dataset": df["dataset"].to_numpy()[idx],
                                "label": lab.to_numpy()[idx], "subject": df["subject_uid"].to_numpy()[idx], "p": p}))

# condition 1: external (whole dataset unseen)
for ds in df["dataset"].unique():
    tr = core & (df["dataset"] != ds).to_numpy()
    te = (df["dataset"] == ds).to_numpy()
    add("1_external", "-", np.flatnonzero(te), fit_predict(X[tr], y[tr], X[te], NJ))

# conditions 2 and 3: 20 grouped splits over every dataset's subjects
for seed in range(20):
    trm, tem = grouped_split(df, seed)
    tr = trm & core
    te = tem  # score all classes of held-out subjects
    add("2_all_absent", "-", np.flatnonzero(te), fit_predict(X[tr], y[tr], X[te], NJ))
    for name, (ds, labels) in PROBES.items():
        extra = trm & lab.isin(labels).to_numpy()
        tr3 = tr | extra
        y3 = y.copy(); y3[extra] = 0
        te3 = tem & (core | lab.isin(labels).to_numpy())
        add("3_added_negative", name, np.flatnonzero(te3), fit_predict(X[tr3], y3[tr3], X[te3], NJ))
S = pd.concat(scores, ignore_index=True)
S.to_csv(OUT / "scores_partA.csv", index=False)

rows = []
for (cond, probe), g in S.groupby(["condition", "probe"]):
    for (ds, l), h in g.groupby(["dataset", "label"]):
        if probe != "-" and not (l in PROBES[probe][1] or l in NON_STRESS + ["Stress"]):
            continue
        r = boot_rate(h["subject"], h["p"] > 0.5)
        r.update(condition=cond, probe=probe, dataset=ds, label=l, mean_score=h["p"].mean(),
                 kind="stress_recall" if l == "Stress" else ("false_stress_probed" if not (l in NON_STRESS) else "false_stress_core"))
        rows.append(r)
A = pd.DataFrame(rows)[["condition", "probe", "dataset", "label", "kind", "n_subjects", "n_windows", "rate", "lo", "hi",
                        "subj_median", "subj_q1", "subj_q3", "mean_score"]].round(3)
A.to_csv(OUT / "partA_specificity_panel.csv", index=False)

# compact panel: probed classes under the three conditions
def pick(cond, probe, ds, l):
    m = A[(A.condition == cond) & (A.probe == probe) & (A.dataset == ds) & (A.label == l)]
    return m.iloc[0] if len(m) else None
panel = []
for name, (ds, labels) in PROBES.items():
    for l in labels:
        r1, r2, r3 = pick("1_external", "-", ds, l), pick("2_all_absent", "-", ds, l), pick("3_added_negative", name, ds, l)
        panel.append(dict(probe_class=f"{ds} {l}", n_subj=r1.n_subjects, n_win=r1.n_windows,
                          ext_rate=r1.rate, ext_ci=f"[{r1.lo},{r1.hi}]", ext_subj_med=r1.subj_median,
                          all_absent_rate=r2.rate, all_absent_ci=f"[{r2.lo},{r2.hi}]", all_absent_subj_med=r2.subj_median,
                          added_neg_rate=r3.rate, added_neg_ci=f"[{r3.lo},{r3.hi}]", added_neg_subj_med=r3.subj_median))
P = pd.DataFrame(panel); P.to_csv(OUT / "partA_panel_compact.csv", index=False)
print("=== PART A: probed classes, false-stress rate (p>0.5) ===\n", P.to_string(index=False))
ref = A[(A.probe == "-") & (A.kind != "false_stress_probed")].pivot_table(index=["dataset", "label"], columns="condition", values="rate")
print("\n=== PART A reference: core non-stress false-stress and Stress recall, conditions 1 & 2 ===\n", ref.round(3).to_string())
# does adding the probed negative cost stress recall in that dataset?
c3 = A[(A.condition == "3_added_negative") & (A.label == "Stress")].pivot_table(index="dataset", columns="probe", values="rate")
print("\n=== PART A: Stress recall per dataset when each probed class is added as negative (cond 3) ===\n", c3.round(3).to_string())

# ---------- Part B ----------
B_rows, per_subj = [], []
for ds, stressors in {"Stress-Predict": ["Stroop", "TSST interview"], "PhysioNet": ["Stroop", "TMCT"]}.items():
    in_ds = (df["dataset"] == ds).to_numpy()
    st = in_ds & (y == 1) & lab.eq("Stress").to_numpy() & df["original_label"].isin(stressors).to_numpy()
    idx = np.flatnonzero(st)
    p_within = np.full(len(df), np.nan)
    for s in df.loc[in_ds, "subject_uid"].unique():
        me = (df["subject_uid"] == s).to_numpy()
        tr = in_ds & core & ~me
        te = st & me
        if te.any():
            p_within[te] = fit_predict(X[tr], y[tr], X[te], NJ)
    tr = core & ~in_ds
    p_ext = np.full(len(df), np.nan); p_ext[st] = fit_predict(X[tr], y[tr], X[st], NJ)
    base = raw[in_ds & lab.eq("Baseline").to_numpy()].groupby("subject_uid")[["hr_mean", "hrv_rmssd", "eda_mean", "cardiac_coverage"]].mean().add_prefix("base_")
    stress_cov = raw[st].groupby("subject_uid")["cardiac_coverage"].mean().rename("stress_cardiac_coverage")
    t = df[st][["subject_uid", "original_label", "self_report_stress_delta"]].copy()
    t["hit_within"], t["hit_ext"] = (p_within[st] > 0.5).astype(int), (p_ext[st] > 0.5).astype(int)
    ps = t.groupby(["subject_uid", "original_label"]).agg(n=("hit_within", "size"), recall_within=("hit_within", "mean"),
                                                         recall_ext=("hit_ext", "mean"), sr_delta=("self_report_stress_delta", "mean")).reset_index()
    ps["dataset"] = ds
    ps = ps.join(base, on="subject_uid").join(stress_cov, on="subject_uid")
    per_subj.append(ps)
    wide = ps.pivot(index="subject_uid", columns="original_label", values=["recall_within", "recall_ext"])
    both = wide.dropna()
    def sp(a, b, n=1000):
        a, b = np.asarray(a, float), np.asarray(b, float)
        ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
        if len(a) < 5: return np.nan, np.nan, np.nan, len(a)
        r = spearmanr(a, b).correlation
        bs = [spearmanr(a[i], b[i]).correlation for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
        return r, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), len(a)
    s1, s2 = stressors
    for model in ("recall_within", "recall_ext"):
        r, lo, hi, n = sp(both[(model, s1)], both[(model, s2)])
        B_rows.append(dict(dataset=ds, question=f"{s1} vs {s2} per-subject recall", model=model, rho=r, lo=lo, hi=hi, n=n))
    for s in stressors:
        w = ps[ps.original_label == s]
        r, lo, hi, n = sp(w.recall_within, w.recall_ext)
        B_rows.append(dict(dataset=ds, question=f"within vs external recall ({s})", model="both", rho=r, lo=lo, hi=hi, n=n))
    pooled = t.groupby("subject_uid").agg(recall_within=("hit_within", "mean"), recall_ext=("hit_ext", "mean"), n=("hit_within", "size"),
                                          sr_delta=("self_report_stress_delta", "mean")).join(base).join(stress_cov)
    r, lo, hi, n = sp(pooled.recall_within, pooled.recall_ext)
    B_rows.append(dict(dataset=ds, question="within vs external recall (pooled stressors)", model="both", rho=r, lo=lo, hi=hi, n=n))
    for cov in ["base_hr_mean", "base_hrv_rmssd", "base_eda_mean", "base_cardiac_coverage", "stress_cardiac_coverage", "sr_delta"]:
        for model in ("recall_within", "recall_ext"):
            r, lo, hi, n = sp(pooled[model], pooled[cov])
            B_rows.append(dict(dataset=ds, question=f"recall vs {cov} (pooled)", model=model, rho=r, lo=lo, hi=hi, n=n))
    for model in ("hit_within", "hit_ext"):
        miss = t.assign(miss=1 - t[model]).groupby("subject_uid")["miss"].sum().sort_values(ascending=False)
        k = max(1, round(0.2 * len(miss)))
        share_win = t.groupby("subject_uid").size().loc[miss.index[:k]].sum() / len(t)
        B_rows.append(dict(dataset=ds, question=f"share of misses from worst 20% subjects (k={k}; their share of stress windows={share_win:.2f})",
                           model=model, rho=miss.iloc[:k].sum() / miss.sum(), lo=np.nan, hi=np.nan, n=len(miss)))
B = pd.DataFrame(B_rows).round(3); B.to_csv(OUT / "partB_miss_consistency.csv", index=False)
pd.concat(per_subj).round(3).to_csv(OUT / "partB_per_subject_recall.csv", index=False)
print("\n=== PART B ===\n", B.to_string(index=False))
PS = pd.concat(per_subj)
print("\n=== PART B: mean recall per stressor ===\n", PS.groupby(["dataset", "original_label"])[["recall_within", "recall_ext"]].mean().round(3).to_string())
