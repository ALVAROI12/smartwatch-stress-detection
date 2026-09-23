"""Hardening fixes 1, 2, 5 for the matched-arousal probe. Reuses committed scores.csv for external/within; recomputes cond3."""
import os, sys, itertools, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from leave_one_dataset_out import normalise, fit_predict, NON_STRESS, EXERCISE
from run_jbhi_experiments import META, grouped_split

OUT = REPO / "outputs" / "tables" / "jbhi_v2" / "contribution_probes" / "hardening"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get("HARMONIZED_CSV", "/Users/octa/Projects/smartwatch-stress-detection/data/processed/combined/harmonized_windows_v2.csv"))
raw = pd.read_csv(DATA)
phys = [c for c in raw.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
cols = [c for c in phys if not c.startswith("temp_")]
df = raw.copy(); df[phys] = normalise(raw, phys, "session")
lab = df["harmonized_label"]; ds_arr, subj = df["dataset"].to_numpy(), df["subject_uid"].to_numpy()
core = lab.isin(NON_STRESS + ["Stress"]).to_numpy(); y = (lab == "Stress").to_numpy(int)
rng = np.random.default_rng(0); NJ = 8
STRESS_DS = ["WESAD", "PhysioNet", "Stress-Predict", "UBFC-Phys", "Campanella2024"]
PROBES = {"exercise": ("PhysioNet", ["Aerobic", "Anaerobic"]), "Hyperventilation": ("Stress-Predict", ["Hyperventilation"]),
          "UBFC Control task": ("UBFC-Phys", ["Control task"]), "Campanella Manual task": ("Campanella2024", ["Manual task"]),
          "EPM Fear": ("EPM-E4", ["Fear"]), "EPM Anger": ("EPM-E4", ["Anger"]), "Baseline/Rest (ref)": (None, ["Baseline", "Rest"])}
sc_old = pd.read_csv(REPO / "outputs/tables/jbhi_v2/contribution_probes/arousal/scores.csv")
assert (sc_old.window_id.to_numpy() == df.window_id.to_numpy()).all()

def ext_scores(fcols):
    X = df[fcols].to_numpy(); p = np.full(len(df), np.nan)
    for ds in df["dataset"].unique():
        tr = core & (ds_arr != ds); te = ds_arr == ds
        p[te] = fit_predict(X[tr], y[tr], X[te], NJ)
    return p

def cond3_scores(fcols):
    X = df[fcols].to_numpy(); out = {}
    for name, (pds, labels) in PROBES.items():
        if pds is None: continue
        acc, cnt = np.zeros(len(df)), np.zeros(len(df)); is_probe = lab.isin(labels).to_numpy()
        for seed in range(20):
            trm, tem = grouped_split(df, seed)
            tr = (trm & core) | (trm & is_probe); y3 = y.copy(); y3[is_probe] = 0; te = tem & (core | is_probe)
            acc[te] += fit_predict(X[tr], y3[tr], X[te], NJ); cnt[te] += 1
        out[name] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return out

hr_z, eda_z = df["hr_mean"].to_numpy(), df["eda_tonic_mean"].to_numpy()
IDX = {"hr+eda": np.where(np.isnan(hr_z), eda_z, (hr_z + eda_z) / 2), "eda_only": eda_z, "hr_only": hr_z}

def bins_of(a, nb=5):
    edges = np.unique(np.quantile(a, np.linspace(0, 1, nb + 1)))
    return np.clip(np.searchsorted(edges, a, side="right") - 1, 0, len(edges) - 2), len(edges) - 1

def wauc(b, nbin, s, yy):
    num = den = 0.0
    for k in range(nbin):
        mm = b == k
        if len(np.unique(yy[mm])) < 2: continue
        num += roc_auc_score(yy[mm], s[mm]) * mm.sum(); den += mm.sum()
    return num / den if den else np.nan

def binned(a, s, yy, sub, n_boot=2000, n_perm=2000):
    """Within-quintile weighted AUROC, subject bootstrap CI, block permutation p (units = subject x bin x class cells)."""
    ok = np.isfinite(a) & np.isfinite(s); a, s, yy, sub = a[ok], s[ok], yy[ok], sub[ok]
    if len(np.unique(yy)) < 2 or len(a) < 20: return dict(n=len(a), auc=np.nan, lo=np.nan, hi=np.nan, p_raw=np.nan, boot=None)
    b, nbin = bins_of(a); full = wauc(b, nbin, s, yy)
    subs = np.unique(sub); bs = []
    for _ in range(n_boot):
        pick = rng.choice(subs, len(subs), replace=True)
        idx = np.concatenate([np.flatnonzero(sub == p) for p in pick])
        bs.append(wauc(b[idx], nbin, s[idx], yy[idx]))
    # permutation: within each bin, units are (subject, class) cells; permute cell labels across cells in that bin
    cell = pd.factorize(pd.Series(list(zip(sub, b, yy))))[0]
    cells = pd.DataFrame({"cell": cell, "b": b, "y": yy}).drop_duplicates("cell").set_index("cell").sort_index()
    null = []
    for _ in range(n_perm):
        yp_cell = cells.y.to_numpy().copy()
        for k in range(nbin):
            m = (cells.b == k).to_numpy(); yp_cell[m] = rng.permutation(yp_cell[m])
        null.append(wauc(b, nbin, s, yp_cell[cell]))
    null = np.array(null); null = null[np.isfinite(null)]
    p = (np.sum(np.abs(null - 0.5) >= abs(full - 0.5)) + 1) / (len(null) + 1)
    return dict(n=len(a), auc=full, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5), p_raw=p, boot=np.array(bs))

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.full(len(p), np.nan); m = np.isfinite(p).sum(); run = 0
    for r, i in enumerate(o):
        if not np.isfinite(p[i]): continue
        run = max(run, (m - r) * p[i]); out[i] = min(1, run)
    return out

def pair_masks(name):
    pds, labels = PROBES[name]
    probe_m = lab.isin(labels).to_numpy() & (ds_arr == pds if pds else core)
    sm = (y == 1) & ((ds_arr == pds) if pds in STRESS_DS else True)
    return sm | probe_m

def run_partA(p_ext, p_c3, p_win, index, tag):
    rows, boots = [], {}
    for score_name, get in [("external", lambda n: p_ext), ("cond3_added_negative", lambda n: p_c3.get(n)), ("within_LOSO", lambda n: p_win)]:
        for name, (pds, labels) in PROBES.items():
            sc = get(name)
            if sc is None or (score_name == "within_LOSO" and (pds == "EPM-E4" or pds is None)): continue
            pair = pair_masks(name); a = IDX[index]
            r = binned(a[pair], sc[pair], y[pair], subj[pair]); boots[(score_name, name)] = r.pop("boot")
            r.update(variant=tag, index=index, score=score_name, probe=name); rows.append(r)
    A = pd.DataFrame(rows)
    A["p_holm"] = np.nan
    for s, g in A.groupby("score"): A.loc[g.index, "p_holm"] = holm(g.p_raw)
    # exercise (cond3) minus each other pair, bootstrap over subjects (independent subject sets)
    ex = boots.get(("cond3_added_negative", "exercise"))
    A["diff_vs_exercise"] = np.nan; A["diff_lo"] = np.nan; A["diff_hi"] = np.nan
    if ex is not None:
        for i, r in A.iterrows():
            bo = boots.get((r.score, r.probe))
            if r.probe == "exercise" or bo is None: continue
            d = ex[:min(len(ex), len(bo))] - bo[:min(len(ex), len(bo))]
            A.loc[i, ["diff_vs_exercise", "diff_lo", "diff_hi"]] = [ex.mean() - bo.mean(), np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)]
    return A.round(4)

# ---------- Fix 1 ----------
p_ext0, p_win0 = sc_old.p_ext.to_numpy(), sc_old.p_win.to_numpy()
p_c30 = cond3_scores(cols)
A1 = run_partA(p_ext0, p_c30, p_win0, "hr+eda", "full_model")
A1.to_csv(OUT / "partA_inference.csv", index=False)
print("=== FIX 1: inference, full model, hr+eda index ===")
print(A1[["score", "probe", "n", "auc", "lo", "hi", "p_raw", "p_holm", "diff_vs_exercise", "diff_lo", "diff_hi"]].to_string(index=False))

# ---------- Fix 2 ----------
eda_fam = [c for c in cols if c.startswith("eda_")]; hr_fam = [c for c in cols if c.startswith(("hr_", "hrv_"))]
VARIANTS = {"disjoint_hr+eda": ("hr+eda", [c for c in cols if c not in ("hr_mean", "eda_tonic_mean", "eda_mean")]),
            "disjoint_hr_only_model_eda": ("hr_only", eda_fam),
            "disjoint_eda_only_model_hr": ("eda_only", [c for c in hr_fam if c != "hr_mean"])}
A2 = []
for tag, (index, fc) in VARIANTS.items():
    print(f"\n--- {tag}: model features {fc}")
    A2.append(run_partA(ext_scores(fc), cond3_scores(fc), None, index, tag))
A2 = pd.concat(A2); A2.to_csv(OUT / "partA_disjoint.csv", index=False)
print("=== FIX 2: feature-disjoint index ===")
print(A2[["variant", "score", "probe", "n", "auc", "lo", "hi", "p_raw", "p_holm"]].to_string(index=False))

# ---------- Fix 5 ----------
# block structure for F1/F3 (copied from the probe)
d = df[["dataset", "subject_uid", "harmonized_label", "timestamp_start", "timestamp_end", "cardiac_coverage"]].copy(); d["i"] = np.arange(len(d))
d = d[~d.harmonized_label.isin(EXERCISE)].sort_values(["subject_uid", "timestamp_start"])
d["blk"] = (d.groupby("subject_uid").apply(lambda g: (g.harmonized_label != g.harmonized_label.shift()) | (g.timestamp_start - g.timestamp_end.shift() > 60)).reset_index(level=0, drop=True)).cumsum()
d["mso"] = (d.timestamp_start - d.groupby("blk").timestamp_start.transform("min")) / 60; d["msoff"] = np.nan
for s, g in d.groupby("subject_uid"):
    offs = g[g.harmonized_label == "Stress"].groupby("blk").timestamp_end.max().sort_values().to_numpy()
    if len(offs) == 0: continue
    ts = g.timestamp_start.to_numpy(); j = np.searchsorted(offs, ts, side="right") - 1
    d.loc[g.index, "msoff"] = np.where(j >= 0, (ts - offs[np.clip(j, 0, None)]) / 60, np.nan)
d.loc[d.dataset == "UBFC-Phys", "msoff"] = np.nan
mso = np.full(len(df), np.nan); msoff = np.full(len(df), np.nan); mso[d.i] = d.mso; msoff[d.i] = d.msoff

def responders(rule):
    out = {}
    for s, g in raw[core].groupby("subject_uid"):
        b, st = g[g.harmonized_label == "Baseline"], g[g.harmonized_label == "Stress"]
        if len(b) < 2 or len(st) == 0: out[s] = np.nan; continue
        zs = []
        for c in ("hr_mean", "eda_tonic_mean"):
            sd = b[c].std(); zs.append((st[c].mean() - b[c].mean()) / sd if np.isfinite(sd) and sd > 0 else np.nan)
        out[s] = rule(np.array(zs))
    return out
RULES = {"lenient_0.5sd_either": lambda z: bool(np.nanmax(z) > 0.5) if np.isfinite(z).any() else np.nan,
         "1.0sd_either": lambda z: bool(np.nanmax(z) > 1.0) if np.isfinite(z).any() else np.nan,
         "1.0sd_both": lambda z: bool(np.all(z > 1.0)) if np.isfinite(z).all() else np.nan,
         "top_half_pooled_z": lambda z: float(np.nanmean(z)) if np.isfinite(z).any() else np.nan}
FBASE = {"F1_recovery": ~((y == 0) & (msoff < 5)), "F2_sensor": df.cardiac_coverage.to_numpy() >= 0.5, "F3_onset": ~((y == 1) & (mso < 1))}
def ba(m, p):
    m = m & np.isfinite(p); return balanced_accuracy_score(y[m], p[m] > 0.5) if len(np.unique(y[m])) == 2 else np.nan
rows5 = []
for rule_name, rule in RULES.items():
    rs = responders(rule); v = pd.Series(subj).map(rs).to_numpy(float)
    if rule_name == "top_half_pooled_z":  # per-dataset median split
        is_resp = np.zeros(len(df), bool)
        for ds in STRESS_DS:
            m = ds_arr == ds; med = np.nanmedian(pd.Series(v[m]).groupby(subj[m]).first())
            is_resp[m] = v[m] >= med
    else:
        is_resp = v == 1
    for ds in STRESS_DS:
        base = core & (ds_arr == ds)
        for score_name, p in (("within_LOSO", p_win0), ("external", p_ext0)):
            F = dict(FBASE, F4_responder=is_resp); names = list(F); ba_set = {}
            for k in range(5):
                for S in itertools.combinations(names, k):
                    m = base.copy()
                    for f in S: m &= F[f]
                    ba_set[S] = ba(m, p)
            sh, n = 0.0, 4; others = names[:3]
            for k in range(4):
                for S in itertools.combinations(others, k):
                    w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                    a_, b_ = ba_set[tuple(x for x in names if x in S + ("F4_responder",))], ba_set[tuple(x for x in names if x in S)]
                    if np.isfinite(a_) and np.isfinite(b_): sh += w * (a_ - b_)
            subs = np.unique(subj[base]); bs = []
            for _ in range(500):
                pick = rng.choice(subs, len(subs), replace=True)
                idx = np.concatenate([np.flatnonzero((subj == q) & base) for q in pick]); yy, pp, fm = y[idx], p[idx], is_resp[idx]; okp = np.isfinite(pp)
                if len(np.unique(yy[okp & fm])) < 2 or len(np.unique(yy[okp])) < 2: continue
                bs.append(balanced_accuracy_score(yy[okp & fm], pp[okp & fm] > 0.5) - balanced_accuracy_score(yy[okp], pp[okp] > 0.5))
            total = ba_set[tuple(names)] - ba_set[()]
            rows5.append(dict(rule=rule_name, dataset=ds, score=score_name, n_subjects=len(subs), n_responders=int(np.unique(subj[base & is_resp]).size),
                              ba_all=ba_set[()], F4_single=ba_set[("F4_responder",)], F4_gain=ba_set[("F4_responder",)] - ba_set[()],
                              F4_gain_lo=np.percentile(bs, 2.5) if bs else np.nan, F4_gain_hi=np.percentile(bs, 97.5) if bs else np.nan,
                              F4_shapley=sh, all_filters_gain=total, F4_shapley_share=sh / total if total else np.nan, ba_all_filters=ba_set[tuple(names)]))
B5 = pd.DataFrame(rows5).round(3); B5.to_csv(OUT / "partB_responder_sensitivity.csv", index=False)
print("\n=== FIX 5: responder-filter sensitivity ===")
print(B5.to_string(index=False))
