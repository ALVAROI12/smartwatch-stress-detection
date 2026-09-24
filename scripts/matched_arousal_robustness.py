"""Robustness of the matched-arousal test and specificity panel (JBHI review items REV-3 and REV-6).

Every matched-arousal row reports the within-quintile AUROC of a stress-vs-state pair, the same statistic for a
reference pair scored by the same model, their difference (Δref) and a 95% CI / two-sided p from a joint subject
bootstrap (subjects resampled within dataset, one draw shared by both pairs), as in reviewer_reanalyses.py.
Bins are arousal-index quintiles of each pair; the index is mean(z hr_mean, z eda_tonic_mean), EDA alone when HR is missing.

1  Graded synthetic positive control (session z-scores, external = leave-one-dataset-out scores).
   score:   add d x (pooled within-class SD of logit score) to the stress windows of a random half of the pair's
            stress subjects; bins and index untouched, so this is the ideal case of a stress-only signal that the
            model fully uses. Δ = injected minus uninjected pair (same windows, joint bootstrap).
   feature: add d (within-subject SD units) to one non-index feature, the one least correlated with the index,
            in the stress windows of a random half of all stress subjects in every dataset, retrain the external
            models, Δ = injected minus d = 0 retrained. Tells what size of real non-arousal signal the pipeline sees.
2  Missingness: complete case (HR present, index = HR+EDA) and HR-missing stratum (index = EDA only); same scores as
   Table IV (committed external scores, added-negative scores recomputed); only evaluation windows are restricted.
3  Normalisation: raw features and baseline-referenced z-scores (normalise(..., "baseline") in leave_one_dataset_out.py:
   mean/SD of the subject's Baseline windows). Index for raw = global z of raw hr_mean and eda_tonic_mean. EPM-E4 has no
   Baseline windows, so its probes are not available under baseline scaling. False-stress rates: external = mean(p > 0.5);
   added-negative = pooled over the 20 grouped splits' test windows (as the specificity panel). Exercise = Aerobic+Anaerobic.
4  Class size (session): exercise added as negative but subsampled to 33 or 80 training windows per split; and a joint
   model with all six probe classes added as negatives, unweighted and with equal total weight (80 windows each).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from compare_models import holm
from leave_one_dataset_out import NON_STRESS, fit_predict, normalise
from probe_normalisation import model
from run_jbhi_experiments import META, grouped_split

TABLES = REPO / "outputs" / "tables" / "jbhi_v2"
OUT = TABLES / "matched_arousal_robustness"
cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument("--input", type=Path, default=Path(os.environ.get("HARMONIZED_CSV", REPO / "data/processed/combined/harmonized_windows_v2.csv")))
cli.add_argument("--n-boot", type=int, default=2000)
cli.add_argument("--n-jobs", type=int, default=8)
args = cli.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
T0, NJ = time.time(), args.n_jobs

raw = pd.read_csv(args.input)
phys = [c for c in raw.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
cols = [c for c in phys if not c.startswith("temp_")]
lab = raw["harmonized_label"]; ds_arr, subj = raw["dataset"].to_numpy(), raw["subject_uid"].to_numpy()
core = lab.isin(NON_STRESS + ["Stress"]).to_numpy(); y = (lab == "Stress").to_numpy(int)
STRESS_DS = ["WESAD", "PhysioNet", "Stress-Predict", "UBFC-Phys", "Campanella2024"]
PROBES = {"exercise": ("PhysioNet", ["Aerobic", "Anaerobic"]), "Hyperventilation": ("Stress-Predict", ["Hyperventilation"]),
          "UBFC Control task": ("UBFC-Phys", ["Control task"]), "Campanella Manual task": ("Campanella2024", ["Manual task"]),
          "EPM Fear": ("EPM-E4", ["Fear"]), "EPM Anger": ("EPM-E4", ["Anger"])}
REF = "Baseline/Rest (ref)"
sub_ds = pd.Series(ds_arr, index=subj).groupby(level=0).first()


def probe_mask(name):
    if name == REF: return lab.isin(["Baseline", "Rest"]).to_numpy() & core
    pds, labels = PROBES[name]; return lab.isin(labels).to_numpy() & (ds_arr == pds)


def pair_mask(name):
    pds = PROBES[name][0] if name != REF else None
    return ((y == 1) & ((ds_arr == pds) if pds in STRESS_DS else True)) | probe_mask(name)


def zfeat(how):
    return normalise(raw, phys, how)[cols].to_numpy(float)


def index_of(X, how):
    h, e = X[:, cols.index("hr_mean")], X[:, cols.index("eda_tonic_mean")]
    if how == "raw": h, e = (h - np.nanmean(h)) / np.nanstd(h), (e - np.nanmean(e)) / np.nanstd(e)
    return np.where(np.isnan(h), e, (h + e) / 2)


# ---------- matched-arousal machinery (same bins and statistic as matched_arousal_hardening.py) ----------
def bins_of(a, nb=5):
    edges = np.unique(np.quantile(a, np.linspace(0, 1, nb + 1)))
    return np.clip(np.searchsorted(edges, a, side="right") - 1, 0, len(edges) - 2), len(edges) - 1


def auc(yy, s):
    n1 = yy.sum(); n0 = len(yy) - n1
    return (rankdata(s)[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def wauc(b, nbin, s, yy):
    num = den = 0.0
    for k in range(nbin):
        m = b == k; yk = yy[m]
        if 0 < yk.sum() < len(yk): num += auc(yk, s[m]) * m.sum(); den += m.sum()
    return num / den if den else np.nan


class Pair:
    def __init__(self, mask, score, index):
        ok = mask & np.isfinite(index) & np.isfinite(score)
        self.s, self.y, sub = score[ok], y[ok], subj[ok]
        self.n_state = int((self.y == 0).sum())
        self.b, self.nbin = bins_of(index[ok]) if ok.sum() >= 20 and 0 < self.y.sum() < len(self.y) else (None, 0)
        self.auc = wauc(self.b, self.nbin, self.s, self.y) if self.nbin else np.nan
        self.by_sub = {u: np.flatnonzero(sub == u) for u in np.unique(sub)}

    def boot(self, picks):
        idx = np.concatenate([self.by_sub[u] for u in picks if u in self.by_sub])
        return wauc(self.b[idx], self.nbin, self.s[idx], self.y[idx])


def boot_diff(P, R, n_boot):
    if not (np.isfinite(P.auc) and np.isfinite(R.auc)): return dict(auc=P.auc, auc_ref=R.auc)
    units = sub_ds[sorted(set(P.by_sub) | set(R.by_sub))]
    groups = [g.index.to_numpy() for _, g in units.groupby(units)]
    rng, d = np.random.default_rng(0), np.empty(n_boot)
    for i in range(n_boot):
        picks = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        d[i] = P.boot(picks) - R.boot(picks)
    return dict(auc=P.auc, auc_ref=R.auc, diff=P.auc - R.auc, lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5),
                p=min(1, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))))


def run(jobs, n_boot=None):
    """jobs: list of (meta dict, probe Pair, reference Pair) -> DataFrame, bootstraps in parallel."""
    res = Parallel(n_jobs=NJ)(delayed(boot_diff)(P, R, n_boot or args.n_boot) for _, P, R in jobs)
    return pd.DataFrame([dict(m, n_state=P.n_state, **r) for (m, P, _), r in zip(jobs, res)])


def matched_jobs(meta, scores, index, names=list(PROBES), extra=True):
    """scores: array (same model for every probe) or dict name -> array (one model per probe, ref scored by it)."""
    out = []
    for n in names:
        s = scores[n] if isinstance(scores, dict) else scores
        out.append((dict(meta, probe=n), Pair(pair_mask(n) & extra, s, index), Pair(pair_mask(REF) & extra, s, index)))
    return out


# ---------- scoring ----------
def ext_scores(X):
    p = np.full(len(raw), np.nan)
    for ds in np.unique(ds_arr):
        tr = core & (ds_arr != ds); te = ds_arr == ds; p[te] = fit_predict(X[tr], y[tr], X[te], NJ)
    return p


def fit_w(Xtr, ytr, w, Xte):
    return model().set_params(n_jobs=NJ).fit(Xtr, ytr, sample_weight=w).predict_proba(Xte)[:, 1]


def added_neg(X, added, subsample=None, equal_weight=None):
    """added: bool mask of windows used as extra negatives. Returns mean score per window and pooled (split x window) hit rate arrays."""
    acc, hit, cnt = np.zeros(len(raw)), np.zeros(len(raw)), np.zeros(len(raw))
    for seed in range(20):
        trm, tem = grouped_split(raw, seed)
        extra = trm & added
        if subsample:
            keep = np.random.default_rng(seed).choice(np.flatnonzero(extra), min(subsample, extra.sum()), replace=False)
            extra = np.zeros(len(raw), bool); extra[keep] = True
        tr = (trm & core) | extra; y3 = y.copy(); y3[added] = 0; te = tem & (core | added)
        if equal_weight:  # each added class (label) totals `equal_weight` windows; stress balances all negatives
            w = np.ones(len(raw))
            for l in np.unique(lab[extra]): m = extra & (lab == l).to_numpy(); w[m] = equal_weight / m.sum()
            neg = w[tr & (y3 == 0)].sum(); w[tr & (y3 == 1)] = neg / (tr & (y3 == 1)).sum()
            p = fit_w(X[tr], y3[tr], w[tr], X[te])
        else:
            p = fit_predict(X[tr], y3[tr], X[te], NJ)
        acc[te] += p; hit[te] += p > 0.5; cnt[te] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan), hit, cnt


def rate_ext(p, name): m = probe_mask(name) & np.isfinite(p); return (p[m] > 0.5).mean() if m.any() else np.nan
def rate_c3(hit, cnt, name): m = probe_mask(name); return hit[m].sum() / cnt[m].sum() if cnt[m].sum() else np.nan


def show(title, D, keep, fname):
    D = D.copy()
    if "p" in D:
        D["p_holm"] = np.nan
        for _, g in D.groupby([c for c in ("analysis", "norm", "score", "variant") if c in D]): D.loc[g.index, "p_holm"] = holm(g.p).to_numpy()
    D.round(4).to_csv(OUT / fname, index=False)
    print(f"\n=== {title} ({time.time() - T0:.0f}s) ===\n" + D[[c for c in keep if c in D]].round(3).to_string(index=False), flush=True)


# ---------- session scores: committed external, recomputed added-negative ----------
XS = zfeat("session"); IS = index_of(XS, "session")
sc = pd.read_csv(TABLES / "contribution_probes/arousal/scores.csv")
assert (sc.window_id.to_numpy() == raw.window_id.to_numpy()).all()
p_ext = {"session": sc.p_ext.to_numpy()}
c3 = {"session": {n: added_neg(XS, probe_mask(n)) for n in PROBES}}
print(f"session scores done ({time.time() - T0:.0f}s)", flush=True)

# ---------- 1: graded synthetic positive control ----------
D_GRID = [0.1, 0.2, 0.3, 0.5, 1.0]
rng = np.random.default_rng(1)
logit = np.log(np.clip(p_ext["session"], 1e-6, 1 - 1e-6) / np.clip(1 - p_ext["session"], 1e-6, 1))
jobs = []
for pname in (REF, "UBFC Control task"):
    pm = pair_mask(pname) & np.isfinite(IS) & np.isfinite(logit)
    sd = np.sqrt(np.mean([logit[pm & (y == k)].var(ddof=1) for k in (0, 1)]))
    ss = np.unique(subj[pm & (y == 1)]); half = rng.choice(ss, len(ss) // 2, replace=False)
    tgt = pm & (y == 1) & np.isin(subj, half); base = Pair(pm, logit, IS)
    for d in D_GRID:
        jobs.append((dict(analysis="score", pair=pname, d=d, n_subj_injected=len(half)), Pair(pm, logit + d * sd * tgt, IS), base))
cand = [c for c in cols if c not in ("hr_mean", "eda_tonic_mean", "eda_mean") and np.isfinite(XS[core, cols.index(c)]).mean() >= 0.8]
cors = {c: abs(pd.Series(XS[core, cols.index(c)]).corr(pd.Series(IS[core]), method="spearman")) for c in cand}
feat = min(cors, key=cors.get)
ss = np.unique(subj[y == 1]); half = rng.choice(ss, len(ss) // 2, replace=False); tgt = (y == 1) & np.isin(subj, half)
p0 = ext_scores(XS)
for d in D_GRID:
    Xd = XS.copy(); Xd[tgt, cols.index(feat)] += d; pd_ = ext_scores(Xd)
    for pname in (REF, "UBFC Control task"):
        pm = pair_mask(pname)
        jobs.append((dict(analysis=f"feature:{feat}", pair=pname, d=d, n_subj_injected=len(half)), Pair(pm, pd_, IS), Pair(pm, p0, IS)))
R1 = run(jobs).rename(columns={"auc": "auc_injected", "auc_ref": "auc_uninjected"})
R1["detected"] = R1.lo > 0
print(f"feature for injection: {feat} (|Spearman rho| with index {cors[feat]:.3f})")
show("1 positive control (Δ = injected - uninjected)", R1, ["analysis", "pair", "d", "auc_uninjected", "auc_injected", "diff", "lo", "hi", "p", "detected"], "1_positive_control.csv")

# ---------- 2: complete case and missingness strata ----------
hr_ok = np.isfinite(XS[:, cols.index("hr_mean")])
jobs = []
for stratum, m in (("all", True), ("complete_case_hr", hr_ok), ("hr_missing_eda_index", ~hr_ok)):
    jobs += matched_jobs(dict(score="external", stratum=stratum), p_ext["session"], IS, extra=m)
    jobs += matched_jobs(dict(score="added_negative", stratum=stratum), {n: c3["session"][n][0] for n in PROBES}, IS, extra=m)
R2 = run(jobs)
chk = pd.read_csv(TABLES / "reviewer_reanalyses/A_reference_relative.csv").replace({"cond3_added_negative": "added_negative"})
m = R2[R2.stratum == "all"].merge(chk, on=["score", "probe"])
assert np.allclose(m.auc, m.auc_probe, atol=1e-3) and np.allclose(m.auc_ref_x, m.auc_ref_y, atol=1e-3), "does not reproduce A_reference_relative.csv"
show("2 missingness strata", R2.assign(variant=R2.stratum), ["score", "stratum", "probe", "n_state", "auc", "auc_ref", "diff", "lo", "hi", "p"], "2_missingness_strata.csv")

# ---------- 3: normalisation sensitivity ----------
rows, jobs = [], []
for how in ("session", "raw", "baseline"):
    X = XS if how == "session" else zfeat(how); I = IS if how == "session" else index_of(X, how)
    if how != "session":
        p_ext[how] = ext_scores(X); c3[how] = {n: added_neg(X, probe_mask(n)) for n in PROBES}
    na = [n for n in PROBES if how == "baseline" and PROBES[n][0] == "EPM-E4"]  # no Baseline windows -> all-NaN features
    for n in PROBES:
        rows.append(dict(norm=how, probe=n, ext_false_stress=np.nan if n in na else rate_ext(p_ext[how], n),
                         added_neg_false_stress=np.nan if n in na else rate_c3(*c3[how][n][1:], n)))
    ok = [n for n in PROBES if n not in na]
    jobs += matched_jobs(dict(norm=how, score="external"), p_ext[how], I, ok)
    jobs += matched_jobs(dict(norm=how, score="added_negative"), {n: c3[how][n][0] for n in ok}, I, ok)
show("3a normalisation: false-stress rate (p > 0.5)", pd.DataFrame(rows), ["norm", "probe", "ext_false_stress", "added_neg_false_stress"], "3a_normalisation_panel.csv")
show("3b normalisation: matched test", run(jobs), ["norm", "score", "probe", "n_state", "auc", "auc_ref", "diff", "lo", "hi", "p"], "3b_normalisation_matched.csv")

# ---------- 4: class size and weighting (session) ----------
ex = probe_mask("exercise"); rows, jobs = [], []
variants = {"full (Table III/IV)": c3["session"]["exercise"]}
for k in (33, 80): variants[f"exercise subsampled to {k}"] = added_neg(XS, ex, subsample=k)
for v, (p, hit, cnt) in variants.items():
    rows.append(dict(variant=v, probe="exercise", added_neg_false_stress=rate_c3(hit, cnt, "exercise")))
    jobs += matched_jobs(dict(variant=v), p, IS, ["exercise"])
all_added = np.logical_or.reduce([probe_mask(n) for n in PROBES])
for v, kw in (("joint, unweighted", {}), ("joint, equal weight 80/class", dict(equal_weight=80))):
    p, hit, cnt = added_neg(XS, all_added, **kw)
    rows += [dict(variant=v, probe=n, added_neg_false_stress=rate_c3(hit, cnt, n)) for n in PROBES]
    jobs += matched_jobs(dict(variant=v), p, IS)
R4 = pd.DataFrame(rows).merge(run(jobs), on=["variant", "probe"], how="left")
show("4 class size / weighting (added-negative)", R4, ["variant", "probe", "added_neg_false_stress", "auc", "auc_ref", "diff", "lo", "hi", "p"], "4_class_size_weighting.csv")
print(f"\ntotal runtime {time.time() - T0:.0f}s")
