"""Per-dataset reference for the matched-arousal test (Table IV).

Table IV compares each probe pair with a reference pair (stress vs Baseline/Rest) pooled over the five stress
datasets. Pooling puts cross-dataset stress/rest comparisons inside each arousal bin, and the pooled value sits below
every single dataset's (kwon_reconciliation/table4_reference_by_dataset.csv). Here:
1  reference per dataset: within-bin AUROC, subject-bootstrap 95% CI, for external (LODO other-4, committed
   scores.csv p_ext), within-LOSO (p_win) and each added-negative model (cond3, recomputed as in
   reviewer_reanalyses.py); bins are quintiles of the HR+EDA index on the pair itself, as in Table IV.
2  Δref for same-dataset probes against their own dataset's reference (same stress windows, joint subject
   bootstrap within the dataset); EPM clips against the pooled reference, and against a reference with the same
   stress windows and pooling (stress of the five datasets vs Baseline/Rest of those datasets).
3  pooling decomposition: pooled reference vs window-weighted mean of per-dataset references, per-dataset bins
   and pooled-pair bins.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from compare_models import holm
from leave_one_dataset_out import NON_STRESS, fit_predict, normalise
from run_jbhi_experiments import META, grouped_split

TABLES = REPO / "outputs" / "tables" / "jbhi_v2"
OUT = TABLES / "matched_arousal_per_dataset"
cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument("--input", type=Path, default=Path(os.environ.get("HARMONIZED_CSV", REPO / "data/processed/combined/harmonized_windows_v2.csv")))
cli.add_argument("--n-boot", type=int, default=2000)
args = cli.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

# ---------- same setup as reviewer_reanalyses.py / matched_arousal_hardening.py ----------
raw = pd.read_csv(args.input)
phys = [c for c in raw.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
cols = [c for c in phys if not c.startswith("temp_")]
df = raw.copy(); df[phys] = normalise(raw, phys, "session")
lab = df["harmonized_label"]; ds_arr, subj = df["dataset"].to_numpy(), df["subject_uid"].to_numpy()
core = lab.isin(NON_STRESS + ["Stress"]).to_numpy(); y = (lab == "Stress").to_numpy(int)
STRESS_DS = ["WESAD", "PhysioNet", "Stress-Predict", "UBFC-Phys", "Campanella2024"]
PROBES = {"exercise": ("PhysioNet", ["Aerobic", "Anaerobic"]), "Hyperventilation": ("Stress-Predict", ["Hyperventilation"]),
          "UBFC Control task": ("UBFC-Phys", ["Control task"]), "Campanella Manual task": ("Campanella2024", ["Manual task"]),
          "EPM Fear": ("EPM-E4", ["Fear"]), "EPM Anger": ("EPM-E4", ["Anger"])}
REF_LABELS = ["Baseline", "Rest"]
hr_z, eda_z = df["hr_mean"].to_numpy(), df["eda_tonic_mean"].to_numpy()
IDX = np.where(np.isnan(hr_z), eda_z, (hr_z + eda_z) / 2)
sc_old = pd.read_csv(TABLES / "contribution_probes/arousal/scores.csv")
assert (sc_old.window_id.to_numpy() == df.window_id.to_numpy()).all()


def cond3_scores(fcols):  # copied from reviewer_reanalyses.py (that script runs on import)
    X = df[fcols].to_numpy(); out = {}
    for name, (pds, labels) in PROBES.items():
        acc, cnt = np.zeros(len(df)), np.zeros(len(df)); is_probe = lab.isin(labels).to_numpy()
        for seed in range(20):
            trm, tem = grouped_split(df, seed)
            tr = (trm & core) | (trm & is_probe); y3 = y.copy(); y3[is_probe] = 0; te = tem & (core | is_probe)
            acc[te] += fit_predict(X[tr], y3[tr], X[te], 8); cnt[te] += 1
        out[name] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return out


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


class Pair:
    """Stress-vs-state pair: finite windows, quintile bins on the pair itself, per-subject window indices."""
    def __init__(self, mask, score):
        ok = mask & np.isfinite(IDX) & np.isfinite(score)
        self.s, self.y, self.ds, sub = score[ok], y[ok], ds_arr[ok], subj[ok]
        self.b, self.nbin = bins_of(IDX[ok])
        self.auc = wauc(self.b, self.nbin, self.s, self.y)
        self.n, self.n_stress, self.n_subj = int(ok.sum()), int(self.y.sum()), len(np.unique(sub))
        self.by_sub = {u: np.flatnonzero(sub == u) for u in np.unique(sub)}

    def boot(self, picks):
        idx = np.concatenate([self.by_sub[u] for u in picks if u in self.by_sub])
        return wauc(self.b[idx], self.nbin, self.s[idx], self.y[idx])

    def auc_in(self, d):  # this pair's bins, windows of dataset d only
        m = self.ds == d
        return wauc(self.b[m], self.nbin, self.s[m], self.y[m])


is_ref = lab.isin(REF_LABELS).to_numpy()
stress_in = lambda dss: (y == 1) & np.isin(ds_arr, dss)
ref_mask = lambda dss: stress_in(dss) | (is_ref & np.isin(ds_arr, dss))
probe_mask = lambda name: (stress_in([PROBES[name][0]]) if PROBES[name][0] in STRESS_DS else stress_in(STRESS_DS)) \
    | (lab.isin(PROBES[name][1]).to_numpy() & (ds_arr == PROBES[name][0]))
rng = np.random.default_rng(0)
sub_ds = pd.Series(ds_arr, index=subj).groupby(level=0).first()


def draws(pairs, n):
    """Subject bootstrap stratified by dataset, one draw shared by all pairs; returns (n, len(pairs))."""
    units = sub_ds[sorted(set().union(*[p.by_sub for p in pairs]))]
    groups = [g.index.to_numpy() for _, g in units.groupby(units)]
    out = np.empty((n, len(pairs)))
    for i in range(n):
        picks = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        out[i] = [p.boot(picks) for p in pairs]
    return out


p_c3 = cond3_scores(cols)
pd.DataFrame({"window_id": df.window_id} | {f"p_cond3_{k}": v for k, v in p_c3.items()}).to_csv(OUT / "cond3_scores.csv", index=False)
SCORES = {"external": sc_old.p_ext.to_numpy(), "within_LOSO": sc_old.p_win.to_numpy()} | {f"cond3[{k}]": v for k, v in p_c3.items()}
inf = pd.read_csv(TABLES / "contribution_probes/hardening/partA_inference.csv")
T4 = pd.read_csv(TABLES / "kwon_reconciliation/table4_reference_by_dataset.csv")
A_old = pd.read_csv(TABLES / "reviewer_reanalyses/A_reference_relative.csv")
inf = inf[(inf.variant == "full_model") & (inf["index"] == "hr+eda")]

# ---------- 1 + 3: per-dataset reference and pooling decomposition ----------
rows1, rows3 = [], []
for sname, sc in SCORES.items():
    pooled = Pair(ref_mask(STRESS_DS), sc)
    if sname in ("external", "within_LOSO"):  # Table IV reference reproduced (0.584 / 0.784)
        want = T4[(T4.dataset == "pooled (Table IV)") & (T4.score == sname)].auc_hr_eda_quintiles.iloc[0]
        assert abs(want - pooled.auc) < 1e-3, (sname, want, pooled.auc)
    else:
        want = A_old[(A_old.score == "cond3_added_negative") & (A_old.probe == sname[6:-1])].auc_ref.iloc[0]
        assert abs(want - pooled.auc) < 1e-3, (sname, want, pooled.auc)
    per = {d: Pair(ref_mask([d]), sc) for d in STRESS_DS}
    for d, P in per.items():
        bs = draws([P], args.n_boot)[:, 0]
        rows1.append(dict(score=sname, dataset=d, n=P.n, n_stress=P.n_stress, n_subj=P.n_subj, auc=P.auc,
                          lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5), auc_pooled_bins=pooled.auc_in(d)))
    w = np.array([per[d].n for d in STRESS_DS])
    own = np.array([per[d].auc for d in STRESS_DS]); pb = np.array([pooled.auc_in(d) for d in STRESS_DS])
    rows3.append(dict(score=sname, n=pooled.n, pooled=pooled.auc, wmean_own_bins=np.average(own, weights=w),
                      wmean_pooled_bins=np.average(pb, weights=w), min_dataset=own.min(), max_dataset=own.max(),
                      pooling_effect=pooled.auc - np.average(own, weights=w)))
R1 = pd.DataFrame(rows1).round(4); R1.to_csv(OUT / "per_dataset_reference.csv", index=False)
R3 = pd.DataFrame(rows3).round(4); R3.to_csv(OUT / "pooling_decomposition.csv", index=False)

# ---------- 2: Δref against the comparable reference ----------
rows2 = []
for sname in ("external", "cond3_added_negative", "within_LOSO"):
    for name, (pds, _) in PROBES.items():
        if sname == "within_LOSO" and pds not in STRESS_DS: continue  # no within-dataset stress in EPM-E4
        sc = SCORES[f"cond3[{name}]"] if sname == "cond3_added_negative" else SCORES[sname]
        pr = Pair(probe_mask(name), sc)
        want = inf[(inf.score == sname) & (inf.probe == name)]
        if len(want): assert abs(want.auc.iloc[0] - pr.auc) < 1e-3, (sname, name, want.auc.iloc[0], pr.auc)
        if pds in STRESS_DS:
            refs = {f"{pds} only (comparable)": ref_mask([pds]), "pooled 5 datasets (Table IV)": ref_mask(STRESS_DS)}
        else:  # EPM: stress windows come from all five datasets, so the same-stress/same-pooling reference is the pooled pair
            used = sorted(set(pr.ds[pr.y == 1]))
            refs = {f"pooled, same stress windows ({len(used)} ds, comparable)": ref_mask(used)}
        for rname, rm in refs.items():
            rf = Pair(rm, sc)
            same_stress = np.array_equal(np.sort(pr.s[pr.y == 1]), np.sort(rf.s[rf.y == 1]))
            b = draws([pr, rf], args.n_boot); d = b[:, 0] - b[:, 1]
            rows2.append(dict(score=sname, probe=name, reference=rname, same_stress_windows=same_stress, n_probe=pr.n - pr.n_stress,
                              n_ref=rf.n - rf.n_stress, n_subj=len(set(pr.by_sub) | set(rf.by_sub)), auc_probe=pr.auc,
                              auc_ref=rf.auc, diff=pr.auc - rf.auc, diff_lo=np.nanpercentile(d, 2.5), diff_hi=np.nanpercentile(d, 97.5),
                              p_boot=min(1, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0)))))
R2 = pd.DataFrame(rows2); R2["p_holm"] = np.nan
comp = R2.reference.str.contains("comparable")
for _, g in R2[comp].groupby("score"): R2.loc[g.index, "p_holm"] = holm(g.p_boot).to_numpy()  # Holm over the 6 comparable rows
R2 = R2.round(4); R2.to_csv(OUT / "delta_same_dataset.csv", index=False)

with pd.option_context("display.width", 250, "display.max_rows", 200):
    print(R1.to_string(index=False)); print(R3.to_string(index=False)); print(R2.to_string(index=False))
print(f"runtime {time.time() - t0:.0f} s")
