"""Reviewer re-analyses (JBHI), all on existing data and committed scores.

A  reference-relative matched-arousal test: within-bin AUROC of each probe pair minus that of the
   "baseline and rest" reference pair, same model scores and quintile bins as matched_arousal_hardening.py,
   subject bootstrap (2000 draws, subjects resampled within dataset, one draw shared by both pairs).
B  equivalence (TOST) of each within-bin AUROC to 0.5 at margins 0.05 and 0.10 (bootstrap 90% CI inside
   the margin), and the minimum detectable deviation at 80% power, 2.80 x bootstrap SD.
C  label-free arousal index as a detector under LODO: AUROC per dataset; BA with the threshold that maximises
   BA on the other four datasets; averaged over the same 20 test-subject sets as leave_one_dataset_out.py.
D  Nadeau-Bengio corrected CIs and p for the LODO transfer cost (within minus other-4 BA, HR/HRV/EDA).
E  heart-rate missingness by dataset and class, and how often the index falls back to EDA alone.
F  channel-disjoint matched-arousal rows extracted from partA_disjoint.csv.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from compare_models import TEST_TRAIN_RATIO, corrected_ttest, holm
from leave_one_dataset_out import EXERCISE, NON_STRESS, fit_predict, normalise
from run_jbhi_experiments import META, grouped_split

TABLES = REPO / "outputs" / "tables" / "jbhi_v2"
HARD = TABLES / "contribution_probes" / "hardening"
OUT = TABLES / "reviewer_reanalyses"
cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument("--input", type=Path, default=Path(os.environ.get("HARMONIZED_CSV", REPO / "data/processed/combined/harmonized_windows_v2.csv")))
cli.add_argument("--n-boot", type=int, default=2000)
args = cli.parse_args()
OUT.mkdir(parents=True, exist_ok=True)

# ---------- same setup as matched_arousal_hardening.py ----------
raw = pd.read_csv(args.input)
phys = [c for c in raw.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
cols = [c for c in phys if not c.startswith("temp_")]
df = raw.copy(); df[phys] = normalise(raw, phys, "session")
lab = df["harmonized_label"]; ds_arr, subj = df["dataset"].to_numpy(), df["subject_uid"].to_numpy()
core = lab.isin(NON_STRESS + ["Stress"]).to_numpy(); y = (lab == "Stress").to_numpy(int)
STRESS_DS = ["WESAD", "PhysioNet", "Stress-Predict", "UBFC-Phys", "Campanella2024"]
PROBES = {"exercise": ("PhysioNet", ["Aerobic", "Anaerobic"]), "Hyperventilation": ("Stress-Predict", ["Hyperventilation"]),
          "UBFC Control task": ("UBFC-Phys", ["Control task"]), "Campanella Manual task": ("Campanella2024", ["Manual task"]),
          "EPM Fear": ("EPM-E4", ["Fear"]), "EPM Anger": ("EPM-E4", ["Anger"]), "Baseline/Rest (ref)": (None, ["Baseline", "Rest"])}
REF = "Baseline/Rest (ref)"
hr_z, eda_z = df["hr_mean"].to_numpy(), df["eda_tonic_mean"].to_numpy()
IDX = {"hr+eda": np.where(np.isnan(hr_z), eda_z, (hr_z + eda_z) / 2), "eda_only": eda_z, "hr_only": hr_z}
sc_old = pd.read_csv(TABLES / "contribution_probes/arousal/scores.csv")
assert (sc_old.window_id.to_numpy() == df.window_id.to_numpy()).all()
p_ext = sc_old.p_ext.to_numpy()


def cond3_scores(fcols):  # copied from matched_arousal_hardening.py (that script runs on import)
    X = df[fcols].to_numpy(); out = {}
    for name, (pds, labels) in PROBES.items():
        if pds is None: continue
        acc, cnt = np.zeros(len(df)), np.zeros(len(df)); is_probe = lab.isin(labels).to_numpy()
        for seed in range(20):
            trm, tem = grouped_split(df, seed)
            tr = (trm & core) | (trm & is_probe); y3 = y.copy(); y3[is_probe] = 0; te = tem & (core | is_probe)
            acc[te] += fit_predict(X[tr], y3[tr], X[te], 8); cnt[te] += 1
        out[name] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return out


def pair_mask(name):
    pds, labels = PROBES[name]
    probe_m = lab.isin(labels).to_numpy() & (ds_arr == pds if pds else core)
    return ((y == 1) & ((ds_arr == pds) if pds in STRESS_DS else True)) | probe_m


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
    """One stress-vs-state pair: finite windows, fixed quintile bins, per-subject window indices."""
    def __init__(self, mask, score):
        ok = mask & np.isfinite(IDX["hr+eda"]) & np.isfinite(score)
        self.s, self.y, sub = score[ok], y[ok], subj[ok]
        self.b, self.nbin = bins_of(IDX["hr+eda"][ok])
        self.auc = wauc(self.b, self.nbin, self.s, self.y)
        self.by_sub = {u: np.flatnonzero(sub == u) for u in np.unique(sub)}

    def boot(self, picks):
        idx = np.concatenate([self.by_sub[u] for u in picks if u in self.by_sub])
        return wauc(self.b[idx], self.nbin, self.s[idx], self.y[idx])


rng = np.random.default_rng(0)
sub_ds = pd.Series(ds_arr, index=subj).groupby(level=0).first()
p_c3 = cond3_scores(cols)
inf = pd.read_csv(HARD / "partA_inference.csv")
rowsA, rowsB = [], []
for score_name, get in (("external", lambda n: p_ext), ("cond3_added_negative", lambda n: p_c3[n])):
    for name in [n for n in PROBES if n != REF]:
        sc = get(name); pr, rf = Pair(pair_mask(name), sc), Pair(pair_mask(REF), sc)
        units = sub_ds[sorted(set(pr.by_sub) | set(rf.by_sub))]
        groups = [g.index.to_numpy() for _, g in units.groupby(units)]
        bp, br = np.empty(args.n_boot), np.empty(args.n_boot)
        for i in range(args.n_boot):  # stratified by dataset, one draw shared by both pairs
            picks = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            bp[i], br[i] = pr.boot(picks), rf.boot(picks)
        d = bp - br
        ref_row = inf[(inf.score == score_name) & (inf.probe == name)].iloc[0]
        assert abs(ref_row.auc - pr.auc) < 1e-3, (score_name, name, ref_row.auc, pr.auc)  # same scores and bins
        rowsA.append(dict(score=score_name, probe=name, auc_probe=pr.auc, auc_ref=rf.auc, diff=pr.auc - rf.auc,
                          diff_lo=np.nanpercentile(d, 2.5), diff_hi=np.nanpercentile(d, 97.5),
                          p_boot_diff=min(1, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))),
                          p_perm_raw=ref_row.p_raw, p_perm_holm=ref_row.p_holm))
        B_items = [(name, pr, bp)]
        if score_name != "external" or name == "exercise":  # external reference is the same for every probe: once
            B_items.append((REF if score_name == "external" else f"{REF} [{name} model]", rf, br))
        for pname, P, bs in B_items:
            lo90, hi90, sd = np.nanpercentile(bs, 5), np.nanpercentile(bs, 95), np.nanstd(bs, ddof=1)
            rowsB.append({"score": score_name, "probe": pname, "auc": P.auc, "lo90": lo90, "hi90": hi90, "boot_sd": sd,
                          "equiv_0.05": bool(lo90 > 0.45 and hi90 < 0.55), "equiv_0.10": bool(lo90 > 0.40 and hi90 < 0.60),
                          "mdd_80pct": 2.80 * sd})
A = pd.DataFrame(rowsA); A["p_boot_diff_holm"] = np.nan
for s, g in A.groupby("score"): A.loc[g.index, "p_boot_diff_holm"] = holm(g.p_boot_diff).to_numpy()
A.round(4).to_csv(OUT / "A_reference_relative.csv", index=False)
B = pd.DataFrame(rowsB).round(4); B.to_csv(OUT / "B_equivalence_power.csv", index=False)
print("=== A ===\n" + A.round(3).to_string(index=False)); print("=== B ===\n" + B.round(3).to_string(index=False))

# ---------- C: arousal index alone under LODO ----------
lo = df[core & np.isin(ds_arr, STRESS_DS)].reset_index(drop=True)  # same windows as leave_one_dataset_out.py
lo_idx = {k: v[core & np.isin(ds_arr, STRESS_DS)] for k, v in IDX.items()}
ly = (lo.harmonized_label == "Stress").to_numpy(int); lds = lo.dataset.to_numpy()


def best_threshold(s, t):
    cand = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    return cand[np.argmax([balanced_accuracy_score(t, s > c) for c in cand])]


lodo = pd.read_csv(TABLES / "leave_one_dataset_out_per_split.csv")
lodo = lodo[(lodo.features == "hr_hrv_eda") & (lodo.trained_on == "other_datasets_only")].set_index(["test_dataset", "split"])
rowsC = []
for name, a in lo_idx.items():
    for target in STRESS_DS:
        tgt = lds == target; src = ~tgt & np.isfinite(a); thr = best_threshold(a[src], ly[src])
        t_idx = np.flatnonzero(tgt); per = []
        for seed in range(20):
            _, test = grouped_split(lo.iloc[t_idx], seed); te = t_idx[test]; te = te[np.isfinite(a[te])]
            if len(set(ly[te])) < 2: continue
            per.append((seed, balanced_accuracy_score(ly[te], a[te] > thr), roc_auc_score(ly[te], a[te])))
        per = pd.DataFrame(per, columns=["split", "ba", "auc"]).set_index("split")
        m = lodo.loc[target].reindex(per.index)
        diff, _, p = corrected_ttest(m.balanced_accuracy.to_numpy(), per.ba.to_numpy())
        d = m.balanced_accuracy.to_numpy() - per.ba.to_numpy()
        se = np.sqrt((1 / len(d) + TEST_TRAIN_RATIO) * d.var(ddof=1))  # Nadeau-Bengio corrected SE
        ci95 = diff + np.array([-1, 1]) * stats.t.ppf(0.975, len(d) - 1) * se
        ci90 = diff + np.array([-1, 1]) * stats.t.ppf(0.95, len(d) - 1) * se  # TOST at alpha 0.05
        ok = tgt & np.isfinite(a)
        rowsC.append(dict(index=name, dataset=target, coverage=ok.sum() / tgt.sum(), threshold=thr,
                          auroc_pooled=roc_auc_score(ly[ok], a[ok]), ba_pooled=balanced_accuracy_score(ly[ok], a[ok] > thr),
                          ba_splits=per.ba.mean(), auroc_splits=per.auc.mean(), model_ba=m.balanced_accuracy.mean(),
                          model_auroc=m.auroc.mean(), model_minus_index_ba=diff, ci95_lo=ci95[0], ci95_hi=ci95[1], ci90_lo=ci90[0], ci90_hi=ci90[1],
                          equiv_005=bool(ci90[0] > -0.05 and ci90[1] < 0.05), p_nb=p, n_splits=len(per)))
C = pd.DataFrame(rowsC); C["p_nb_holm"] = np.nan
for s, g in C.groupby("index"): C.loc[g.index, "p_nb_holm"] = holm(g.p_nb).to_numpy()
C.round(4).to_csv(OUT / "C_arousal_index_lodo.csv", index=False)
print("=== C ===\n" + C.round(3).to_string(index=False))

# ---------- D: corrected CIs for the LODO transfer cost ----------
ps = pd.read_csv(TABLES / "leave_one_dataset_out_per_split.csv")
ps = ps[ps.features == "hr_hrv_eda"].pivot_table(index=["test_dataset", "split"], columns="trained_on", values="balanced_accuracy")
rowsD = []
for target, g in ps.groupby(level=0):
    d = (g.within_dataset - g.other_datasets_only).to_numpy(); n = len(d)
    diff, t, p = corrected_ttest(g.within_dataset.to_numpy(), g.other_datasets_only.to_numpy())
    half = stats.t.ppf(0.975, n - 1) * np.sqrt((1 / n + TEST_TRAIN_RATIO) * d.var(ddof=1))
    rowsD.append(dict(dataset=target, within=g.within_dataset.mean(), other4=g.other_datasets_only.mean(), cost=diff,
                      lo=diff - half, hi=diff + half, t=t, p=p, n_splits=n, splits_other4_worse=int((d > 0).sum())))
D = pd.DataFrame(rowsD); D["p_holm"] = holm(D.p).to_numpy()
D.round(4).to_csv(OUT / "D_lodo_cost_corrected.csv", index=False)
print("=== D ===\n" + D.round(4).to_string(index=False))

# ---------- E: heart-rate missingness ----------
E = pd.DataFrame({"dataset": ds_arr, "label": lab, "hr_raw_nan": raw.hr_mean.isna(), "hr_z_nan": np.isnan(hr_z),
                  "eda_z_nan": np.isnan(eda_z)})
E["index_eda_fallback"] = E.hr_z_nan & ~E.eda_z_nan; E["index_missing"] = np.isnan(IDX["hr+eda"])
E = E.groupby(["dataset", "label"]).agg(n=("label", "size"), hr_raw_nan=("hr_raw_nan", "mean"), hr_z_nan=("hr_z_nan", "mean"),
                                        index_eda_fallback=("index_eda_fallback", "mean"), index_missing=("index_missing", "mean")).round(3)
E.to_csv(OUT / "E_hr_missingness.csv")
print("=== E ===\n" + E.to_string())

# ---------- F: channel-disjoint matched-arousal ----------
F = pd.read_csv(HARD / "partA_disjoint.csv")
F = F[F.variant.isin(["disjoint_hr_only_model_eda", "disjoint_eda_only_model_hr"])]
F = F[["variant", "index", "score", "probe", "n", "auc", "lo", "hi", "p_raw", "p_holm"]]
F.to_csv(OUT / "F_channel_disjoint.csv", index=False)
print("=== F ===\n" + F.to_string(index=False))
