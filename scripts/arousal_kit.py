"""Arousal-controlled evaluation kit (novelty plan step 2).

Reuses the committed external and within-dataset scores (contribution_probes/arousal/scores.csv),
so no model is retrained. Five parts, tables in contribution_probes/arousal_kit/:
  1. index_baseline.csv   label-free arousal index alone vs the model score, per dataset
  2. equivalence_mde.csv  90% CI against a +-0.10 equivalence bound, and a synthetic positive
                          control giving the smallest detectable within-bin AUROC per pair
  3. probe_vs_rest.csv    each non-stress probe state against baseline/rest at matched arousal
  4. iso_hr.csv           Kwon et al. (2026) iso-heart-rate test (5-bpm within-subject bins)
  5. residual.csv         stress score split into an arousal part and a residual

`arousal_controlled_auroc` is the reusable function: any score, any arousal index.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _bins(index, n_bins):
    edges = np.unique(np.quantile(index, np.linspace(0, 1, n_bins + 1)))
    return np.clip(np.searchsorted(edges, index, side="right") - 1, 0, len(edges) - 2), len(edges) - 1


def _weighted_auc(b, nbin, s, y):
    num = den = 0.0
    for k in range(nbin):
        m = b == k
        if len(np.unique(y[m])) < 2:
            continue
        num += roc_auc_score(y[m], s[m]) * m.sum()
        den += m.sum()
    return num / den if den else np.nan


def arousal_controlled_auroc(scores, index, labels, subjects, n_bins=5, n_boot=2000, seed=0, ci=95):
    """AUROC of `scores` for binary `labels` inside quantile bins of `index`, weighted by bin size.

    Bins are fixed on the pooled pair; the CI is a subject bootstrap. A value near 0.5 means the
    score carries nothing about the label beyond the arousal index. Returns auc, lo, hi, se, n and
    the bootstrap draws.
    """
    s, a, y, g = map(np.asarray, (scores, index, labels, subjects))
    ok = np.isfinite(s) & np.isfinite(a)
    s, a, y, g = s[ok], a[ok], y[ok].astype(int), g[ok]
    if len(np.unique(y)) < 2 or len(y) < 20:
        return dict(n=len(y), auc=np.nan, lo=np.nan, hi=np.nan, se=np.nan, boot=np.array([]))
    b, nbin = _bins(a, n_bins)
    auc = _weighted_auc(b, nbin, s, y)
    rng = np.random.default_rng(seed)
    rows = {u: np.flatnonzero(g == u) for u in np.unique(g)}
    units = list(rows)
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[units[i]] for i in rng.integers(0, len(units), len(units))])
        boot.append(_weighted_auc(b[idx], nbin, s[idx], y[idx]))
    boot = np.array(boot, float)
    q = (100 - ci) / 2
    return dict(n=len(y), auc=auc, lo=np.nanpercentile(boot, q), hi=np.nanpercentile(boot, 100 - q),
                se=np.nanstd(boot), boot=boot)


def iso_hr_subsample(hr_bpm, labels, subjects, rng, width=5):
    """Kwon et al. (2026): within each subject, 5-bpm HR bins; subsample the majority class per bin."""
    keep = []
    d = pd.DataFrame({"hr": hr_bpm, "y": labels, "g": subjects}).dropna()
    d["bin"] = np.floor(d.hr / width)
    for _, cell in d.groupby(["g", "bin"]):
        pos, neg = cell.index[cell.y == 1], cell.index[cell.y == 0]
        k = min(len(pos), len(neg))
        if k:
            keep += list(rng.choice(pos, k, replace=False)) + list(rng.choice(neg, k, replace=False))
    return np.array(sorted(keep), int)


def main():
    from leave_one_dataset_out import NON_STRESS, normalise
    from run_jbhi_experiments import META

    out = REPO / "outputs" / "tables" / "jbhi_v2" / "contribution_probes" / "arousal_kit"
    out.mkdir(parents=True, exist_ok=True)
    data = Path(os.environ.get("HARMONIZED_CSV",
                               "/Users/octa/Projects/smartwatch-stress-detection/data/processed/combined/harmonized_windows_v2.csv"))
    raw = pd.read_csv(data)
    phys = [c for c in raw.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
    df = raw.copy()
    df[phys] = normalise(raw, phys, "session")
    sc = pd.read_csv(REPO / "outputs/tables/jbhi_v2/contribution_probes/arousal/scores.csv")
    assert (sc.window_id.to_numpy() == df.window_id.to_numpy()).all(), "scores.csv is not aligned with the feature table"

    lab, ds, subj = df.harmonized_label, df.dataset.to_numpy(), df.subject_uid.to_numpy()
    core = lab.isin(NON_STRESS + ["Stress"]).to_numpy()
    y = (lab == "Stress").to_numpy(int)
    hr_z, eda_z = df.hr_mean.to_numpy(), df.eda_tonic_mean.to_numpy()
    idx = np.where(np.isnan(hr_z), eda_z, (hr_z + eda_z) / 2)
    p_ext, p_win = sc.p_ext.to_numpy(), sc.p_win.to_numpy()
    rest = lab.isin(["Baseline", "Rest"]).to_numpy()
    stress_ds = ["WESAD", "PhysioNet", "Stress-Predict", "UBFC-Phys", "Campanella2024"]
    probes = {"Exercise": ("PhysioNet", ["Aerobic", "Anaerobic"]), "Hyperventilation": ("Stress-Predict", ["Hyperventilation"]),
              "UBFC control task": ("UBFC-Phys", ["Control task"]), "Lego manual task": ("Campanella2024", ["Manual task"]),
              "Fear clips": ("EPM-E4", ["Fear"]), "Anger clips": ("EPM-E4", ["Anger"]),
              "Baseline/rest (reference)": (None, ["Baseline", "Rest"])}

    def pair_of(name):
        pds, labels = probes[name]
        probe = lab.isin(labels).to_numpy() & ((ds == pds) if pds else core)
        stress = (y == 1) & ((ds == pds) if pds in stress_ds else True)
        return stress | probe

    def boot_auc(s, yy, g, n_boot=2000, seed=0):
        rng = np.random.default_rng(seed)
        rows = {u: np.flatnonzero(g == u) for u in np.unique(g)}
        units = list(rows)
        return np.array([roc_auc_score(yy[i], s[i]) if len(np.unique(yy[i])) == 2 else np.nan
                         for i in (np.concatenate([rows[units[j]] for j in rng.integers(0, len(units), len(units))])
                                   for _ in range(n_boot))])

    # 1. Index baseline -------------------------------------------------------------------------
    rows = []
    for d_ in stress_ds:
        m = core & (ds == d_) & np.isfinite(idx) & np.isfinite(p_ext) & np.isfinite(p_win)
        yy, g = y[m], subj[m]
        r = dict(dataset=d_, n_windows=int(m.sum()), n_subjects=len(np.unique(g)))
        boots = {}
        for name, s in (("index", idx[m]), ("external", p_ext[m]), ("within", p_win[m])):
            r[f"auroc_{name}"] = roc_auc_score(yy, s)
            boots[name] = boot_auc(s, yy, g)
            r[f"auroc_{name}_lo"], r[f"auroc_{name}_hi"] = np.nanpercentile(boots[name], [2.5, 97.5])
        for name in ("external", "within"):
            dlt = boots[name] - boots["index"]  # same resamples (same seed), so paired
            r[f"{name}_minus_index"] = r[f"auroc_{name}"] - r["auroc_index"]
            r[f"{name}_minus_index_lo"], r[f"{name}_minus_index_hi"] = np.nanpercentile(dlt, [2.5, 97.5])
        rows.append(r)
    t1 = pd.DataFrame(rows).round(3)
    t1.to_csv(out / "index_baseline.csv", index=False)
    print("=== 1. Arousal index alone vs model (AUROC, stress vs core non-stress) ===\n", t1.to_string(index=False))

    # 2. Equivalence bounds and minimum detectable effect ---------------------------------------
    shifts = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    logit = np.log(np.clip(p_ext, 1e-4, 1 - 1e-4) / (1 - np.clip(p_ext, 1e-4, 1 - 1e-4)))
    rows = []
    for name in probes:
        m = pair_of(name)
        r0 = arousal_controlled_auroc(p_ext[m], idx[m], y[m], subj[m], ci=90)
        r = dict(pair=f"Stress vs {name}", n=r0["n"], n_subjects=len(np.unique(subj[m])), auc=r0["auc"],
                 ci90_lo=r0["lo"], ci90_hi=r0["hi"], equivalent_within_0_10=bool(r0["lo"] > 0.4 and r0["hi"] < 0.6))
        r["mde_auc"] = r["mde_shift"] = np.nan
        for dlt in shifts[1:]:  # positive control: add dlt to the logit of every stress window in the pair
            inj = logit[m] + dlt * y[m]
            ri = arousal_controlled_auroc(inj, idx[m], y[m], subj[m], n_boot=500)
            if ri["lo"] > 0.5 and ri["auc"] - r0["auc"] > 0:
                r["mde_shift"], r["mde_auc"] = dlt, ri["auc"]
                break
        rows.append(r)
    t2 = pd.DataFrame(rows).round(3)
    t2.to_csv(out / "equivalence_mde.csv", index=False)
    print("\n=== 2. Equivalence (90% CI inside 0.40-0.60) and minimum detectable within-bin AUROC ===\n", t2.to_string(index=False))

    # 3. Probe state vs baseline/rest at matched arousal ----------------------------------------
    rows = []
    for name, (pds, labels) in probes.items():
        if pds is None:
            continue
        probe = lab.isin(labels).to_numpy() & (ds == pds)
        ref = rest & ((ds == pds) if (rest & (ds == pds)).any() else core)
        m = probe | ref
        r0 = arousal_controlled_auroc(p_ext[m], idx[m], probe[m].astype(int), subj[m])
        rows.append(dict(probe=name, rest_from=pds if (rest & (ds == pds)).any() else "all datasets",
                         n_probe=int(probe.sum()), n_rest=int(ref.sum()), auc_pooled=roc_auc_score(probe[m], p_ext[m]),
                         auc_matched=r0["auc"], lo=r0["lo"], hi=r0["hi"]))
    t3 = pd.DataFrame(rows).round(3)
    t3.to_csv(out / "probe_vs_rest.csv", index=False)
    print("\n=== 3. Probe state (positive) vs baseline/rest, external score, within arousal bins ===\n", t3.to_string(index=False))

    # 4. Iso-HR test (Kwon et al. 2026) on our data --------------------------------------------
    hr_bpm = raw.hr_mean.to_numpy()
    rows = []
    targets = [(f"{d_}: stress vs non-stress", core & (ds == d_)) for d_ in stress_ds]
    targets += [(f"Stress vs {n}", pair_of(n)) for n in probes if probes[n][0] is not None]
    for name, m in targets:
        ii = np.flatnonzero(m & np.isfinite(p_ext) & np.isfinite(hr_bpm))
        r = dict(test=name, n_all=len(ii), auroc_unmatched=roc_auc_score(y[ii], p_ext[ii]) if len(np.unique(y[ii])) == 2 else np.nan)
        aucs, kept = [], []
        for seed in range(7):
            k = iso_hr_subsample(hr_bpm[ii], y[ii], subj[ii], np.random.default_rng(seed))
            if len(k) and len(np.unique(y[ii][k])) == 2:
                aucs.append(roc_auc_score(y[ii][k], p_ext[ii][k]))
                kept.append(len(k))
        r.update(n_matched_mean=np.mean(kept) if kept else 0, auroc_iso_hr=np.mean(aucs) if aucs else np.nan,
                 auroc_iso_hr_sd=np.std(aucs) if aucs else np.nan)
        rows.append(r)
    t4 = pd.DataFrame(rows).round(3)
    t4.to_csv(out / "iso_hr.csv", index=False)
    print("\n=== 4. Iso-HR (5-bpm within-subject bins, 7 seeds), external score ===\n", t4.to_string(index=False))

    # 5. Arousal + residual decomposition of the external stress score -------------------------
    rows = []
    for d_ in stress_ds:
        m = core & (ds == d_) & np.isfinite(idx) & np.isfinite(p_ext)
        a, s, yy, g = idx[m], logit[m], y[m], subj[m]
        A = np.c_[np.ones(len(a)), a]
        beta = np.linalg.lstsq(A, s, rcond=None)[0]  # label-free fit on this dataset's core windows
        res = s - A @ beta
        r2 = 1 - res.var() / s.var()
        bres = boot_auc(res, yy, g)
        rows.append(dict(dataset=d_, r2_score_on_index=r2, auroc_score=roc_auc_score(yy, s), auroc_index=roc_auc_score(yy, a),
                         auroc_residual=roc_auc_score(yy, res), auroc_residual_lo=np.nanpercentile(bres, 2.5),
                         auroc_residual_hi=np.nanpercentile(bres, 97.5)))
    t5 = pd.DataFrame(rows).round(3)
    t5.to_csv(out / "residual.csv", index=False)
    print("\n=== 5. logit(external score) = a + b*index + residual; AUROC of each part ===\n", t5.to_string(index=False))


if __name__ == "__main__":
    main()
