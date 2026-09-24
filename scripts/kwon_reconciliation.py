"""Reconcile our matched-arousal probe with Kwon et al. (2026) iso-HR matching, WESAD only.

Scores are the committed ones (contribution_probes/arousal/scores.csv: p_win = WESAD LOSO, p_ext = trained on the
other four stress datasets); no model is refitted, so matching here is an evaluation subset, whereas Kwon refit on it.
Kwon matching: per subject, raw hr_mean (bpm) in 5-bpm bins, majority class subsampled to the minority count per
bin, 7 seeds, pooled AUROC on the matched set. Quintile rows reuse the binned AUROC of matched_arousal_probe.py
(bins on the pooled pair, session-z index, within-bin AUROC weighted by bin size). Extra rows apply Kwon-style
within-subject matching on tonic EDA (0.5 SD bins of the session-z value) and on HR x EDA jointly.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from leave_one_dataset_out import normalise

OUT = REPO / "outputs" / "tables" / "jbhi_v2" / "kwon_reconciliation"
SCORES = REPO / "outputs" / "tables" / "jbhi_v2" / "contribution_probes" / "arousal" / "scores.csv"
NEGS = {"baseline": ["Baseline"], "baseline+meditation (Kwon)": ["Baseline", "Meditation"],
        "all_non_stress": ["Baseline", "Meditation", "Amusement"]}
SEEDS, N_BOOT = range(7), 500
rng = np.random.default_rng(0)


def boot_ci(stat, sub):
    subs = np.unique(sub)
    groups = [np.flatnonzero(sub == s) for s in subs]
    bs = [stat(np.concatenate([groups[i] for i in rng.integers(0, len(subs), len(subs))])) for _ in range(N_BOOT)]
    return np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def safe_auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def binned_auc(a, s, y, idx=None, nb=5):
    """Quintile edges fixed on the full pair; weighted within-bin AUROC on the (resampled) rows idx."""
    edges = np.unique(np.quantile(a, np.linspace(0, 1, nb + 1)))
    b = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, len(edges) - 2)
    idx = np.arange(len(a)) if idx is None else idx
    num = den = 0.0
    for k in range(len(edges) - 1):
        m = idx[b[idx] == k]
        v = safe_auc(y[m], s[m])
        if np.isfinite(v):
            num += v * len(m); den += len(m)
    return num / den if den else np.nan


def iso_match(keys, y, sub, seed):
    """Kwon: within subject x bin, keep min(n_stress, n_neg) windows of each class, random subsample."""
    r = np.random.default_rng(seed)
    keep = []
    for _, g in pd.DataFrame({"k": keys, "y": y, "s": sub}).groupby(["s", "k"]):
        pos, neg = g.index[g.y == 1].to_numpy(), g.index[g.y == 0].to_numpy()
        n = min(len(pos), len(neg))
        if n:
            keep += list(r.choice(pos, n, replace=False)) + list(r.choice(neg, n, replace=False))
    keep = np.sort(np.array(keep, int))
    assert (y[keep] == 1).sum() == (y[keep] == 0).sum()
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path(os.environ.get(
        "HARMONIZED_CSV", REPO / "data/processed/combined/harmonized_windows_v2.csv")))
    raw = pd.read_csv(ap.parse_args().input)
    z = normalise(raw, ["hr_mean", "eda_tonic_mean"], "session")  # same per-column z as matched_arousal_probe.py
    sc = pd.read_csv(SCORES)
    assert (sc.window_id.to_numpy() == raw.window_id.to_numpy()).all(), "scores.csv does not match --input"
    w = (raw.dataset == "WESAD").to_numpy()
    d = pd.DataFrame({"sub": raw.subject_uid, "lab": raw.harmonized_label, "hr": raw.hr_mean,
                      "hr_z": z.hr_mean, "eda_z": z.eda_tonic_mean, "within_LOSO": sc.p_win, "external": sc.p_ext})[w]
    d["hrdx"] = np.where(d.hr_z.isna(), d.eda_z, (d.hr_z + d.eda_z) / 2)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, seed_rows = [], []
    for neg_name, negs in NEGS.items():
        p = d[d.lab.isin(negs + ["Stress"])].reset_index(drop=True)
        y, sub = (p.lab == "Stress").to_numpy(int), p["sub"].to_numpy()
        hr_ok = p.hr.notna().to_numpy()
        hr_bin, eda_bin = np.floor(p.hr / 5), np.floor(p.eda_z / 0.5)
        for score in ("within_LOSO", "external"):
            s = p[score].to_numpy()
            base = dict(negatives=neg_name, score=score)
            def add(cond, rows_idx, stat, **kw):
                lo, hi = boot_ci(lambda i: stat(rows_idx[i]), sub[rows_idx])
                rows.append(base | dict(condition=cond, n=len(rows_idx), n_stress=int(y[rows_idx].sum()),
                                        auc=stat(rows_idx), lo=lo, hi=hi) | kw)
            allr = np.arange(len(p))
            pooled = lambda i: safe_auc(y[i], s[i])
            add("a: no matching", allr, pooled)
            add("a': no matching, HR available", allr[hr_ok], pooled)
            for key_name, keys, ok in [("b: Kwon iso-HR (5 bpm, within subject)", hr_bin, hr_ok),
                                       ("f: Kwon-style iso-EDA (0.5 SD, within subject)", eda_bin, p.eda_z.notna().to_numpy()),
                                       ("g: Kwon-style iso-HR x EDA (within subject)", hr_bin.astype(str) + "_" + eda_bin.astype(str), hr_ok)]:
                idx = allr[ok]
                aucs, ns, cis = [], [], []
                for seed in SEEDS:
                    keep = idx[iso_match(np.asarray(keys)[idx], y[idx], sub[idx], seed)]
                    aucs.append(pooled(keep)); ns.append(len(keep)); cis.append(boot_ci(lambda i: pooled(keep[i]), sub[keep]))
                    seed_rows.append(base | dict(condition=key_name, seed=seed, n=len(keep), auc=aucs[-1]))
                rows.append(base | dict(condition=key_name, n=float(np.mean(ns)), n_stress=float(np.mean(ns)) / 2,
                                        auc=np.mean(aucs), auc_sd=np.std(aucs, ddof=1), n_subjects=len(np.unique(sub[keep])),
                                        lo=np.mean([c[0] for c in cis]), hi=np.mean([c[1] for c in cis])))
            for cond, col in [("c: quintiles, HR-only index", "hr_z"), ("d: quintiles, HR+EDA index", "hrdx"),
                              ("e: quintiles, EDA-only index", "eda_z")]:
                ok = allr[p[col].notna().to_numpy()]
                a, ss, yy = p[col].to_numpy()[ok], s[ok], y[ok]
                lo, hi = boot_ci(lambda i: binned_auc(a, ss, yy, i), sub[ok])
                rows.append(base | dict(condition=cond, n=len(ok), n_stress=int(yy.sum()), auc=binned_auc(a, ss, yy), lo=lo, hi=hi))
    R = pd.DataFrame(rows).round(3)
    R.to_csv(OUT / "kwon_reconciliation.csv", index=False)
    pd.DataFrame(seed_rows).round(3).to_csv(OUT / "kwon_matching_seeds.csv", index=False)
    # Where does Table IV's pooled 0.584 / 0.784 come from? Same reference pair (stress vs Baseline/Rest, HR+EDA
    # index quintiles), pooled over datasets as in the probe, then one dataset at a time. Point estimates only.
    a_all = np.where(z.hr_mean.isna(), z.eda_tonic_mean, (z.hr_mean + z.eda_tonic_mean) / 2)
    lab_all, y_all = raw.harmonized_label.to_numpy(), (raw.harmonized_label == "Stress").to_numpy(int)
    pair_all = np.isin(lab_all, ["Stress", "Baseline", "Rest"]) & (raw.dataset != "EPM-E4").to_numpy()
    dec = []
    for ds in ["pooled (Table IV)"] + sorted(raw.dataset[pair_all & (y_all == 1)].unique()):
        m = pair_all & ((raw.dataset == ds).to_numpy() if ds[0] != "p" else True)
        for score, col in (("within_LOSO", "p_win"), ("external", "p_ext")):
            s = sc[col].to_numpy()
            dec.append(dict(dataset=ds, score=score, n=int(m.sum()), n_stress=int(y_all[m].sum()),
                            auc_unmatched=safe_auc(y_all[m], s[m]), auc_hr_eda_quintiles=binned_auc(a_all[m], s[m], y_all[m])))
    D = pd.DataFrame(dec).round(3)
    D.to_csv(OUT / "table4_reference_by_dataset.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(R[["negatives", "score", "condition", "n", "n_stress", "auc", "auc_sd", "lo", "hi"]].to_string(index=False))
        print(D.to_string(index=False))


if __name__ == "__main__":
    main()
