#!/usr/bin/env python3
"""EDA quality screen (Kleckner et al. 2018 rules) and an HR/HRV-only arm (REV-7, REV-14).

1. Quality flags per 60 s window, from the raw E4 EDA/TEMP at 4 Hz, on exactly the windows of the feature table:
   extract_features.py's own loaders are reused, with windows_for_recording swapped for a function that walks the
   same segments with the same WINDOW/STEP but computes flags instead of features. Rules (Kleckner Table 1):
     R1 EDA not within 0.05-60 uS; R2 EDA slope faster than +-10 uS/s (sample to sample); R3 skin temperature not
     within 30-40 C (not applicable where TEMP is absent: UBFC-Phys, PhysioNet STRESS f07); R4 5 s either side of R1-R3.
   Deviation: Kleckner low-pass filter the 32 Hz Q-sensor signal at 0.35 Hz first; the E4 signal is used unfiltered
   (4 Hz), which makes R2 stricter. Flat window: raw EDA range within the window < 0.05 uS (the SCR minimum Kleckner
   use for R1). A window is rejected when more than REJECT_FRAC of its samples are invalid.
2. Reruns on screened windows: LODO (HR/HRV/EDA, leave_one_dataset_out.py logic), specificity-panel false-stress rates
   (external and added-negative, specificity_panel_probe.py logic), matched-arousal within-bin AUROC and delta vs the
   baseline/rest reference (reviewer_reanalyses.py part A logic). "none" (all windows) is run too as a reproduction check.
3. HR/HRV-only arm on all windows: LODO and specificity panel.
Screened windows are dropped before per-subject z-scoring, so the scaling reference excludes them as well.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import extract_features as ef  # noqa: E402
from leave_one_dataset_out import EXERCISE, NON_STRESS, fit_predict, normalise  # noqa: E402
from run_jbhi_experiments import META, grouped_split  # noqa: E402

REJECT_FRAC = 0.10
FLAT_US = 0.05
PAD = 5 * ef.FS["EDA"]
STRESS_DS = ["WESAD", "PhysioNet", "Stress-Predict", "UBFC-Phys", "Campanella2024"]
KEY = ["dataset", "subject_id", "harmonized_label", "timestamp_start"]
NJ = 8


# ---------------------------------------------------------------- 1. quality flags from raw signals
def dilate(mask: np.ndarray) -> np.ndarray:
    return np.convolve(mask.astype(float), np.ones(2 * PAD + 1), "same") > 0


def quality_windows(signals, segments, meta, ppg_valid=True):
    """Drop-in for ef.windows_for_recording: same window grid, quality flags instead of features."""
    eda, fs = signals["EDA"], ef.FS["EDA"]
    r1 = (eda < 0.05) | (eda > 60)
    jump = np.abs(np.diff(eda)) * fs > 10
    r2 = np.zeros(len(eda), bool); r2[1:] |= jump; r2[:-1] |= jump
    r3 = np.zeros(len(eda), bool)
    temp = signals.get("TEMP") if ppg_valid else None  # f07: protection dock on, TEMP invalid (NO_PPG_TEMP)
    if temp is not None:
        n = min(len(temp), len(eda)); r3[:n] = (temp[:n] < 30) | (temp[:n] > 40)
    bad_all, bad_eda = dilate(r1 | r2 | r3), dilate(r1 | r2)
    duration = min(len(signals[k]) / ef.FS[k] for k in signals)
    rows = []
    for seg_start, seg_end, _, harmonized in segments:
        start = max(0.0, seg_start)
        while start + ef.WINDOW <= min(seg_end, duration):
            sl = slice(int(start * fs), int((start + ef.WINDOW) * fs))
            w = eda[sl]
            rows.append(meta | {"harmonized_label": harmonized, "timestamp_start": start, "temp_available": temp is not None,
                                "r1_range": r1[sl].mean(), "r2_slope": r2[sl].mean(), "r3_temp": r3[sl].mean(),
                                "invalid_frac": bad_all[sl].mean(), "invalid_frac_eda_rules": bad_eda[sl].mean(),
                                "raw_eda_range": w.max() - w.min(), "raw_eda_mean": w.mean()})
            start += ef.STEP
    return rows


def window_quality(data_root: Path, features: pd.DataFrame, cache: Path) -> pd.DataFrame:
    if not cache.exists():
        ef.windows_for_recording = quality_windows  # the extractors look this name up at call time
        q = pd.DataFrame([r for fn in ef.EXTRACTORS.values() for r in fn(data_root)])
        q.to_csv(cache, index=False)
    q = pd.read_csv(cache)
    for d in (q, features):
        d["timestamp_start"] = d["timestamp_start"].round(3)
    q = q.drop(columns=["subject_uid"]).drop_duplicates(KEY)
    out = features.merge(q, on=KEY, how="left", validate="one_to_one")
    missing = out["invalid_frac"].isna().sum()
    assert missing == 0, f"{missing} feature windows have no raw-signal match"
    out["reject_kleckner"] = out["invalid_frac"] > REJECT_FRAC
    out["reject_eda_rules"] = out["invalid_frac_eda_rules"] > REJECT_FRAC
    out["flat"] = out["raw_eda_range"] < FLAT_US
    return out


# ---------------------------------------------------------------- 2/3. analyses (logic of the committed scripts)
def lodo(raw: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    """leave_one_dataset_out.py defaults: session z-scoring without exercise, Stress vs NON_STRESS, 20 splits."""
    df = raw[~raw["harmonized_label"].isin(EXERCISE)].copy()
    phys = sorted({c for cols in feature_sets.values() for c in cols})
    df[phys] = normalise(df, phys, "session")
    df = df[df["harmonized_label"].isin(NON_STRESS + ["Stress"]) & df["dataset"].isin(STRESS_DS)].reset_index(drop=True)
    y = (df["harmonized_label"] == "Stress").to_numpy(int)
    rows = []
    for fname, cols in feature_sets.items():
        x = df[cols].to_numpy()
        for target in STRESS_DS:
            t_idx, o_idx = np.flatnonzero(df["dataset"] == target), np.flatnonzero(df["dataset"] != target)
            for seed in range(20):
                train, test = grouped_split(df.iloc[t_idx], seed)
                truth = y[t_idx[test]]
                if len(set(truth)) < 2:
                    continue
                for name, idx in (("within_dataset", t_idx[train]), ("other_datasets_only", o_idx),
                                  ("all_datasets", np.concatenate([t_idx[train], o_idx]))):
                    p = fit_predict(x[idx], y[idx], x[t_idx[test]], NJ)
                    rows.append({"features": fname, "test_dataset": target, "trained_on": name, "split": seed,
                                 "balanced_accuracy": balanced_accuracy_score(truth, p > 0.5), "auroc": roc_auc_score(truth, p)})
    return pd.DataFrame(rows)


PANEL = {"Campanella Manual task": ("Campanella2024", ["Manual task"]), "UBFC Control task": ("UBFC-Phys", ["Control task"]),
         "Hyperventilation": ("Stress-Predict", ["Hyperventilation"]), "PhysioNet Aerobic": ("PhysioNet", ["Aerobic"]),
         "PhysioNet Anaerobic": ("PhysioNet", ["Anaerobic"]), "EPM-E4 emotions": ("EPM-E4", ["Anger", "Fear", "Happiness", "Sadness"])}
MATCHED = {"exercise": ("PhysioNet", ["Aerobic", "Anaerobic"]), "Hyperventilation": ("Stress-Predict", ["Hyperventilation"]),
           "UBFC Control task": ("UBFC-Phys", ["Control task"]), "Campanella Manual task": ("Campanella2024", ["Manual task"]),
           "EPM Fear": ("EPM-E4", ["Fear"]), "EPM Anger": ("EPM-E4", ["Anger"])}
REF = (None, ["Baseline", "Rest"])


class Setup:
    """Probe-script setup: session z-scoring over the whole table (exercise borrows protocol statistics)."""
    def __init__(self, raw: pd.DataFrame, cols: list[str]):
        phys = [c for c in raw.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
        self.df = raw.reset_index(drop=True).copy()
        self.df[phys] = normalise(self.df, phys, "session")
        self.X = self.df[cols].to_numpy()
        self.lab, self.ds, self.subj = self.df["harmonized_label"], self.df["dataset"].to_numpy(), self.df["subject_uid"].to_numpy()
        self.core = self.lab.isin(NON_STRESS + ["Stress"]).to_numpy()
        self.y = (self.lab == "Stress").to_numpy(int)

    def external(self) -> np.ndarray:
        p = np.full(len(self.df), np.nan)
        for d in np.unique(self.ds):
            tr, te = self.core & (self.ds != d), self.ds == d
            p[te] = fit_predict(self.X[tr], self.y[tr], self.X[te], NJ)
        return p

    def added_negative(self, labels: list[str]) -> np.ndarray:
        """Mean held-out score over 20 grouped splits with the probed class as a training negative."""
        acc, cnt = np.zeros(len(self.df)), np.zeros(len(self.df))
        is_probe = self.lab.isin(labels).to_numpy()
        y3 = np.where(is_probe, 0, self.y)
        for seed in range(20):
            trm, tem = grouped_split(self.df, seed)
            tr, te = trm & (self.core | is_probe), tem & (self.core | is_probe)
            acc[te] += fit_predict(self.X[tr], y3[tr], self.X[te], NJ); cnt[te] += 1
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def rate_ci(sub, hit, rng, n=1000):
    cnt = pd.Series(np.asarray(hit, float)).groupby(np.asarray(sub)).agg(["sum", "size"])
    bs = [(c := cnt.iloc[rng.integers(0, len(cnt), len(cnt))])["sum"].sum() / c["size"].sum() for _ in range(n)]
    return np.mean(hit), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(cnt)


def panel(s: Setup, p_ext: np.ndarray, p_add: dict[str, np.ndarray], rng) -> pd.DataFrame:
    rows = []
    for name, (d, labels) in PANEL.items():
        for label in labels:
            m = (s.ds == d) & (s.lab == label).to_numpy()
            for cond, p in (("external", p_ext), ("added_negative", p_add[name])):
                r, lo, hi, n_sub = rate_ci(s.subj[m], p[m] > 0.5, rng)
                rows.append(dict(probe_class=f"{d} {label}", condition=cond, n_subjects=n_sub, n_windows=int(m.sum()),
                                 false_stress=r, lo=lo, hi=hi))
    for d in STRESS_DS:  # stress recall under the external model, for context
        m = (s.ds == d) & (s.y == 1)
        r, lo, hi, n_sub = rate_ci(s.subj[m], p_ext[m] > 0.5, rng)
        rows.append(dict(probe_class=f"{d} Stress (recall)", condition="external", n_subjects=n_sub, n_windows=int(m.sum()),
                         false_stress=r, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def bins_of(a, nb=5):
    edges = np.unique(np.quantile(a, np.linspace(0, 1, nb + 1)))
    return np.clip(np.searchsorted(edges, a, side="right") - 1, 0, len(edges) - 2), len(edges) - 1


def wauc(b, nbin, sc, yy):
    num = den = 0.0
    for k in range(nbin):
        mm = b == k
        if len(np.unique(yy[mm])) < 2:
            continue
        num += roc_auc_score(yy[mm], sc[mm]) * mm.sum(); den += mm.sum()
    return num / den if den else np.nan


class Pair:  # as reviewer_reanalyses.py part A (that script runs on import, so the logic is copied)
    def __init__(self, s: Setup, mask, score, arousal):
        ok = mask & np.isfinite(arousal) & np.isfinite(score)
        self.s, self.y, sub = score[ok], s.y[ok], s.subj[ok]
        self.b, self.nbin = bins_of(arousal[ok])
        self.auc = wauc(self.b, self.nbin, self.s, self.y)
        self.n = int(ok.sum())
        self.by_sub = {u: np.flatnonzero(sub == u) for u in np.unique(sub)}

    def boot(self, picks):
        idx = np.concatenate([self.by_sub[u] for u in picks if u in self.by_sub])
        return wauc(self.b[idx], self.nbin, self.s[idx], self.y[idx])


def pair_mask(s: Setup, probe):
    d, labels = probe
    probe_m = s.lab.isin(labels).to_numpy() & (s.ds == d if d else s.core)
    return ((s.y == 1) & ((s.ds == d) if d in STRESS_DS else True)) | probe_m


def matched(s: Setup, p_ext, p_add: dict[str, np.ndarray], rng, n_boot) -> pd.DataFrame:
    hr, eda = s.df["hr_mean"].to_numpy(), s.df["eda_tonic_mean"].to_numpy()
    arousal = np.where(np.isnan(hr), eda, (hr + eda) / 2)
    sub_ds = pd.Series(s.ds, index=s.subj).groupby(level=0).first()
    rows = []
    for score_name in ("external", "cond3_added_negative"):
        for name, probe in MATCHED.items():
            sc = p_ext if score_name == "external" else p_add[name]
            pr, rf = Pair(s, pair_mask(s, probe), sc, arousal), Pair(s, pair_mask(s, REF), sc, arousal)
            units = sub_ds[sorted(set(pr.by_sub) | set(rf.by_sub))]
            groups = [g.index.to_numpy() for _, g in units.groupby(units)]
            d = np.empty(n_boot)
            for i in range(n_boot):
                picks = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
                d[i] = pr.boot(picks) - rf.boot(picks)
            rows.append(dict(score=score_name, probe=name, n_pair=pr.n, auc_probe=pr.auc, auc_ref=rf.auc,
                             delta_ref=pr.auc - rf.auc, delta_lo=np.nanpercentile(d, 2.5), delta_hi=np.nanpercentile(d, 97.5)))
    return pd.DataFrame(rows)


def lodo_summary(per_split: pd.DataFrame) -> pd.DataFrame:
    return per_split.groupby(["features", "test_dataset", "trained_on"], sort=False)[["balanced_accuracy", "auroc"]].mean().round(3).reset_index()


# Table I demographics, verbatim from the descriptor full texts in sources/ (git-ignored). Counts describe each
# descriptor's cohort, not necessarily the subjects kept here (n_used is added from the feature table).
DEMOGRAPHICS = [
    ("WESAD", "schmidt-2018-wesad.txt:220-222", 15, "27.5 +- 2.4 (mean +- SD)", 12, 3,
     "The remaining 15 subjects had a mean age of 27.5 ± 2.4 years. Twelve subjects were male and the other three subjects were female."),
    ("PhysioNet", "hongn-2025-scidata-article.txt:152", 36, "19-30 (range)", None, None,
     "Participants were males and females between 19 and 30 years. Out of the 36 volunteers participating in the study, all 36 "
     "completed the stress protocol, 31 completed the anaerobic session, and 30 completed the aerobic session."),
    ("Stress-Predict", "iqbal-2022-stress-predict.txt:121", 35, "32 +- 8.2 (mean +- SD)", 10, 25,
     "There were 25 women and 10 men participants (mean age = 32 ± 8.2 years) in the created dataset. The data collection "
     "protocol was not followed properly for one (1) participant, and their data were removed from further analysis."),
    ("UBFC-Phys", "meziatisabour-2021-ubfc-phys.txt:265-269 (two-column extraction, left column)", 56,
     "21.8 +- 3.11 (mean +- SD); 19-38 (range)", 10, 46,
     "56 healthy subjects (12 participants were eliminated due to technical problems or data sharing refusal). Participants are "
     "all aged between 19 and 38 (mean age is 21.8 and standard deviation is 3.11). Among these participants 46 are female and 10 male."),
    ("Campanella2024", "campanella-2024-empatica-e4-stress-dataset.txt:265", 29, "20-60 (range)", 21, 8,
     "A total of 29 subjects (21 male and 8 female) from 20 to 60 years of age have been enrolled for the study to ensure variety."),
    ("EPM-E4", "garciamoreno-2020-epm-e4.txt:14 (Zenodo record)", 53, "not reported", None, None,
     "derived from 53 participants using the Empatica E4 wearable"),
]


def physionet_subject_info(data_root: Path) -> dict:
    """PhysioNet Table 1 is an image in the article text; the dataset's subject-info.csv has per-subject values."""
    s = pd.read_csv(next(data_root.glob("wearable-device-dataset/*/subject-info.csv"))).iloc[:36]
    s.columns = [c.strip() for c in s.columns]
    age = pd.to_numeric(s["Age"], errors="coerce")
    return dict(file_age=f"{age.mean():.1f} +- {age.std():.1f} (mean +- SD, n={age.notna().sum()}); {age.min():.0f}-{age.max():.0f}",
                file_male=int((s["Gender"] == "m").sum()), file_female=int((s["Gender"] == "f").sum()))


def write_demographics(data_root: Path, feats: pd.DataFrame, path: Path) -> None:
    d = pd.DataFrame(DEMOGRAPHICS, columns=["dataset", "source_line", "n_descriptor", "age_descriptor", "male", "female", "quote"])
    d["n_used_here"] = d["dataset"].map(feats.groupby("dataset")["subject_uid"].nunique())
    pn = physionet_subject_info(data_root)
    d.loc[d.dataset == "PhysioNet", ["age_from_subject_info_csv", "male", "female"]] = [pn["file_age"], pn["file_male"], pn["file_female"]]
    d.to_csv(path, index=False)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", type=Path, default=Path("/Users/octa/Projects/smartwatch-stress-detection/data/processed/combined/harmonized_windows_v2.csv"))
    cli.add_argument("--data-root", type=Path, default=Path("/Users/octa/Projects/smartwatch-stress-detection"))
    cli.add_argument("--output-dir", type=Path, default=REPO / "outputs/tables/jbhi_v2/eda_quality_hr_arm")
    cli.add_argument("--n-boot", type=int, default=1000)
    args = cli.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    feats = pd.read_csv(args.input)
    write_demographics(args.data_root, feats, out / "demographics.csv")
    q = window_quality(args.data_root, feats, out / "window_quality_raw.csv")
    q[["window_id"] + KEY + ["temp_available", "r1_range", "r2_slope", "r3_temp", "invalid_frac", "invalid_frac_eda_rules",
       "raw_eda_range", "reject_kleckner", "reject_eda_rules", "flat"]].to_csv(out / "window_quality.csv", index=False)
    rates = q.groupby(["dataset", "harmonized_label"]).agg(
        n=("flat", "size"), any_r1=("r1_range", lambda v: (v > 0).mean()), any_r2=("r2_slope", lambda v: (v > 0).mean()),
        any_r3=("r3_temp", lambda v: (v > 0).mean()), temp_available=("temp_available", "mean"),
        reject_kleckner=("reject_kleckner", "mean"), reject_eda_rules=("reject_eda_rules", "mean"), flat=("flat", "mean"),
        reject_or_flat=("flat", lambda v: (v | q.loc[v.index, "reject_kleckner"]).mean())).round(3)
    rates.to_csv(out / "rejection_rates.csv")
    print(rates.to_string(), f"\n[{time.time() - t0:.0f}s]", flush=True)

    phys = [c for c in feats.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
    hr_eda = [c for c in phys if not c.startswith("temp_")]
    hr_only = [c for c in phys if c.startswith(("hr_", "hrv_"))]
    screens = {"none": q, "kleckner": q[~q.reject_kleckner], "eda_rules_only": q[~q.reject_eda_rules], "kleckner_or_flat": q[~(q.reject_kleckner | q.flat)]}
    lodo_rows, panel_rows, matched_rows = [], [], []
    for sname, sub in screens.items():
        rng = np.random.default_rng(0)
        feature_sets = {"hr_hrv_eda": hr_eda} | ({"hr_hrv_only": hr_only} if sname == "none" else {})
        lodo_rows.append(lodo_summary(lodo(sub[feats.columns], feature_sets)).assign(screen=sname))
        for fname, cols in feature_sets.items():
            s = Setup(sub[feats.columns], cols)
            p_ext = s.external()
            p_add = {n: s.added_negative(labels) for n, (_, labels) in (PANEL | MATCHED).items()}
            panel_rows.append(panel(s, p_ext, p_add, rng).assign(screen=sname, features=fname))
            if fname == "hr_hrv_eda":
                matched_rows.append(matched(s, p_ext, p_add, rng, args.n_boot).assign(screen=sname))
        print(f"screen {sname} done [{time.time() - t0:.0f}s]", flush=True)
    L = pd.concat(lodo_rows); L.to_csv(out / "lodo_summary.csv", index=False)
    P = pd.concat(panel_rows).round(3); P.to_csv(out / "specificity_panel.csv", index=False)
    M = pd.concat(matched_rows).round(3); M.to_csv(out / "matched_arousal.csv", index=False)
    print(L.pivot_table(index=["screen", "features", "test_dataset"], columns="trained_on", values="balanced_accuracy").to_string())
    print(P.pivot_table(index="probe_class", columns=["features", "screen", "condition"], values="false_stress").to_string())
    print(M.to_string(index=False))
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
