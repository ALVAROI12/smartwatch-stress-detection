"""Novelty plan step 3: does wrist HRV move in the same direction as chest-ECG HRV under stress (WESAD)?

Input: outputs/tables/jbhi_v2/wrist_vs_ecg_per_window.csv (validate_cardiac_against_ecg.py; windows with
usable wrist beats only). Per subject: mean over TSST windows minus mean over Baseline windows, for wrist PPG
and chest ECG. Subject-bootstrap 95% CIs. Also wrist-ECG agreement within each condition.
Output: outputs/tables/jbhi_v2/wrist_hrv_direction.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
T = REPO / "outputs" / "tables" / "jbhi_v2"


def boot_ci(v, n=5000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    return np.percentile(rng.choice(v, (n, len(v))).mean(1), [2.5, 97.5])


def main():
    w = pd.read_csv(T / "wrist_vs_ecg_per_window.csv")
    w = w[w.label.isin(["Baseline", "stress"]) | w.label.str.lower().isin(["baseline", "stress"])].copy()
    w["cond"] = np.where(w.label.str.lower() == "stress", "stress", "baseline")
    rows = []
    for m in ("hr_mean", "hrv_rmssd", "hrv_sdnn"):
        per = w.groupby(["subject", "cond"])[[f"ppg_{m}", f"ecg_{m}"]].mean().unstack("cond")
        d_ppg = (per[(f"ppg_{m}", "stress")] - per[(f"ppg_{m}", "baseline")]).dropna()
        d_ecg = (per[(f"ecg_{m}", "stress")] - per[(f"ecg_{m}", "baseline")]).dropna()
        both = d_ppg.index.intersection(d_ecg.index)
        r = dict(measure=m, n_subjects=len(both),
                 ecg_delta=d_ecg[both].mean(), ppg_delta=d_ppg[both].mean(),
                 same_sign_subjects=int((np.sign(d_ecg[both]) == np.sign(d_ppg[both])).sum()))
        r["ecg_lo"], r["ecg_hi"] = boot_ci(d_ecg[both])
        r["ppg_lo"], r["ppg_hi"] = boot_ci(d_ppg[both])
        for c in ("baseline", "stress"):
            s = w[w.cond == c][[f"ppg_{m}", f"ecg_{m}"]].dropna()
            r[f"r_{c}"] = s.corr().iloc[0, 1]
            r[f"n_windows_{c}"] = len(s)
        rows.append(r)
    t = pd.DataFrame(rows).round(3)
    t.to_csv(T / "wrist_hrv_direction.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
