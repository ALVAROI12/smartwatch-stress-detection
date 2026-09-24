"""Novelty plan step 4: stressor-potency dose-response.

Unit = subject x stressor (original label). Response = mean arousal index (per-subject session z of hr_mean and
eda_tonic_mean, as in arousal_kit.py) over that stressor's windows minus the mean over the subject's Baseline
windows. Recall = share of the stressor's windows with score > 0.5 (external and within models, from
contribution_probes/arousal/scores.csv). Questions:
  1. Does recall rise with the subject's response (Spearman, subject-cluster bootstrap)?
  2. Once response is known, does dataset identity still explain recall (Delta R^2, subject-cluster bootstrap)?
  3. Across stressors, does mean response order mean recall?
Output: outputs/tables/jbhi_v2/contribution_probes/stressor_potency/{units,summary,by_stressor}.csv
Caveat: the detector is close to an arousal index (arousal_kit.py), so (1) is expected; (2) is the test.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def r2(X, y, w):
    X = np.c_[np.ones(len(y)), X]
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    res = y - X @ beta
    ybar = np.average(y, weights=w)
    return 1 - np.sum(w * res ** 2) / np.sum(w * (y - ybar) ** 2)


def main():
    from leave_one_dataset_out import normalise
    from run_jbhi_experiments import META

    out = REPO / "outputs" / "tables" / "jbhi_v2" / "contribution_probes" / "stressor_potency"
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(Path(os.environ.get("HARMONIZED_CSV",
                           "/Users/octa/Projects/smartwatch-stress-detection/data/processed/combined/harmonized_windows_v2.csv")))
    phys = [c for c in raw.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
    df = raw.copy()
    df[phys] = normalise(raw, phys, "session")
    sc = pd.read_csv(REPO / "outputs/tables/jbhi_v2/contribution_probes/arousal/scores.csv")
    assert (sc.window_id.to_numpy() == df.window_id.to_numpy()).all()
    df["idx"] = np.where(df.hr_mean.isna(), df.eda_tonic_mean, (df.hr_mean + df.eda_tonic_mean) / 2)
    df["p_ext"], df["p_win"] = sc.p_ext.to_numpy(), sc.p_win.to_numpy()

    base = df[df.harmonized_label == "Baseline"].groupby("subject_uid").idx.mean()
    st = df[(df.harmonized_label == "Stress") & df.idx.notna()]
    st = st[st.original_label != "Subtract"]  # one window in PhysioNet
    u = st.groupby(["dataset", "original_label", "subject_uid"]).agg(
        n=("idx", "size"), stress_idx=("idx", "mean"),
        recall_ext=("p_ext", lambda p: (p > 0.5).mean()), recall_win=("p_win", lambda p: (p > 0.5).mean())).reset_index()
    u["response"] = u.stress_idx - u.subject_uid.map(base)
    u = u.dropna(subset=["response"])
    u["stressor"] = u.dataset + ": " + u.original_label.str.replace(r"\s*\(.*\)", "", regex=True)
    u.round(4).to_csv(out / "units.csv", index=False)

    rng = np.random.default_rng(0)
    subs = u.subject_uid.unique()
    rows_of = {s: np.flatnonzero(u.subject_uid.to_numpy() == s) for s in subs}
    D = pd.get_dummies(u.dataset, drop_first=True).to_numpy(float)
    summ = []
    for model in ("ext", "win"):
        y, x, w = u[f"recall_{model}"].to_numpy(), u.response.to_numpy(), u.n.to_numpy(float)
        rho = spearmanr(x, y).statistic
        r2_resp, r2_ds, r2_both = r2(x[:, None], y, w), r2(D, y, w), r2(np.c_[x, D], y, w)
        boots = []
        for _ in range(2000):
            i = np.concatenate([rows_of[s] for s in rng.choice(subs, len(subs))])
            if len(np.unique(u.dataset.to_numpy()[i])) < len(u.dataset.unique()):
                continue
            boots.append((spearmanr(x[i], y[i]).statistic,
                          r2(np.c_[x[i], D[i]], y[i], w[i]) - r2(x[i][:, None], y[i], w[i]),
                          r2(np.c_[x[i], D[i]], y[i], w[i]) - r2(D[i], y[i], w[i])))
        b = np.array(boots)
        summ.append(dict(model=model, n_units=len(u), n_subjects=len(subs), rho=rho,
                         rho_lo=np.percentile(b[:, 0], 2.5), rho_hi=np.percentile(b[:, 0], 97.5),
                         r2_response=r2_resp, r2_dataset=r2_ds, r2_both=r2_both,
                         delta_r2_dataset_given_response=r2_both - r2_resp,
                         delta_lo=np.percentile(b[:, 1], 2.5), delta_hi=np.percentile(b[:, 1], 97.5),
                         delta_r2_response_given_dataset=r2_both - r2_ds,
                         delta_resp_lo=np.percentile(b[:, 2], 2.5), delta_resp_hi=np.percentile(b[:, 2], 97.5)))
    s = pd.DataFrame(summ).round(3)
    s.to_csv(out / "summary.csv", index=False)
    g = u.groupby("stressor").agg(n_subjects=("subject_uid", "nunique"), response=("response", "mean"),
                                  recall_ext=("recall_ext", "mean"), recall_win=("recall_win", "mean")).round(3)
    g = g.sort_values("response")
    g.to_csv(out / "by_stressor.csv")
    rs = spearmanr(g.response, g.recall_ext).statistic, spearmanr(g.response, g.recall_win).statistic
    print(s.to_string(index=False))
    print(g.to_string())
    print(f"stressor-level Spearman (k={len(g)}): external {rs[0]:.2f}, within {rs[1]:.2f}")


if __name__ == "__main__":
    main()
