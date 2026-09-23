"""Trained two-feature arousal model under LODO: logistic regression on per-subject z-scored hr_mean and
eda_tonic_mean, against the untrained index and the HR/HRV/EDA XGBoost model (same windows and splits).

Missing heart rate is set to 0 (the subject's own mean after z-scoring); no missingness indicator, so the
model cannot use missing pulse as a motion cue. Class-balanced weights, threshold 0.5, as the XGBoost model.
Writes outputs/tables/jbhi_v2/two_feature_lr/{per_split,summary}.csv.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from compare_models import TEST_TRAIN_RATIO, holm  # noqa: E402
from leave_one_dataset_out import NON_STRESS, normalise  # noqa: E402
from run_jbhi_experiments import META, grouped_split  # noqa: E402

TABLES = REPO / "outputs" / "tables" / "jbhi_v2"
OUT = TABLES / "two_feature_lr"


def nb(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired Nadeau-Bengio corrected test of a - b over splits, with 95% CI and 90% CI (TOST at +/-0.05)."""
    d = a - b
    se = np.sqrt((1 / len(d) + TEST_TRAIN_RATIO) * d.var(ddof=1))
    t95, t90 = stats.t.ppf(0.975, len(d) - 1), stats.t.ppf(0.95, len(d) - 1)
    p = 2 * stats.t.sf(abs(d.mean() / se), len(d) - 1) if se > 0 else 1.0
    return dict(diff=d.mean(), ci95_lo=d.mean() - t95 * se, ci95_hi=d.mean() + t95 * se,
                equiv_005=bool(d.mean() - t90 * se > -0.05 and d.mean() + t90 * se < 0.05), p=p)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", type=Path, default=Path(os.environ.get("HARMONIZED_CSV", REPO / "data/processed/combined/harmonized_windows_v2.csv")))
    args = cli.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input)
    phys = [c for c in raw.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_", "temp_"))]
    df = raw.copy(); df[phys] = normalise(raw, phys, "session")
    df = df[df["harmonized_label"].isin(NON_STRESS + ["Stress"])]
    datasets = [d for d, g in df.groupby("dataset") if (g["harmonized_label"] == "Stress").any()]
    df = df[df["dataset"].isin(datasets)].reset_index(drop=True)
    y = (df["harmonized_label"] == "Stress").to_numpy(int)
    x = np.column_stack([df["hr_mean"].fillna(0.0), df["eda_tonic_mean"]])
    index = np.where(df["hr_mean"].isna(), df["eda_tonic_mean"], (df["hr_mean"] + df["eda_tonic_mean"]) / 2)

    xgb = pd.read_csv(TABLES / "leave_one_dataset_out_per_split.csv")
    xgb = xgb[(xgb.features == "hr_hrv_eda") & (xgb.trained_on == "other_datasets_only")].set_index(["test_dataset", "split"])

    rows, coefs = [], []
    for target in datasets:
        tgt, oth = np.flatnonzero(df["dataset"] == target), np.flatnonzero(df["dataset"] != target)
        lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(x[oth], y[oth])
        coefs.append(dict(test_dataset=target, coef_hr=lr.coef_[0, 0], coef_eda=lr.coef_[0, 1], intercept=lr.intercept_[0]))
        cand = np.unique(np.quantile(index[oth], np.linspace(0.01, 0.99, 197)))
        thr = cand[np.argmax([balanced_accuracy_score(y[oth], index[oth] > c) for c in cand])]
        for seed in range(20):
            _, test = grouped_split(df.iloc[tgt], seed)
            te = tgt[test]
            if len(set(y[te])) < 2:
                continue
            p = lr.predict_proba(x[te])[:, 1]
            rows.append(dict(test_dataset=target, split=seed,
                             lr_ba=balanced_accuracy_score(y[te], p > 0.5), lr_auroc=roc_auc_score(y[te], p),
                             index_ba=balanced_accuracy_score(y[te], index[te] > thr), index_auroc=roc_auc_score(y[te], index[te]),
                             xgb_ba=xgb.loc[(target, seed), "balanced_accuracy"], xgb_auroc=xgb.loc[(target, seed), "auroc"]))
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_split.csv", index=False)

    summary = []
    for target, g in per.groupby("test_dataset", sort=False):
        base = dict(test_dataset=target, n_splits=len(g), lr_ba=g.lr_ba.mean(), lr_auroc=g.lr_auroc.mean(),
                    index_ba=g.index_ba.mean(), xgb_ba=g.xgb_ba.mean(), xgb_auroc=g.xgb_auroc.mean())
        for tag, a, b in (("xgb_minus_lr", g.xgb_ba, g.lr_ba), ("lr_minus_index", g.lr_ba, g.index_ba)):
            base |= {f"{tag}_{k}": v for k, v in nb(a.to_numpy(), b.to_numpy()).items()}
        summary.append(base)
    summary = pd.DataFrame(summary).merge(pd.DataFrame(coefs), on="test_dataset")
    for tag in ("xgb_minus_lr", "lr_minus_index"):
        summary[f"{tag}_p_holm"] = holm(summary[f"{tag}_p"]).to_numpy()
    summary.round(4).to_csv(OUT / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
