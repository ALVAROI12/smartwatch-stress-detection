#!/usr/bin/env python3
"""Per-user few-shot calibration of a model trained on OTHER datasets (contribution 4b).

For each target dataset, one XGBoost model is trained on the other datasets only (as the
"other_datasets_only" arm of leave_one_dataset_out.py: stress vs all non-stress, HR/HRV/EDA features,
per-subject scaling). Each target user then gives k labelled windows per class (support), chosen as in
domain_adaptation.split_support_query: the first k of each class in time (default) or at random, with
query windows within WINDOW_SEC + gap of a support window dropped. Every method is scored on the SAME
query windows:

  source_only   the external model at threshold 0.5
  threshold     external model; per-user threshold at the midpoint of its mean support scores per class
  refit         XGBoost refitted on the other datasets + this user's support windows, the support given
                SUPPORT_SHARE of the total training weight (fixed in advance, not tuned)
  support_only  nearest class centroid on the user's support windows alone (no other data)

Pooled balanced accuracy over each dataset's query windows; differences to source_only get a 95% CI and
a two-sided p from a subject-level bootstrap, Holm-corrected within each normalisation and selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from compare_models import holm
from leave_one_dataset_out import EXERCISE, NON_STRESS, normalise
from probe_normalisation import model
from run_jbhi_experiments import META, REPO_ROOT

# After xgboost: loading torch first makes XGBoost's fit segfault on macOS (clashing OpenMP runtimes).
from domain_adaptation import split_support_query  # noqa: E402

SUPPORT_SHARE = 0.1
METHODS = ["source_only", "threshold", "refit", "support_only"]


def class_weights(y: np.ndarray) -> np.ndarray:
    return np.where(y == 1, (y == 0).sum() / max(1, (y == 1).sum()), 1.0)


def centroid_score(xs: np.ndarray, ys: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Distance to the non-stress centroid minus distance to the stress centroid, ignoring missing features."""
    dist = [np.sqrt(np.nanmean((xq - np.nanmean(xs[ys == c], axis=0)) ** 2, axis=1)) for c in (0, 1)]
    return np.nan_to_num(dist[0] - dist[1])


def bootstrap(per_window: pd.DataFrame, method: str, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """95% CI and two-sided p for BA(method) - BA(source_only), resampling subjects."""
    rng, groups = np.random.default_rng(seed), dict(tuple(per_window.groupby("subject_uid")))
    subjects, diffs = list(groups), []
    for _ in range(n):
        s = pd.concat([groups[k] for k in rng.choice(subjects, len(subjects))])
        if s["y"].nunique() == 2:
            diffs.append(balanced_accuracy_score(s["y"], s[method] > 0.5)
                         - balanced_accuracy_score(s["y"], s["source_only"] > 0.5))
    diffs = np.array(diffs)
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return *np.percentile(diffs, [2.5, 97.5]).round(3), p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "combined" / "harmonized_windows_v2.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "tables" / "jbhi_v2" / "fewshot")
    parser.add_argument("--normalisation", choices=["session", "raw"], default="session")
    parser.add_argument("--support", choices=["chronological", "random"], default="chronological")
    parser.add_argument("--gap", type=float, default=30.0)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    tag = f"{args.normalisation}_{args.support}_gap{args.gap:g}"

    df = pd.read_csv(args.input, low_memory=False)
    df = df[~df["harmonized_label"].isin(EXERCISE)]
    cols = [c for c in df.columns if c not in META and c.startswith(("hr_", "hrv_", "eda_"))]
    df[cols] = normalise(df, cols, args.normalisation)
    df = df[df["harmonized_label"].isin(NON_STRESS + ["Stress"])]
    df = df[df["dataset"].isin(df.loc[df["harmonized_label"] == "Stress", "dataset"].unique())].reset_index(drop=True)
    x, y = df[cols].to_numpy(), (df["harmonized_label"] == "Stress").to_numpy(int)
    files, starts = df["subject_id"].astype(str).to_numpy(), df["timestamp_start"].to_numpy()

    rows = []
    for target in sorted(df["dataset"].unique()):
        t_mask = (df["dataset"] == target).to_numpy()
        o_idx = np.flatnonzero(~t_mask)
        w_src = class_weights(y[o_idx])
        source = model().set_params(n_jobs=args.n_jobs).fit(x[o_idx], y[o_idx], sample_weight=w_src)
        p_all = source.predict_proba(x)[:, 1]
        for k in args.k:
            rng = np.random.default_rng(k)
            for subject in df.loc[t_mask, "subject_uid"].unique():
                idx = np.flatnonzero(t_mask & (df["subject_uid"] == subject).to_numpy())
                if min((y[idx] == 0).sum(), (y[idx] == 1).sum()) < k:
                    continue
                support, query = split_support_query(idx, y, k, rng, files, starts, args.support, args.gap)
                if len(query) == 0:
                    continue
                ys = y[support]
                mid = (p_all[support][ys == 0].mean() + p_all[support][ys == 1].mean()) / 2
                w_sup = np.full(len(support), SUPPORT_SHARE * w_src.sum() / (1 - SUPPORT_SHARE) / len(support))
                refit = model().set_params(n_jobs=args.n_jobs).fit(
                    np.vstack([x[o_idx], x[support]]), np.concatenate([y[o_idx], ys]),
                    sample_weight=np.concatenate([w_src, w_sup]))
                rows.append(pd.DataFrame({
                    "dataset": target, "k": k, "subject_uid": subject, "y": y[query],
                    "source_only": p_all[query],
                    "threshold": p_all[query] - mid + 0.5,  # shifted so 0.5 is the user's threshold
                    "refit": refit.predict_proba(x[query])[:, 1],
                    "support_only": 0.5 + centroid_score(x[support], ys, x[query])}))
            print(f"done {target} k={k}", flush=True)
    per_window = pd.concat(rows, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_window.to_csv(args.output_dir / f"fewshot_windows_{tag}.csv", index=False)

    summary = []
    for (target, k), g in per_window.groupby(["dataset", "k"]):
        base = {"dataset": target, "k": k, "subjects": g["subject_uid"].nunique(), "query_windows": len(g),
                "query_stress": int(g["y"].sum())}
        if g["y"].nunique() < 2:
            summary.append({**base, "method": "not evaluable (one class in query)"})
            continue
        for method in METHODS:
            row = {**base, "method": method, "balanced_accuracy": round(balanced_accuracy_score(g["y"], g[method] > 0.5), 3),
                   "auroc_pooled": round(roc_auc_score(g["y"], g[method]), 3)}
            if method != "source_only":
                row["diff_lo"], row["diff_hi"], row["p_boot"] = bootstrap(g, method)
            summary.append(row)
    summary = pd.DataFrame(summary)
    tested = summary["p_boot"].notna() if "p_boot" in summary else pd.Series(False, index=summary.index)
    summary.loc[tested, "p_holm"] = holm(summary.loc[tested, "p_boot"]).round(3)
    summary.to_csv(args.output_dir / f"fewshot_summary_{tag}.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
