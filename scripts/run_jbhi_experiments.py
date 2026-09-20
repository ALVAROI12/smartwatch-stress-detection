#!/usr/bin/env python3
"""Subject-independent evaluation suite for the JBHI revision.

Runs on the output of relabel_windows.py and writes one CSV per experiment:

  harmonization_table   Dataset | Original Label | Harmonized Label | Subjects | Windows
  leakage_check         window-level random split vs subject-grouped split on the legacy 6-class task
  repeated_splits       20 subject-grouped 80/20 splits + LOSO, every task x modality set
  cross_dataset         train on one dataset, test on the other, shared labels only (Baseline vs Stress)

All splits group by subject_uid, so no participant is ever in both train and test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
META = {"window_id", "subject_id", "dataset", "label", "timestamp_start", "timestamp_end",
        "subject_uid", "original_label", "harmonized_label", "purity",
        "self_report_stress", "self_report_stress_delta", "self_report_validated", "sam_valence", "sam_arousal"}
EXERCISE = {"Aerobic", "Anaerobic"}
MIN_PURITY = 0.8


def modality_sets(features: list[str]) -> dict[str, list[str]]:
    return {
        "all_modalities": features,
        "acc_only": [f for f in features if f.startswith("acc_")],
        "all_except_acc": [f for f in features if not f.startswith("acc_")],
        "physiology_only": [f for f in features if f.startswith(("hr_", "hrv_", "eda_", "temp_"))],
    }


def tasks(df: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, str]]:
    clean = df[(df["purity"] >= MIN_PURITY) & (df["harmonized_label"] != "Excluded")]
    # ponytail: classes with too few pure windows to evaluate (EPM Neutral: 2) are dropped, not merged.
    clean = clean[clean.groupby("harmonized_label")["purity"].transform("size") >= 30]
    affective = clean[~clean["harmonized_label"].isin(EXERCISE)]
    shared = clean[clean["harmonized_label"].isin(["Baseline", "Stress"]) & clean["dataset"].isin(["WESAD", "PhysioNet"])]
    return {
        "legacy_6class_session_labels": (df, "label"),
        "harmonized_all_classes": (clean, "harmonized_label"),
        "affective_only_no_exercise": (affective, "harmonized_label"),
        "within_WESAD": (clean[clean["dataset"] == "WESAD"], "harmonized_label"),
        "within_PhysioNet_stress_session": (
            clean[(clean["dataset"] == "PhysioNet") & ~clean["harmonized_label"].isin(EXERCISE)], "harmonized_label"),
        "within_EPM_emotions": (clean[clean["dataset"] == "EPM-E4"], "harmonized_label"),
        "shared_Baseline_vs_Stress_pooled": (shared, "harmonized_label"),
    }


def subject_zscore(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Label-free per-subject standardisation (needs only that subject's own unlabelled windows)."""
    out = df.copy()
    grouped = out.groupby("subject_uid")[features]
    out[features] = (out[features] - grouped.transform("mean")) / grouped.transform("std").replace(0, 1).fillna(1)
    return out


def fit_predict(x_train, y_train, x_test, seed: int):
    # Fixed, untuned hyper-parameters: tuning on these splits would leak test subjects.
    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                          random_state=seed, n_jobs=8, verbosity=0)
    model.fit(x_train, y_train)
    return model.predict(x_test), model.predict_proba(x_test)


def score(y_true, y_pred) -> dict[str, float]:
    return {"accuracy": accuracy_score(y_true, y_pred), "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}


def grouped_split(data: pd.DataFrame, seed: int, test_size: float = 0.2):
    """Hold out test_size of the subjects of every dataset."""
    rng = np.random.default_rng(seed)
    test_subjects: list[str] = []
    for _, subjects in data.groupby("dataset")["subject_uid"].unique().items():
        shuffled = rng.permutation(subjects)
        test_subjects += list(shuffled[: max(1, round(len(shuffled) * test_size))])
    is_test = data["subject_uid"].isin(test_subjects).to_numpy()
    return ~is_test, is_test


def summarise(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    def agg(group: pd.DataFrame) -> pd.Series:
        out = {"n": len(group)}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = group[metric].to_numpy()
            half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
            out |= {f"{metric}_mean": values.mean(), f"{metric}_sd": values.std(ddof=1) if len(values) > 1 else np.nan,
                    f"{metric}_ci95_low": values.mean() - half, f"{metric}_ci95_high": values.mean() + half}
        return pd.Series(out)
    return rows.groupby(keys, sort=False).apply(agg, include_groups=False).reset_index()


def harmonization_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.groupby(["dataset", "original_label", "harmonized_label"]).agg(
        n_subjects=("subject_uid", "nunique"), n_windows=("purity", "size"),
        n_windows_purity_ge_0_8=("purity", lambda p: int((p >= MIN_PURITY).sum()))).reset_index()
    return table.rename(columns={"dataset": "Dataset", "original_label": "Original Label",
                                 "harmonized_label": "Harmonized Label", "n_subjects": "Number of Subjects",
                                 "n_windows": "Number of Windows"})


def leakage_check(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Reproduce the thesis hold-out (random 15% of windows) next to a subject-grouped 15% hold-out."""
    labels = pd.factorize(df["label"])[0]
    rows = []
    for seed in range(10):
        tr, te = train_test_split(np.arange(len(df)), test_size=0.15, stratify=labels, random_state=seed)
        pred, _ = fit_predict(df[features].to_numpy()[tr], labels[tr], df[features].to_numpy()[te], seed)
        rows.append({"split": "random_windows_15pct (thesis notebook 12)", **score(labels[te], pred)})
        train, test = grouped_split(df, seed, test_size=0.15)
        pred, _ = fit_predict(df.loc[train, features].to_numpy(), labels[train], df.loc[test, features].to_numpy(), seed)
        rows.append({"split": "unseen_subjects_15pct", **score(labels[test], pred)})
    return summarise(pd.DataFrame(rows), ["split"])


def repeated_and_loso(df: pd.DataFrame, features: list[str], n_splits: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, loso_rows, loso_summary = [], [], []
    for normalisation in ("none", "subject_zscore"):
        frame = subject_zscore(df, features) if normalisation == "subject_zscore" else df
        for task, (data, column) in tasks(frame).items():
            y = pd.factorize(data[column])[0]
            for modality, cols in modality_sets(features).items():
                x = data[cols].to_numpy()
                for seed in range(n_splits):
                    train, test = grouped_split(data, seed)
                    pred, _ = fit_predict(x[train], y[train], x[test], seed)
                    rows.append({"task": task, "normalisation": normalisation, "modality": modality,
                                 "n_classes": len(set(y)), "n_windows": len(data), **score(y[test], pred)})
                if modality not in ("all_modalities", "physiology_only"):
                    continue  # LOSO is the expensive protocol: run it for the two headline sets only
                subjects = data["subject_uid"].to_numpy()
                truth, guess = [], []
                for subject in np.unique(subjects):
                    test = subjects == subject
                    if len(set(y[~test])) < len(set(y)):
                        continue
                    pred, _ = fit_predict(x[~test], y[~test], x[test], 0)
                    truth.append(y[test]); guess.append(pred)
                    loso_rows.append({"task": task, "normalisation": normalisation, "modality": modality,
                                      "subject": subject, "accuracy": accuracy_score(y[test], pred)})
                pooled = score(np.concatenate(truth), np.concatenate(guess))
                loso_summary.append({"task": task, "normalisation": normalisation, "modality": modality,
                                     "n_subjects": len(truth), **{f"pooled_{k}": v for k, v in pooled.items()},
                                     "mean_subject_accuracy": float(np.mean([accuracy_score(t, g) for t, g in zip(truth, guess)]))})
                print(f"  {task:36s} {normalisation:15s} {modality:16s} LOSO pooled bal_acc={pooled['balanced_accuracy']:.3f}")
    return (summarise(pd.DataFrame(rows), ["task", "normalisation", "modality", "n_classes", "n_windows"]),
            pd.DataFrame(loso_rows), pd.DataFrame(loso_summary))


def cross_dataset(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for normalisation in ("none", "subject_zscore"):
        frame = subject_zscore(df, features) if normalisation == "subject_zscore" else df
        for label_scheme, negatives in (("Baseline_vs_Stress", ["Baseline"]), ("Baseline+Rest_vs_Stress", ["Baseline", "Rest"])):
            data = frame[(frame["purity"] >= MIN_PURITY) & frame["harmonized_label"].isin(negatives + ["Stress"])
                         & frame["dataset"].isin(["WESAD", "PhysioNet"])]
            y = (data["harmonized_label"] == "Stress").to_numpy(int)
            for source, target in (("WESAD", "PhysioNet"), ("PhysioNet", "WESAD")):
                train, test = (data["dataset"] == source).to_numpy(), (data["dataset"] == target).to_numpy()
                for modality, cols in modality_sets(features).items():
                    x = data[cols].to_numpy()
                    pred, proba = fit_predict(x[train], y[train], x[test], 0)
                    rows.append({"labels": label_scheme, "normalisation": normalisation, "train": source, "test": target,
                                 "modality": modality, "n_train": int(train.sum()), "n_test": int(test.sum()),
                                 **score(y[test], pred), "auroc": roc_auc_score(y[test], proba[:, 1])})
    return pd.DataFrame(rows)


def cross_dataset_arousal(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """WESAD <-> EPM-E4 on the one label both collected: the participant's own SAM arousal (high = above midpoint 5)."""
    rows = []
    for normalisation in ("none", "subject_zscore"):
        frame = subject_zscore(df, features) if normalisation == "subject_zscore" else df
        data = frame[(frame["purity"] >= MIN_PURITY) & frame["sam_arousal"].notna() & (frame["harmonized_label"] != "Excluded")]
        y = (data["sam_arousal"] > 5).to_numpy(int)
        for source, target in (("WESAD", "EPM-E4"), ("EPM-E4", "WESAD")):
            train, test = (data["dataset"] == source).to_numpy(), (data["dataset"] == target).to_numpy()
            for modality, cols in modality_sets(features).items():
                x = data[cols].to_numpy()
                pred, proba = fit_predict(x[train], y[train], x[test], 0)
                rows.append({"labels": "SAM_arousal_high_vs_low", "normalisation": normalisation, "train": source,
                             "test": target, "modality": modality, "n_train": int(train.sum()), "n_test": int(test.sum()),
                             "test_pct_high": round(100 * y[test].mean(), 1), **score(y[test], pred),
                             "auroc": roc_auc_score(y[test], proba[:, 1])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "combined" / "harmonized_windows.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "tables" / "jbhi")
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--validated-only", action="store_true",
                        help="Keep Stress windows only where the participant's own rating rose over baseline.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.validated_only:
        # z-scoring still sees every window; only the evaluated label set shrinks
        df.loc[~df["self_report_validated"].astype(bool), ["harmonized_label", "purity"]] = "Excluded", 0.0
    features = [c for c in df.columns if c not in META]
    train, test = grouped_split(df, 0)
    assert not set(df.loc[train, "subject_uid"]) & set(df.loc[test, "subject_uid"]), "subject leaked across split"

    harmonization_table(df).to_csv(args.output_dir / "harmonization_table.csv", index=False)
    leakage_check(df, features).to_csv(args.output_dir / "leakage_check.csv", index=False)
    cross_dataset(df, features).to_csv(args.output_dir / "cross_dataset_shared_labels.csv", index=False)
    cross_dataset_arousal(df, features).to_csv(args.output_dir / "cross_dataset_sam_arousal.csv", index=False)
    repeated, loso, loso_summary = repeated_and_loso(df, features, args.n_splits)
    repeated.to_csv(args.output_dir / "repeated_subject_splits.csv", index=False)
    loso.to_csv(args.output_dir / "loso_per_subject.csv", index=False)
    loso_summary.to_csv(args.output_dir / "loso_summary.csv", index=False)
    print(f"tables written to {args.output_dir}")


if __name__ == "__main__":
    main()
