#!/usr/bin/env python3
"""Evaluate XGBoost modality baselines on frozen subject-grouped repeated splits and LOSO folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "combined" / "combined_dataset_filled.csv"
DEFAULT_SPLITS_DIR = REPO_ROOT / "outputs" / "splits"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "tables" / "frozen_split_evaluation"
DEFAULT_XGB_CONFIG = REPO_ROOT / "outputs" / "models" / "best_hyperparameters.json"
METADATA_COLUMNS = {"subject_id", "dataset", "window_id", "timestamp_start", "timestamp_end", "label"}
PHYSIOLOGY_PREFIXES = ("hr_", "hrv_", "eda_", "temp_")
SUPPORTED_XGB_PARAMS = {
    "objective",
    "base_score",
    "booster",
    "colsample_bylevel",
    "colsample_bynode",
    "colsample_bytree",
    "device",
    "eval_metric",
    "gamma",
    "grow_policy",
    "importance_type",
    "interaction_constraints",
    "learning_rate",
    "max_bin",
    "max_cat_threshold",
    "max_cat_to_onehot",
    "max_delta_step",
    "max_depth",
    "max_leaves",
    "min_child_weight",
    "missing",
    "monotone_constraints",
    "multi_strategy",
    "n_estimators",
    "n_jobs",
    "num_parallel_tree",
    "random_state",
    "reg_alpha",
    "reg_lambda",
    "sampling_method",
    "scale_pos_weight",
    "subsample",
    "tree_method",
    "validate_parameters",
    "verbosity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Combined feature table CSV.")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR, help="Frozen split directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for result tables.")
    parser.add_argument(
        "--xgb-config",
        type=Path,
        default=DEFAULT_XGB_CONFIG,
        help="Optional JSON containing the XGBoost parameters under the 'XGBoost' key.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_xgb_classifier(config_path: Path, seed: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise ImportError("xgboost is required to run frozen split evaluation.") from exc

    params = {
        "objective": "multi:softprob",
        "n_estimators": 100,
        "random_state": seed,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "n_jobs": -1,
        "device": "cpu",
    }
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        params.update(
            {
                key: value
                for key, value in config.get("XGBoost", {}).items()
                if value is not None and key in SUPPORTED_XGB_PARAMS
            }
        )
        params["device"] = "cpu"
        params["random_state"] = seed
    return XGBClassifier(**params)


def confidence_interval(values: pd.Series) -> tuple[float, float, float]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return np.nan, np.nan, np.nan
    mean = float(clean.mean())
    if len(clean) == 1:
        return mean, 0.0, mean
    std = float(clean.std(ddof=1))
    ci = 1.96 * std / np.sqrt(len(clean))
    return mean, std, ci


def get_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    candidate_cols = [column for column in df.columns if column not in METADATA_COLUMNS | {"group_id"}]
    feature_cols = [column for column in candidate_cols if pd.api.types.is_numeric_dtype(df[column])]
    acc_cols = [column for column in feature_cols if column.startswith("acc_")]
    non_acc_cols = [column for column in feature_cols if not column.startswith("acc_")]
    physiology_cols = [
        column for column in feature_cols if column.startswith(PHYSIOLOGY_PREFIXES)
    ]
    feature_sets = {
        "all_modalities": feature_cols,
        "accelerometer_only": acc_cols,
        "all_except_accelerometer": non_acc_cols,
        "physiology_only": physiology_cols,
    }
    empty = [name for name, cols in feature_sets.items() if not cols]
    if empty:
        raise ValueError(f"Missing features for modality sets: {empty}")
    return feature_sets


def prepare_dataframe(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Pass the real combined dataset CSV with --input."
        )
    df = pd.read_csv(input_path)
    required = {"dataset", "subject_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["group_id"] = df["dataset"].astype(str) + "::" + df["subject_id"].astype(str)
    return df


def evaluate_single_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_group_ids: list[str],
    test_group_ids: list[str],
    model,
) -> dict[str, object]:
    train_df = df[df["group_id"].isin(train_group_ids)].copy()
    test_df = df[df["group_id"].isin(test_group_ids)].copy()
    if train_df.empty or test_df.empty:
        return {"status": "empty_partition"}

    train_labels = train_df["label"].astype(str)
    test_labels = test_df["label"].astype(str)
    unseen_labels = sorted(set(test_labels.unique()) - set(train_labels.unique()))
    if unseen_labels:
        return {
            "status": "unseen_test_labels",
            "unseen_test_labels": " | ".join(unseen_labels),
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
        }

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].to_numpy())
    X_test = scaler.transform(test_df[feature_cols].to_numpy())

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_labels)
    y_test = encoder.transform(test_labels)

    if len(np.unique(y_train)) < 2:
        return {"status": "single_train_class", "n_train_rows": len(train_df), "n_test_rows": len(test_df)}

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "status": "ok",
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "n_train_groups": train_df["group_id"].nunique(),
        "n_test_groups": test_df["group_id"].nunique(),
        "n_train_labels": train_labels.nunique(),
        "n_test_labels": test_labels.nunique(),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
    }


def evaluate_assignments(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    split_column: str,
    feature_sets: dict[str, list[str]],
    xgb_config: Path,
    seed: int,
    protocol_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_id, split_df in assignments.groupby(split_column, sort=True):
        train_ids = split_df.loc[split_df["partition"] == "train", "group_id"].drop_duplicates().tolist()
        test_ids = split_df.loc[split_df["partition"] == "test", "group_id"].drop_duplicates().tolist()
        for modality_name, feature_cols in feature_sets.items():
            modality_seed = seed + int(split_id) * 1_000 + sum(ord(char) for char in modality_name)
            result = evaluate_single_split(
                df=df,
                feature_cols=feature_cols,
                train_group_ids=train_ids,
                test_group_ids=test_ids,
                model=load_xgb_classifier(xgb_config, modality_seed),
            )
            result.update(
                {
                    "protocol": protocol_name,
                    "split_id": split_id,
                    "modality": modality_name,
                    "n_features": len(feature_cols),
                }
            )
            rows.append(result)
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (protocol, modality), group in results.groupby(["protocol", "modality"], sort=True):
        ok = group[group["status"] == "ok"].copy()
        skipped = group[group["status"] != "ok"].copy()
        skip_reasons = (
            " | ".join(
                f"{status}:{count}"
                for status, count in skipped["status"].value_counts().sort_index().items()
            )
            if not skipped.empty
            else ""
        )
        macro_mean, macro_std, macro_ci = confidence_interval(ok["macro_f1"])
        bal_mean, bal_std, bal_ci = confidence_interval(ok["balanced_accuracy"])
        acc_mean, acc_std, acc_ci = confidence_interval(ok["accuracy"])
        summaries.append(
            {
                "protocol": protocol,
                "modality": modality,
                "n_total_splits": int(len(group)),
                "n_evaluable_splits": int(len(ok)),
                "n_skipped_splits": int((group["status"] != "ok").sum()),
                "skip_reasons": skip_reasons,
                "macro_f1_mean": macro_mean,
                "macro_f1_std": macro_std,
                "macro_f1_ci95": macro_ci,
                "balanced_accuracy_mean": bal_mean,
                "balanced_accuracy_std": bal_std,
                "balanced_accuracy_ci95": bal_ci,
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "accuracy_ci95": acc_ci,
            }
        )
    return pd.DataFrame(summaries).sort_values(["protocol", "modality"])


def build_skipped_summary(results: pd.DataFrame) -> pd.DataFrame:
    skipped = results[results["status"] != "ok"].copy()
    if skipped.empty:
        return pd.DataFrame(columns=["protocol", "modality", "status", "n_splits"])
    return (
        skipped.groupby(["protocol", "modality", "status"], sort=True)
        .size()
        .reset_index(name="n_splits")
        .sort_values(["protocol", "modality", "status"])
    )


def main() -> None:
    args = parse_args()
    df = prepare_dataframe(args.input)
    feature_sets = get_feature_sets(df)

    repeated_assignments = pd.read_csv(args.splits_dir / "repeated_subject_group_assignments.csv")
    loso_assignments = pd.read_csv(args.splits_dir / "loso_subject_group_assignments.csv")

    repeated_results = evaluate_assignments(
        df=df,
        assignments=repeated_assignments,
        split_column="split_id",
        feature_sets=feature_sets,
        xgb_config=args.xgb_config,
        seed=args.seed,
        protocol_name="repeated_subject_group",
    )
    loso_results = evaluate_assignments(
        df=df,
        assignments=loso_assignments,
        split_column="fold_id",
        feature_sets=feature_sets,
        xgb_config=args.xgb_config,
        seed=args.seed,
        protocol_name="loso",
    )

    repeated_summary = summarize_results(repeated_results)
    loso_summary = summarize_results(loso_results)
    full_summary = pd.concat([repeated_summary, loso_summary], ignore_index=True)
    headline = full_summary.copy()
    skipped_summary = build_skipped_summary(pd.concat([repeated_results, loso_results], ignore_index=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    repeated_results.to_csv(args.output_dir / "repeated_split_results.csv", index=False)
    repeated_summary.to_csv(args.output_dir / "repeated_split_summary.csv", index=False)
    loso_results.to_csv(args.output_dir / "loso_results.csv", index=False)
    loso_summary.to_csv(args.output_dir / "loso_summary.csv", index=False)
    skipped_summary.to_csv(args.output_dir / "skipped_split_summary.csv", index=False)
    headline.to_csv(args.output_dir / "headline_subject_grouped_results.csv", index=False)
    (args.output_dir / "modality_feature_sets.json").write_text(
        json.dumps(feature_sets, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote frozen split evaluation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
