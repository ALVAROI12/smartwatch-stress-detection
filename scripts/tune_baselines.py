#!/usr/bin/env python3
"""Nested, subject-grouped hyper-parameter tuning for the baseline classifiers.

Outer loop: the same 20 subject-grouped 80/20 splits as run_jbhi_experiments.py (identical seeds, so
models can be compared with paired tests). Inner loop: 3-fold subject-grouped CV on the TRAINING
subjects only; Optuna maximises balanced accuracy there. Test subjects never influence tuning,
imputation or scaling.

Cross-dataset rows tune on the source dataset alone and test once on the target dataset.
Results are appended per configuration, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from run_jbhi_experiments import META, REPO_ROOT, grouped_split, modality_sets, score, subject_zscore, summarise, tasks

optuna.logging.set_verbosity(optuna.logging.WARNING)
MLP_LAYERS = {"32": (32,), "64": (64,), "64-32": (64, 32), "128-64": (128, 64)}


def build(model: str, trial: optuna.Trial | optuna.trial.FixedTrial, seed: int):
    if model == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50), max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0), colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10), gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True), random_state=seed, n_jobs=1, verbosity=0)
    if model == "random_forest":
        forest = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 600, step=100), max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            class_weight="balanced", random_state=seed, n_jobs=1)
        return make_pipeline(SimpleImputer(strategy="median"), forest)
    mlp = MLPClassifier(hidden_layer_sizes=MLP_LAYERS[trial.suggest_categorical("layers", list(MLP_LAYERS))],
                        alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                        learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                        max_iter=300, early_stopping=True, random_state=seed)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), mlp)


def fit(model: str, estimator, x, y):
    if model == "xgboost":  # trees get class balance through weights; RF uses class_weight; sklearn's MLP has neither
        return estimator.fit(x, y, sample_weight=compute_sample_weight("balanced", y))
    return estimator.fit(x, y)


def tune_and_test(model: str, x_train, y_train, groups, x_test, y_test, n_trials: int, seed: int) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # joblib workers do not inherit the parent setting
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(x_train, y_train, groups))

    def objective(trial: optuna.Trial) -> float:
        scores = []
        for tr, va in folds:
            if len(set(y_train[tr])) < len(set(y_train)):
                continue
            estimator = fit(model, build(model, trial, seed), x_train[tr], y_train[tr])
            scores.append(score(y_train[va], estimator.predict(x_train[va]))["balanced_accuracy"])
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    final = fit(model, build(model, optuna.trial.FixedTrial(study.best_params), seed), x_train, y_train)
    return {**score(y_test, final.predict(x_test)), "inner_cv_balanced_accuracy": study.best_value,
            "best_params": json.dumps(study.best_params)}


def configurations(df: pd.DataFrame, features: list[str], task_names: list[str], modalities: list[str]):
    for normalisation in ("none", "subject_zscore"):
        frame = subject_zscore(df, features) if normalisation == "subject_zscore" else df
        all_tasks = tasks(frame)
        for task in task_names:
            for modality in modalities:
                yield task, normalisation, modality, all_tasks[task]
        shared, column = all_tasks["shared_Baseline_vs_Stress_pooled"]
        for source, target in (("WESAD", "PhysioNet"), ("PhysioNet", "WESAD")):
            for modality in modalities:
                yield f"cross_dataset_{source}_to_{target}", normalisation, modality, (shared, column)


def run_configuration(task, data, column, cols, model, n_trials, n_splits) -> list[dict]:
    y = pd.factorize(data[column], sort=True)[0]
    x, groups = data[cols].to_numpy(float), data["subject_uid"].to_numpy()
    jobs = []
    if task.startswith("cross_dataset_"):
        source, target = task.removeprefix("cross_dataset_").split("_to_")
        train, test = (data["dataset"] == source).to_numpy(), (data["dataset"] == target).to_numpy()
        splits = [(seed, train, test) for seed in range(5)]  # one fixed split; seeds vary the search and the model
    else:
        splits = [(seed, *grouped_split(data, seed)) for seed in range(n_splits)]
    for seed, train, test in splits:
        jobs.append(delayed(tune_and_test)(model, x[train], y[train], groups[train], x[test], y[test], n_trials, seed))
    results = Parallel(n_jobs=8)(jobs)
    return [{"split": seed, **res} for (seed, _, _), res in zip(splits, results)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO_ROOT / "data" / "processed" / "combined" / "harmonized_windows_v2.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "tables" / "jbhi_v2")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--tasks", nargs="+", default=["shared_Baseline_vs_Stress_pooled", "within_WESAD",
                                                       "within_PhysioNet_stress_session", "affective_only_no_exercise"])
    parser.add_argument("--modalities", nargs="+", default=["all_modalities", "physiology_only", "acc_only"])
    parser.add_argument("--models", nargs="+", default=["xgboost", "random_forest", "mlp"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_split_path = args.output_dir / "tuned_baselines_per_split.csv"

    df = pd.read_csv(args.input)
    features = [c for c in df.columns if c not in META]
    done = set()
    if per_split_path.exists():
        done = set(map(tuple, pd.read_csv(per_split_path)[["task", "normalisation", "modality", "model"]].drop_duplicates().to_numpy()))

    for task, normalisation, modality, (data, column) in configurations(df, features, args.tasks, args.modalities):
        for model in args.models:
            if (task, normalisation, modality, model) in done:
                continue
            rows = run_configuration(task, data, column, modality_sets(features)[modality], model, args.n_trials, args.n_splits)
            out = pd.DataFrame(rows).assign(task=task, normalisation=normalisation, modality=modality, model=model)
            out.to_csv(per_split_path, mode="a", header=not per_split_path.exists(), index=False)
            print(f"{task:42s} {normalisation:15s} {modality:16s} {model:14s} "
                  f"bal_acc={out['balanced_accuracy'].mean():.3f} ±{out['balanced_accuracy'].std():.3f}", flush=True)

    summarise(pd.read_csv(per_split_path), ["task", "normalisation", "modality", "model"]).to_csv(
        args.output_dir / "tuned_baselines_summary.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
