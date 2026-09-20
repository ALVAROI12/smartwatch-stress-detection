#!/usr/bin/env python3
"""Generate repeated subject-grouped splits and LOSO folds from the combined feature table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "combined" / "combined_dataset_filled.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "splits"
REQUIRED_COLUMNS = {"dataset", "subject_id", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Combined feature table CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where frozen split files will be written.",
    )
    parser.add_argument("--n-splits", type=int, default=20, help="Number of repeated train/test splits.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of subject groups per dataset for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. "
            "The repository ignores data/processed/, so pass --input to a local combined dataset CSV."
        )

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.reset_index().rename(columns={"index": "row_index"})


def stable_dataset_offset(dataset: str) -> int:
    return sum(ord(char) for char in dataset) * 1009


def build_group_table(df: pd.DataFrame) -> pd.DataFrame:
    groups = (
        df.assign(group_id=df["dataset"].astype(str) + "::" + df["subject_id"].astype(str))
        .groupby(["dataset", "subject_id", "group_id"], dropna=False)
        .agg(
            n_windows=("row_index", "size"),
            labels=("label", lambda values: " | ".join(sorted({str(v) for v in values}))),
        )
        .reset_index()
        .sort_values(["dataset", "subject_id"])
    )
    return groups


def validate_group_counts(groups: pd.DataFrame) -> None:
    too_small = groups.groupby("dataset").size()
    invalid = too_small[too_small < 2]
    if not invalid.empty:
        dataset_list = ", ".join(f"{dataset} ({count})" for dataset, count in invalid.items())
        raise ValueError(
            "Each dataset needs at least 2 subject groups for grouped train/test splits. "
            f"Invalid datasets: {dataset_list}."
        )


def max_unique_test_signatures(groups: pd.DataFrame, test_size: float) -> int:
    total = 1
    for _, dataset_groups in groups.groupby("dataset", sort=True):
        n_groups = len(dataset_groups)
        n_test = int(np.ceil(n_groups * test_size))
        n_test = min(max(1, n_test), n_groups - 1)
        total *= math.comb(n_groups, n_test)
    return total


def per_dataset_split(
    dataset_groups: pd.DataFrame,
    split_id: int,
    test_size: float,
    seed: int,
    attempt: int = 0,
) -> tuple[list[str], list[str]]:
    n_test = int(np.ceil(len(dataset_groups) * test_size))
    n_test = min(max(1, n_test), len(dataset_groups) - 1)
    rng = np.random.default_rng(
        seed
        + split_id * 100_003
        + stable_dataset_offset(dataset_groups["dataset"].iat[0])
        + attempt * 10_007
    )
    shuffled = rng.permutation(dataset_groups["group_id"].tolist())
    test_groups = sorted(shuffled[:n_test].tolist())
    train_groups = sorted(shuffled[n_test:].tolist())
    return train_groups, test_groups


def generate_repeated_splits(
    groups: pd.DataFrame,
    n_splits: int,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    assignments = []
    summaries = []
    seen_signatures: set[tuple[tuple[str, ...], ...]] = set()

    for split_id in range(n_splits):
        split_assignments = []
        signature_parts = []
        for attempt in range(max(100, n_splits * 5)):
            split_assignments = []
            signature_parts = []

            for dataset, dataset_groups in groups.groupby("dataset", sort=True):
                train_groups, test_groups = per_dataset_split(
                    dataset_groups,
                    split_id,
                    test_size,
                    seed,
                    attempt=attempt,
                )
                signature_parts.append(tuple(test_groups))

                for partition, partition_groups in (("train", train_groups), ("test", test_groups)):
                    subset = dataset_groups[dataset_groups["group_id"].isin(partition_groups)].copy()
                    subset["split_id"] = split_id
                    subset["partition"] = partition
                    split_assignments.append(subset)

            signature = tuple(signature_parts)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                break
        else:
            raise RuntimeError(
                "Unable to find a new unique repeated split with the current seed and test size."
            )

        split_df = pd.concat(split_assignments, ignore_index=True)
        split_df["duplicate_test_signature"] = False
        assignments.append(split_df)

        summary = (
            split_df.groupby(["split_id", "partition", "dataset"], dropna=False)
            .agg(
                n_subject_groups=("group_id", "nunique"),
                n_windows=("n_windows", "sum"),
                duplicate_test_signature=("duplicate_test_signature", "max"),
            )
            .reset_index()
        )
        summaries.append(summary)

    return pd.concat(assignments, ignore_index=True), pd.concat(summaries, ignore_index=True), len(seen_signatures)


def write_loso_outputs(output_dir: Path, groups: pd.DataFrame) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = output_dir / "loso_subject_group_assignments.csv"
    fieldnames = ["dataset", "subject_id", "group_id", "n_windows", "labels", "fold_id", "partition"]
    base_records = groups.reset_index(drop=True).to_dict("records")
    dataset_totals = (
        groups.groupby("dataset", dropna=False)
        .agg(
            n_subject_groups=("group_id", "nunique"),
            n_windows=("n_windows", "sum"),
        )
        .to_dict("index")
    )

    summary_rows = []
    with assignments_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for fold_id, test_record in enumerate(base_records):
            for record in base_records:
                partition = "test" if record["group_id"] == test_record["group_id"] else "train"
                writer.writerow(
                    {
                        "dataset": record["dataset"],
                        "subject_id": record["subject_id"],
                        "group_id": record["group_id"],
                        "n_windows": record["n_windows"],
                        "labels": record["labels"],
                        "fold_id": fold_id,
                        "partition": partition,
                    }
                )

            for dataset, totals in dataset_totals.items():
                test_subject_groups = 1 if dataset == test_record["dataset"] else 0
                test_windows = test_record["n_windows"] if dataset == test_record["dataset"] else 0

                if test_subject_groups:
                    summary_rows.append(
                        {
                            "fold_id": fold_id,
                            "partition": "test",
                            "dataset": dataset,
                            "n_subject_groups": test_subject_groups,
                            "n_windows": test_windows,
                        }
                    )

                summary_rows.append(
                    {
                        "fold_id": fold_id,
                        "partition": "train",
                        "dataset": dataset,
                        "n_subject_groups": totals["n_subject_groups"] - test_subject_groups,
                        "n_windows": totals["n_windows"] - test_windows,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "loso_subject_group_summary.csv", index=False)
    return summary


def write_outputs(
    output_dir: Path,
    repeated_assignments: pd.DataFrame,
    repeated_summary: pd.DataFrame,
    manifest: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    repeated_assignments.to_csv(output_dir / "repeated_subject_group_assignments.csv", index=False)
    repeated_summary.to_csv(output_dir / "repeated_subject_group_summary.csv", index=False)

    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.n_splits < 1:
        raise ValueError("--n-splits must be at least 1.")

    df = load_dataset(args.input)
    groups = build_group_table(df)
    validate_group_counts(groups)
    unique_signature_limit = max_unique_test_signatures(groups, args.test_size)
    if args.n_splits > unique_signature_limit:
        raise ValueError(
            f"Requested {args.n_splits} repeated splits but only {unique_signature_limit} unique test signatures "
            "are possible with the current dataset group counts and --test-size."
        )
    repeated_assignments, repeated_summary, n_unique_signatures = generate_repeated_splits(
        groups=groups,
        n_splits=args.n_splits,
        test_size=args.test_size,
        seed=args.seed,
    )
    loso_summary = write_loso_outputs(args.output_dir, groups)

    manifest = {
        "input_file": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "n_rows": int(len(df)),
        "n_subject_groups": int(len(groups)),
        "datasets": sorted(groups["dataset"].unique().tolist()),
        "n_repeated_splits": int(args.n_splits),
        "n_unique_repeated_test_signatures": int(n_unique_signatures),
        "max_unique_repeated_test_signatures": int(unique_signature_limit),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "n_loso_folds": int(loso_summary["fold_id"].nunique()),
    }

    write_outputs(
        output_dir=args.output_dir,
        repeated_assignments=repeated_assignments,
        repeated_summary=repeated_summary,
        manifest=manifest,
    )

    print(f"Wrote split outputs to {args.output_dir}")
    print(f"Repeated splits: {args.n_splits}")
    print(f"LOSO folds: {loso_summary['fold_id'].nunique()}")


if __name__ == "__main__":
    main()
