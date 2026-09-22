#!/usr/bin/env python3
"""Audit dataset/label harmonization and confounding from the combined feature table.

If timestamp columns are unavailable, overlap outputs are still emitted but explicitly report
that overlap could not be audited from the input file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "combined" / "combined_dataset_filled.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "tables" / "harmonization_audit"
REQUIRED_COLUMNS = {"dataset", "subject_id", "label"}
SOURCE_LABEL_COLUMNS = ("original_label", "source_label", "protocol")

INFERRED_ORIGINAL_LABELS = {
    ("WESAD", "Baseline"): "baseline",
    ("WESAD", "Stress"): "stress",
    ("WESAD", "Amusement"): "amusement",
    ("EPM-E4", "Emotion"): "emotion elicitation",
    ("PhysioNet", "Stress"): "STRESS",
    ("PhysioNet", "Aerobic"): "AEROBIC",
    ("PhysioNet", "Anaerobic"): "ANAEROBIC",
}

JUSTIFICATIONS = {
    ("WESAD", "baseline", "Baseline"): "WESAD baseline segment mapped directly to the harmonized baseline/rest state.",
    ("WESAD", "stress", "Stress"): "WESAD stress task mapped directly to the harmonized stress state.",
    ("WESAD", "amusement", "Amusement"): "WESAD amusement segment kept separate from generic emotion labels.",
    ("EPM-E4", "emotion elicitation", "Emotion"): "EPM-E4 windows are labeled as emotion elicitation in the current pipeline.",
    ("PhysioNet", "STRESS", "Stress"): "PhysioNet STRESS protocol mapped directly to the harmonized stress state.",
    ("PhysioNet", "AEROBIC", "Aerobic"): "PhysioNet AEROBIC protocol mapped directly to the harmonized aerobic state.",
    ("PhysioNet", "ANAEROBIC", "Anaerobic"): "PhysioNet ANAEROBIC protocol mapped directly to the harmonized anaerobic state.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Combined feature table CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where audit tables will be written. If timestamp columns are missing, overlap outputs note that the audit was skipped.",
    )
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
    return df


def pick_source_label_column(df: pd.DataFrame) -> str | None:
    for column in SOURCE_LABEL_COLUMNS:
        if column in df.columns:
            return column
    return None


def infer_original_labels(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    source_column = pick_source_label_column(df)
    if source_column is not None:
        source_values = df[source_column].astype("string")
        mapping_sources = pd.Series(source_column, index=df.index, dtype="object")
        missing_mask = source_values.isna() | (source_values.str.strip() == "")
        if missing_mask.any():
            fallback_values = [
                INFERRED_ORIGINAL_LABELS.get((dataset, label), label)
                for dataset, label in zip(df.loc[missing_mask, "dataset"], df.loc[missing_mask, "label"])
            ]
            source_values.loc[missing_mask] = fallback_values
            mapping_sources.loc[missing_mask] = "inferred_builtin_mapping"
        return source_values.astype(str), mapping_sources

    inferred = [
        INFERRED_ORIGINAL_LABELS.get((dataset, label), label)
        for dataset, label in zip(df["dataset"], df["label"])
    ]
    return (
        pd.Series(inferred, index=df.index, dtype="object"),
        pd.Series("inferred_builtin_mapping", index=df.index, dtype="object"),
    )


def sorted_join(values: Iterable[str]) -> str:
    unique_values = sorted({str(value) for value in values})
    return " | ".join(unique_values)


def normalize_original_labels(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["original_label"] = (
        normalized["original_label"]
        .astype("string")
        .fillna("UNRESOLVED_SOURCE_LABEL")
        .replace("", "UNRESOLVED_SOURCE_LABEL")
        .astype(str)
    )
    return normalized


def build_harmonization_table(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_original_labels(df)
    table = (
        df.groupby(["dataset", "original_label", "label"], dropna=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            n_windows=("subject_id", "size"),
            mapping_source=("mapping_source", sorted_join),
        )
        .reset_index()
        .rename(columns={"label": "harmonized_label"})
        .sort_values(["dataset", "original_label", "harmonized_label"])
    )
    table["justification"] = table.apply(
        lambda row: JUSTIFICATIONS.get(
            (row["dataset"], row["original_label"], row["harmonized_label"]),
            "Verify this mapping manually; the combined table does not preserve finer-grained source labels.",
        ),
        axis=1,
    )
    return table


def build_label_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        df.groupby("label", dropna=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            datasets=("dataset", sorted_join),
            n_subjects=("subject_id", "nunique"),
            n_windows=("subject_id", "size"),
        )
        .reset_index()
        .sort_values(["n_datasets", "label"], ascending=[True, True])
    )
    coverage["confounded_with_dataset"] = coverage["n_datasets"] == 1
    return coverage


def build_dataset_label_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = (
        df.groupby(["dataset", "label"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
        .sort_index(axis=1)
    )
    return matrix


def build_shared_labels_table(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    datasets = list(matrix.index)
    for i, dataset_a in enumerate(datasets):
        labels_a = set(matrix.columns[matrix.loc[dataset_a] > 0])
        for dataset_b in datasets[i + 1 :]:
            labels_b = set(matrix.columns[matrix.loc[dataset_b] > 0])
            shared = sorted(labels_a & labels_b)
            rows.append(
                {
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "n_shared_labels": len(shared),
                    "shared_labels": " | ".join(shared),
                }
            )
    return pd.DataFrame(rows)


def build_subject_label_table(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_original_labels(df)
    subject_table = (
        df.groupby(["dataset", "subject_id"], dropna=False)
        .agg(
            n_windows=("subject_id", "size"),
            n_harmonized_labels=("label", "nunique"),
            harmonized_labels=("label", sorted_join),
            original_labels=("original_label", sorted_join),
        )
        .reset_index()
        .sort_values(["dataset", "subject_id"])
    )
    return subject_table


def coerce_timestamp_series(series: pd.Series) -> pd.Series:
    valid_mask = series.notna()
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric[valid_mask].notna().all():
        return numeric

    datetimes = pd.to_datetime(series, errors="coerce")
    if datetimes[valid_mask].notna().all():
        return datetimes

    raise ValueError(
        f"Could not parse timestamp column '{series.name}' as numeric or datetime values."
    )


def build_overlap_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not {"timestamp_start", "timestamp_end"}.issubset(df.columns):
        summary = pd.DataFrame(
            [
                {
                    "has_overlap_columns": False,
                    "groups_checked": 0,
                    "groups_with_overlap": 0,
                    "windows_checked": 0,
                    "overlapping_windows": 0,
                    "overlap_ratio": 0.0,
                }
            ]
        )
        return pd.DataFrame(), summary

    df = df.copy()
    df["timestamp_start"] = coerce_timestamp_series(df["timestamp_start"])
    df["timestamp_end"] = coerce_timestamp_series(df["timestamp_end"])

    overlap_rows = []
    windows_checked = 0
    overlapping_windows = 0
    groups_checked = 0
    groups_with_overlap = 0

    grouped = df.groupby(["dataset", "subject_id", "label"], dropna=False)
    for (dataset, subject_id, label), group in grouped:
        groups_checked += 1
        ordered = group.sort_values(["timestamp_start", "timestamp_end"]).reset_index(drop=True)
        if len(ordered) <= 1:
            continue

        prior_max_end = ordered["timestamp_end"].cummax().shift(1)
        overlap_mask = ordered["timestamp_start"] <= prior_max_end
        n_overlap = int(overlap_mask.fillna(False).sum())
        windows_checked += max(len(ordered) - 1, 0)
        overlapping_windows += n_overlap
        if n_overlap > 0:
            groups_with_overlap += 1

        overlap_rows.append(
            {
                "dataset": dataset,
                "subject_id": subject_id,
                "harmonized_label": label,
                "n_windows": len(ordered),
                "overlapping_windows": n_overlap,
                "overlap_ratio": n_overlap / max(len(ordered) - 1, 1),
            }
        )

    details = pd.DataFrame(overlap_rows)
    if not details.empty:
        details = details.sort_values(
            ["overlap_ratio", "dataset", "subject_id", "harmonized_label"],
            ascending=[False, True, True, True],
        )
    summary = pd.DataFrame(
        [
            {
                "has_overlap_columns": True,
                "groups_checked": groups_checked,
                "groups_with_overlap": groups_with_overlap,
                "windows_checked": windows_checked,
                "overlapping_windows": overlapping_windows,
                "overlap_ratio": overlapping_windows / windows_checked if windows_checked else 0.0,
            }
        ]
    )
    return details, summary


def write_outputs(
    output_dir: Path,
    harmonization: pd.DataFrame,
    coverage: pd.DataFrame,
    matrix: pd.DataFrame,
    shared_labels: pd.DataFrame,
    subjects: pd.DataFrame,
    overlap_details: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    manifest: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlap_details_path = output_dir / "window_overlap_details.csv"

    harmonization.to_csv(output_dir / "harmonization_table.csv", index=False)
    coverage.to_csv(output_dir / "label_dataset_coverage.csv", index=False)
    matrix.to_csv(output_dir / "dataset_label_matrix.csv")
    shared_labels.to_csv(output_dir / "pairwise_shared_labels.csv", index=False)
    subjects.to_csv(output_dir / "subject_label_summary.csv", index=False)
    overlap_summary.to_csv(output_dir / "window_overlap_summary.csv", index=False)
    if not overlap_details.empty:
        overlap_details.to_csv(overlap_details_path, index=False)
    elif overlap_details_path.exists():
        overlap_details_path.unlink()

    with (output_dir / "audit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)
    df["original_label"], df["mapping_source"] = infer_original_labels(df)
    df = normalize_original_labels(df)

    harmonization = build_harmonization_table(df)
    coverage = build_label_coverage_table(df)
    matrix = build_dataset_label_matrix(df)
    shared_labels = build_shared_labels_table(matrix)
    subjects = build_subject_label_table(df)
    overlap_details, overlap_summary = build_overlap_tables(df)

    manifest = {
        "input_file": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "n_rows": int(len(df)),
        "n_subjects": int(df[["dataset", "subject_id"]].drop_duplicates().shape[0]),
        "n_datasets": int(df["dataset"].nunique()),
        "n_labels": int(df["label"].nunique()),
        "mapping_sources": sorted(df["mapping_source"].unique().tolist()),
        "labels_confounded_with_dataset": coverage.loc[
            coverage["confounded_with_dataset"], "label"
        ].tolist(),
        "overlapping_windows_detected": bool(
            overlap_summary["groups_with_overlap"].iat[0] > 0 if not overlap_summary.empty else False
        ),
    }

    write_outputs(
        output_dir=args.output_dir,
        harmonization=harmonization,
        coverage=coverage,
        matrix=matrix,
        shared_labels=shared_labels,
        subjects=subjects,
        overlap_details=overlap_details,
        overlap_summary=overlap_summary,
        manifest=manifest,
    )

    print(f"Wrote audit outputs to {args.output_dir}")
    print(f"Input rows: {len(df)}")
    print(f"Labels confounded with dataset: {manifest['labels_confounded_with_dataset']}")
    if not overlap_summary.empty:
        print(
            "Overlapping windows detected: "
            f"{manifest['overlapping_windows_detected']} "
            f"(ratio={overlap_summary['overlap_ratio'].iat[0]:.4f})"
        )


if __name__ == "__main__":
    main()
