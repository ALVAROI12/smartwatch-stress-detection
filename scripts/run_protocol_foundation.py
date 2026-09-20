#!/usr/bin/env python3
"""Run the harmonization audit and frozen split generation, then summarize the paper risks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "combined" / "combined_dataset_filled.csv"
DEFAULT_AUDIT_DIR = REPO_ROOT / "outputs" / "tables" / "harmonization_audit"
DEFAULT_SPLITS_DIR = REPO_ROOT / "outputs" / "splits"
DEFAULT_SUMMARY = REPO_ROOT / "outputs" / "tables" / "protocol_foundation_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Combined feature table CSV.")
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR, help="Audit output directory.")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR, help="Frozen split output directory.")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY, help="Markdown summary path.")
    parser.add_argument("--n-splits", type=int, default=20, help="Number of repeated subject-grouped splits.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of subject groups per dataset for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def build_recommendation(coverage: pd.DataFrame, shared_labels: pd.DataFrame) -> str:
    confounded_labels = coverage.loc[coverage["confounded_with_dataset"], "label"].tolist()
    non_confounded_labels = coverage.loc[~coverage["confounded_with_dataset"], "label"].tolist()
    max_shared = int(shared_labels["n_shared_labels"].max()) if not shared_labels.empty else 0

    if confounded_labels and (not non_confounded_labels or max_shared <= 1):
        return (
            "Recommendation: treat the full 6-class task as a confounded diagnostic result and pivot the main paper "
            "contribution to a narrower fair task built only from genuinely shared labels."
        )
    if confounded_labels:
        return (
            "Recommendation: report the full 6-class task only as supporting/confounded analysis, and make the main "
            "headline result the subset of labels shared across datasets."
        )
    return "Recommendation: the label space appears sufficiently shared to keep the full task as the main target."


def main() -> None:
    args = parse_args()

    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "harmonization_audit.py"),
            "--input",
            str(args.input),
            "--output-dir",
            str(args.audit_dir),
        ]
    )
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "generate_frozen_splits.py"),
            "--input",
            str(args.input),
            "--output-dir",
            str(args.splits_dir),
            "--n-splits",
            str(args.n_splits),
            "--test-size",
            str(args.test_size),
            "--seed",
            str(args.seed),
        ]
    )

    harmonization = pd.read_csv(args.audit_dir / "harmonization_table.csv")
    coverage = pd.read_csv(args.audit_dir / "label_dataset_coverage.csv")
    shared_labels = pd.read_csv(args.audit_dir / "pairwise_shared_labels.csv")
    overlap = pd.read_csv(args.audit_dir / "window_overlap_summary.csv").iloc[0]
    split_manifest = json.loads((args.splits_dir / "split_manifest.json").read_text(encoding="utf-8"))

    dataset_specific = coverage.loc[coverage["confounded_with_dataset"], "label"].tolist()
    shared_pairs = shared_labels.to_dict("records")
    overlap_message = (
        "Window overlap detected"
        if bool(overlap["has_overlap_columns"]) and int(overlap["groups_with_overlap"]) > 0
        else "No overlap detected"
    )
    if not bool(overlap["has_overlap_columns"]):
        overlap_message = "Window overlap could not be audited because timestamp columns were missing"

    summary_lines = [
        "# Protocol Foundation Summary",
        "",
        f"- Input CSV: `{args.input}`",
        f"- Harmonization rows: {len(harmonization)}",
        f"- Frozen repeated splits: {args.n_splits}",
        f"- LOSO folds: {split_manifest['n_loso_folds']}",
        "",
        "## Key paper-risk answers",
        "",
        f"1. **Dataset-specific labels**: {', '.join(dataset_specific) if dataset_specific else 'None'}",
        f"2. **Window overlap**: {overlap_message} (overlap ratio = {float(overlap['overlap_ratio']):.4f})",
        "3. **Genuinely shared labels across datasets**:",
    ]

    for row in shared_pairs:
        labels = row["shared_labels"] if isinstance(row["shared_labels"], str) and row["shared_labels"] else "None"
        summary_lines.append(
            f"   - {row['dataset_a']} vs {row['dataset_b']}: {labels} (n={int(row['n_shared_labels'])})"
        )

    summary_lines.extend(
        [
            "",
            "## Protocol decisions",
            "",
            f"- Frozen split directory: `{args.splits_dir}`",
            "- Use these split files as the only train/test protocol for future experiments.",
            f"- {build_recommendation(coverage, shared_labels)}",
            "",
            "## Paper table inputs",
            "",
            f"- Harmonization CSV: `{args.audit_dir / 'harmonization_table.csv'}`",
            f"- Coverage CSV: `{args.audit_dir / 'label_dataset_coverage.csv'}`",
            f"- Shared-label CSV: `{args.audit_dir / 'pairwise_shared_labels.csv'}`",
        ]
    )

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
