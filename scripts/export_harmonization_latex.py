#!/usr/bin/env python3
"""Export a paper-ready LaTeX harmonization table from harmonization_table.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "tables" / "harmonization_audit" / "harmonization_table.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "tables" / "harmonization_audit" / "harmonization_table.tex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="harmonization_table.csv path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination .tex file.")
    parser.add_argument(
        "--caption",
        default="Dataset-to-label harmonization used for the combined stress-detection analysis.",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:harmonization-table",
        help="LaTeX table label.",
    )
    return parser.parse_args()


def latex_escape(value: object) -> str:
    normalized = " ".join(str(value).splitlines())
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in normalized)


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.input)
    required = [
        "dataset",
        "original_label",
        "harmonized_label",
        "n_subjects",
        "n_windows",
        "justification",
    ]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(args.caption)}}}",
        rf"\label{{{' '.join(args.label.splitlines())}}}",
        r"\begin{tabular}{lllrrp{7cm}}",
        r"\hline",
        r"Dataset & Original Label & Harmonized Label & \#Subjects & \#Windows & Justification \\",
        r"\hline",
    ]

    for row in table[required].itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    latex_escape(row.dataset),
                    latex_escape(row.original_label),
                    latex_escape(row.harmonized_label),
                    latex_escape(row.n_subjects),
                    latex_escape(row.n_windows),
                    latex_escape(row.justification),
                ]
            )
            + r" \\"
        )

    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote LaTeX table to {args.output}")


if __name__ == "__main__":
    main()
