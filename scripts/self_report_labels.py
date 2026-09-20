#!/usr/bin/env python3
"""Attach participants' own ratings to the harmonized windows.

Protocol labels say what the experimenter did; self-reports say whether it worked on that person.
Adds to harmonized_windows.csv:

  self_report_stress        rating for the window's stage (PhysioNet: stress 1-10; WESAD: PANAS "Stressed" 1-5)
  self_report_stress_delta  that rating minus the same subject's Baseline rating
  self_report_validated     Stress windows: delta >= --min-delta. Every other class: True.
  sam_valence, sam_arousal  SAM 1-9 (WESAD per condition, EPM-E4 per film clip; PhysioNet has none)

Also writes self_reports_tidy.csv and manipulation_check.csv (did ratings rise from baseline to stressor?).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSIONET = ("wearable-device-dataset/"
             "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1")
EPM_SAM = "EPM-E4/questionnaires/preprocessed/Ficha_Evaluacion_Participante_SAM_Refactored.csv"
WESAD_CONDITION = {"Base": "Baseline", "TSST": "Stress", "Fun": "Amusement"}  # meditation has no windows here
PANAS_STRESSED = 21  # 0-based position of the "Stressed" item (wesad_readme.pdf, section III.2)
STAI_REVERSED = (0, 3, 5)  # "at ease", "relaxed", "pleasant"; 4-point scale


def wesad_reports(root: Path) -> pd.DataFrame:
    rows = []
    for quest in sorted((root / "WESAD").glob("S*/S*_quest.csv")):
        lines = [line.strip().lstrip("# ").split(";") for line in quest.read_text().splitlines()]
        order = next(line[1:6] for line in lines if line[0] == "ORDER")
        by_name = {name: [line[1:] for line in lines if line[0] == name] for name in ("PANAS", "STAI", "DIM")}
        for i, condition in enumerate(order):
            if condition not in WESAD_CONDITION:
                continue
            stai = [int(v) for v in by_name["STAI"][i][:6]]
            rows.append({"dataset": "WESAD", "subject_uid": f"WESAD:{quest.parent.name}",
                         "stage": WESAD_CONDITION[condition],
                         "stress": int(by_name["PANAS"][i][PANAS_STRESSED]),
                         "stai6": sum(5 - v if j in STAI_REVERSED else v for j, v in enumerate(stai)),
                         "sam_valence": int(by_name["DIM"][i][0]), "sam_arousal": int(by_name["DIM"][i][1])})
    return pd.DataFrame(rows)


def physionet_reports(root: Path) -> pd.DataFrame:
    frames = []
    for version in ("v1", "v2"):
        wide = pd.read_csv(root / PHYSIONET / f"Stress_Level_{version}.csv", index_col=0)
        long = wide.rename_axis("subject").reset_index().melt("subject", var_name="stage", value_name="stress")
        frames.append(long)
    out = pd.concat(frames, ignore_index=True)
    out["dataset"], out["subject_uid"] = "PhysioNet", "PhysioNet:" + out["subject"]
    return out.drop(columns="subject")


def epm_reports(root: Path) -> pd.DataFrame:
    sam = pd.read_csv(root / EPM_SAM)
    return pd.DataFrame({"dataset": "EPM-E4", "subject_uid": "EPM-E4:" + sam["ID"].astype(str), "stage": sam["EMOTION"],
                         "sam_valence": sam["VALENCE"], "sam_arousal": sam["AROUSAL"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "tables" / "jbhi")
    parser.add_argument("--windows", default="harmonized_windows_v2.csv", help="Window table inside data/processed/combined.")
    parser.add_argument("--min-delta", type=float, default=1.0, help="Minimum rise over own baseline for a valid Stress window.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = pd.concat([wesad_reports(args.data_root), physionet_reports(args.data_root), epm_reports(args.data_root)],
                        ignore_index=True)
    baseline = reports[reports["stage"] == "Baseline"].set_index("subject_uid")["stress"]
    reports["stress_delta"] = reports["stress"] - reports["subject_uid"].map(baseline)
    reports.to_csv(args.output_dir / "self_reports_tidy.csv", index=False)

    stressors = reports[reports["stress_delta"].notna() & ~reports["stage"].isin(["Baseline", "Amusement"])]
    check = stressors.groupby(["dataset", "stage"]).agg(
        n_subjects=("stress_delta", "size"), mean_delta=("stress_delta", "mean"), sd_delta=("stress_delta", "std"),
        pct_rose=("stress_delta", lambda d: 100 * (d >= args.min_delta).mean())).round(2).reset_index()
    check.to_csv(args.output_dir / "manipulation_check.csv", index=False)
    print(check.to_string(index=False))

    path = args.data_root / "data" / "processed" / "combined" / args.windows
    windows = pd.read_csv(path).drop(columns=["self_report_stress", "self_report_stress_delta", "self_report_validated",
                                              "sam_valence", "sam_arousal"], errors="ignore")
    # WESAD and PhysioNet-Baseline windows match on harmonized label; PhysioNet stages and EPM clips on original label.
    key = windows["original_label"].where(windows["dataset"] != "WESAD", windows["harmonized_label"])
    merged = windows.assign(stage=key).merge(
        reports.rename(columns={"stress": "self_report_stress", "stress_delta": "self_report_stress_delta"})[
            ["subject_uid", "stage", "self_report_stress", "self_report_stress_delta", "sam_valence", "sam_arousal"]],
        on=["subject_uid", "stage"], how="left").drop(columns="stage")
    assert len(merged) == len(windows), "self-report merge duplicated windows"
    is_stress = merged["harmonized_label"] == "Stress"
    merged["self_report_validated"] = ~is_stress | (merged["self_report_stress_delta"] >= args.min_delta)
    merged.to_csv(path, index=False)

    pure_stress = merged[is_stress & (merged["purity"] >= 0.8)]
    print("\nStress windows (purity >= 0.8) kept after self-report validation:")
    print(pure_stress.groupby("dataset").agg(windows=("purity", "size"), validated=("self_report_validated", "sum"),
                                             subjects=("subject_uid", "nunique")).to_string())


if __name__ == "__main__":
    main()
