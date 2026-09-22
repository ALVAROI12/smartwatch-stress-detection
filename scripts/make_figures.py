#!/usr/bin/env python3
"""Draw the current result figures from the committed tables in outputs/tables/jbhi_v2/.

Writes PNG and PDF files to outputs/figures/jbhi_v2/. Needs only the committed CSVs, not the raw data.
Figures from the old notebook pipeline live in legacy/outputs/figures/ and are not valid results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "outputs" / "tables" / "jbhi_v2"
OUT = REPO_ROOT / "outputs" / "figures" / "jbhi_v2"

# Validated categorical slots (dataviz reference palette, light mode); marker shapes add a second cue.
SERIES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "D")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DATASETS = {"WESAD": "WESAD", "PhysioNet": "PhysioNet", "Stress-Predict": "Stress-Predict",
            "UBFC-Phys": "UBFC-Phys", "Campanella2024": "Campanella"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "savefig.facecolor": "#fcfcfb", "legend.frameon": False})


def save(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def dot_rows(ax, frame: pd.DataFrame, columns: dict[str, str], xlabel: str) -> None:
    """One row per dataset, one marker per series, joined by a thin line; value labels on the first and last series."""
    rows = list(frame.index)[::-1]
    offsets = [0.18, 0.0, -0.18][: len(columns)]  # small vertical offset so equal values never hide each other
    for y, name in enumerate(rows):
        values = frame.loc[name, list(columns)]
        ax.plot([values.min(), values.max()], [y, y], color=GRID, lw=2, zorder=1)
    for (col, label), (color, marker), dy in zip(columns.items(), SERIES, offsets):
        ax.scatter(frame.loc[rows, col], [y + dy for y in range(len(rows))], s=56, color=color, marker=marker,
                   label=label, edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.2)


def leakage() -> None:
    t = pd.read_csv(TABLES / "leakage_check.csv")
    names = ["Windows split at random\n(old thesis pipeline)", "Unseen subjects\n(subject-grouped split)"]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for i, (metric, label) in enumerate((("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"))):
        color, _ = SERIES[i]
        means, lo, hi = (t[f"{metric}_{s}"].to_numpy() for s in ("mean", "ci95_low", "ci95_high"))
        y = [j + (i - 0.5) * 0.36 for j in range(2)]
        ax.barh(y, means, height=0.34, color=color, label=label, xerr=[means - lo, hi - means],
                error_kw={"ecolor": MUTED, "lw": 1})
        for yy, m, top in zip(y, means, hi):
            ax.text(top + 0.015, yy, f"{m:.2f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(2), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Six-class task, 10 repeats (mean, 95% CI)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "fig1_window_vs_subject_split")


def lodo() -> None:
    t = pd.read_csv(TABLES / "leave_one_dataset_out_summary.csv")
    t = t[t["features"] == "hr_hrv_eda"].pivot(index="test_dataset", columns="trained_on", values="balanced_accuracy")
    t = t.rename(index=DATASETS).loc[list(DATASETS.values())]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    dot_rows(ax, t, {"within_dataset": "Other subjects, same dataset", "other_datasets_only": "Other datasets only",
                     "all_datasets": "All datasets"},
             "Balanced accuracy on held-out subjects (HR/HRV/EDA, 20 splits)")
    save(fig, "fig2_leave_one_dataset_out")


def normalisation() -> None:
    frames = {}
    for tag, label in (("session_check", "Whole-session z-score"), ("causal", "Past windows only"), ("raw", "No per-subject scaling")):
        s = pd.read_csv(TABLES / "novelty_experiments" / f"leave_one_dataset_out_summary_{tag}.csv")
        s = s[(s["features"] == "hr_hrv_eda") & (s["trained_on"] == "other_datasets_only")]
        frames[label] = s.set_index("test_dataset")["balanced_accuracy"]
    t = pd.DataFrame(frames).rename(index=DATASETS).loc[list(DATASETS.values())]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    dot_rows(ax, t, {c: c for c in t.columns}, "External balanced accuracy (trained on the other datasets)")
    save(fig, "fig3_normalisation")


def exercise() -> None:
    s = pd.read_csv(TABLES / "novelty_experiments" / "leave_one_dataset_out_summary_session_exercise.csv")
    s = s[(s["features"] == "hr_hrv_eda") & (s["test_dataset"] == "PhysioNet")].set_index("trained_on")
    order = {"within_dataset": "Other PhysioNet subjects\n(exercise in training)",
             "all_datasets": "All datasets\n(exercise in training)",
             "other_datasets_only": "Other datasets only\n(no exercise in training)"}
    values = 100 * s.loc[list(order), "exercise_called_stress"]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.barh(range(3), values, height=0.55, color=SERIES[0][0])
    for y, v in enumerate(values):
        ax.text(v + 1.5, y, f"{v:.1f}%", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(3), list(order.values()))
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("PhysioNet exercise windows called stress (%)")
    save(fig, "fig4_exercise_false_alarms")


def wrist_vs_ecg() -> None:
    t = pd.read_csv(TABLES / "wrist_vs_ecg_summary.csv").set_index("label").drop(index="all")
    t = t.loc[["Baseline", "Amusement", "Meditation", "Stress"]]
    panels = (("pct_with_valid_ppg", "Windows with usable wrist PPG (%)", "{:.0f}%"),
              ("hr_mean_mae", "Heart-rate error vs chest ECG (bpm)", "{:.1f}"),
              ("hrv_rmssd_r", "RMSSD correlation with chest ECG (r)", "{:.2f}"))
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8), sharey=True)
    for ax, (col, title, fmt) in zip(axes, panels):
        ax.barh(range(len(t)), t[col], height=0.55, color=SERIES[0][0])
        for y, v in enumerate(t[col]):
            ax.text(v, y, " " + fmt.format(v), va="center", color=INK, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlim(0, t[col].max() * 1.3)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(t)), t.index)
    axes[0].invert_yaxis()
    fig.suptitle("WESAD: Empatica E4 wrist PPG vs RespiBAN chest ECG, per condition", x=0.01, ha="left", fontsize=10)
    save(fig, "fig5_wrist_vs_chest_ecg")


def few_shot() -> None:
    rows = {}
    for support in ("random", "chronological"):
        t = pd.read_csv(TABLES / f"domain_adaptation_physiology_{support}_gap30.csv")
        t = t[(t["train"] == "PhysioNet") & (t["test"] == "WESAD") & (t["normalisation"] == "subject_zscore")].set_index("method")
        rows["No target labels (source only)"] = t.loc["source_only", "balanced_accuracy_mean"]
        key = "5 labelled windows per class,\nfirst in time + 30 s gap" if support == "chronological" \
            else "5 labelled windows per class,\npicked at random"
        rows[key] = t.loc["finetune_k5 (uses target labels)", "balanced_accuracy_mean"]
    values = pd.Series(rows)
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.barh(range(len(values)), values, height=0.55, color=SERIES[0][0])
    for y, v in enumerate(values):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(values)), values.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("WESAD balanced accuracy, trained on PhysioNet")
    save(fig, "fig6_few_shot_personalisation")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for draw in (leakage, lodo, normalisation, exercise, wrist_vs_ecg, few_shot):
        draw()
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")


if __name__ == "__main__":
    main()
