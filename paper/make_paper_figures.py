"""Figures for the JBHI manuscript. Reads committed tables only; writes PDFs next to this file."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T = Path(__file__).resolve().parents[1] / "outputs" / "tables" / "jbhi_v2"
OUT = Path(__file__).parent
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 8, "legend.fontsize": 7, "legend.frameon": False, "pdf.fonttype": 42})
ORDER = ["WESAD", "PhysioNet", "Stress-Predict", "UBFC-Phys", "Campanella2024"]
SHORT = {"WESAD": "WESAD", "PhysioNet": "PhysioNet", "Stress-Predict": "Stress-\nPredict", "UBFC-Phys": "UBFC-\nPhys",
         "Campanella2024": "Campanella"}


def ci(s):  # "[a,b]" -> (a, b)
    a, b = s.strip("[]").split(",")
    return float(a), float(b)


def fig1_lodo():
    d = pd.read_csv(T / "leave_one_dataset_out_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, feat, title in zip(axes, ["hr_hrv_eda", "physiology"], ["HR/HRV/EDA features", "HR/HRV/EDA + temperature"]):
        x = np.arange(len(ORDER))
        for k, (name, off, c) in enumerate([("within_dataset", -0.25, "#4c72b0"), ("other_datasets_only", 0, "#dd8452"),
                                             ("all_datasets", 0.25, "#55a868")]):
            v = [d[(d.features == feat) & (d.test_dataset == ds) & (d.trained_on == name)].balanced_accuracy.item() for ds in ORDER]
            ax.bar(x + off, v, 0.25, color=c, label=name.replace("_", " "))
        ax.set_xticks(x, [SHORT[o] for o in ORDER])
        ax.set_ylim(0.5, 1.0)
        ax.set_title(title)
        if feat == "physiology":
            ax.annotate("no temperature\nchannel in target", xy=(3, 0.589), xytext=(2.2, 0.62), fontsize=6.5,
                        arrowprops=dict(arrowstyle="->", lw=0.6))
    axes[0].set_ylabel("Balanced accuracy (held-out subjects)")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_lodo.pdf")


def fig2_panel():
    p = pd.read_csv(T / "contribution_probes/specificity/partA_panel_compact.csv")
    keep = ["PhysioNet Aerobic", "PhysioNet Anaerobic", "Campanella2024 Manual task", "UBFC-Phys Control task",
            "Stress-Predict Hyperventilation", "EPM-E4 Fear", "EPM-E4 Anger", "EPM-E4 Sadness", "EPM-E4 Happiness"]
    lab = {"PhysioNet Aerobic": "Aerobic exercise", "PhysioNet Anaerobic": "Anaerobic exercise",
           "Campanella2024 Manual task": "Lego manual task", "UBFC-Phys Control task": "Speech/arith., no evaluation",
           "Stress-Predict Hyperventilation": "Hyperventilation", "EPM-E4 Fear": "Fear clips", "EPM-E4 Anger": "Anger clips",
           "EPM-E4 Sadness": "Sadness clips", "EPM-E4 Happiness": "Happiness clips"}
    p = p.set_index("probe_class").loc[keep]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(keep))
    for off, col, cic, c, name in [(0.27, "ext_rate", "ext_ci", "#dd8452", "dataset absent from training"),
                                   (0, "all_absent_rate", "all_absent_ci", "#8172b2", "class absent, dataset present"),
                                   (-0.27, "added_neg_rate", "added_neg_ci", "#55a868", "class added as negative")]:
        lo = np.array([ci(s)[0] for s in p[cic]]); hi = np.array([ci(s)[1] for s in p[cic]])
        v = p[col].to_numpy()
        ax1.barh(y + off, v, 0.25, color=c, label=name, xerr=[v - lo, hi - v], error_kw=dict(lw=0.6))
    ax1.set_yticks(y, [lab[k] for k in keep]); ax1.invert_yaxis()
    ax1.set_xlabel("False-stress rate (p > 0.5)"); ax1.set_xlim(0, 1)
    ax1.axvline(0.5, color="grey", lw=0.5, ls=":")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=6)
    ax1.set_title("(a) Specificity panel")

    m = pd.read_csv(T / "contribution_probes/arousal/partA_matched_arousal.csv")
    m = m[m["index"] == "hr+eda"]
    rows = [("exercise", "same_dataset", "cond3_added_negative", "Exercise (model shown exercise)"),
            ("Baseline/Rest (ref)", "pooled_stress", "external", "Baseline / rest (reference)"),
            ("EPM Fear", "pooled_stress", "external", "Fear clips"),
            ("UBFC Control task", "same_dataset", "external", "Speech/arith., no evaluation"),
            ("Hyperventilation", "same_dataset", "external", "Hyperventilation"),
            ("EPM Anger", "pooled_stress", "external", "Anger clips"),
            ("Campanella Manual task", "same_dataset", "external", "Lego manual task")]
    vals, los, his, names = [], [], [], []
    for probe, scope, score, name in rows:
        r = m[(m.probe == probe) & (m.stress_scope == scope) & (m.score == score)].iloc[0]
        vals.append(r.auc_binned); los.append(r.lo); his.append(r.hi); names.append(name)
    vals, los, his = map(np.array, (vals, los, his))
    y2 = np.arange(len(rows))
    cols = ["#55a868"] + ["#dd8452"] * (len(rows) - 1)
    ax2.barh(y2, vals, 0.55, color=cols, xerr=[vals - los, his - vals], error_kw=dict(lw=0.6))
    ax2.set_yticks(y2, names); ax2.invert_yaxis()
    ax2.axvline(0.5, color="k", lw=0.6)
    ax2.set_xlim(0.2, 1.0); ax2.set_xlabel("AUROC within arousal quintiles")
    ax2.set_title("(b) Matched-arousal test")
    fig.tight_layout()
    fig.savefig(OUT / "fig_panel.pdf")


def fig3_time():
    s = pd.read_csv(T / "contribution_probes/timeprobe/time_probe_summary.csv")
    w = pd.read_csv(T / "contribution_probes/timeprobe/warmup_trajectory_2min_bins.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    ax = axes[0]
    ds = ["WESAD", "PhysioNet", "Stress-Predict", "Campanella2024"]
    x = np.arange(len(ds))
    for off, model, c, name in [(-0.25, "time_only_within", "#c44e52", "time since start only"),
                                (0, "std_within", "#4c72b0", "physiology, within"),
                                (0.25, "std_external", "#dd8452", "physiology, external")]:
        v = [s[(s.dataset == d) & (s.model == model)].ba.item() for d in ds]
        ax.bar(x + off, v, 0.25, color=c, label=name)
    ax.set_xticks(x, [SHORT[d] for d in ds], fontsize=6.5); ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Balanced accuracy"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=6, ncol=1)
    ax.set_title("(a) Time versus physiology")
    for ax, feat, lab, title in [(axes[1], "temp_mean_dm", "Skin temperature (°C, subject-centred)", "(b) Temperature"),
                                 (axes[2], "eda_tonic_mean_dm", "Tonic EDA (µS, subject-centred)", "(c) Tonic EDA")]:
        for d, c in zip(["Campanella2024", "PhysioNet", "Stress-Predict", "WESAD", "EPM-E4"],
                        ["#55a868", "#4c72b0", "#dd8452", "#8172b2", "#937860"]):
            g = w[(w.dataset == d) & (w.bin <= 30)]
            ax.plot(g.bin + 1, g[feat], "-o", ms=2, lw=0.9, color=c, label=d.replace("2024", ""))
        ax.axvspan(0, 10, color="grey", alpha=0.12, lw=0)
        ax.set_xlabel("Minutes since first protocol window"); ax.set_ylabel(lab, fontsize=7); ax.set_title(title)
    axes[1].set_ylim(-4.5, 4.5); axes[2].set_ylim(-1.2, 1.2)
    axes[2].legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_time.pdf")


def fig4_budget():
    b = pd.read_csv(T / "contribution_probes/arousal/partB_error_budget.csv")
    rows = [("PhysioNet", "within_LOSO"), ("PhysioNet", "external"), ("Stress-Predict", "within_LOSO"),
            ("Stress-Predict", "external"), ("WESAD", "external")]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for key, c, name in [("F1_recovery_shapley", "#4c72b0", "recovery (F1)"), ("F2_sensor_shapley", "#dd8452", "sensor (F2)"),
                         ("F3_onset_shapley", "#55a868", "onset (F3)"), ("F4_responder_shapley", "#8172b2", "responder (F4)")]:
        v = np.array([max(b[(b.dataset == d) & (b.score == m)][key].item(), 0) for d, m in rows])
        ax.barh(y, v, 0.55, left=left, color=c, label=name); left += v
    resid = np.array([1 - b[(b.dataset == d) & (b.score == m)].ba_all_filters.item() for d, m in rows])
    ax.barh(y, resid, 0.55, left=left, color="#cccccc", label="unexplained")
    ax.set_yticks(y, [f"{d.replace('2024', '')}, {m.replace('_LOSO', '')}" for d, m in rows]); ax.invert_yaxis()
    ax.set_xlabel("Share of 1 − balanced accuracy"); ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_budget.pdf")


if __name__ == "__main__":
    fig1_lodo(); fig2_panel(); fig3_time(); fig4_budget()
    assert all((OUT / f).exists() for f in ["fig_lodo.pdf", "fig_panel.pdf", "fig_time.pdf", "fig_budget.pdf"])
    print("ok")
