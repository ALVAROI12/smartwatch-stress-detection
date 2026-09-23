# Verification of manuscript rewrite (`git diff 3922e90 HEAD -- paper/main.tex`)

Worktree: `.claude/worktrees/gitignore-graphify`, HEAD `cebf35d`. T = `outputs/tables/jbhi_v2/`, RR = `T/reviewer_reanalyses/`, CP = `T/contribution_probes/`, S = `sources/`, SN = `sources/novelty/`.

## 1. Claim table

| main.tex line | Claim | Source | Value in source | Verdict |
|---|---|---|---|---|
| 25 | 132 subjects | T/dataset_audit.csv, panel n_subj | 15+35 (36 minus f13)+34+19+29 = 132 | OK |
| 25, 37, 126, 294 | Transfer cost 0.015–0.05, none significant, upper 95% bounds 0.08–0.13 | RR/D_lodo_cost_corrected.csv | cost 0.0148–0.0517; hi 0.0761–0.1322; min p_holm 0.1962 | OK |
| 25, 128 | Index "performs as well"; no model-minus-index diff survives Holm | RR/C_arousal_index_lodo.csv (hr+eda) | p_nb_holm 0.1376–1.0 | OK |
| 25 | Matched-arousal reference AUROC 0.58 | RR/A, B; CP/hardening/partA_inference.csv | 0.5839 (external) | OK, but see Inconsistency I1 |
| 25 | "these nulls exclude only deviations above 0.12" | RR/B mdd_80pct, non-separable pairs | 0.121–0.217 | WRONG (incomplete: 0.12–0.22, as in lines 211 and 291) |
| 25 | Exercise called stress 69%, 8–10% once in training | CP/specificity/partA_panel_compact.csv | 0.692/0.695; 0.099/0.081 | OK |
| 33 | Xiao: "BA falling from 0.88 to 0.53–0.69 across seven wrist datasets" | SN/xiao-2025 Table 6 + abstract | 0.8771 is ERM in-distribution on WESAD; 0.53/0.56/0.69 are three OOD sets (Control, Predose, Postdose); the paper uses seven datasets, not all wrist-worn, and not all in that comparison | WRONG (conflated; also contradicts line 41 wording) |
| 33, 41 | Mishra: cost about 0.01 between two E4 studies, same stressor | SN/mishra-2020 Table 4 | S3eda LOSO 0.94 vs S4eda→S3 0.93; S4eda LOSO 0.98 vs S3eda→S4 0.97 (median AUROC) | OK (AUROC, not BA) |
| 33 | Kwon: near-perfect transfer to WESAD, near-chance on Stress-Predict | SN/kwon-2026 abstract | "approximately 0.90 with WESAD as the target but 0.52–0.57 with Stress-Predict or Nurse" | OK ("near-perfect" is generous for 0.90) |
| 33, 47 | Kwon matched HR within subject, 5-bpm bins; 0.955→0.938, EDA+temp 0.935; SP EDA+temp 0.607 | SN/kwon-2026-iso-hr l.239, 583 | "Within each subject, windows are binned into 5 bpm HR bins"; "0.955→0.938±0.009 … 0.935±0.015 … 0.607±0.016" | OK |
| 35, 283 | Transfer cost small "once labels and per-subject normalisation are right" | line 132 (raw features −0.01 to 0.05), line 134 (1 of 16 label-fix diffs significant) | own results contradict both causes | WRONG (internal inconsistency, see I3) |
| 44 | Aydoğan: PhysioNet stress sessions mix rest and stressor blocks | SN/aydogan-2026 l.217 | "these recordings contain both rest and stress-induction blocks" | OK |
| 44 | Aydoğan: XGBoost labels 82.6% of exercise windows stress; 6–7% with exercise as class | SN/aydogan-2026 l.430, 680 | "XGB … classified 82.6% of held-out exercise-session windows as stress"; "aerobic and anaerobic false-stress rates were 0.062 and 0.070" | OK |
| 44 | Mishra exclude post-stressor rest for residual arousal | SN/mishra-2020 fn 6 | "such rest periods will contain some residual physiological arousal of the preceding stressor" | OK |
| 44 | Richer: pre-stress baseline confounded with sequence effects | SN/richer-2024 l.173 | "could be attributed to the different tasks, or to sequence effects" | OK |
| 44 | Tognotti: inflation 3–13 points | SN/tognotti-2026 l.469 | "0.03–0.13 mean balanced accuracy percentage points" | OK (BA units 0.03–0.13; "points" ambiguous) |
| 44 | Schmidt 2019 quote "sufficient to estimate the intensity level of an activity" | S/schmidt-2019-affect-review.txt | verbatim present | OK |
| 44 | Vos: apart from two studies incl. Mishra none tested on a totally unseen dataset | S/vos-2023 l.1113 | "None of these studies apart from Mishra et al. [41] and Liapis et al. [47], tested generalization … on a totally" | OK |
| 44 | Vos "estimate that 'at least 34 test subjects would be required to achieve 80% statistical power'" | S/vos-2023 l.1087–1089 | "Iqbal et al. [39] specifically performed a power analysis and similarly concluded that at least 34 test subjects would be required…" | WRONG (misattributed: Vos report Iqbal et al.'s power analysis) |
| 47 | Tervonen: retraining XGBoost +1.7 BA points on WESAD with 1–3 windows per class | SN/tervonen-2026 l.182, 248, 94 | +1.7 %-points is the average over all datasets and tasks, not WESAD; "first 1-3 samples … of each separate stimulus" | WRONG (text predates the diff, but the line was rewritten: drop "on WESAD") |
| 47 | Akkaya: few-shot benefit did not survive prospective chronological protocol | SN/akkaya-2026 (abstract only) | abstract only | UNVERIFIABLE (known abstract-only) |
| 47 | Milstein RMSSD r=0.42 in conversation vs 0.74–0.81 at rest | SN/milstein-2020 l.380, Table 2 | "lower (0.42 and 0.46) during the conversation"; baselines 0.743, 0.812 | OK |
| 47 | Watanabe: TSST worst condition for E4 beat detection | SN/watanabe-2025 | not re-checked (pre-existing claim) | UNVERIFIABLE here |
| 47 | Sosa quotes | no local full text; docs/literature/contribution-gap-scan.md (marked FT) | quotes match gap-scan transcription | UNVERIFIABLE locally (full text not in sources/) |
| 47 | Sosa: "the reviewed studies did not test classifiers on such states" | contribution-gap-scan.md | "reviewed studies mostly did not test classifiers on non-stress arousal" | WRONG (overstated: "most of the reviewed studies") |
| 52 | WESAD 15, TSST standing | S/schmidt-2018-wesad | pre-existing | OK |
| 52 | PhysioNet: Stroop and TMCT | S/hongn-2025 l.161; Table I | protocol also has serial subtraction, which Table I counts as Stress | WRONG (incomplete; Table I lists "Stroop, TMCT, subtraction") |
| 52 | Stress-Predict: hyperventilation listed among stress-inducing tasks | S/iqbal-2022 l.23 | "three different stress-inducing tasks (i.e., Stroop colour word test, Trier Social Stress Test and Hyperventilation Provocation Test session)" | OK |
| 52 | UBFC test: job interview + countdown by 17; control: holiday speech + countdown by 10; spoken aloud; "imply a high stress level" | S/meziatisabour-2021 l.211–221, 745 | all present verbatim ("pronounce the countdown numbers out loud") | OK |
| 52 | Campanella: 3 Lego tasks for "mental strain", "one with a spoken countdown" | S/campanella-2024 l.275 | "count backwards from 180 … to zero"; "to create the mental strain that employees may experience" | "spoken" UNVERIFIABLE (descriptor does not say aloud) |
| 52 | Campanella CV presentation in untagged tail, excluded | scripts/extract_features.py l.303–333 (fixed 300 s tail) | consistent with the code | OK |
| 101 | Pooled BA 0.83 (NB CI 0.79–0.88), 0.81 (0.76–0.86), 0.75 (0.68–0.82) | T/repeated_subject_splits.csv (shared_Baseline_vs_Stress_pooled, none); recomputed mean ± t19·sd·sqrt(1/20+0.25) | 0.831 [0.786, 0.877]; 0.810 [0.757, 0.864]; 0.749 [0.676, 0.822] | OK |
| 101 | LOSO 0.85, 0.81, 0.75 | T/loso_only_summary.csv | 0.8495, 0.8143, 0.7548 | OK |
| 101 | p_holm = 1.0 for modality differences | T/significance_modalities.csv (none) | all 1.0 | OK |
| 101 | 108-configuration tuning run | T/tuned_baselines_summary.csv | 108 rows | OK |
| 101 | Within BA 0.916/0.882/0.877/0.776/0.697 | T/leave_one_dataset_out_summary.csv | same | OK |
| 101 | EDA rise 0.4 z PhysioNet, 0.7 z SP, 1.7–2.2 z others | T/stress_effect_size_all_datasets.csv eda_mean | 0.39, 0.72; 1.72–2.24 | OK |
| 116–120 | Table II all 35 cells (model, index, +temperature) | T/leave_one_dataset_out_summary.csv; RR/C ba_splits | all match (index UBFC 0.7625 shown as 0.762) | OK |
| 126 | Costs 0.03/0.03/0.05/0.04/0.015; all-5 vs within | RR/D; LODO summary | 0.0316/0.0302/0.0517/0.0394/0.0148; 0.689 vs 0.697, 0.839 vs 0.877 | OK |
| 126 | p_holm ≥ 0.20; SP 0.052 [0.003, 0.101]; 3–7 held-out subjects per split | RR/D; scripts/run_jbhi_experiments.py grouped_split (20%, rounded) | 0.1962; 0.0517 [0.0028, 0.1006]; round(0.2·n) = 3–7 | OK |
| 128 | Index external BA 0.836/0.743/0.657/0.762/0.877; AUROC 0.73–0.95 | RR/C ba_splits, auroc_splits | 0.8362/0.7434/0.6569/0.7625/0.8766; 0.731–0.951 | OK |
| 128 | Model beats index by 0.048 (WESAD), 0.075 (UBFC), +0.003 PhysioNet, trails SP and Campanella; p_holm ≥ 0.14 | RR/C | 0.048, 0.0748, 0.0026, −0.0113, −0.0091; min 0.1376 | OK |
| 128 | EDA-only index loses on PhysioNet (−0.15); HR-only loses on UBFC, Campanella | RR/C | 0.1472 (p_holm 0.0007); 0.2635 (0.0001), 0.3189 (0.033) | OK |
| 154–162 | Table III, 27 rates and CIs | CP/specificity/partA_panel_compact.csv | all match after rounding | OK |
| 154–162 | "(subjects / windows)" | same | windows are the condition-1 counts; conditions 2–3 use about 4× more windows (e.g. Lego 2760) | minor: label ambiguous |
| 164 | In-task non-stress 0.08–0.29 / 0.05–0.25 | CP/specificity/partA_specificity_panel.csv (false_stress_core) | 0.084–0.291; 0.046–0.252 | OK |
| 165 | Stress recall 0.53–0.91 / 0.63–0.95; added negative "within −0.05" | same | 0.529–0.907; 0.629–0.947; worst drop −0.046 | OK |
| 170 | 69–70% vs Aydoğan 71–83% | panel; SN/aydogan l.430, 685 | 0.692/0.695; 71.0% (WESAD→PhysioNet RF), 82.6% | OK |
| 170 | BA 0.746→0.583, p_holm 0.011 | T/novelty_experiments/exercise_negatives_physionet.csv | 0.746→0.583, p_holm 0.0106 | OK |
| 170 | UBFC 0.55–0.57; hypervent 0.42→0.54–0.55; Lego 0.43→0.33; EPM fall 0.05–0.10 | panel | 0.55/0.573/0.563; 0.415→0.538/0.547; 0.427→0.328; 0.053–0.101 | OK |
| 170 | Recall 0.74→0.70 (Lego), 0.74→0.71, 0.92→0.89 (EPM); exercise raises SP 0.63→0.70, UBFC 0.74→0.79 | panel | 0.744→0.698; 0.735→0.709; 0.917→0.886; SP 0.629→0.708/0.697; UBFC 0.744→0.798/0.788 | OK (SP 0.70–0.71, UBFC 0.79–0.80) |
| 170 | 1.7 windows/subject, 1 window/subject, 8 subjects | panel | 53/31 = 1.71; 33/33; 8 | OK |
| 191 | Ref 0.584 [0.54, 0.64], p_holm 0.004, MDD 0.07 | partA_inference; RR/B | 0.5839 [0.5407, 0.6374], 0.0035, 0.0654 | OK |
| 192 | Exercise ext Δ ref **+0.18** [+0.08, +0.25] | RR/A | diff 0.1747 [0.0814, 0.252] | WRONG (+0.17) |
| 192 | Exercise ext within-bin CI [0.67, **0.82**] | partA_inference (the source of every other row) | [0.6716, 0.8299] | WRONG (0.83; 0.82 comes from partA_matched_arousal.csv, mixing sources) |
| 193–197 | Fear, UBFC, Hypervent, Anger, Lego external rows (AUROC, CI, Δ, p_holm, MDD) | partA_inference; RR/A, B | all match | OK |
| 199 | Ref 0.71–0.72 (trained with probe) | RR/B | 0.7089–0.7243 | OK |
| 200, 201, 203, 204, 205 | Exercise, fear, hypervent, Lego, anger rows (trained) | partA_inference; RR/A, B | all match | OK |
| 202 | UBFC control (trained) Δ **−0.16** [−0.26, −0.05] | RR/A | −0.1546 [−0.2629, −0.0496] | WRONG (−0.15) |
| 208 | MDD = 2.8 bootstrap SD; only hypervent equivalent within ±0.10 (90% CI [0.45, 0.60]); none within ±0.05 | RR/B | 0.0654/0.0233 = 2.81; equiv_0.10 True only for ext hypervent [0.449, 0.596]; equiv_0.05 all False | OK |
| 211 | "0.16–0.29 below it with CIs excluding zero" | RR/A | 0.155, 0.164, 0.218, 0.292; all hi < 0 | WRONG (0.15–0.29) |
| 211 | "carry little stress information beyond arousal magnitude, whatever the comparator" | RR/B cond3 ref 0.71–0.72; CP/arousal/partA_matched_arousal.csv within_LOSO ref 0.784 | contradicted two sentences later | WRONG (see I1) |
| 211 | MDD 0.12–0.22 | RR/B | 0.121–0.217 | OK |
| 211 | Exercise 0.892 (p_holm 0.003), 0.759 without; fear 0.779 | partA_inference | 0.892/0.003; 0.7587; 0.7787 | OK |
| 211 | Exercise-minus-pair lower bounds 0.18–0.36 | partA_inference diff_lo | 0.179–0.361 | OK |
| 211 | HR-only index 0.91, EDA-only 0.82 (trained); 0.65, 0.54 (external) | CP/arousal/partA_matched_arousal.csv exercise same_dataset | 0.912, 0.818; 0.654, 0.538 | OK |
| 213 | Disjoint: hypervent and UBFC 0.45–0.54 in every cell | CP/hardening/partA_disjoint.csv (= RR/F) | 0.4537–0.539 | OK |
| 213 | Lego 0.73 ext, 0.84 trained under HR index; 259 windows | same | 0.7264, 0.8389; n = 259 | OK |
| 213 | 63% of Campanella windows lack pulse | T/dataset_audit.csv; RR/E | 63.1% | OK |
| 213 | Anger separable under EDA index once trained (0.74); ext cardiac model exercise 0.29 | partA_disjoint | 0.7425 (p_holm 0.003); 0.2873 | OK |
| 213 | Median HR index 3.9 vs 1.2 | partA_matched_arousal (hr_only exercise) | 3.887 vs 1.227 | OK |
| 213 | Removing hr_mean, eda_tonic_mean, eda_mean changes no row by > 0.09 | partA_disjoint (disjoint_hr+eda) vs partA_inference (full_model) | max \|Δ\| 0.085 (anger trained, Lego ext) | OK |
| 213 | WESAD 63% stress vs 10% baseline windows lack pulse | RR/E | 0.633 vs 0.102 | OK |
| 213 | Exercise median index 4.8 vs 0.8; bins cover lowest three quintiles | partA_matched_arousal | 4.781 vs 0.794; bins "0.81 0.64 0.83 na na" | OK |
| 275 | 69.5% by other-4 model; 4–6% with PhysioNet subjects in training | T/novelty_experiments/leave_one_dataset_out_summary_session_exercise.csv | 0.695; 0.043 (within), 0.064 (all) | OK |
| 283 | Stress from rest "only weakly (0.584)" | RR/B | external only; 0.71–0.72 trained, 0.78 within-LOSO | see I1 |
| 283 | "sixth (PhysioNet) to a third (SP)" of WESAD EDA response | stress_effect_size_all_datasets.csv | 0.39/2.24 = 0.17; 0.72/2.24 = 0.32 | OK |
| 283 | Recovery, sensor, onset filters recover at most 0.06; strict responder ≤ 0.09 PhysioNet, nothing on SP | CP/arousal/partB_error_budget.csv; CP/hardening/partB_responder_sensitivity.csv | F1+F2+F3 Shapley ≤ 0.057; 1.0sd_both PhysioNet 0.094, SP −0.02 ext / +0.032 within (CI includes 0) | OK |
| 283 | Richer: HPA axis assessed through cortisol | SN/richer-2024 l.12, 14 | "slower HPA axis … release of the hormone cortisol" | OK |
| 283 | "our reference row … suggests that this residual [Kwon, WESAD within-subject] is largely electrodermal arousal" | partA_matched_arousal within_LOSO ref | 0.784 within-LOSO (Kwon's setting is within-dataset); 0.584 is external | WRONG (unsupported comparison, see I1) |
| 285 | PPV 0.28 / 0.11 / 0.05 (prev 0.05, recall 0.75, FSR 0.10/0.33/0.69) | recomputed | 0.283, 0.107, 0.054 | OK (table range is actually 0.08–0.70) |
| 289 | f-TSST matches TSST structure and cognitive demand without social-evaluative elements; within subject, randomised | SN/richer-2024 l.53, 37 | "as similar as possible to the TSST with similar structure and cognitive demands but social-evaluative elements are removed"; "within-design setting in randomized order" | OK |
| 291 | Milstein SCL r = 0.61, 0.40, n.s. 0.30 | SN/milstein-2020 l.884 | r(28) = 0.606, 0.399 (p = 0.03), 0.298 (p = 0.11) | OK |
| 291 | "strict definitions discard half the subjects" | partB_responder_sensitivity | PhysioNet keeps 17–18/35; SP keeps 12–17/34 (discards 50–65%) | OK (approximate) |
| 291 | Raw and causal keep cost on long recordings, lose about 0.10 on short | line 132; novelty_experiments | consistent with line 132 | OK |

## 2. Inconsistencies between sections

- **I1. How well stress separates from rest at matched arousal.** Line 211 says the wrist features "carry little stress information beyond arousal magnitude, whatever the comparator". Two sentences later the same paragraph reports a reference of 0.71–0.72 for the trained model and says "the model learns to separate stress from rest beyond arousal". The within-LOSO reference, which is not reported, is 0.784 (`partA_matched_arousal.csv`). The abstract (25), Discussion (283) and Conclusion (294) quote only the external 0.58 as "only weakly". The trained reference is higher mainly because the target dataset is in training in condition 3, not because the probe was added. The table header at line 199 hides this. Line 283 then compares the external 0.584 with Kwon's within-dataset WESAD result.
- **I2. Magnitude of the nulls.** The abstract (25) says "exclude only deviations above 0.12". Line 211 and the Limitations (291) say 0.12–0.22.
- **I3. Cause of the small transfer cost.** The Introduction (35: "small once labels and per-subject normalisation are right") and the Discussion (283: "Once labels are protocol-anchored and features are normalised per subject … transfer is cheap") disagree with the Results. At line 132 the cost stays small with raw features (−0.01 to 0.05). At line 134 only 1 of 16 label-fix differences is significant, and the text says "the label audit is a correctness contribution".
- **I4. Methods (lines ~94–96, unchanged) are out of date.** "Four evaluation rules" does not define the index-only LODO baseline, the reference pair and Δ-ref, the permutation test, the MDD or the equivalence margins, which Results and Reporting norms now rely on. The Methods sentence "An AUROC near 0.5 means the model has learned nothing beyond arousal magnitude" conflicts with the new reading, in which the comparison is against the reference (0.58), not 0.5.
- **I5. Xiao is described twice, differently:** line 33 ("0.88 to 0.53–0.69 across seven wrist datasets") and line 41 ("train on one or two of seven wrist datasets … 0.53–0.69 out of distribution").
- **I6. PhysioNet stressors:** line 52 lists Stroop and TMCT; Table I lists Stroop, TMCT and subtraction.
- **I7. Table III window counts** are the condition-1 counts only.

## 3. Leftover wording from the old thesis

- `paper/make_paper_figures.py` l.50 and l.73 label the UBFC-Phys control version **"Speech/arith., no evaluation"** in both Fig. 2 panels, and the compiled `fig_panel.pdf` shows it. This contradicts line 170 ("an easier social task that its authors expect to be stressful"). The figures were regenerated at 11:54, before the rewrite.
- Line 82 (unchanged): hyperventilation is "a respiratory manoeuvre with no evaluative component". This is tolerable, but it is the only remaining "evaluative" framing of a probe.
- No remaining "evaluative stress does not transfer" or "cannot separate" statement in main.tex. "No better from … Lego/anger" (25, 283) understates the result: those AUROCs are below 0.5, so the model ranks Lego and anger above stress.

## 4. LaTeX and PDF layout (paper/main.pdf, 10 pp.)

- Table III (p. 5): the `L` column is too narrow. Every probe label wraps onto two lines ("PhysioNet aerobic (30 / ↵ 2143)"), and so do both reference rows. Shorten the labels or widen the column (for example `@{}p{0.34\columnwidth}ccc@{}`), or move the "(subj / win)" counts into their own column.
- Table IV: "Baseline and rest (ref.)" wraps. Acceptable.
- Fig. 1 (p. 4) and Fig. 3 (p. 7) predate the rewrite. Legends overlap the bars and the Fig. 3(b) y-label is clipped.
- Fig. 2 needs regenerating after the label fix above.

## 5. Required fixes (exact replacement text)

1. **Line 192:** `+0.18 [+0.08, +0.25]` → `+0.17 [+0.08, +0.25]`, and `0.759 [0.67, 0.82]` → `0.759 [0.67, 0.83]`.
2. **Line 202:** `\m0.16 [\m0.26, \m0.05]` → `\m0.15 [\m0.26, \m0.05]`.
3. **Line 211:** `stay at 0.43--0.57, 0.16--0.29 below it` → `stay at 0.43--0.57, 0.15--0.29 below it`.
4. **Line 211:** `reaches only 0.584 for the external model. The wrist features therefore carry little stress information beyond arousal magnitude, whatever the comparator.` → `reaches only 0.584 for the external model (0.71--0.72 once the target dataset is in training; 0.78 within dataset). A model that has not seen the target dataset therefore carries little stress information beyond arousal magnitude.`
5. **Line 211:** `Once the model has been trained with the probed state as a negative, the reference rises to 0.71--0.72` → `When the probed state is added as a negative, which also puts the target dataset in training, the reference rises to 0.71--0.72`. Mirror this in the Table IV sub-header at line 199: `Model trained on all datasets with the probed state as a negative (ref. 0.71--0.72)`.
6. **Line 25 (abstract):** `these nulls exclude only deviations above 0.12` → `these nulls exclude only deviations of 0.12--0.22`. Also `At matched arousal the model separates stress from baseline and rest only weakly (AUROC 0.58)` → `At matched arousal a model that has not seen the target dataset separates stress from baseline and rest only weakly (AUROC 0.58)`.
7. **Lines 283 and 294:** add the same qualifier: `only weakly (0.584 externally, 0.71--0.72 in distribution)`. Line 283: replace `our reference row, which matches heart rate and EDA together, suggests that this residual is largely electrodermal arousal` with `our external reference row, which matches heart rate and EDA together, is weak (0.58), but within dataset the same contrast reaches 0.78, so part of that residual survives joint matching in distribution`.
8. **Line 35:** `the transfer cost is small once labels and per-subject normalisation are right, and that` → `the transfer cost is small, with or without per-subject normalisation and label fixes, and that`. **Line 283:** `Once labels are protocol-anchored and features are normalised per subject, the distribution of that arousal is similar across laboratory datasets` → `The distribution of that arousal is similar across laboratory datasets`.
9. **Line 44 (Vos):** `and estimate that ``at least 34 test subjects would be required to achieve 80\% statistical power''` → `and cite the power analysis of Iqbal et al.~\cite{iqbal2022}, which concluded that ``at least 34 test subjects would be required to achieve 80\% statistical power''`.
10. **Line 33 (Xiao):** `Xiao et al.~\cite{xiao2025} report balanced accuracy falling from 0.88 to 0.53--0.69 across seven wrist datasets` → `Xiao et al.~\cite{xiao2025} report balanced accuracy falling from 0.88 on WESAD to 0.53--0.69 on three out-of-distribution datasets`.
11. **Line 47 (Tervonen):** `gained 1.7 balanced-accuracy points on WESAD with 1--3 windows per class` → `gained on average 1.7 balanced-accuracy points across datasets and tasks with the first 1--3 samples of each stimulus`.
12. **Line 47 (Sosa):** `the reviewed studies did not test classifiers on such states` → `most reviewed studies did not test classifiers on such states`. Add the Sosa full text to `sources/novelty/` so the quotes can be verified.
13. **Line 52:** `Stroop and the Trier Mental Challenge Test (TMCT) of mental arithmetic under time pressure, seated` → `Stroop, the Trier Mental Challenge Test (TMCT) of mental arithmetic under time pressure and serial subtraction, seated`. Also `one with a spoken countdown` → `one with a countdown`.
14. **Table III header (line ~152):** `Probed state (subjects / windows)` → `Probed state (subjects / windows, dataset-absent run)`.
15. **Methods, Four evaluation rules:** after the matched-arousal definition add `Each probe pair is compared with the stress-versus-baseline/rest pair under the same model ($\Delta$ ref, joint subject bootstrap), tested against 0.5 by a within-bin permutation test with Holm correction, and reported with its minimal detectable deviation (80\% power) and a $\pm$0.10 equivalence test. \emph{Index-only baseline.} The same index, thresholded on the training datasets, is scored under LODO and compared with the model by paired Nadeau--Bengio tests.` Then replace `An AUROC near 0.5 means the model has learned nothing beyond arousal magnitude for that pair.` with `An AUROC at the level of the reference pair means the model separates that state no better than it separates rest.`
16. **`paper/make_paper_figures.py` l.50 and l.73:** `"Speech/arith., no evaluation"` → `"UBFC-Phys control version"`. Then rerun the script and `tectonic paper/main.tex`.
