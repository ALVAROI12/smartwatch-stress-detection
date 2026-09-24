# Under-used findings inventory (JBHI revision, `paper/main.tex`)

Scope: CLAUDE.md, STATUS.md, docs/*.md, every CSV in `outputs/tables/` (jbhi_v1, jbhi_v2 and subfolders, jbhi_v2_self_report_validated), `sources/jbhi-chat-scratch/`, and the manuscript text. Read-only; two small pandas computations were run on `wrist_vs_ecg_per_window.csv` (marked NEW below). "In paper?" is from reading/grepping `paper/main.tex`. No CSV filename outside the ten already cited is named in the manuscript, so the check was done on the numbers.

## A. Top 5 under-used findings

1. **Wrist HRV reverses sign under speech; wrist HR sees less than half the true cardiac response.** This is new, and it combines three results.
   - Chest ECG on the same WESAD windows: RMSSD falls from baseline to TSST (median 44.9 to 36.7 ms; per subject −11.6 ms, rising in only 6/15). Wrist RMSSD *rises* (89 to 93 ms; per subject +26.5 ms, rising in 9/14). SDNN behaves the same way (wrist +25.6 ms, ECG −2.9 ms).
   - Median per-subject HR rise: wrist +9.2 bpm, ECG +21.3 bpm. Wrist shows an HR rise in 12/14 subjects, ECG in 15/15.
   - During stress, SDNN agreement drops to r = 0.12 and RMSSD to r = 0.38 (bias +52 ms). Only 10 of 311 TSST windows have coverage ≥ 0.8 (`wrist_vs_ecg_summary.csv`).
   - `stress_effect_size_by_dataset.csv`: wrist HRV z *increases* under stress on WESAD (RMSSD +0.86, SDNN +0.99, pNN50 +1.21) and *decreases* on seated PhysioNet (−0.42, −0.54, −0.31). The same feature therefore carries opposite meanings across datasets, and the detector still transfers. That says EDA and HR level carry the model.
   - The paper gives only r = 0.53, 37% usable PPG and a −7 bpm bias, as a methods check.
   - Novelty: **high**. It is a wrist-specific, ECG-referenced claim: "HRV from the wrist during speech stressors measures motion, not vagal withdrawal". It also backs the arousal thesis (the model cannot be using HRV as vagal tone). Prior work (Milstein 2020, Watanabe 2025) reports agreement, not the sign reversal.

2. **Source diversity does what few-shot personalisation does.**
   - Single source PhysioNet→WESAD: source-only 0.70 BA, and k = 5 chronological few-shot lifts it to 0.86 (+0.17; `domain_adaptation_physiology_chronological_gap30.csv`).
   - Four-dataset source, no target labels: WESAD external 0.884–0.89, and few-shot adds +0.003 to +0.008 (`fewshot/fewshot_summary_session_chronological_gap30.csv`).
   - The paper reports each null separately and never puts them side by side.
   - Novelty: **med-high**. It gives a mechanism for the personalisation null and a practical rule ("pool datasets before personalising"). It needs a matched rerun with the same features and task (the single-source run uses the physiology set on the three-dataset task).

3. **Self-reported arousal does not transfer, while protocol arousal does.**
   - `cross_dataset_sam_arousal.csv`: predicting SAM high vs low arousal across WESAD↔EPM-E4 gives BA 0.48–0.66 and AUROC 0.50–0.78, with WESAD→EPM-E4 at chance (0.51–0.54).
   - `manipulation_check.csv`: only 47% of PhysioNet subjects report more stress after the opinion tasks, 61% after Stroop, 89% after TMCT and 80% after WESAD TSST.
   - `jbhi_v2_self_report_validated/`: dropping stressor windows without a self-reported rise leaves pooled LOSO BA unchanged (0.849 to 0.849 all modalities, 0.814 to 0.811 physiology).
   - `mild_stress_*`: within PhysioNet, self-rated intensity does not predict recall.
   - None of this is in the paper except "self-report is the wrong ruler".
   - Novelty: **high** for the thesis. It separates the three constructs (exposure, physiological arousal, felt stress) with data, which the paper currently argues only conceptually.
   - Caveat: the self-report-validated run predates the label fixes (PhysioNet has 36 subjects there).

4. **Recovery contamination acts through training labels, not test windows. Baseline-referenced normalisation is the most confounded "personalisation".**
   - Second-half-rest labels raise external PhysioNet from 0.746 to 0.797 and make external exceed within (0.797 vs 0.770) (`label_fix_sensitivity/resthalf_*`). The test-side F1 filter buys only +0.015.
   - Personal-baseline normalisation gives Campanella 0.988 within, and costs 0.096/0.132 externally on Stress-Predict/UBFC-Phys (`normalisation_vs_session`). Baseline-delta features call 50–83% of later rest windows stress (`order_confound_check.csv`).
   - The paper omits the rest-half sensitivity entirely, although CLAUDE.md says "report both", and it omits the baseline-normalisation variant.
   - Novelty: **medium**. It sharpens the error-budget and time-confound sections and gives a concrete protocol recommendation.

5. **The label audit and benchmark release: the most citable asset, still unreleased.**
   - All the pieces exist: seven descriptor-level fixes plus the hyperventilation retag, before/after tables, frozen-split generator plus tests, and threshold/prevalence diagnostics.
   - The paper's Data Availability section says "audited label tables, the 20 frozen subject-grouped splits … are released … (branch jbhi-revision)". **No split or label file is tracked in git.** `outputs/splits/` does not exist, and `data/processed/` is git-ignored.
   - Novelty: **high** as a resource (contribution_map rank 2). Nearest prior work (Liu & Ning 2026, Shahriar 2025) released no split files. The claim needs to be made true before submission.

## B. Full inventory

| # | Finding | Numbers | Source file | In paper? | Novelty | Why |
|---|---|---|---|---|---|---|
| 1 | Wrist HRV sign reversal during TSST vs ECG | wrist RMSSD +26.5 ms vs ECG −11.6 ms per subject; SDNN +25.6 vs −2.9 | `jbhi_v2/wrist_vs_ecg_per_window.csv` (NEW computation) | No | High | ECG-referenced proof that wrist HRV under speech is artefact; supports "arousal via EDA/HR" |
| 2 | Wrist HR attenuates stress reactivity | median rise 9.2 bpm (wrist) vs 21.3 (ECG); 12/14 vs 15/15 subjects rise | same (NEW) | Partly (−7 bpm bias only) | High | The effect size seen by the wrist is about half the true cardiac response, which feeds the "stressor potency" ceiling |
| 3 | Stress-window HRV agreement collapses | SDNN r 0.12, RMSSD r 0.38, bias +52 ms; meditation SDNN r 0.81 | `wrist_vs_ecg_summary.csv` | Partly (overall r 0.53) | Med-high | State-dependent validity; an argument for dropping or gating HRV |
| 4 | HRV effect direction flips across datasets | WESAD RMSSD/SDNN/pNN50 z +0.86/+0.99/+1.21; PhysioNet −0.42/−0.54/−0.31 | `stress_effect_size_by_dataset.csv` | No (only EDA and HR cited from the file) | High (combined with 1–3) | Transfer works despite contradictory HRV, so HRV is not the carrier |
| 5 | Chest-ECG ceiling concentrated in low-coverage windows | wrist HR only, cov < 0.5: 0.50; ECG only 0.93; EDA only 0.891 | `timeprobe/wesad_ecg_ceiling_summary.csv` | Yes | Low | Already used |
| 6 | Single-source few-shot gains vs multi-source nulls | PhysioNet→WESAD 0.70 → 0.86 (k = 5); four-source WESAD 0.884, +0.003 to +0.008 | `domain_adaptation_physiology_chronological_gap30.csv`, `fewshot/*` | No (each null reported alone) | Med-high | "Diversity substitutes for personalisation" |
| 7 | Few-shot replaces normalisation under raw features | refit +0.02 to +0.13 raw; user-only beats source by 0.21 on UBFC-Phys | `fewshot_summary_raw_*` | No | Medium | Explains why the personalisation literature reports gains (no per-subject scaling) |
| 8 | SAM arousal cross-dataset near chance | BA 0.48–0.66, AUROC 0.50–0.78 | `cross_dataset_sam_arousal.csv` (both folders) | No | High | Detector tracks physiological, not felt, arousal; third construct |
| 9 | Manipulation check: many subjects report no stress | PhysioNet opinion 47%, Stroop 61%, TMCT 89%; WESAD 80% | `jbhi_v1_thesis_features/manipulation_check.csv` | No | Medium | Label validity of "stress" blocks; counterpoint to the responder filter |
| 10 | Self-report-validated relabelling changes nothing | pooled LOSO 0.849 → 0.849; physiology 0.814 → 0.811 | `jbhi_v2_self_report_validated/loso_summary.csv` | No | Medium | Felt-stress filtering ≠ physiological-responder filtering (which gains 0.09) |
| 11 | WESAD: the most-stressed subjects are detected *least* | ρ(rise, recall) within −0.49 [−0.87, 0.17], external −0.60 [−0.84, −0.08]; "strong" band recall 0.50/0.62 vs mild 0.99 | `mild_stress/mild_stress_summary.csv`, `_by_band.csv` | **Misreported** (see C2) | Medium (n = 15) | An inverted self-report/physiology relation is interesting; small n |
| 12 | Rest-half labels: recovery hurts via training labels | external PhysioNet 0.746 → 0.797 (> within 0.770) | `label_fix_sensitivity/resthalf_*` | No | Medium | Test-side F1 gives +0.015 only; training-label cleaning matters more |
| 13 | Baseline-referenced normalisation inflates and fails to transfer | Campanella 0.988/0.952; external cost 0.096 (SP), 0.132 (UBFC) | `novelty_experiments/normalisation_vs_session.csv` | No | Medium | The common "personal baseline" practice is the most time-confounded |
| 14 | Position-matching removes time only partly | time-only BA after matching: Stress-Predict 0.881, PhysioNet 0.620, WESAD 0.602 | `timeprobe/time_probe_summary.csv` | No (only the physiology side reported) | Medium | The proposed "position-matched" rule is insufficient on Stress-Predict; must be stated |
| 15 | WESAD post-stressor carry-over on the external model | false stress 0.204 at ≥10 min after TSST vs 0.100 before any stressor (within 0.021/0.025) | `arousal/partC_onset_offset_curves.csv` | No | Medium | Long-lasting post-stress arousal and the order confound on the "easy" dataset |
| 16 | Baseline EDA predicts external detectability on Stress-Predict | ρ 0.48 [0.18, 0.70] | `specificity/partB_miss_consistency.csv` | **Contradicted** (see C3) | Low-med | EDA floor/non-responder; must be reported honestly |
| 17 | Within/external models agree on hard subjects | PhysioNet ρ 0.64 [0.37, 0.82]; Stroop 0.72 | `partB_miss_consistency.csv` | Yes (0.64) | Low | Used |
| 18 | Stress-Predict Stroop nearly undetectable externally | recall 0.24 | `partB_per_subject_recall.csv` (contribution_map §3) | No | Low-med | Task-potency evidence within one dataset |
| 19 | Specificity panel "all datasets, state absent" column | exercise 0.61/0.58 (vs 0.69 external); others ≈ external | `specificity/partA_panel_compact.csv` | No (Methods names three conditions, the table shows two) | Low | Seeing the same subjects' stress protocol does not fix exercise; completes the table |
| 20 | Hyperventilation false stress is bimodal | external rate 0.415 but subject median 0.0 | `partA_panel_compact.csv` | No | Low | Averages hide a few subjects; cell is small |
| 21 | Label-prevalence shift | PhysioNet target stress rate 0.143 vs source 0.315; UBFC 0.49 | `threshold_transfer_probe_summary.csv` | Partly | Low | Supports the threshold discussion |
| 22 | Accelerometer alone 0.75 BA on baseline-vs-stress | 0.749 [0.72, 0.78] | `repeated_subject_splits.csv` | Yes | Low | Used |
| 23 | Source-only "fixes" (missingness cue, modality fusion, monotonic constraints) | all within ±0.03 of plain XGBoost pooled; fusion with temperature hurts (0.645) | `source_only_fixes_probe.csv` = scratch `shortcut_probe.csv` | No | Low | Another negative; could go into Table VI |
| 24 | Normalisation variants on WESAD↔PhysioNet (older pipeline) | trailing 10-min delta 0.59/0.72; baseline delta 0.65/0.65 | `normalisation_probe.csv`, scratch `baseline_delta_probe.csv` | No | Low | Superseded by five-dataset normalisation table |
| 25 | WESAD-trained model on PhysioNet stages | TMCT 57%, Stroop 54%, baseline 20%, rest 17%/12% | `wesad_model_on_physionet_stages.csv` | No | Low | Two-dataset precursor of the panel |
| 26 | Leakage magnitude in BA | random windows BA 0.807 vs subject-held-out 0.355 | `leakage_check.csv` | Partly (accuracy 94%/49%) | Low | BA gap (0.45) is the stronger statement |
| 27 | Exercise separable by cardiac signature | HR-only index 0.65, EDA-only 0.54 | hardening tables | Yes | – | Used |
| 28 | Warm-up since donning, WESAD offset | baseline starts 17–49 min after switch-on | `wesad_warmup/*` | Yes | – | Used |
| 29 | Kwon setting reproduced | SP external AUROC 0.67 vs 0.56; z-scoring WESAD 0.834 → 0.914 | `kwon_setting_summary.csv` | Yes | – | Used |

## C. Inconsistencies between paper and tables

1. **Model-comparison count.** The paper says "0 of 48 comparisons after Holm correction (`significance_models.csv`)". The file has 72 comparisons, and 1 is significant (affective_only_no_exercise, physiology, MLP vs XGBoost, p_holm 0.038). CLAUDE.md says "1 of 72". Fix: state "1 of 72, on the nine-class task only; 0 of 18 on the stress task".
2. **Mild-stress row, Table VI.** The paper reads "PhysioNet 54 units, WESAD 15: ρ = −0.08 [−0.42, 0.21] and 0.02 [−0.30, 0.31]". Both of those numbers are PhysioNet (within and external). WESAD is −0.49 [−0.87, 0.17] within and **−0.60 [−0.84, −0.08] external, and that CI excludes 0**. "Mild tasks not missed more" still holds, but WESAD shows the reverse pattern, and the paper hides it.
3. **Detectability predictors, Table VI.** "Recall vs baseline HR, RMSSD, EDA, PPG coverage: |ρ| ≤ 0.35, CIs include 0." Stress-Predict external recall vs baseline EDA is ρ = 0.483 [0.184, 0.701]. Base cardiac coverage external −0.347 [−0.671, 0.034] is borderline.
4. **Exercise false-stress with exercise in training.** The abstract and panel say "69% to 8%" (8–10%, `partA_panel_compact`, all-dataset probe model). Table VI says "69.5%; 4–6% with exercise in training" (`exercise_negatives_physionet`: within 4.3%, all 6.4%). Both are correct for different training sets, but the paper uses them without explaining the difference.
5. **Specificity panel conditions.** Methods defines three training conditions. Table III and the text report two. The "all datasets, state absent" column exists (exercise 0.61/0.58).
6. **Pooled subject count.** The Limitations section says "about 118". Table I sums to 15 + 35 + 34 + 19 + 29 = 132, or 165 with EPM-E4.
7. **Leakage example.** The paper says the earlier pipeline "reported 94% accuracy". `leakage_check.csv` reproduces 92.1% (BA 0.807) for random windows. The 94.53% is the historical README figure. Cite both, or give the reproduced value.
8. **Release claim.** Data Availability promises audited label tables and 20 frozen splits on `jbhi-revision`. None is tracked in git: `outputs/splits/` is absent and `data/processed/` is ignored.
9. **Minor.** CLAUDE.md still gives UBFC-Phys external AUROC 0.928 and oracle 0.909 (pre-fix). The paper's 0.931/0.910 matches the current `threshold_transfer_probe_summary.csv`, so the paper is right and CLAUDE.md is stale. The error-budget table matches the Shapley columns of `partB_error_budget.csv`. LODO Table II, the panel, matched-arousal, time-probe and Kwon numbers all match their files.

## D. Suggested use

- Add a short "What the wrist sees" subsection (items 1–4, 8–10) that makes the three-construct argument empirical: exposure vs physiological arousal vs felt stress. This is the strongest un-mined support for the central claim.
- Put items 6–7 into the negative-results discussion as the mechanism for the personalisation null (after one matched rerun).
- Add a rest-half row and a baseline-normalisation row as sensitivity analyses (items 12–13), and state the position-matching caveat (item 14).
- Fix C1–C8 before the advisor sees the draft. C2, C3 and C8 are the ones a reviewer would catch.
