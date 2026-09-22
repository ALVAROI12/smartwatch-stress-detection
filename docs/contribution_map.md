# Where to contribute: a map of open ground for wrist stress detection

Written 2026-09-22 from commit `590f671` plus three new probes run the same day. Five agents worked in parallel: a conceptual critique (`docs/what_wrist_detectors_detect.md`), a literature gap scan (`docs/literature/contribution-gap-scan.md`), and three data probes whose scripts are in `scripts/` and whose tables are in `outputs/tables/jbhi_v2/contribution_probes/`. All probes reuse the leave-one-dataset-out defaults: whole-session per-subject z-scoring, HR/HRV/EDA features, XGBoost, threshold 0.5, 20 subject-grouped splits, subject bootstrap for confidence intervals. Nothing here changes a committed result; each probe reproduces the committed numbers it overlaps with (external PhysioNet exercise false-stress 0.69, TMCT within recall 0.63).

## 1. The finding that reframes the paper

**Once autonomic arousal is matched, the wrist model cannot tell stress from non-evaluative tasks.** The model can tell stress from exercise, and only from exercise.

`scripts/matched_arousal_probe.py`, Part A. A label-free arousal index (mean of z-scored `hr_mean` and `eda_tonic_mean`) is computed per window. Stress windows are pooled with each non-stress arousal class, binned into arousal quintiles, and the model's stress score is scored by AUROC inside each bin. An AUROC near 0.5 inside bins means the model has learned nothing beyond arousal magnitude for that pair.

| Stress versus | Windows | Pooled AUROC | Within-bin AUROC [95% CI], external model | Class coefficient after arousal control |
|---|---|---|---|---|
| Campanella Lego manual task | 667 | 0.754 | **0.355 [0.25, 0.54]** (Lego scores higher) | −0.73 [−1.05, −0.49] |
| UBFC-Phys same tasks without evaluation | 80 | 0.587 | 0.577 [0.49, 0.66] | 0.27 [0.03, 0.52] |
| Stress-Predict hyperventilation | 53 | 0.551 | 0.523 [0.43, 0.62] | 0.12 [−0.30, 0.59] |
| EPM-E4 anger clips | 33 | 0.643 | 0.390 [0.32, 0.52] | −0.43 [−0.87, 0.01] |
| EPM-E4 fear clips | 394 | 0.716 | 0.633 [0.58, 0.69] | 0.31 [0.01, 0.60] |
| Baseline and rest (reference) | 3839 | 0.770 | 0.584 [0.54, 0.64] | 0.53 [0.32, 0.73] |
| PhysioNet exercise, model shown exercise | 3763 | 0.947 | **0.892 [0.84, 0.93]** | 5.0 [4.3, 5.6] |

Exercise is the exception because its cardiac signature differs (heart rate up without the EDA pattern); the HR-only index carries that residual. Every seated, non-evaluative arousal state is indistinguishable from stress at matched arousal.

The same picture from the raw false-stress rates (`scripts/specificity_panel_probe.py`, Part A). Adding a class to the training negatives collapses exercise false alarms (0.69 to 0.08–0.10) but does not help the others:

| Probed class | Subjects / windows | External model | Class added as training negative |
|---|---|---|---|
| UBFC-Phys control group, speech and arithmetic without evaluation | 8 / 80 | 0.55 [0.46, 0.66] | 0.56 [0.45, 0.67] |
| Stress-Predict hyperventilation | 31 / 53 | 0.42 [0.26, 0.59] | 0.55 [0.36, 0.72] |
| Campanella Lego manual tasks | 29 / 667 | 0.43 [0.38, 0.48] | 0.33 [0.27, 0.39] |
| EPM-E4 anger / fear / sadness / happiness | 33 each | 0.42 / 0.35 / 0.37 / 0.24 | 0.35 / 0.25 / 0.30 / 0.18 |
| PhysioNet aerobic / anaerobic | 30–31 / 3763 | 0.69 / 0.70 | 0.10 / 0.08 |

Reference false-stress on in-training non-stress classes is 0.06–0.29; stress recall is unchanged or higher when a probed class is added (exercise negatives raise Stress-Predict recall 0.63 to 0.70 and UBFC-Phys 0.74 to 0.79).

The literature scan found this untested: reviews name the arousal-versus-stress problem (Sosa et al., 2026, *Sensors*), single confounders have been scored one at a time (exercise: Aydoğan & Villagra Povina, 2026; Kwon et al., 2026), but no paper scores one fixed subject-independent detector across a panel of non-stress arousal blocks, and no paper does the matched-arousal test. The UBFC-Phys contrast (same tasks, evaluation removed) is the cleanest version and is between-subject; a within-subject version needs new data (Section 4).

Two caveats. The arousal index is built from two of the model's own features, so within-bin AUROC measures what HRV and SCR features add beyond level. The hyperventilation and UBFC-Phys cells are small (53 and 80 windows); Lego and the EPM-E4 clips are the statistical backbone.

## 2. Time in session is a stronger predictor than physiology on the two hard datasets

`scripts/time_in_session_probe.py`. Minutes since the first window of the stress-protocol recording, alone, as the only feature:

| Dataset | Time-only within | Physiology within | Physiology external | Physiology + time, external |
|---|---|---|---|---|
| Campanella | 1.000 | 0.995 | 0.900 | 0.978 |
| PhysioNet | **0.826** | 0.776 | 0.742 | 0.517 |
| Stress-Predict | **0.947** | 0.698 | 0.656 | 0.560 |
| WESAD | 0.900 | 0.916 | 0.896 | 0.452 |

Protocol position beats wrist physiology within dataset on PhysioNet (16 of 20 splits) and Stress-Predict (20 of 20), and a model given time collapses externally because timing does not transfer. Any within-dataset pipeline that leaks time (raw temperature, drift features, no per-subject scaling) inherits this. The committed physiology result survives a position-matched test (non-stress test windows restricted to the minute range of that dataset's stress windows): within-dataset changes are at most −0.03 and external at most −0.04, so the small transfer cost is not riding on time.

The warm-up half (`warmup_drift.csv`): baseline is recorded inside the first 10 minutes in 100% of Campanella and PhysioNet windows and 98% of Stress-Predict, while skin temperature is still rising (0.02–0.07 °C/min) and tonic EDA is still rising in three datasets. EPM-E4, whose clock starts at donning, shows a 7 °C temperature swing over 30 minutes. The literature scan found no wearable-stress paper that treats time since donning as a covariate; the vendor "10–15 minutes to equilibrium" claim could not be traced to a primary source. This is the quantitative form of the order confound (novelty search item 3b) and it also explains why temperature features track time and should be dropped.

## 3. The ceiling is the stressor, not the pipeline

`scripts/matched_arousal_probe.py`, Part B: test-side filters on out-of-sample scores, no retraining, Shapley-averaged over the 24 filter orders. F1 drops rest windows within 5 minutes of a stressor offset; F2 keeps windows with cardiac coverage ≥ 0.5; F3 drops the first 60 s of each stressor; F4 keeps only physiological responders (label-using, diagnostic only).

| Dataset, model | BA all | F1 recovery | F2 sensor | F3 onset | F4 responder | All four | Residual error |
|---|---|---|---|---|---|---|---|
| PhysioNet within | 0.776 | +0.002 | −0.001 | +0.016 | +0.019 | 0.825 | 0.175 |
| PhysioNet external | 0.746 | +0.014 | +0.003 | +0.036 | +0.008 | 0.810 | 0.190 |
| Stress-Predict within | 0.710 | +0.016 | +0.012 | +0.018 | +0.011 | 0.771 | 0.229 |
| Stress-Predict external | 0.653 | +0.021 | −0.003 | +0.016 | +0.005 | 0.671 | 0.329 |
| WESAD external | 0.887 | 0 | +0.046 | +0.002 | 0 | 0.934 | 0.066 |

Recovery contamination, sensor loss, onset latency and non-response together buy 0.05–0.06 on PhysioNet and Stress-Predict, leaving 0.17–0.23 error that none of them explains. On WESAD the only sizeable term is the sensor (+0.046 external), and the chest-ECG ceiling probe (`wesad_ecg_ceiling_summary.csv`) agrees: replacing wrist cardiac features with chest ECG raises WESAD from 0.916 to 0.956, so motion-robust wrist HR is worth at most about +0.04 there, and EDA alone already gives 0.89.

Two further negatives close off directions that the advisor sheet left open. Subject-level detectability is not a trait: per-subject stress recall does not correlate across stressors (Stress-Predict Stroop vs TSST ρ −0.14 [−0.51, 0.24]; PhysioNet Stroop vs TMCT 0.44 [−0.04, 0.78]), is not predicted by baseline HR, RMSSD, EDA, cardiac coverage or self-reported stress rise, and the worst 20% of subjects hold only 37–61% of misses (`partB_miss_consistency.csv`). Stress-Predict Stroop is nearly undetectable externally (recall 0.24), which is a task effect. Onset latency is real (Stress-Predict recall 0.47 in the first minute versus 0.77 after three; PhysioNet 0.63 versus 0.77) and recovery false alarms concentrate in the first minute after offset (PhysioNet within 0.26 versus 0.06–0.09 later), but too few windows sit there to move balanced accuracy.

Together with the effect-size table (PhysioNet EDA +0.4 z versus WESAD +2.2 z) and the literature scan's finding that Stress-Predict's interviewers "were friendly and kind to the participant" (Iqbal et al., 2022), the residual is best read as stressor potency: seated, silent, weakly evaluative stressors produce less of what the wrist can see. Raw-signal representation learning is not a documented escape: the PULSE paper (Zhao et al., 2025) is WESAD-only, where trees already reach AUROC 0.99, and the one deep model scored on the Hongn PhysioNet dataset under leave-one-subject-out (Akkaya, 2026) reaches AUROC 0.59–0.60, below gradient boosting.

## 4. Ranked directions

Scored after the probes. "Evidence" is what this repo already holds; "cost" is student-months.

| Rank | Direction | Novelty (from gap scan) | Evidence in hand | Cost | Verdict |
|---|---|---|---|---|---|
| 1 | **Specificity panel plus matched-arousal test as the paper's thesis and as a proposed reporting standard** | Unexplored as a panel; reviews call for it | Done (Sections 1–3) | Weeks (write-up) | Make this the JBHI revision's central claim. Title candidates in `what_wrist_detectors_detect.md` §5. |
| 2 | **Verified five-dataset benchmark release**: audited protocol-stage labels, frozen 20-split subject groups, specificity panel, leakage tests, loader | Unexplored; nearest is Liu & Ning (2026, EDA-only, code pending) | All artefacts exist | 1–2 months | Companion resource paper (Scientific Data or NeurIPS Datasets & Benchmarks). Release label and split files plus code, not raw data. Highest citations per effort. |
| 3 | **Warm-up and time-in-session as a covariate** | Unexplored | Done (Section 2) | Weeks | One section of the JBHI paper, plus a recommended minimum settling time and a position-matched evaluation rule. |
| 4 | **Counterbalanced UTSA lab study**: 2 × 2 speech × evaluation, matched-HR cycling, n-back without feedback, Latin-square order, baseline at start and end, E4 plus chest ECG plus palm EDA reference, two visits | Unexplored with E4; EmoWork (2025) has a speech-without-workload baseline and EmpkinS has TSST vs friendly-TSST but neither has both with E4 | None | 12–18 months, IRB, about 40 participants | The pioneering move. Only a within-subject design can say whether wrist EDA carries an evaluative-threat signature that Section 1 could not find between subjects. Becomes the external test set nobody has. |
| 5 | Open-set or selective classification: stress vs known non-stress vs "unrecognised arousal", abstention gated by PPG usability, coverage–accuracy curves under leave-one-dataset-out | Partly explored (Farahani et al., 2026, within-dataset, null gains) | Scores and coverage per window exist | 1–2 months | The deployment form of rank 1. Honest, cheap, one table plus a figure. |
| 6 | Recovery-aware or time-decayed labels (positive-unlabelled treatment of rest) | Unexplored under subject-independent validation | Second-half-rest result, offset curves | 2–3 months | Section 3 says the payoff is at most +0.02 on balanced accuracy. Worth a sensitivity analysis, not a method paper. |
| 7 | Motion-robust wrist HR during speech | Mature signal-processing literature | ECG ground truth in WESAD | 3–4 months | Capped at about +0.04 on WESAD. Engineering chapter at most. |
| 8 | Foundation models or self-supervised encoders under leave-one-dataset-out on the hard datasets | Unexplored on Hongn, Stress-Predict, UBFC-Phys, Campanella | None | 2–3 months plus GPU | Run one row with a public frozen PPG encoder (PaPaGei, Pulse-PPG) as a feature extractor. A negative result strengthens Section 3; a positive one overturns it. Do it as a check, not a bet. |
| 9 | Subject-level detectability trait | Partly explored (WESAD anecdotes) | Done, negative | – | Closed. Report the null in one paragraph. |
| 10 | Few-shot personalisation, domain adaptation, per-user thresholds, self-rated mild stress | – | Done, negative | – | Closed. |

## 5. What to do this week

1. Rewrite the JBHI contribution list around Sections 1–3: (i) small cross-dataset transfer cost once labels and normalisation are right; (ii) the detector transfers because it detects arousal, shown by the specificity panel and the matched-arousal test; (iii) the within-dataset ceiling is stressor potency, shown by the error budget; (iv) time in session and warm-up as a measured confound with a position-matched evaluation rule; (v) the audited benchmark. Drop "method contribution" as a goal for this paper; the methods tested (few-shot, DA, thresholds) are reported as leakage-controlled negatives.
2. Add the Stress-Predict hyperventilation relabelling to the label audit: the descriptor paper tags it as a "stress-inducing task"; this pipeline already keeps it separate, and Section 1 shows why that matters.
3. Correct one premise in the advisor sheet: PULSE did not report 0.965 on PhysioNet; it is WESAD-only. The "raw-signal representation learning" option should be presented as an untested check, not a documented path to the ceiling.
4. Decide with the advisor whether to write the benchmark paper in parallel and whether to start the IRB for the counterbalanced study.

## Reproduction

```
cd <repo>
.venv/bin/python scripts/specificity_panel_probe.py     # about 1 min
.venv/bin/python scripts/time_in_session_probe.py       # about 2 min
.venv/bin/python scripts/matched_arousal_probe.py       # about 4 min
```

Set `HARMONIZED_CSV` if the feature table is not at the thesis-worktree path. The 17 MB per-window score file from the specificity probe is regenerated, not committed.
