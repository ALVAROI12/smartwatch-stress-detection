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

## 6. Conceptual framing (summary of `what_wrist_detectors_detect.md`)

A wrist E4 measures pulse rate and a weak vagal proxy (wrist RMSSD r = 0.53 against chest ECG), wrist EDA (agreement with palmar EDA about r = 0.3), skin temperature and movement. None of these is stress. The field attaches three constructs to one label: stressor exposure (what every dataset actually labels), autonomic arousal (what the sensor sees and the model learns), and subjective stress (self-report, available per task only in PhysioNet and WESAD). Every anomaly in this repo follows from equating them: the low PhysioNet and Stress-Predict ceilings (seated, silent, weakly evaluative stressors produce a fifth of WESAD's EDA response), the mild-stress null (self-report is the wrong ruler for an arousal detector), the few-shot and domain-adaptation nulls (nothing to align once labels are right and features are z-scored per subject; personalisation cannot recover a response that did not happen), and the exercise false alarms.

Four theories were stated with refutation conditions and are now tested:

| Theory | Prediction | Result | Status |
|---|---|---|---|
| (a) Arousal-only: specificity comes only from context | Within-bin AUROC about 0.5 against any non-stress arousal state | 0.36–0.63 for all seated states; 0.89 for exercise | Holds for seated states; exercise is separable by cardiac signature |
| (b) Social-evaluative EDA signature | UBFC evaluated vs non-evaluated tasks separable at matched arousal, on EDA features | 0.58 [0.49, 0.66]; EDA-only index 0.59 | Not supported between subjects; wrist EDA may lack the resolution, so undecidable without a palm reference |
| (c) Detectability is a subject trait | Per-subject recall correlates across stressors; misses concentrated | ρ −0.14 to 0.44, CIs cross 0; worst 20% hold 37–61% of misses | Refuted on these data |
| (d) Error sits at the edges (onset, recovery) | Recall lowest in first minute; false alarms peak after offset | True (Stress-Predict 0.47 vs 0.77; PhysioNet rest 0.26 vs 0.06–0.09) but ≤0.02 of balanced accuracy | Real, small |

The honest reframing: a wristband detects a change in autonomic arousal reliably across people and protocols; whether that arousal is stress is an inference that needs context the wrist does not have. The proposed norm is that every stress detector publishes a specificity panel (false-stress under exercise, speech without evaluation, cognitive load without threat, strong emotion), the way an assay reports cross-reactivity.

## 7. Closest prior work per direction (from `docs/literature/contribution-gap-scan.md`)

Full-text quotes unless marked abstract-only. Verdicts are the scout's, checked against this repo's earlier novelty search.

| Direction | Closest work | What they did | Gap that remains |
|---|---|---|---|
| Specificity panel | Sosa et al. (2026, *Sensors*, review): "Elevated sympathetic activation… can therefore arise not only during stress, but also during non-stressful states such as excitement, engagement, or novelty detection." Aydoğan & Villagra Povina (2026): exercise only. Sahu et al. (2025, arXiv): cross-activity anxiety AUROC 0.82 within to 0.59–0.62 across activities, chest ECG. Liu et al. (2026, arXiv): 2 × 2 difficulty × time pressure, N = 29, 4-class LOSO accuracy 0.27. | One confounder at a time, or one study | No fixed subject-independent detector scored across a panel; no matched-arousal test. Stress-Predict's descriptor (Iqbal et al., 2022) tags hyperventilation as a "stress-inducing task" and describes its interviewers as "friendly and kind to the participant". |
| Warm-up / time since donning | Tognotti et al. (2026, *Front Digit Health*): test-inclusive baseline normalisation inflates BA by 3–13 points. Brandebusemeyer et al. (2026, *Sensors*): 5-min acclimatisation used, not analysed. Parry & Briganti (2026, medRxiv): passive hydration under the band raises apparent skin conductance. | Normalisation leakage; device validation | No wearable-stress paper uses time since donning as a covariate or checks the first baseline block for temperature or EDA drift. Vendor "10–15 min to equilibrium" claim untraceable to a primary source. |
| Benchmark release | Liu & Ning (2026, arXiv): 26 EDA datasets, LOSO and LODO on five, "The code will be made publicly available upon acceptance." Akkaya (2026, *BMC MIDM*, abstract only): three E4 datasets, leakage-controlled LOSO. Shahriar (2025, arXiv): three E4 datasets, "a unified benchmark", no split files. FEEL (Singh et al., 2026): 19 datasets, arousal/valence. | Pipelines, not artefacts | No released set of harmonised labels, frozen subject splits and protocol-stage anchors for wrist stress. |
| Counterbalanced speech × evaluation with E4 | EmoWork (2025, *Sci Data*, controlled access): baseline B2 "participants read a neutral script in a natural speaking tone", E4 + Polar H10, condition order randomised. EmpkinS (Richer et al., 2025, OSF): TSST vs friendly-TSST, ECG + ICG only. Toner et al. (2023, arXiv): E4, explicitly vs implicitly evaluative conversations, randomised order, availability unclear. EmpathicSchool (2025, *Sci Data*): "We could not mitigate [the order effect] by testing different task orders". | Partial designs | No public counterbalanced E4 dataset contrasting TSST with a friendly or placebo TSST plus exercise and non-evaluative cognitive load. |
| Selective classification / abstention | Farahani et al. (2026, arXiv): conformal-style gate on EDA–BVP–TEMP coupling, WESAD and Stress-Predict LOSO, false positives 29 to 27 and 94 to 92 (p = .16, .32); "an interpretable signal of unsupported physiology… rather than a stand-alone safety guarantee". Akkaya (2026): few-shot recalibration cuts Nurse ECE 0.22 to 0.07 retrospectively, not prospectively. | Within dataset, null gains | Nothing gated by PPG usability with a coverage–accuracy curve, nothing under leave-one-dataset-out. |
| Recovery-aware labels | Abdel-Ghaffar et al. (2025, arXiv, Fitbit, not public): arousal-event onset labels because events occur "well into the recovery period". Skat-Rørdam et al. (2025, arXiv): window-based F1 with temporal tolerance. Kyprakis et al. (2026): MIL over 3-month bags, oncology. | Event labels on private data; metrics | No positive-unlabelled or time-decayed label treatment of rest under subject-independent validation on protocol datasets. |
| Foundation models | PULSE (Zhao et al., 2025, arXiv): WESAD only, LOSO AUROC 0.989, 60 s windows at 0.25 s stride, pretraining fold hygiene not stated. Pulse-PPG (Saha et al., 2025): WESAD stress vs baseline+amusement, precision 0.87 / recall 0.89, split protocol not stated. NormWear (2024): WESAD AUROC 76 vs 66 for statistical features, one 80/20 subject-stratified split. UME EDA foundation model (Alchieri et al., 2026): "similar balanced accuracy to… handcrafted features". Akkaya (2026, abstract): CNN below gradient boosting on Hongn, AUROC about 0.59–0.60. | Saturated on WESAD | No foundation model scored on Hongn, Stress-Predict, UBFC-Phys or Campanella under LOSO or LODO. |
| Detectability as a trait | Yildiz & Subasi (2026, bioRxiv): WESAD S2, S3, S9 low responders, "a characteristic of the monitored population rather than a deficiency of any specific algorithm". Sah & Ghasemzadeh (2021): S14 55.9%. Veit et al. (1997): four-year stability of HR reactivity r = .76. | Single-dataset anecdotes | Cross-stressor test now done here and negative. |

Access notes from the scout: PubMed Central switched to a CAPTCHA mid-session, so four full texts came through the PubMed tool; Springer/BMC, MDPI and Empatica support pages were blocked, so Akkaya (2026) and the Stress-Predict Data paper are abstract-level.

## 8. Recommended paper structure for the JBHI revision

1. Introduction: stress detectors are reported as detecting stress; we ask what they detect. Cite Vos et al. (2023) and Schmidt et al. (2019) for validation practice, Kwon et al. (2026) and Prajod et al. (2024) for the transfer debate, Sosa et al. (2026) for the arousal problem.
2. Data and label audit: five E4 datasets, seven descriptor-level label fixes plus the hyperventilation tag, protocol-stage anchors, harmonisation table. Cite Aydoğan & Villagra Povina (2026) as the independent PhysioNet audit and Mishra et al. (2020) for recovery exclusion.
3. Methods: wrist HR/HRV validated against chest ECG (methods check), per-subject z-scoring stated as transductive with raw and causal sensitivity, 20 subject-grouped splits plus LOSO, Nadeau–Bengio tests with Holm.
4. Result 1, transfer: leave-one-dataset-out cost 0.015–0.05 balanced accuracy; the missing-temperature threshold shift; label fixes as suggestive, not causal; Kwon setting reproduced.
5. Result 2, what transfers: specificity panel and matched-arousal test (Section 1). Exercise separable, seated states not.
6. Result 3, the ceiling: time-only classifier and warm-up (Section 2), error budget (Section 3), chest-ECG bound, negative results for few-shot, domain adaptation, thresholds, self-rated intensity and detectability trait as leakage-controlled nulls.
7. Discussion: arousal transfers, stress does not; specificity panel as a reporting norm; position-matched evaluation and a settling period as protocol rules; what a counterbalanced study must contain; limits (wrist EDA resolution, between-subject UBFC contrast, small hyperventilation cell).
8. Release: audited labels, frozen splits, probe scripts.

## 9. Hardening (2026-09-22, evening): what changed after fixes 1, 2, 4 and 5

Scripts `scripts/matched_arousal_hardening.py` and `scripts/warmup_since_donning.py`; tables in `outputs/tables/jbhi_v2/contribution_probes/hardening/` and `.../wesad_warmup/`.

- **Inference for the matched-arousal test (fix 1).** Subject-clustered permutation test against 0.5 with Holm correction: only exercise (0.89, p_holm 0.003), fear clips once in training (0.78, p_holm 0.003) and the baseline/rest reference are separable at matched arousal. UBFC control, hyperventilation, Lego and anger are not distinguishable from chance (p_holm 0.13–0.94). The earlier wording "Lego scores higher than stress" is dropped; Lego's raw p = 0.03 does not survive Holm. The exercise-minus-pair differences all exclude zero (lower bounds 0.18–0.36).
- **Feature-disjoint index (fix 2).** With the two index features and `eda_mean` removed from the model, the four non-separable pairs stay at 0.41–0.52; exercise 0.88, fear 0.78. Conclusion unchanged. Fear's residual sits in the cardiac features (0.79 with an EDA-only index and a cardiac-only model) as much as in EDA (0.71), so Section 1 should say "seated non-evaluative tasks and hyperventilation", not "every seated state".
- **WESAD warm-up since recording start (fix 4).** WESAD's synchronised protocol starts 16–26 min after the raw E4 recording (median 21, found by exact match of the EDA stream), so its baseline begins 17–49 min after switch-on (median 33) and no baseline window lies within 15 min. The warm-up itself is largest on WESAD (+1.8 °C pooled between minutes 1–5 and 30–40; +0.3 to +6.5 °C per subject). Baseline overlaps warm-up on Campanella (100% within 10 min of recording start), Stress-Predict (97%) and PhysioNet (68% within 10 min, 90% within 15), not on WESAD. Section 2's "100% PhysioNet" was measured from the first protocol window, not from recording start; use 68/90. WESAD is also the one dataset where the time-only classifier does not beat physiology, which is consistent.
- **Responder filter (fix 5).** 0.5 SD and 1.0 SD on either channel select the same subjects and gain 0.01–0.02. 1.0 SD on both channels (17 of 35 PhysioNet subjects) gains +0.094 [0.05, 0.15] within and +0.065 [0.01, 0.13] externally on PhysioNet, nothing on Stress-Predict (−0.02 external). Section 3 should read: a strict responder filter recovers up to 0.09 on PhysioNet and nothing on Stress-Predict; the residual stays 0.13–0.19 and 0.23–0.31.
- **Manuscript.** `paper/main.tex` (IEEE JBHI format, figures from `paper/make_paper_figures.py`) incorporates all four fixes.
