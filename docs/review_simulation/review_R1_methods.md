# Peer Review Report

## Manuscript Information
- **Title**: What Do Wrist-Worn Stress Detectors Detect? Arousal Transfers Across Five Empatica E4 Datasets, Evaluative Stress Does Not
- **Manuscript ID**: n/a (simulated IEEE JBHI panel)
- **Review Date**: 2026-09-23
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 1 (Methodology)

### Reviewer Identity
Biostatistician and ML-evaluation methodologist: subject-level cross-validation, cross-dataset generalisation, resampling inference and multiple-comparison control.

### Review Focus
Whether the evaluation design (subject-grouped splits, LODO, Nadeau–Bengio tests, Holm families, subject bootstraps) supports the headline numbers, and whether the two new "reporting standards" (specificity panel, matched-arousal test) and the Shapley error budget are valid for the causal-sounding conclusions drawn from them. I checked the manuscript against the committed scripts (`leave_one_dataset_out.py`, `specificity_panel_probe.py`, `matched_arousal_probe.py`, `matched_arousal_hardening.py`, `run_jbhi_experiments.py`, `compare_models.py`) and tables under `outputs/tables/jbhi_v2/`. I also recomputed a few quantities from the committed per-split and window tables (stated where used).

---

## Overall Assessment

### Recommendation
- [x] **Major Revision**

### Confidence Score
4. Core expertise for the evaluation and inference questions. I have less expertise in E4 signal processing, and I did not rerun the model-training pipelines.

### Calibration Status
`NOT_CALIBRATED`

### Summary Assessment
The paper evaluates an XGBoost HR/HRV/EDA stress detector on five E4 datasets with audited labels, subject-grouped splits and leave-one-dataset-out (LODO) testing. It reports a small transfer cost (0.015–0.05 BA). It then argues from a specificity panel, a matched-arousal test, a time-only classifier and a Shapley error budget that the detector measures arousal, not evaluative stress. The evaluation hygiene is well above the field norm. It holds out whole subjects, fixes hyperparameters, runs nested tuning, uses corrected resampled tests and subject bootstraps, reports the leakage demonstration, and includes feature-disjoint and responder-rule sensitivity analyses. The inferential chain behind the title claim is weaker than the prose suggests:
1. The central claims are nulls ("cannot separate", "none different from chance after Holm"), supported by non-significance. There are no equivalence margins or power analysis, and Holm correction makes such nulls easier to obtain.
2. The matched-arousal test barely separates the contrast the model was trained on (stress vs baseline/rest: within-bin AUROC 0.58). So it has little ability to tell "arousal-only" from "stress mostly expressed as arousal".
3. Per-subject whole-session z-scoring largely removes between-subject differences in response magnitude. That matters for the only evaluative contrast (UBFC control, 8 subjects, between-subject).
4. The "added negative" result tracks the added class's window count.
5. Cardiac missingness differs strongly by class, which affects both the model and the arousal index.
6. One robustness statement misreports the committed table.
7. The headline LODO costs have no uncertainty. Nadeau–Bengio-corrected intervals I computed from the committed per-split file reach 0.08–0.13.

All of these can be repaired by reanalysis and reframing, so I recommend major revision rather than rejection.

---

## Strengths

### S1: Subject-grouped evaluation throughout, with an explicit leakage demonstration
Every result uses subject-held-out splits. The paper quantifies the collapse from record-wise splitting (94% to 49% accuracy), and the three participants recorded in two files are merged so no person appears on both sides of a split. `grouped_split` holds out 20% of subjects within every dataset, and LODO scores the *same* held-out subjects under within, external and pooled training, so the comparisons are paired.
**Evidence Anchor**: text: §III "the same model scores 49\% (balanced accuracy 0.36) when 15\% of subjects are held out"

### S2: Fixed hyperparameters with a separate nested-tuning check
Tuning is not done on the evaluation splits (`fit_predict` comment: "tuning on these splits would leak test subjects"). A separate 108-configuration nested run (108 rows in `tuned_baselines_summary.csv`, consistent with the text) confirms the model choice does not drive conclusions.
**Evidence Anchor**: text: §IV-C "a nested-tuning comparison of XGBoost, random forest and a multilayer perceptron found one significant difference in 72 comparisons after Holm correction"

### S3: Diagnosis of a threshold shift as a missing-channel artefact
The UBFC-Phys result is well analysed. AUROC is kept (0.931) and oracle-threshold BA is 0.910, while BA collapses to 0.589 because temperature is missing and falls to XGBoost's default branch. This is a correct separation of ranking from calibration and a useful cautionary example for the field.
**Evidence Anchor**: table: Table II — UBFC-Phys, + temperature, Other 4 = 0.589

### S4: Sensitivity analyses that go beyond the usual
Raw and causal normalisation variants, feature-disjoint reruns of the matched-arousal test, responder-rule sensitivity (`partB_responder_sensitivity.csv`), position-matched negatives and a chest-ECG ceiling are all reported, some with results that cut against the thesis (Lego becomes separable under one variant; normalisation loses 0.10 on short recordings).
**Evidence Anchor**: text: §V-D "the test was repeated with those features (and \texttt{eda\_mean}) removed from the model"

### S5: Subject-level uncertainty for window-level rates
Probe false-stress rates and matched-arousal AUROCs carry subject-cluster bootstrap CIs rather than window-level binomial CIs, which respects the 50%-overlapping, within-subject-correlated windows.
**Evidence Anchor**: table: Table III caption — "Subject-Bootstrap 95\% CI"

---

## Weaknesses

### W1: Central claims are inferred from non-significance; Holm correction is used in the direction that favours the null
**Problem**: The abstract, §V-D and the Conclusion state that at equal arousal the detector "cannot separate" stress from non-evaluated tasks, hyperventilation, Lego or anger, because $p_{holm}$ is 0.13–0.62. That is absence of evidence. The intervals do not exclude meaningful separability: UBFC control within-bin AUROC 0.577 [0.50, 0.67], hyperventilation 0.523 [0.44, 0.61], anger 0.39 [0.31, 0.52], Lego 0.355 [0.24, 0.54]. Holm correction controls false *positives*. Applied to a claim of *no* effect, it inflates p-values and makes the desired null easier to reach. Uncorrected, Lego is already significant (p_raw = 0.0315, `partA_inference.csv`), in the reverse direction.
**Evidence Anchor**: text: Abstract "AUROC 0.36--0.58, none different from chance after Holm correction"
**Why it matters**: The title claim ("Evaluative Stress Does Not [transfer]") and contribution (2) rest on these nulls. As stated, they are not supported.
**Suggestion**: Pre-specify an equivalence margin (for example |AUROC − 0.5| < 0.10, or relative to the reference contrast, see W2). Report TOST or CI-inclusion equivalence decisions per pair, with uncorrected p-values alongside Holm. Rephrase every "cannot separate" as "no separation was detected; the data are compatible with AUROC up to X". State for each pair the smallest effect the design could have detected (see W7).
**Severity**: Critical (repairable by reanalysis and rewording; the empirical estimates themselves may survive)
**Confidence**: 5 — core expertise: hypothesis testing and multiplicity

### W2: The matched-arousal test has low sensitivity even for the trained contrast, and there is no positive control at a realistic effect size
**Problem**: On the reference pair (stress vs baseline/rest, which the model was *trained* to separate), within-bin AUROC is only 0.584 [0.54, 0.64]. The probe pairs (0.52–0.58) sit inside that range. Conditioning on an arousal index built from the physiological channels removes most of whatever the model uses, *including* for the contrast it detects well. The test therefore cannot distinguish "the detector learned arousal only" from "stress is expressed physiologically mostly as arousal, which is all a wrist can see". Those are different conclusions for the paper's thesis. Exercise (0.89) is not a useful positive control: it differs by a gross cross-channel pattern (high HR, movement), not by a subtle stress-specific signature.
**Evidence Anchor**: table: Table IV — "Baseline and rest (reference) & 3839 & 0.770 & 0.584 [0.54, 0.64]"
**Why it matters**: Without a calibrated sensitivity, a within-bin AUROC near 0.5 is uninformative about stress specificity. The Discussion sentence "What the detector has learned is arousal magnitude" is an interpretation the test cannot support in isolation.
**Suggestion**:
1. Report every probe AUROC as a *difference from the reference contrast*, with a joint subject bootstrap.
2. Add a synthetic positive control: inject a known non-arousal feature shift of graded size into stress windows and show the test's detection curve.
3. Reframe the finding as "at matched arousal, stress is no more separable from X than from rest". That is defensible and still interesting.

**Severity**: Major
**Confidence**: 4 — core expertise: test design and validity; the adjacent physiology is a judgement call

### W3: Whole-session per-subject z-scoring removes between-subject response magnitude, which undermines the UBFC control contrast and cross-dataset effect sizes
**Problem**: `normalise(..., "session")` standardises each subject over all windows of that subject's protocol recording. Consider a subject whose session is one rest block plus two task blocks (UBFC-Phys: T1, T2, T3), where within-block noise is small relative to the block shift. Task windows then receive z ≈ +0.7 and rest ≈ −1.4 *whatever the size of the shift*. A test-group subject with a large evaluative response and a control-group subject with a small one are mapped to similar z-profiles. The "UBFC control called stress at 0.55" result and the null matched-arousal result for that pair are therefore partly expected from the normalisation alone. The same mechanism affects:
- the probe-heavy Campanella sessions (667 Lego vs 102 stress windows, so Lego sets the subject mean);
- EPM-E4, whose session contains only emotion clips with no rest, so "arousal" is relative to other clips;
- the cross-dataset effect sizes used to argue "stressor potency" ("EDA rises by 0.4 z in PhysioNet against 2.2 z in WESAD"), since session composition differs across datasets (PhysioNet 1193 rest vs 232 stress windows).

**Evidence Anchor**: text: §IV-B "Features are z-scored per subject using all windows of that subject's stress-protocol recording, without labels"
**Why it matters**: The only evaluative-vs-non-evaluative contrast in the paper is between subjects. For that contrast, the normalisation is the thing most likely to erase the effect the paper says is absent.
**Suggestion**: Rerun the specificity panel and matched-arousal test for UBFC control, Lego and EPM under baseline-referenced scaling (already implemented: `--normalisation baseline`) and raw features. Report the between-group difference in raw stressor-minus-rest HR and EDA for UBFC test vs control. Express cross-dataset effect sizes as within-subject standardised mean differences relative to baseline SD, not session z.
**Severity**: Major
**Confidence**: 4 — core expertise in standardisation artefacts; the size of the effect here is not yet measured

### W4: "Adding the state to the negatives does not help" is confounded with the added class's window count
**Problem**: In condition 3, the probed class is appended to the training negatives. `fit_predict` reweights only stress vs non-stress in aggregate, so each added class carries weight in proportion to its window count. The added class sizes are exercise 3763, Lego 667, fear 394, UBFC control 80 (about 6 training subjects per split), hyperventilation 53 and anger 33. The classes that "collapse" or become separable (exercise; fear in the matched test) are the large ones. The classes that "do not change" are those that make up ≤1% of the negatives. The paper reads this as a property of the states ("No other state behaves this way").
**Evidence Anchor**: table: Table III — "UBFC-Phys speech + arithmetic, no evaluation (8 / 80) & 0.55 [0.46, 0.66] & 0.56 [0.45, 0.67]"
**Why it matters**: This is the evidence for contribution (2) and for the claim that exercise is special. With this design, the result may say more about sample size than about physiology.
**Suggestion**: Run two checks. (a) Subsample exercise to 33–80 windows from 6–8 subjects and show whether exercise false-stress still collapses. (b) Upweight each probed class to equal total weight with the core negatives. Report the rate against added-class size.
**Severity**: Major
**Confidence**: 5 — core expertise: class weighting and design confounds

### W5: Cardiac missingness differs by class, which affects both the classifier and the arousal index
**Problem**: I tabulated `hr_mean` missingness by class from the feature table:
- WESAD: stress 63%, baseline 10%.
- UBFC-Phys: stress 39%, control 21%, baseline 11%.
- Campanella: Lego 70%, stress 44%.
- PhysioNet: exercise 26–28%, stress 9%.
- EPM-E4: 2–5%.

This has two consequences. (a) XGBoost's missing-value branch can use "no usable pulse" (that is, wrist motion during speech) as a stress cue. The paper documents this mechanism itself for temperature. (b) The arousal index is defined as the (HR+EDA)/2 mean where HR exists and EDA alone otherwise. So stress and probe windows are binned on *different index definitions* in different proportions, which is differential measurement within bins. The reversed Lego AUROC (0.355) and the reversed anger AUROC (0.39) may reflect this.
**Evidence Anchor**: text: §IV-D "mean of per-subject z-scored heart rate and tonic EDA, or tonic EDA alone where heart rate is missing"
**Why it matters**: Matching on an index whose definition depends on a class-correlated missingness pattern does not produce matched arousal, and a model that exploits missingness is not an "arousal detector".
**Suggestion**: Report missingness by class. Run the matched-arousal test (i) on the complete-case subset with the full index, and (ii) with an index that uses EDA only for every window, stratified by missingness. Refit the detector with cardiac missingness neutralised (imputation plus a missingness indicator that is then ablated) and report the change in probe false-stress rates.
**Severity**: Major
**Confidence**: 4 — core expertise in missing-data mechanisms; figures computed by me from the committed feature table

### W6: The robustness text misreports the feature-disjoint variants
**Problem**: The text says that the two further variants "give the same pattern, except that Lego becomes separable (0.73–0.84)". In `partA_disjoint.csv` (EDA-only index, cardiac-only model):
- anger with the probed class added as a negative is 0.743, $p_{holm}$ = 0.003, so it is separable;
- the difference between fear and exercise is −0.005 [−0.11, 0.12] (external) and −0.075 [−0.18, 0.05] (cond3), so exercise is no longer distinguishable from fear;
- external exercise is 0.287, $p_{holm}$ = 0.0035, so it is significantly separable *in reverse*.

Under the HR-only index, external exercise is only 0.62 ($p_{holm}$ = 0.054).
**Evidence Anchor**: text: §V-D "give the same pattern, except that Lego becomes separable (0.73--0.84) under the first variant"
**Why it matters**: The claim that exercise and fear are the only separable states depends on the index and model variant. Readers are told otherwise.
**Suggestion**: Report all variants in a supplementary table with raw and Holm p-values. Correct the sentence. Discuss why an EDA-only index with a cardiac-only model makes anger separable and puts exercise below chance.
**Severity**: Major
**Confidence**: 5 — read directly from the committed table

### W7: Small cells, few clusters, and CIs of uncertain coverage
**Problem**: The probe cells are small:
- UBFC control: 8 subjects in a between-subject contrast with 11 stress subjects.
- Hyperventilation: 53 windows from 31 subjects (1.7 per subject; subject-median false-stress rate 0.0 against a pooled 0.415, so a few subjects drive the rate).
- Anger: 33 windows, one per subject.

Percentile subject-bootstrap CIs with 8–11 clusters are known to under-cover. The permutation test for UBFC control permutes class labels across only about 19 subject cells per bin, so its resolution and power are limited. No power or minimum-detectable-effect statement is given, yet these are the cells that carry the evaluative claim.
**Evidence Anchor**: text: §V-C "The hyperventilation (1.7 windows per subject) and UBFC-Phys control (8 subjects) cells are small; Lego and the fear clips carry the statistics"
**Why it matters**: The key evaluative contrast is statistically uninformative at the effect sizes that would matter.
**Suggestion**: Report the minimum detectable within-bin AUROC at 80% power for each pair (by simulation under the observed cluster structure). Use a subject-level mixed model (GLMM with subject random intercept) or wild cluster bootstrap for rates. For anger, report the per-subject binary outcome with an exact CI. Demote UBFC control and hyperventilation to exploratory in the abstract.
**Severity**: Major
**Confidence**: 5 — core expertise: cluster inference

### W8: Headline LODO transfer costs carry no uncertainty; corrected intervals are wide
**Problem**: Table II and the abstract give point costs of 0.015–0.05 BA with no CI or test. The text says they are "within the error expected", but no figure is given. From `leave_one_dataset_out_per_split.csv` (HR/HRV/EDA, within minus other-4, 20 paired splits, Nadeau–Bengio variance factor 1/20 + 0.25 as in `compare_models.py`) I obtain the following 95% CIs:

| Target | Cost | 95% CI | p |
|---|---|---|---|
| WESAD | 0.032 | [−0.013, 0.076] | |
| PhysioNet | 0.030 | [−0.036, 0.096] | |
| Stress-Predict | 0.052 | [0.003, 0.101] | 0.039 (uncorrected) |
| UBFC-Phys | 0.039 | [−0.053, 0.132] | |
| Campanella | 0.015 | [−0.076, 0.106] | |

The data are compatible with costs up to 0.08–0.13, the size of the literature drops the paper argues against.

Two further points:
- The external model's training set is identical across all 20 seeds (all subjects of the other four datasets). The 0.25 test/train ratio in the NB factor was derived for resampled train/test splits and has no clear meaning for the external arm. A per-subject paired analysis (subject-level BA or AUROC difference, clustered bootstrap) is the cleaner design here.
- The overall 95% CIs in §V-A ("0.83 (95% CI 0.81–0.85)") come from `summarise`, a naive t-interval over 20 overlapping resamples, which is too narrow.

**Evidence Anchor**: table: Table II — Within vs Other 4 columns, no CI or test reported
**Why it matters**: "Small transfer cost" is contribution (1). It currently rests on point estimates whose corrected intervals include the effects the paper disputes.
**Suggestion**: Report per-target costs with corrected or subject-cluster CIs. Pre-state an equivalence margin (for example 0.05 BA) and give TOST results. Replace naive split CIs with NB-corrected intervals or state that they are uncorrected. The paper's own sentence "we claim no large cost, not equality" should be backed by these intervals.
**Severity**: Major
**Confidence**: 5 — core expertise; recomputed from the committed table

### W9: Error budget: "at most 0.06" is a point estimate, and "stressor potency" is inferred by elimination
**Problem**:
- The Shapley values in Table V have no CIs. Only single-filter gains are bootstrapped, so "at most 0.06" is a point estimate, not an upper bound.
- Filters F1 and F3 remove windows of one class only. Their possible BA gain is capped by the fraction of windows removed, which the paper acknowledges ("too few windows sit there to move BA"). The budget therefore measures prevalence × error, not the size of the mechanism.
- F4 uses test labels and changes the evaluated population. Averaging a population-selection "gain" with window-level filters in one Shapley decomposition mixes estimands.
- The residual (0.17–0.33) is then attributed to "stressor potency". The only direct evidence is a dataset-level association across five datasets (EDA effect sizes vs ceilings). That is an ecological inference with n = 5, and the effect sizes are in session-z units (W3).

**Evidence Anchor**: text: Abstract "Wrist detectors transfer because they detect autonomic arousal; the ceiling is stressor potency"
**Why it matters**: The conclusion is phrased as causal attribution, but the analysis can only show that four named factors do not explain the residual.
**Suggestion**:
1. Bootstrap the Shapley values at the subject level.
2. Report each filter's removed fraction and its error rate before and after.
3. Keep F4 out of the Shapley decomposition and report it separately.
4. Test potency within dataset: relate per-subject physiological response magnitude (baseline-SD units) to per-subject recall with a mixed model across all five datasets.
5. Soften "the ceiling is stressor potency" to "is not explained by the tested pipeline factors".

**Severity**: Major
**Confidence**: 4 — core expertise: decomposition estimands

### W10: The position-matched rule does not remove the time confound it is proposed to control
**Problem**: The paper proposes position-matched negatives as an evaluation rule, and says the physiology model "survives" them. In `time_probe_summary.csv`, however, the time-only classifier *still* reaches BA 0.881 on Stress-Predict under position matching (physiology: 0.670), and 0.62 on PhysioNet. The rule could not be applied to Campanella or UBFC-Phys (no matched values in the table), and the text does not say so.
**Evidence Anchor**: text: §V-E "The physiology model survives position-matched negatives with changes of at most \m0.03 within and \m0.04 externally"
**Why it matters**: A proposed reporting rule that leaves time-only BA at 0.88 on the dataset where the confound is worst does not do what it claims. "The LODO result is not riding on time" also needs the time-only matched number beside it.
**Suggestion**: Report the time-only BA under matching for every dataset, and state where matching was impossible. Consider stricter matching (per-subject, within-minute strata, or a covariate-adjusted model with minutes-since-start as a stratifier). Either strengthen the rule or present it as necessary but not sufficient.
**Severity**: Major
**Confidence**: 5 — read directly from the committed table

### W11: Pooled-stress pairs mix scores from different models
**Problem**: In `pair_masks`, probe classes outside the five stress datasets (EPM anger and fear) and the baseline/rest reference are paired with stress windows pooled from all five datasets. Under the "external" score, each stress window is scored by its own leave-its-dataset-out model, while EPM windows are scored by the all-five model. Within-bin AUROC then compares scores from six models whose calibration differs (the paper reports mean external stress scores from 0.18 to 0.42 across targets). Stress and clip windows also come from different datasets, sessions and z-reference sets (W3).
**Evidence Anchor**: dataset: `matched_arousal_hardening.py` `pair_masks` — `sm = (y == 1) & ((ds_arr == pds) if pds in STRESS_DS else True)`
**Why it matters**: The anger and fear results, and the reference contrast, partly reflect differences in model calibration and dataset.
**Suggestion**: Score all windows of a cross-dataset pair with one model (for example the all-five model on held-out subjects, as in cond3), or rank-normalise scores within model. State that the EPM contrasts are between dataset and between subject.
**Severity**: Major
**Confidence**: 4 — read from code; size of impact unmeasured

### W12: A fixed 0.5 threshold is used for false-stress rates despite documented miscalibration across datasets
**Problem**: All specificity-panel rates are P(p > 0.5). The paper shows that external score levels shift between targets (UBFC case) and that an oracle threshold recovers 0.02–0.05 BA. Probe false-stress rates at 0.5 therefore mix specificity with calibration offset, and the "in-task non-stress 0.08–0.29" reference is on a different calibration for each dataset.
**Evidence Anchor**: text: §IV-C "The threshold is 0.5 throughout"
**Why it matters**: A comparison across the panel rows is partly a comparison of calibration.
**Suggestion**: Also report each probe's false-stress rate at the threshold that gives a fixed specificity (for example 90%) on that dataset's core negatives, and the threshold-free AUROC of stress vs probe (already computed as "Pooled" in Table IV).
**Severity**: Minor
**Confidence**: 4 — core expertise

### W13: The analysis history is post hoc and the test families are chosen by the analyst
**Problem**: The matched-arousal analysis was "hardened" after initial results (script docstring: "Hardening fixes 1, 2, 5"). Holm families are defined "within each model type" (about 7 tests). Three indices × three model variants × three score types were run, and the Discussion summarises a subset. The paper does not say which analyses were pre-planned.
**Evidence Anchor**: text: Table IV note "Holm-corrected within each model type"
**Why it matters**: The risk is selective emphasis rather than false positives, since the claims are mostly nulls.
**Suggestion**: Add a short analysis-history statement (primary: hr+eda index, external score; everything else sensitivity). Give the total number of matched-arousal tests run.
**Severity**: Minor
**Confidence**: 4 — core expertise

### W14: Reproducibility gaps
**Problem**:
- The feature table that every probe consumes (`data/processed/combined/harmonized_windows_v2.csv`) is not among the listed released artefacts. Only label tables, splits and scripts are named.
- Several probe scripts carry uncommitted modifications in the working tree.
- Table V's "All" BA values (for example Stress-Predict within 0.710, external 0.653) differ from Table II (0.697, 0.646) because the error budget uses LOSO and single-model external scores. This is not stated.
- §V-A names EPM-E4 in a baseline-vs-stress task to which it contributes no subjects (15 + 35 = 50).

**Evidence Anchor**: text: Data and Code Availability "The audited label tables, the 20 frozen subject-grouped splits, the feature-extraction and evaluation scripts"
**Why it matters**: Independent reproduction of the probe tables requires re-extracting features from five raw datasets.
**Suggestion**:
- Release the window-level feature table (derived data, as licences permit) and tag the exact commit.
- Note in each table caption whether it uses LOSO or 20-split scores.
- Correct the three-dataset description.

**Severity**: Minor
**Confidence**: 5 — checked repository state and tables

---

## Detailed Comments

### Research Questions & Hypotheses
The two questions (does the detector transfer; does it detect anything beyond arousal) are clear. The second is posed as a hypothesis whose confirmation is a null. It needs equivalence framing and a calibrated test (W1, W2).

### Research Design
LODO with paired within/external/pooled arms on the same held-out subjects is the right design. The specificity panel is a good idea, but condition 3 needs balanced weighting (W4). The matched-arousal test needs a positive control and stratification by missingness (W2, W5). The only evaluative contrast is between subjects, and normalisation weakens it (W3).

### Sampling Strategy
There are 132 subjects in the five stress datasets. The probe cells are very small (W7). There is no power analysis anywhere, although several conclusions are nulls.

### Data Collection
The label audit is careful. The wrist-vs-ECG validation (MAE 2.4 bpm, RMSSD r = 0.53) is appropriate. Differential PPG loss by class is documented for WESAD TSST but not carried into the analysis design (W5).

### Analysis Methods
- NB-corrected t-tests with Holm are appropriate for model and modality comparisons.
- For LODO external arms, the NB ratio is not meaningful (fixed training set). Use subject-level paired analyses (W8).
- The permutation scheme (labels permuted across subject×bin×class cells within bins, bins fixed) is exchangeable under a within-bin null and is a reasonable choice. Its resolution is low for the 8-subject UBFC cell.
- Weighting by bin size ignores bins with only one class, which changes the target population for exercise (matched bins cover only the lowest three quintiles). Report common support per pair.

### Results Presentation
- Table II needs CIs.
- Table IV should add a "vs reference" column and uncorrected p.
- The column labelled $n$ in Table IV is the probe window count, while the tests run on the pair (for example 904 for hyperventilation, including 851 stress windows). Label both.

### Reproducibility
Seeds are fixed, splits are frozen, and scripts reproduce the committed tables from a feature table that is not released (W14).

### Methodological Fallacies Detected
- Absence of evidence taken as evidence of absence (W1).
- Ecological inference from five dataset-level points to a "stressor potency" mechanism (W9).
- Confounding of a design factor (added-class size) with the construct (W4).
- Differential measurement of the matching variable (W5).

---

## Questions for Authors
1. For the UBFC-Phys test vs control groups, what is the between-group difference in raw (unnormalised) stressor-minus-T1 HR and EDA? Does the matched-arousal result for this pair survive baseline-referenced scaling?
2. If exercise is subsampled to the size of the UBFC control cell (about 80 windows, 6–8 training subjects), does its false-stress rate still collapse when it is added as a negative?
3. What fraction of each class's windows has a missing cardiac index? Does the Lego/anger reverse-direction AUROC persist in the complete-case subset?
4. What equivalence margin for "small transfer cost" would you defend a priori, and do the per-target corrected CIs fall inside it?

---

## Minor Issues

### Language / Grammar
- §V-E, warm-up paragraph: the sentence starting "subject-centred skin temperature rises at 0.02–0.08 °C/min in the first 10 min on all four datasets with a continuous clock keeps rising..." is missing a conjunction and is hard to parse.

### Figures and Tables
- Table III: give n windows per condition. Conditions 2 and 3 pool 20 splits in which each subject appears about 4 times, so the effective n differs from condition 1.
- Table IV: add bins-with-both-classes / common support for each pair.
- Fig. 4: a negative Shapley value drawn as zero hides information. Show it below the axis.

### Layout
- Abstract: "(69\% to 8\%)" merges aerobic and anaerobic figures (0.69–0.70 → 0.08–0.10). State the range.

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | quality_rubrics (methodology lens only) | NOT_ASSESSED | — | Outside methodology remit | — | no |
| Methodological Rigor | statistical_reporting_standards; methodology protocol Steps 2–4 | PARTLY_MEETS | table: Table II; text: §IV-D matched-arousal definition | Split hygiene and tests are strong. The null-inference framework, normalisation artefact, class-size confound and missingness-dependent index are not addressed | I did not rerun training; W3–W5 impact sizes are unmeasured | yes — W1–W5 must be resolved |
| Evidence Sufficiency | statistical_reporting_standards §CI/power | DOES_NOT_MEET | table: Table IV UBFC/hyperventilation rows; recomputed LODO CIs | The key evaluative contrast rests on 8 subjects, and headline costs lack CIs that would show them to be small | Could be repaired by equivalence analysis if the CIs turn out narrow | yes |
| Argument Coherence | methodology protocol Step 5 | PARTLY_MEETS | text: Abstract "the ceiling is stressor potency" | Conclusions go beyond what elimination-based analyses support. One robustness statement contradicts the committed table (W6) | — | yes |
| Writing Quality | — | NOT_ASSESSED | — | Outside remit | — | no |
| Literature Integration | — | NOT_ASSESSED | — | Reviewer 2 remit | — | no |
| Significance & Impact | — | NOT_ASSESSED | — | Reviewer 3 remit | — | no |

**Recommendation rationale**: Major Revision. The decision-bearing unresolved items are W1 (null inference; Critical but repairable), W2–W5 (validity of the matched-arousal test and the specificity panel), W6 (misreported robustness) and W8 (uncertainty of the headline LODO costs). None needs new data collection. All need reanalysis with existing scripts and options, and the causal wording in the title, abstract and conclusion needs to be brought down to the level the evidence supports. The evaluation hygiene (S1–S5) does not offset these, but it makes the repairs feasible.

---

## Arithmetic Receipts
no_recomputable_statistics: The manuscript reports p-values from permutation tests, Holm-adjusted NB corrected t-tests and bootstrap CIs without the test statistics, df or means with SDs that p_from_test_statistic, grim, grimmer or n_from_df require. The NB-corrected LODO intervals in W8 are my own recomputation from the committed per-split table, not a check of a reported statistic.
