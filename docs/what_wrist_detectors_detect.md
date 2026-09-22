# What does a wrist stress detector detect? A conceptual analysis for the JBHI revision and beyond

Written 2026-09-22 from the repo state at commit 590f671 (correction sheet, novelty experiments, novelty search, LODO tables, harmonization table). No new data were run. Citations are limited to papers named in the repo documents; anything from memory is marked *(unverified)*.

## 1. Construct validity: three things called "stress"

A wrist Empatica E4 measures four things, none of which is stress:

- **BVP**: pulse rate, and a noisy vagal proxy (RMSSD; r = 0.53 against chest ECG in our own validation, +34 ms bias). Corrupted by wrist motion and, during speech, by the arm and jaw movement that accompanies talking. Only 37% of WESAD stress windows keep usable PPG.
- **EDA at the ventral wrist**: eccrine sweat-gland activity, purely sympathetic-cholinergic, but from a site with far fewer glands than the palm. Agreement with palmar EDA is weak (Milstein & Gordon 2020: SCL r = 0.30 during conversation; van Lier 2020: cross-correlation 0.25). Rises with cognitive effort, speech, ambient temperature, and movement.
- **Skin temperature**: slow peripheral vasoconstriction proxy, dominated by ambient conditions and by the warm-up after donning the device.
- **Accelerometer**: movement. Transfers at AUROC 0.52–0.59 across datasets; useful only as a confound detector.

So the instrument sees *autonomic arousal*, mostly its sympathetic branch, through a low-fidelity window. The field then attaches three different constructs to the label "stress":

1. **Stressor exposure**: the protocol stage (TSST, Stroop, arithmetic). This is what every one of our five datasets actually labels.
2. **Autonomic arousal**: what the sensor sees and what the model learns.
3. **Subjective stress / appraisal**: self-report, available only in PhysioNet (per-task 1–10), WESAD (PANAS) and as pre/post STAI in Stress-Predict.

A fourth construct, HPA-axis activation (cortisol), is what Dickerson & Kemeny (2004) *(certain, but not in the repo)* showed responds specifically to social-evaluative threat plus uncontrollability; none of our datasets measured it. The claim "we detect stress from the wrist" silently equates 1, 2 and 3. Every one of our anomalies falls out of that equation.

### The five datasets, operationalised

| Dataset | Stressor | Social-evaluative threat | Speech | Posture | Non-stress arousal states present | Stress effect size (z) | Within / external BA |
|---|---|---|---|---|---|---|---|
| WESAD | TSST: speech + arithmetic before a panel | Strong | Yes | Standing | Amusement (clips), meditation | EDA +2.2, HR +2.2 | 0.916 / 0.884 |
| UBFC-Phys (test group) | Speech + arithmetic under evaluation | Present (test) vs absent (ctrl) | Yes, both groups | Seated | Ctrl group does the same tasks without evaluation | not computed | 0.877 / 0.837 |
| Stress-Predict | Stroop (silent) + TSST-style interview | Mixed: none (Stroop) / present (interview) | Interview only | Seated | Hyperventilation (arousal, no threat), relax | not computed | 0.697 / 0.646 |
| PhysioNet (Hongn 2025) | Stroop, TMCT arithmetic with time pressure, short opinion task | Weak (computer feedback) | Mostly no | Seated | Aerobic and anaerobic exercise, video rest | EDA +0.4, HR +1.7 | 0.776 / 0.746 |
| Campanella 2024 | Backward subtraction | Weak (experimenter present) | Probably aloud *(unverified)* | Seated | Lego manual tasks with and without countdown | not computed | 0.882 / 0.867 |
| EPM-E4 | none (emotion clips) | none | No | Seated | Anger, fear, happiness, sadness | n/a | n/a |

Two things stand out. First, no dataset separates speech from evaluation except UBFC-Phys, and there the separation is between subjects. Second, the two datasets with the lowest ceilings are exactly the two whose stressors are seated, mostly silent and weakly evaluative. The "difficulty" of PhysioNet and Stress-Predict is not a modelling problem; it is that their stressors produce less of the thing the wrist can see.

### Which findings the conflation explains

- **Low PhysioNet and Stress-Predict ceilings.** Under the arousal reading, separability is a function of arousal magnitude, and PhysioNet's EDA response is one fifth of WESAD's. A detector of construct 2 cannot exceed the arousal the stressor produced.
- **Mild-stress null.** Self-rated intensity (construct 3) did not predict misses. That is what you expect if the model reads construct 2 and constructs 2 and 3 are weakly coupled. Campbell & Ehlert (2012) *(from memory, unverified)* found subjective and physiological stress responses agree in only a minority of studies. The null is not "there is no mild-stress problem"; it is "self-report is the wrong ruler".
- **Few-shot null.** After per-subject z-scoring the person-specific level is gone. What five windows could teach is the person's response *shape*; but if the person did not respond, no number of their labelled windows creates a response to learn. Personalisation cannot buy back a missing signal.
- **Domain adaptation null.** CORAL, MMD and DANN align marginal feature distributions. Once labels are correct and features are z-scored per subject, the marginals are already close (transfer cost 0.015–0.05); what differs across datasets is label-conditional separability, i.e. stressor potency. DA has nothing to align.
- **Exercise false alarms (69.5%).** The cleanest evidence that the model detects arousal: a state with large arousal and no stress is called stress. Adding exercise to training fixes it (4–6%) only because the model then learns the *accelerometer-free physiological signature of exercise* as a second negative class, not because it learned stress.
- **UBFC-Phys control group.** Not yet scored, but decisive: same tasks, same speech, no evaluation. This is the only within-corpus contrast between construct 1 and construct 2.
- **Order confound.** Baseline is first in every dataset, so time in session covaries with arousal (warm-up, habituation, recovery). A baseline-referenced model calls 51–85% of post-stressor rest "stress" because recovery *is* residual arousal.
- **Missing-temperature threshold shift.** Orthogonal to the conflation; it is a modelling artefact and stays as a pitfall.

## 2. Falsifiable theories of wrist-detectable stress

A theory is worth having only if some result on our data would refute it.

### (a) Arousal-only: wrist detectors detect sympathetic arousal; stress specificity comes only from context

Predictions:
- The false-stress rate on any arousing non-stress state (exercise, hyperventilation, control-group speech, fear clips, Lego under countdown) rises monotonically with that state's arousal effect size.
- Within-dataset ceiling is a monotone function of the stressor's HR/EDA effect size. Already consistent: WESAD d ≈ 2.2, BA 0.92; PhysioNet EDA 0.4, BA 0.78.
- **Matched-arousal test.** Bin all windows by a label-free arousal index (mean of per-subject z-scored hr_mean and eda_tonic_mean). Within each bin, the AUROC of stress versus non-stress-arousal states (exercise, hyperventilation, control speech) is about 0.5.

Refutation: a state with large arousal and a low false-stress rate, or within-bin AUROC clearly above 0.5, would show the model has learned something beyond arousal magnitude.

Existing tables that bear on it: `exercise_negatives_physionet.csv` (69.5%), `stress_effect_size_by_dataset.csv`, `threshold_transfer_probe_summary.csv` (ranking survives, so whatever is learned transfers).

### (b) Social-evaluative signature: evaluative threat produces an EDA pattern the wrist can separate from non-evaluative arousal

Predictions:
- UBFC-Phys test-group speech is separable from control-group speech on EDA features beyond tonic level (SCR count and amplitude pattern), not only on HR.
- Models trained on evaluative stressors (WESAD, UBFC test) transfer to each other with lower cost than to PhysioNet. Partly consistent: external UBFC 0.837 and WESAD 0.884 versus PhysioNet 0.746, but Campanella (weakly evaluative) is also 0.867, which cuts against it.
- In the matched-arousal test, stress versus control speech keeps AUROC above 0.5 while stress versus exercise does not.

Refutation: control-group speech is called stress at the same rate that test-group speech is recalled. Caveat: because wrist EDA agrees with palmar EDA at r ≈ 0.3, a null could be instrument failure rather than theory failure. Say so; the palm-reference lab study in Section 4 is the only way to separate these.

### (c) Trait reactivity: detectability is a subject property

Predictions:
- Per-subject stress recall correlates across stressors within a person (Stroop versus TSST in Stress-Predict, Stroop versus TMCT in PhysioNet) and across models (within versus external).
- Misses are concentrated: the worst 20% of subjects produce most of the missed stress windows.
- Per-subject recall is predictable from label-free baseline physiology (low baseline HRV, high baseline EDA) or from the size of the person's own physiological response.

Refutation: per-subject recall uncorrelated across stressors (ρ near zero with a CI excluding 0.4) and misses spread evenly. Bears on: `loso_per_subject.csv`, the mild-stress per-unit tables. Blunted cardiovascular reactivity is an established psychophysiology construct *(Carroll and Phillips' work; certain of existence, specifics unverified)*, but the ML literature reviewed in the repo never treats the ceiling as a property of people.

### (d) Temporal theory: the error is at the edges

Predictions: recall is lowest in the first window of each stressor (onset latency) and false alarms peak in the first windows after stressor offset (recovery), so a large share of the error is temporal rather than physiological. Refutation: flat recall and false-alarm curves against time since onset and offset. O'Brien et al. (2026) saw the offset peak qualitatively on Campanella; the second-half-rest result (0.746 to 0.797 external on PhysioNet) is the quantitative hint.

These four are not exclusive. The likely truth is (a) plus (d) plus some (c), with (b) undecidable on wrist EDA.

## 3. The ceiling: an error budget

Within-dataset error (1 − BA) is about 0.08 on WESAD, 0.22 on PhysioNet, 0.30 on Stress-Predict. Candidate terms, ranked by the evidence already in the repo:

1. **Stressor potency (weak arousal).** Strongest evidence: the effect-size table. Seated arithmetic moves EDA by 0.4 z. Nothing downstream can recover a response that is a fifth the size.
2. **Recovery contamination of rest labels.** Direct evidence: second-half rests raise external PhysioNet from 0.746 to 0.797; baseline-referenced features call 51–85% of post-stressor rest "stress". This term is partly a labelling choice and partly real physiology.
3. **Non-responders.** Indirect evidence only: self-report rose in 89% (TMCT) and 80% (TSST), but that is construct 3. No per-subject physiological response magnitude has been computed.
4. **Sensor loss during speech.** Direct for WESAD (37% usable) and probably large for the Stress-Predict interview; small for PhysioNet, whose stressors are silent. Per-dataset cardiac_coverage by class would settle it in minutes.
5. **Residual label error.** Stress-Predict's tag snapping (180 s) and its weak within-dataset AUROC (0.77) suggest more is wrong there than elsewhere.
6. **Onset latency.** No evidence yet; cheap to check.
7. **Window-level noise.** 60 s windows are short for HRV; a floor, not a lever.

### Estimating the terms

All but the recovery term are test-side filters on predictions we already have, so they need no retraining. Using the standard 20 grouped splits and hr_hrv_eda features:

- Recovery term: BA with second-half rests only minus BA with all rests (already have +0.05 for PhysioNet external).
- Sensor term: BA on windows with high cardiac_coverage minus BA on all; on WESAD additionally chest-ECG HR/HRV versus wrist HR/HRV with the same EDA.
- Onset term: BA excluding the first 60 s of each stressor minus BA on all.
- Responder term (oracle): BA restricted to subjects whose own stressor-minus-baseline HR or EDA delta exceeds a threshold, minus BA on all. This is an upper bound on "what if everyone responded" and is label-using, so it is a diagnostic, never a method.
- Potency term: the residual after the above, compared across datasets and correlated with effect size.

The terms are not additive. Report them as a sequence and, since they are test-side filters, average over filter orders (Shapley-style; 4 filters, 24 orders, no retraining). The result is the first error budget for wrist stress detection that says how much of the ceiling is people, labels, sensor and stressor. That is new information regardless of which term wins.

## 4. Where one student can contribute most, 2026–2028

Scores 1–5; higher is better for novelty, evidence in hand, fit and pioneering value; higher is *worse* for cost and risk.

| # | Direction | Novelty | Evidence | Cost | Risk | Fit | Pioneering | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | Specificity panel / arousal taxonomy across five datasets (exercise, hyperventilation, control speech, manual task, emotion clips), with matched-arousal AUROC | 4 | 4 | 1 | 1 | JBHI, TAFFC | 4 | **Do now.** Turns the exercise result into a general claim about what detectors detect. Aydoğan did exercise on one pair; nobody has the panel. |
| 2 | Error budget of the ceiling (Section 3) | 4 | 4 | 1 | 1 | JBHI | 4 | **Do now.** Cheap, diagnostic, and it decides which method is worth building. |
| 3 | Subject-level detectability trait | 3 | 3 | 1 | 3 | JBHI, Psychophysiology | 3 | Do now as a probe; n is small (15–35 per dataset), so treat as hypothesis-generating. If misses are trait-like, the field has been optimising the wrong thing. |
| 4 | Verified multi-dataset benchmark release: fixed protocol-stage labels, frozen subject splits, leakage tests, corrected tests, specificity panel | 2 as method, 5 as infrastructure | 5 | 3 | 1 | Sci Data, NeurIPS D&B, IMWUT | 3 | Highest citations per effort. Release label tables, split files and code, not the raw data (licences). FEEL (Singh 2026) is arousal/valence over 19 datasets; nothing stress-specific with audited labels exists. |
| 5 | Recovery- and time-aware labelling (PU or noisy-negative learning with time since stressor as the noise covariate) | 4 | 3 | 3 | 3 | JBHI methods | 3 | Promising but dangerous: a method that "improves" by down-weighting hard negatives is label editing unless evaluated on clean second-half rests and on held-out datasets. Fix the evaluation protocol before the method. |
| 6 | Selective classification / open-set recognition: stress vs known non-stress vs "unrecognised arousal", with abstention on low signal quality | 3 | 3 | 2 | 2 | JBHI, IMWUT | 3 | The natural deployment form of #1. More honest than any accuracy number. |
| 7 | Conformal prediction under dataset shift with subject-level exchangeability | 3 | 2 | 2 | 2 | JBHI | 2 | A wrapper. Worth one table, not a paper. |
| 8 | Motion-robust wrist HR during speech (accelerometer-aided PPG) | 2 | 4 | 4 | 3 | Sensors, TBME | 1 | Mature signal-processing literature *(SpaMA, TROIKA and successors; certain of existence, unverified here)*. Incremental, but it is the only lever on the 63% of lost WESAD stress windows and it has ECG ground truth. Engineering chapter, not a thesis. |
| 9 | Raw-signal SSL / PPG foundation models under our LODO | 2 | 1 | 4 | 4 | IMWUT | 2 | Industry will out-scale a student. Do the cheap version: a public pretrained PPG encoder as a frozen feature extractor, one LODO row. **The single most important literature check is Zhao et al. (2025, PULSE): LOSO AUROC 0.965 on PhysioNet versus our 0.854.** If their labels are protocol-stage and their split is clean, our ceiling is a feature ceiling and Section 3 is wrong. If they used session-level labels, it is the defect-D confound. |
| 10 | A counterbalanced lab study at UTSA | 5 | 0 | 5 | 3 | Sci Data + JBHI/TAFFC | 5 | **The pioneering move.** Every public E4 dataset has fixed order and conflates speech with evaluation. Design below. |
| 11 | Ambulatory validation | 5 | 0 | 5 | 5 | IMWUT | 5 | The label problem (EMA sparsity, no ground truth) is unsolved; not for this revision. A later chapter, only after #10 defines what is being validated. |

Blunt summary: #5, #7, #8 and #9 are incremental. #1–#3 are cheap and change the paper's thesis. #4 is the citation engine. #10 is the only thing here that makes the field different afterwards.

### The lab study that would break the confounds

Within-subject, two visits, about 40 participants, Empatica E4 plus chest ECG plus a palm or finger EDA reference, salivary cortisol at four points if budget allows, continuous slider self-report, and a 2 × 2 factorial of **speech** (yes/no) × **evaluation** (panel present/absent), plus **exercise** (cycling at matched HR) and a **cognitive-load-without-threat** task (n-back with no feedback), with recovery periods long enough to track the return to baseline. Order counterbalanced by Latin square, with a baseline at both the start and the end of each visit. The second visit repeats the protocol to estimate test-retest reliability of each person's reactivity (theory c). This single dataset answers (a), (b), (c) and (d) at once, provides the first order-matched baseline for the wrist, and lets the palm reference say whether wrist EDA lost a signal that was there. It also becomes the external test set that no current benchmark can provide.

## 5. The one big idea: two honest reframings

The field's implicit claim is that a wristband detects stress. The defensible claim is narrower and more useful: **a wristband detects a change in autonomic arousal reliably across people and protocols; whether that arousal is stress is an inference that needs context the wrist does not have.** The proposal that follows from this is a reporting norm: every stress detector should publish a *specificity panel*, the false-stress rate under exercise, speech without evaluation, cognitive load without threat, and strong emotion, the way a diagnostic assay reports cross-reactivity. That norm is a contribution one student can own.

### Title A

**What Do Wrist-Worn Stress Detectors Detect? A Five-Dataset Specificity Panel**

*Abstract.* Wrist-worn stress detectors are trained to separate stressor periods from rest and are reported as detecting stress. We ask what they detect. Across five public Empatica E4 datasets (WESAD, PhysioNet, Stress-Predict, UBFC-Phys, Campanella; 118 subjects) with audited protocol-stage labels, subject-held-out and leave-one-dataset-out evaluation, and per-subject normalisation of heart-rate and electrodermal features, cross-dataset transfer costs only 0.015–0.05 balanced accuracy. Yet the same models call physical exercise stress in 70% of windows, and we measure the corresponding rates for hyperventilation, speech without evaluation, manual tasks under time pressure and induced emotions. Within bins of matched autonomic arousal, stress and non-stress arousal are [separable / not separable]. An error budget of the within-dataset ceiling attributes [x]% to stressor potency, [y]% to post-stressor recovery labelled as rest, [z]% to sensor loss during speech and [w]% to subjects who did not respond. We conclude that wrist detectors transfer because they detect autonomic arousal, and that stress specificity must come from context or from protocols designed to separate arousal from threat. We propose a specificity panel as a reporting standard and release audited labels, frozen subject splits and code.

### Title B

**Arousal Transfers, Stress Does Not: The Ceiling of Wrist Stress Detection Is the Stressor, Not the Model**

*Abstract.* Reports of near-chance cross-dataset transfer in wearable stress detection have motivated domain adaptation and personalisation. We show on five E4 datasets that, once dataset labels are corrected and features are normalised per subject, transfer cost is small, and that domain adaptation, few-shot personalisation and per-user thresholds add nothing significant. What remains is a within-dataset ceiling of 0.70–0.78 balanced accuracy on seated, silent, weakly evaluative stressors versus 0.88–0.92 on social-evaluative ones, matched by a fivefold difference in electrodermal effect size. We decompose the ceiling into stressor potency, recovery contamination, sensor loss and non-response, show that self-rated intensity does not predict misses, and show that the models respond to any arousing state. The practical conclusion is that the next gains in wrist stress detection will come from protocol design, contextual sensing and honest abstention, not from classifiers. We release the audited five-dataset benchmark.

Either title turns the revision's collection of negative and corrective results into a single positive claim with a test the community can adopt.
