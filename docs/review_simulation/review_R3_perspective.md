# Peer Review Report

## Manuscript Information
- **Title**: What Do Wrist-Worn Stress Detectors Detect? Arousal Transfers Across Five Empatica E4 Datasets, Evaluative Stress Does Not
- **Manuscript ID**: n/a (simulated IEEE JBHI panel)
- **Review Date**: 2026-09-23
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 3 (Perspective)

### Reviewer Identity
Digital-health and clinical-translation researcher: ambulatory physiological monitoring, consumer wearables, regulatory and deployment questions, and measurement-validity frameworks (V3 for biometric monitoring technologies, COSMIN, clinical-laboratory interference testing). I am not a machine-learning methodologist; the statistical design is Reviewer 1's remit and I comment on it only where it decides whether the proposed norms can be used.

### Review Focus
Whether the findings change how wrist stress products should be built, claimed and validated; whether the proposed reporting norms (specificity panel, matched-arousal test, three protocol rules) are specified well enough for another group or a regulator to apply; consumer harm from false stress alerts; how far lab E4 results generalise; and how well the paper uses the measurement-science framing it borrows.

---

## Overall Assessment

### Recommendation
- [ ] Accept
- [ ] Minor Revision
- [x] **Major Revision**: the central finding matters for practice and the evidence behind it is careful. But the paper's stated deliverable, reporting standards, is two sentences long (l. 273), and the paper never turns its false-stress rates into what a user would actually experience.
- [ ] Reject

### Confidence Score
4. Core expertise covers deployment, validation frameworks and harm; I have adjacent (not core) expertise in the classifier evaluation.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Calibration Status
`NOT_CALIBRATED`

### Criterion-Bound Judgements
| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Practical impact for wearable stress products and clinical use | Caller focus (R3 configuration) | PARTLY_MEETS | `section: §VI Discussion, l. 271-273`; `table: Table III` | The result (the detector reports arousal, not evaluative stress) directly affects product claims. The paper stops before translating it into alert burden, PPV or labelling language | Depends on assumed daily-life prevalence | yes: the main practical payoff is left implicit |
| Actionability of proposed reporting norms | Caller focus | DOES_NOT_MEET | `section: §VI "Reporting norms", l. 273`; `section: §IV-D, l. 97` | No required state list, minimum sample, metric definition, operating point, equivalence margin or reporting template | none identified | yes: the abstract names this as a contribution |
| Ethics and consumer harm from false alerts | Caller focus | DOES_NOT_MEET | `text: l. 280 "honest abstention"`; `section: §VI Limitations, l. 277` | No discussion of false-alert harm, alert fatigue or third-party use (employers, insurers). No ethics or secondary-use statement | none identified | yes, as part of a Major revision |
| Generalisability beyond laboratory E4 data | Caller focus | PARTLY_MEETS | `text: l. 277 "All data are laboratory recordings; ambulatory validity is untested."`; `table: Table I` | Acknowledged in one sentence. But the title generalises to all "wrist-worn stress detectors". No demographics are given, and there is no analysis of what consumer devices expose | Consumer-device claims are extrapolation | yes: the title and scope need to change |
| Cross-disciplinary framing (assay cross-reactivity, measurement science) | Caller focus | PARTLY_MEETS | `text: l. 273 "the way an assay reports cross-reactivity"` | The analogy is apt but used only as decoration. The matching validity vocabulary (discriminant validity, interference testing, context of use) is missing | none identified | no: improves the paper, does not decide it |

### Summary Assessment
The authors take five public Empatica E4 datasets and audit their labels. They show that a subject-normalised HR/HRV/EDA detector transfers across datasets at a small cost (0.015 to 0.05 BA). They then ask what transfers. A specificity panel and a matched-arousal test show that the detector rejects exercise once it has been trained on exercise. It does not separate evaluative stress from the same tasks performed without evaluation, from hyperventilation, from manual tasks or from anger clips. From a translation point of view this is the most useful kind of paper: it reframes what a commercial "stress score" measures. The evidence is careful, the negative results are leakage-controlled, and the sensor warm-up confound is a practical finding that deployers can act on. The weaknesses are in the translation layer that the paper itself promises. The "reporting standards" are not specified well enough to apply or audit. The false-stress rates (0.33 to 0.56 on ordinary daily states at a fixed threshold of 0.5) are never converted into positive predictive value or false alerts per day, so the consumer-harm consequence of the paper's own result goes unstated. The title and conclusion generalise from 132 laboratory subjects on a research-grade device that, to my knowledge, is no longer sold, to "wrist-worn stress detectors" as a class, without demographics or a consumer-device mapping. These can be fixed without new data collection, so I recommend Major Revision.

---

## Strengths

### S1: The specificity panel moves the field's question from accuracy to validity
Scoring one fixed detector on arousal states outside its task definition is, in measurement terms, a discriminant-validity test. Consumer stress features have never been subjected to one publicly. The exercise result (0.69 to 0.08 once exercise is a training negative) and the null results for non-evaluated speech and hyperventilation are directly usable by product teams.
**Evidence Anchor**: `table: Table III (l. 139-159)`

### S2: Sensor warm-up identified as a protocol confound with a concrete remedy
The finding that baseline falls inside E4 warm-up on three of five datasets, with WESAD as the natural control, is the kind of result that changes study protocols rather than models. It also explains why time-only classifiers do well.
**Evidence Anchor**: `section: §V-E, l. 211-213`

### S3: A chest-ECG bound on what better wrist sensing can buy
Swapping in chest ECG on the same windows gives about +0.04 on WESAD. This is useful to hardware teams deciding whether motion-robust PPG is worth investing in for this use.
**Evidence Anchor**: `section: §V-F, l. 243`

### S4: Leakage-controlled negative results for personalisation and domain adaptation
Few-shot calibration with windows taken first in time, and per-user thresholds, add nothing significant; random calibration windows produce the only gains. Vendors routinely claim "learns your baseline", and this result is the right caution against that claim.
**Evidence Anchor**: `table: Table V (l. 246-266)`

### S5: Open, reproducible release
Audited labels, frozen subject splits and scripts make the proposed panel re-runnable by others, which a reporting norm needs.
**Evidence Anchor**: `section: Data and Code Availability, l. 283`

---

## Weaknesses

### W1: The reporting norms are not specified well enough to adopt
**Problem**: The proposed standards amount to one sentence listing four state types and three protocol rules. The paper does not say which states are mandatory, how many subjects or windows each cell needs, at what operating point the false-stress rate is reported, what matched-arousal AUROC or equivalence margin counts as "separable", how the arousal index is fixed in advance, or what to report when a state is unavailable. Several of the paper's own cells would fail any reasonable minimum: hyperventilation has 1.7 windows per subject, the UBFC-Phys control has 8 subjects, and anger has 33 windows.
**Evidence Anchor**: `section: §VI "Reporting norms", l. 273`
**Why it matters**: A norm that cannot be audited will not be adopted. Without a template, other groups will run incomparable panels. The abstract (l. 25) claims these as proposed "reporting standards".
**Suggestion**: Add a boxed checklist or table "Specificity reporting for wearable stress detectors". It should give: (a) required states (exercise at matched HR, non-evaluated speech, non-threat cognitive load, emotion induction) and recommended ones (thermal load, postural change, caffeine, hyperventilation); (b) minimum subjects per cell, with the power reasoning; (c) the metric: false-stress rate with a subject-bootstrap CI, at the deployed threshold and at a fixed sensitivity (e.g. recall 0.80); (d) the matched-arousal specification: index defined a priori and disjoint from the model, number of bins, minimum windows per bin, the permutation scheme, and an equivalence margin; (e) a "not available" code. Map the items onto an existing reporting guideline (for example TRIPOD+AI) so journals can require them.
**Severity**: Major
**Confidence**: 5 (core expertise: validation frameworks and reporting guidelines)

### W2: False-stress rates are never translated into user-facing harm
**Problem**: The results imply that a deployed detector would flag a third to a half of ordinary arousing activities (Lego 0.33 to 0.43, non-evaluated speech 0.55, emotion clips 0.18 to 0.42), and even 0.08 to 0.29 of in-task rest. In daily life acute stress is rare relative to these states. The paper never computes PPV or expected false alerts per day, and never discusses harm.
**Evidence Anchor**: `table: Table III (l. 155 "Reference: in-task non-stress states & 0.08--0.29")`
**Why it matters**: Worked through, this is the paper's most important practical consequence. With illustrative numbers (5% of waking minutes stressed, recall 0.75, false-stress rate 0.20), PPV is about 0.17, so roughly five of six alerts would be false. Alert fatigue, nocebo and anxiety effects, and use of "stress" scores by employers or insurers are foreseeable harms that the paper's evidence bears on directly.
**Suggestion**: Add a short "Deployment implications" subsection with a PPV and false-alerts-per-waking-hour table over a range of assumed prevalences and daily state mixes, computed from the measured recall and false-stress rates. Add a paragraph on harms and on labelling: products should describe the output as "physiological arousal", not "stress", and should not trigger interventions on it without context. Note that consumer stress features are typically marketed as general-wellness functions; state what claim language the evidence supports.
**Severity**: Major
**Confidence**: 5 (core expertise: consumer-wearable deployment and harm)

### W3: The title and conclusion generalise beyond the evidence (device, population, setting)
**Problem**: The title says "Wrist-Worn Stress Detectors", but the evidence comes from one research-grade device (E4, raw 4 Hz EDA and 64 Hz BVP), 132 laboratory subjects, and no reported demographics. Consumer watches mostly expose optical HR/HRV at sparse sampling, and EDA only as spot checks, if at all. PPG error depends on skin tone, BMI and motion, and Table I reports none of these. Limitations covers ambulatory validity in one sentence.
**Evidence Anchor**: `section: Title (l. 15)`; `table: Table I (l. 54-80)`; `text: l. 277 "All data are laboratory recordings; ambulatory validity is untested."`
**Why it matters**: Product and clinical readers will read the result as a statement about the devices on their wrists. It may well hold, but for devices without EDA that is an extrapolation.
**Suggestion**: (a) Scope the title, for example "...Detect? Evidence From Five Laboratory Empatica E4 Datasets". (b) Add an HR/HRV-only arm to the LODO and specificity panels as a consumer-device proxy; the EDA-only and cardiac-only models already exist (l. 201). (c) Add age, sex and, where available, skin-tone and BMI columns to Table I, and report false-stress rate by sex where n allows. (d) Add a paragraph on what differs outside the lab: thermoregulatory sweating, caffeine, posture and circadian drift act on wrist EDA and HR. Say which panel states would be needed to cover them.
**Severity**: Major
**Confidence**: 4 (core expertise: consumer device capabilities; demographic effects on PPG are cited from general knowledge, not checked here)

### W4: Non-significance is read as "cannot separate"; a norm needs equivalence
**Problem**: The matched-arousal conclusion that the detector "cannot separate" stress from four states rests on AUROCs not differing from 0.5 after Holm correction. Several CIs are wide (Lego [0.24, 0.54]; anger [0.31, 0.52], n = 33 windows). Lego at 0.355 is below 0.5, which means that at equal arousal Lego is scored as more stress-like than stress itself. That is informative and is not "nothing beyond arousal".
**Evidence Anchor**: `table: Table IV (l. 170-197)`; `text: l. 97 "An AUROC near 0.5 means the model has learned nothing beyond arousal magnitude for that pair."`
**Why it matters**: As a reporting standard, the test must separate "shown equivalent to chance" from "underpowered". In interference testing, "no interference" is declared against a pre-set allowable bias, not from a non-significant difference.
**Suggestion**: Pre-specify an equivalence margin (e.g. |AUROC - 0.5| < 0.1) and report TOST or CI-inside-margin results per pair. Rephrase conclusions (abstract l. 25, conclusion l. 280) as "not shown to separate" where equivalence is not established. Discuss the sub-0.5 Lego and anger values explicitly.
**Severity**: Major
**Confidence**: 3 (adjacent field: applying general inference standards; Reviewer 1 is better placed on the statistics)

### W5: The measurement-science framing is announced but not used
**Problem**: The assay cross-reactivity analogy appears once (l. 273). Its natural counterparts are not used. Interference testing (for example CLSI EP07) reports interference as a function of interferent concentration. V3 separates analytical from clinical validation and requires a stated context of use. COSMIN and Campbell–Fiske would call the matched-arousal test a discriminant-validity test. The paper also never states the reference standard: "stress" is the protocol stage, not a validated criterion such as cortisol or self-report, and Table V shows self-report does not track detection.
**Evidence Anchor**: `text: l. 273 "the way an assay reports cross-reactivity"`; `table: Table V, row "Self-rated stress rise vs window recall"`
**Why it matters**: Without a context of use, "specificity" is undefined: false alarms on exercise matter for an ambulatory product and not for a seated-lab tool. Without a named reference standard, readers cannot tell whether the result is a failure of the detector or of the label construct.
**Suggestion**: (a) Present the panel and matched-arousal test together as an interference curve: false-stress rate against arousal quintile for each state. This is the direct analogue of a dose–interference plot and is computable from existing outputs. (b) Add a short paragraph placing the work in V3: analytical validity is covered by the wrist-vs-chest checks, and the construct/clinical validity of "stress" is what the panel tests. (c) State that the reference standard is the protocol stage, and discuss what an evaluative-threat criterion would require.
**Severity**: Minor
**Confidence**: 5 (core expertise: measurement-validity frameworks)

### W6: The protocol rules lack the numbers the paper already has
**Problem**: The paper recommends "a settling period before baseline" and "an activity gate" but gives no duration and no gate definition. Its own data show temperature rising until about minute 20 and tonic EDA over the first 10 minutes.
**Evidence Anchor**: `section: §VI "Reporting norms", l. 273`; `section: §V-E, l. 213`
**Why it matters**: Study designers need a number. The paper's data can supply one; without it the rule is advisory only.
**Suggestion**: State a recommended minimum settling period derived from Fig. 4 (e.g. at least 15 to 20 min from donning, with the criterion that the slope of temperature and tonic EDA falls below X). Specify the activity gate (accelerometer threshold, or exercise in the negatives) and report the false-stress rate after gating.
**Severity**: Minor
**Confidence**: 4 (core expertise: ambulatory protocol design)

### W7: "Honest abstention" is recommended but not evaluated
**Problem**: The conclusion names abstention as a route to future gains. The paper never evaluates selective prediction, although PPG coverage (F2) and arousal level are natural abstention signals.
**Evidence Anchor**: `text: l. 280 "protocol design, contextual sensing and honest abstention"`
**Why it matters**: For a product, abstaining on low-quality windows is the cheapest mitigation of false alerts. The authors hold the data to quantify it.
**Suggestion**: Add a coverage–accuracy curve (abstain when cardiac coverage < 0.5, or at an intermediate score), or remove the claim.
**Severity**: Minor
**Confidence**: 4 (core expertise: deployment design)

### W8: Missing ethics and data-use statement, and incomplete panel uncertainty
**Problem**: There is no statement on secondary use of the human-subject datasets (licences, ethics exemption). Table III gives no CIs for the exercise and EPM-E4 rows, although other rows have them.
**Evidence Anchor**: `section: Data and Code Availability, l. 282-283`; `table: Table III, rows l. 149 and l. 153`
**Why it matters**: JBHI expects an ethics or data-use statement for human data. Norms that require CIs should model them.
**Suggestion**: Add a one-paragraph ethics statement (secondary analysis of de-identified public data, original approvals, licence compliance). Add bootstrap CIs to every Table III cell.
**Severity**: Minor
**Confidence**: 4 (adjacent: journal policy knowledge)

---

## Detailed Comments

### Assumption Audit
- **Explicit assumptions**: protocol stage equals stress state; per-subject whole-session z-scoring is acceptable (stated as transductive, with sensitivity analyses; well handled).
- **Implicit assumptions**: that a threshold of 0.5 is the relevant operating point (products tune thresholds, so report at matched sensitivity too); that lab arousal states represent daily-life confounders (thermal, postural, pharmacological and circadian sources are absent); that recording start equals donning (acknowledged at l. 213).
- **Paradigmatic assumptions**: that "stress" is a single target a wrist device ought to detect. The paper's own result suggests the product category should be redefined as arousal monitoring with context. The Discussion could say this outright.

### Cross-Disciplinary Connections
- **Parallel research**: clinical-chemistry interference testing (CLSI EP07); V3 for biometric monitoring technologies (Goldsack et al., npj Digit. Med., 2020); COSMIN construct-validity terminology; Campbell–Fiske convergent/discriminant validity. Psychophysiology already distinguishes arousal from appraisal (for example the challenge/threat cardiovascular literature), which bears on the fear-clip cardiac result at l. 271.
- **Borrowing opportunities**: the dose–interference curve (W5a); an "intended use / context of use" statement as in regulatory submissions; a PPV table as in diagnostic-accuracy reporting (STARD).
- **Methodological borrowing**: equivalence testing (TOST) for "no separation" claims; decision-curve analysis to express the value of alerts at different thresholds.

### Practical Impact
- **Real-world application**: high. The result is directly relevant to stress-score features on consumer watches, workplace wellbeing programmes, and any clinical use (for example anxiety monitoring) that would interpret wrist arousal as stress.
- **Implementation feasibility**: the panel is feasible for any group with access to exercise, speech and cognitive-load recordings. The matched-arousal test needs an a priori index and adequate per-bin sampling (W1, W4).
- **Unintended consequences**: detectors labelled "stress" would, on this evidence, produce mostly false alerts in daily life (W2). There is a risk of that data being used in employment or insurance decisions. Conversely, a well-specified panel could become a de facto certification test; the paper should anticipate that and specify it carefully.

### Future directions
The two proposed studies (l. 280) are the right ones. I would add that the ambulatory validation should pre-register the panel states and the PPV target, and that the counterbalanced within-subject study should include a consumer HR-only device alongside the E4-class and chest references, so that the result reaches the devices people actually wear.

---

## Questions for Authors
1. What minimum per-cell sample size would you require for a specificity panel entry to be reportable, and why?
2. At a threshold tuned for recall 0.80, what are the false-stress rates in Table III?
3. How does the HR/HRV-only detector (no EDA) behave on the panel and on the matched-arousal test?
4. Does any dataset allow a sex- or age-stratified false-stress rate?

## Recommendation Rationale
Major Revision. The science supports a significant and practically important claim. The two items the paper presents as its translational contribution, reporting standards and implications for detectors, are underspecified (W1), and the consumer-harm consequence is left unstated (W2). The scope of the claim also exceeds the device and population studied (W3). All three can be fixed from existing data plus writing, which is why I do not recommend rejection.
