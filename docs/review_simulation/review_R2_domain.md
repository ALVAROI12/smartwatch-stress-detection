## Domain Review Report (Peer Reviewer 2)

**Manuscript:** "What Do Wrist-Worn Stress Detectors Detect? Arousal Transfers Across Five Empatica E4 Datasets, Evaluative Stress Does Not" (IEEE JBHI submission, `paper/main.tex`). Line numbers refer to `main.tex`.

### Reviewer Identity
Psychophysiologist and wearable-sensing researcher. Background: Empatica E4 EDA/PPG signal processing, TSST and other laboratory stress paradigms, and the wearable stress-detection literature (WESAD, Schmidt et al., Vos et al. 2023, Prajod et al. 2024). This review covers physiological validity, construct validity of the "stress" and comparator labels, signal-processing adequacy on the wrist, and whether citations are accurate. Statistical design is left to Reviewer 1.

### Overall Recommendation
**Major Revision**

Rationale: the empirical core is careful and useful to the field: descriptor-level label audit, a small leave-one-dataset-out transfer cost, the exercise false-alarm result, the warm-up/time-in-session quantification, and the specificity-panel reporting proposal. The headline construct claim is the problem. The title's "Evaluative Stress Does Not" and the conclusion that the detector "has not learned evaluative threat" go further than the data allow, for four reasons: (i) several "non-evaluative" comparators are, according to their own descriptor papers, evaluated or deliberately stressful tasks; (ii) no manipulation check shows that comparator and stress states differ in psychological stress; (iii) the matched-arousal test conditions on the same peripheral signals that carry any wrist stress response, so a null is expected under both competing hypotheses; (iv) the closest prior work (Kwon et al. 2026 iso-HR matching) and the psychophysiology of social-evaluative threat (HPA axis, friendly-TSST) are not engaged. All four can be fixed by reframing and by analyses on data the authors already hold. None needs new data collection.

### Confidence Score
4. Domain expertise in E4 psychophysiology and stress paradigms. Descriptor claims were checked against the full texts in the author's `sources/` folder (UBFC-Phys, Campanella 2024, Stress-Predict, Hongn 2025, Kwon 2026, Prajod 2024) and the author's `docs/literature/` notes. Some suggested references come from memory and are marked accordingly.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Calibration Status
`NOT_CALIBRATED`

### Criterion-Bound Judgements
| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Literature coverage (classic + recent) | domain_reviewer_agent Step 1 | PARTLY_MEETS | `absence: Related Work and Discussion — expected stress psychophysiology (Dickerson & Kemeny; TSST/f-TSST; Schmidt 2019; Vos 2023); checked §I, §II, §VI, bibliography` | Recent ML transfer papers are well covered; the psychophysiological theory of stress and the canonical wearable reviews are absent | Recency of 2026 citations not independently checked | yes: the central claim is a psychophysiological one |
| Theoretical framework (arousal vs stress) | Step 2 | PARTLY_MEETS | `text: §VI line 271 "it has not learned evaluative threat"` | Arousal is treated as one-dimensional, and stress is never defined beyond "protocol stage". The expected wrist signature of evaluative threat is not stated | — | yes |
| Factual accuracy of dataset/comparator descriptions | Step 3a | DOES_NOT_MEET | `text: §V-C line 161 "performs the same speech and arithmetic tasks as the test group without an evaluating panel"` | Contradicted by UBFC-Phys descriptor (different speech topic, easier countdown, experimenter present with error-stop) | Checked against full text in `sources/` | yes |
| Citation representation | Step 3a | PARTLY_MEETS | `text: §II-A line 41 "Prajod et al. attribute transfer failure to stressor type, on chest ECG"` | Mostly accurate. Prajod's intensity confound, Kwon's iso-HR arousal test and the Sosa scope are misrepresented or omitted | Tervonen 1.7-point figure not checked | partly |
| Terminology precision | Step 3c | PARTLY_MEETS | `text: §I line 33 "pulse rate, a weak vagal proxy"` | HR vs vagal index, HRV vs PRV, "non-evaluative" used loosely | — | no |
| Contribution and positioning | Step 4 | PARTLY_MEETS | `text: §I line 33 "has not, to our knowledge, been tested"` | Real contribution, but the novelty claim ignores Kwon et al.'s explicit arousal-vs-stress test | — | yes |
| Wrist signal-processing validity (EDA/PPG/TEMP) | Step 3 (domain anchors) | PARTLY_MEETS | `text: §IV-A line 88 "EDA features are mean, SD, range, slope, tonic and phasic means"` | PPG is validated against ECG. EDA processing is unspecified and has no artefact handling. The temperature warm-up reading lacks controls for alternative explanations | — | yes (comparators involve hand motion) |

### Summary Assessment
The manuscript asks a question the field has avoided: whether wrist stress detectors trained on stressor-vs-rest labels detect anything beyond autonomic arousal. It answers with a five-dataset E4 benchmark, a specificity panel and a matched-arousal test. Three results are solid and worth publishing: the small transfer cost once labels are audited and features are normalised per subject; the finding that exercise must be in the training negatives; and the demonstration that baselines coincide with sensor warm-up. The psychophysiological interpretation is less secure. The paper never defines stress. It does not state what peripheral signature evaluative threat should have at the wrist; the established discriminating marker is HPA-axis output, which no E4 channel measures. Its "non-evaluative" comparators are, by their descriptors, either still socially evaluated (UBFC-Phys control), explicitly designed as stressors (Campanella Lego with countdown, Stress-Predict hyperventilation provocation), or passive film viewing (EPM-E4), which confounds the contrast with active versus passive coping. Because the arousal index is built from HR and tonic EDA, the matched-arousal test removes the only channel through which a wrist stress response could appear. The paper's own reference row (stress vs baseline/rest, within-bin AUROC 0.58) shows this. The defensible conclusion is narrower and still valuable: wrist HR/HRV/EDA carry little stressor-specific information beyond sympathetic magnitude in these paradigms. The paper should present this as a property of the instrument and the paradigms, and should not claim to show what the detector "learned" about evaluative threat.

### Strengths

### S1: Descriptor-level label audit
Every labelling decision was checked against the descriptor paper. The audit found real errors: WESAD offset, PhysioNet whole-session labels, the Stress-Predict protocol violation, and the hyperventilation tag. The paper also correctly says the audit is a correctness contribution rather than the cause of good transfer. Few wearable-stress papers do this.
**Evidence Anchor**: `text: §III line 82 "Every labelling decision was checked against the dataset's descriptor paper. Seven corrections followed"`

### S2: Wrist cardiac features validated against chest ECG, with honest reporting of PPG loss during speech
The 37% usable-PPG figure for TSST windows and the RMSSD bias agree with Milstein & Gordon (2020), per the author's full-text note (RMSSD r = 0.417 during speech). This matters physiologically: talking and gesturing degrade wrist PPG exactly when the stressor is present.
**Evidence Anchor**: `text: §IV-A line 88 "only 37\% of TSST windows keep usable photoplethysmography (PPG) because speech moves the wrist"`

### S3: Exercise specificity result is physiologically coherent and replicates independent work
The 69–70% exercise false-alarm rate matches Aydoğan & Villagra Povina (2026). It collapses to 8–10% once exercise is a negative, and exercise stays separable whichever channel defines arousal. This fits exercise's metabolically driven HR, which is out of proportion to its EDA.
**Evidence Anchor**: `table: Table III, row "PhysioNet aerobic / anaerobic"`

### S4: The time-in-session and warm-up analysis is new and important for the field
To my knowledge, and per the author's gap scan, no prior E4 work quantifies how far baseline windows fall inside the post-donning settling period. The contrast with WESAD, whose baseline starts 17–49 min after switch-on, makes a natural control.
**Evidence Anchor**: `figure: Fig. 3 (fig_time), panels (b)-(c)`

### S5: The specificity-panel reporting proposal
The analogy to assay cross-reactivity is apt. The proposal is concrete enough to adopt: exercise, speech without evaluation, cognitive load without threat, strong emotion.
**Evidence Anchor**: `text: §VI line 273 "the way an assay reports cross-reactivity"`

### Weaknesses

### W1: The UBFC-Phys control group is misdescribed as the same tasks without evaluation
The paper uses the UBFC-Phys control group as its cleanest "evaluation removed" contrast (abstract; §V-C line 161; §V-D line 199; Discussion line 271, "cannot tell an evaluated speech from an unevaluated one"). The descriptor (Meziati Sabour et al., IEEE TAFFC 2023, p. 626) says otherwise. The control speech task is a different topic ("recall a positive holiday memory ... and persuade the experimenter"). The control arithmetic is an easier task ("countdown starting from 2025 in steps of 10" vs 2023 in steps of 17). In both groups participants count aloud, "were stopped whenever they gave the wrong number", and face a live experimenter. The test group adds a job-interview framing and a fake Skype "jury member". The descriptor authors conclude the tasks "imply a high stress level in both scenarios". The author's own verification notes (`docs/literature/dataset-and-claims-verification.md` §2) already warn: "Do not call the control tasks 'non-evaluative'". The contrast is therefore a higher versus lower dose of social-evaluative threat, with task difficulty confounded. It is not evaluation versus none. A detector that cannot separate these two at matched arousal is expected under any hypothesis.
**Fix:** Describe the UBFC-Phys control as "lower-intensity social-evaluative condition (experimenter present, easier task)" throughout, including the abstract and Table I. Drop "without evaluation", "unevaluated" and "no evaluation" from Tables III–IV and §V–VI. State the task-difficulty confound. Report the descriptor's manipulation check (somatic anxiety differed at p = 0.03; cognitive anxiety did not).
- **Severity**: Major | **Evidence Anchor**: `text: §V-C line 161 "performs the same speech and arithmetic tasks as the test group without an evaluating panel"` | **Confidence**: 5 — verified against the descriptor full text

### W2: Other comparator states are stressors by their own descriptors, and film clips confound active and passive coping
- *Campanella Lego tasks.* The descriptor says the Lego tasks were "created to mimic manufacturing processes ... as well as to create the mental strain that employees may experience". They are time-limited (10 min, 5 min), and the third requires counting backwards aloud from 180. The authors label all tasks "1 for stress". Meanwhile, the Campanella "stress" class (backward subtraction) "does not have a time constraint" and has no stated evaluator. Stress vs Lego is therefore two cognitive/manual stressors, not "a stressor versus a non-evaluative manual task" (line 271). The one-minute CV presentation, the dataset's only overtly evaluative task, is absent from Table I; please say how it was labelled.
- *Stress-Predict hyperventilation.* The Hyperventilation Provocation Test is anxiogenic by design, and hypocapnia directly perturbs HR, pulse amplitude and HRV. Calling it "a respiratory manoeuvre with no evaluative component" (line 82) is correct about evaluation, but "non-stress arousal" understates its aversiveness. It also yields 1.7 windows per subject.
- *EPM-E4 clips.* These are passive, seated viewing. Passive stimulus intake and active task engagement produce different cardiac patterns (Obrist's active/passive coping distinction). The separability of fear clips (0.78) is therefore as consistent with "active task vs passive viewing" as with "a distinct cardiac response to threat films" (line 271). Anger has one 79-s window per subject (33 windows), which cannot support the null in the abstract.

**Fix:** Add a column to Table I grading each state on the Dickerson & Kemeny components: social-evaluative threat, uncontrollability, time pressure, active vs passive, speech, hand motion. Rename the probe classes neutrally (e.g., "time-pressured manual task"). Offer the active/passive-coping explanation for fear. Drop anger from the abstract's null list, or flag it as underpowered.
- **Severity**: Major | **Evidence Anchor**: `text: §VI line 271 "a subtraction task from a Lego task, or a stressor from hyperventilation"` | **Confidence**: 4 — Campanella and Stress-Predict descriptors read in full; coping-mode point from the psychophysiology literature

### W3: The matched-arousal test conditions away the channel through which a wrist stress response would appear
The arousal index is the mean of per-subject z-scored HR and tonic EDA (line 97). At the wrist, those two quantities are the peripheral expression of an acute stress response. Conditioning on them leaves only residual information such as HRV shape, phasic EDA and SD terms. A within-bin AUROC near 0.5 is then predicted both by "the detector learned only arousal" and by "evaluative threat has no wrist signature beyond sympathetic magnitude". The test cannot tell these apart. The paper's own reference row shows this: stress vs baseline/rest, which the detector solves at BA 0.78–0.92, gives within-bin AUROC 0.584 (Table IV). The test therefore does not single out evaluative threat as unlearned; it would give the same answer for any contrast. The feature-disjoint rerun does not address this, because the removed features are still near-collinear with remaining EDA features.
**Fix:** (a) State explicitly what peripheral signature evaluative threat would be expected to show on E4 channels. The psychophysiology literature places the discriminating marker in the HPA axis (cortisol) and, for cardiovascular threat vs challenge, in impedance-derived measures. Neither is available at the wrist, so a wrist null is the prior expectation, not a finding about the detector. (b) Recast the conclusion as "wrist HR/HRV/EDA carry no detectable stressor-specific information beyond sympathetic magnitude in these paradigms" and make the stress vs baseline/rest reference row central to the interpretation. (c) Optionally, match on HR only (as Kwon et al. did) and on EDA only, and report both. Channel-specific matching is more interpretable than a composite.
- **Severity**: Major | **Evidence Anchor**: `table: Table IV, row "Baseline and rest (reference)" within-bin 0.584` | **Confidence**: 4 — conceptual; not a claim about test implementation

### W4: Kwon et al. (2026) already tested stress-specific physiology against HR arousal, which contradicts the gap claim
The Introduction says whether a detector "has learned anything beyond arousal has not, to our knowledge, been tested" (line 33). Related Work (lines 47–49) says no study asked whether stress is separable at equal arousal. Kwon et al. (2026), cited here only for transfer, state in their abstract that they "tested whether binary stress-versus-non-stress performance reflects stress-specific physiology rather than heart-rate (HR) arousal". Their §2.5 uses within-subject iso-HR matching. Their result (§3.6) points the other way: on WESAD, all-feature AUROC falls only from 0.955 to 0.938 after HR matching, and "the matched non-cardiac signal is strong only for laboratory social-evaluative stress (WESAD)". The author's own gap scan (`contribution-gap-scan.md` line 27) calls this "the closest existing 'specificity control'". The present paper's novelty lies elsewhere: non-stress comparator states rather than baseline, a composite index, five datasets. That difference must be argued, and the apparent disagreement with Kwon's WESAD result must be reconciled, probably through HR-only vs HR+EDA matching (see W3).
**Fix:** Add a paragraph in §II-C on Kwon's iso-HR design and result. Remove "has not ... been tested" and state the difference precisely. Report HR-only matching on WESAD stress vs baseline for direct comparison.
- **Severity**: Major | **Evidence Anchor**: `text: §I line 33 "Whether a detector trained on stressor labels has learned anything beyond arousal has not, to our knowledge, been tested"` | **Confidence**: 5 — Kwon full text checked in `sources/novelty/`

### W5: "Stress" is never defined, no manipulation checks are reported, and "stressor potency" is inferred from the same instrument
The paper treats "stress" as protocol stage and "evaluative stress" as a construct, but never defines either. It never mentions the HPA axis or cortisol, and none of the five datasets offers an endocrine check. The conclusion that "the ceiling is stressor potency" (abstract, line 280) rests on smaller EDA/HR effect sizes in PhysioNet and Stress-Predict (line 101). The F4 "non-responder" filter defines responders by HR/EDA change. Both define stress intensity by the arousal the detector measures, which is circular. The paper's own self-report analysis cuts against the potency reading: in PhysioNet, Stroop (mean self-rated rise 0.7/10) is detected as well as arithmetic (2.5), and self-rated intensity does not predict misses (Table V, line 268). The descriptors document weak evaluative content for three of the five "stress" classes: Stress-Predict's interviewers "were friendly and kind" per the author's verification notes, Campanella's subtraction has "no time constraint", and PhysioNet's stressors are seated computer tasks.
**Fix:** (a) Define stress in the Introduction: an appraisal-based definition plus the operational protocol-stage definition, and state which is being detected. (b) Add a table of the manipulation checks available per dataset: WESAD self-reports, UBFC-Phys anxiety scores, PhysioNet per-stage 1–10 ratings, EPM-E4 arousal ratings; Stress-Predict and Campanella have none. Use them to show that comparators are in fact less stressful than the stress class. If they are not, the specificity panel measures something else. (c) Replace "stressor potency" with "physiological response magnitude" unless an independent measure supports potency, and reconcile it with the self-report null. (d) Cite Carroll et al. (2017) on blunted reactivity when discussing non-responders.
- **Severity**: Major | **Evidence Anchor**: `text: §VII line 280 "the within-dataset ceiling appears to be set by the stressor's potency and by subjects who do not respond"` | **Confidence**: 4 — descriptor facts from the author's full-text notes

### W6: EDA processing is unspecified and has no artefact handling, although the key comparators involve hand and arm motion
§IV-A names the EDA features but does not give the decomposition method, filter, SCR onset/amplitude threshold, or any quality control. The author's verification notes confirm that "no EDA artefact rejection is applied". This is a physiological-validity issue, not a reporting nicety. The comparators that drive the headline are the Lego manual tasks, speech with gesturing, and hyperventilation. All of them produce electrode-contact and pressure artefacts in wrist EDA that mimic SCRs and inflate phasic features and SCR counts. 63% of Campanella windows have no usable pulse, so the arousal index there is mostly tonic EDA. A within-bin AUROC below 0.5 for Lego (0.355) means Lego windows look more stress-like than stress windows at matched arousal. Motion-inflated phasic EDA is a plausible cause, and the paper does not discuss it. At 4 Hz, the E4 EDA also resolves SCR onset and rise-time poorly.
**Fix:** Report decomposition method and parameters, following the SPR EDA publication recommendations (Boucsein et al., 2012). Apply a published wrist-EDA quality rule: Kleckner et al. (2018), which the author's notes already propose (0.05–60 µS range, ±10 µS/s slope, 5-s padding), or Taylor et al. (2015). Report invalid and flat-signal (non-responder) rates per dataset and per state. Rerun Tables III–IV on artefact-screened windows, and discuss the sub-0.5 Lego AUROC.
- **Severity**: Major | **Evidence Anchor**: `absence: Methods §IV-A — expected EDA decomposition method, SCR criteria and artefact/quality rule; checked §IV-A, §IV-B, §VI Limitations, Table captions` | **Confidence**: 5 — E4 EDA processing expertise; confirmed by the author's notes

### W7: Comparator states are normalised against reference sessions that differ from the stress windows' sessions
Features are z-scored "using all windows of that subject's stress-protocol recording" (line 91). PhysioNet exercise sessions and EPM-E4 emotion sessions are separate recordings. EPM-E4 contains no stress protocol at all. If exercise and clip windows are scaled by their own session statistics, a clip's z-score is relative to an all-emotion session and a bout's to an all-exercise session. They are then not on the same physiological scale as stress windows, which were scaled against a session containing stress. This directly shifts false-stress rates in Table III and bin membership in Table IV. The domain consequence is that "equal arousal" in the matched test may not mean equal physiological activation.
**Fix:** State which reference each probe state uses. Rerun the panel for exercise and EPM-E4 with raw features, the paper's own non-transductive sensitivity variant, and report whether the conclusions hold. Report raw effect sizes (µS, bpm) alongside z-units for every state, including the 0.4 vs 2.2 z comparison on line 101.
- **Severity**: Major | **Evidence Anchor**: `text: §IV-B line 91 "using all windows of that subject's stress-protocol recording, without labels"` | **Confidence**: 3 — depends on implementation details not in the manuscript

### W8: The warm-up interpretation does not rule out anticipatory arousal or ambient acclimatisation, and one statement is physiologically unclear
Early-session rises in tonic EDA and wrist temperature are attributed to "sensor warm-up" (line 213). At least three mechanisms overlap:
- device and electrode settling: thermopile equilibration, and passive hydration under the band;
- the participant's acclimatisation to room temperature after arriving;
- anticipatory arousal before an announced stressor, well documented for the TSST.

The last is psychological, not a sensor artefact, and would change the recommended remedy. Distal skin temperature also falls under acute stress through vasoconstriction, which works against the warm-up rise. "Every subject's first minute of temperature reads 4–6 °C above skin" is unclear: ambient air is normally below skin temperature, so what does the E4 read above skin in the first minute? A rise of +6.5 °C over 30 min is implausible for skin and points to probe equilibration. If so, the paper should say it, and the claim then concerns the instrument.
**Fix:** Use EPM-E4 as a no-anticipation control. Its clock starts at recording and it has no stressor, so if temperature and tonic EDA follow the same curve over the first 10–20 min there, warm-up is supported over anticipation. Cite a mechanism for EDA settling; the author's notes contain Parry & Briganti on passive hydration. Clarify the "above skin" sentence. Say "settling (device and participant)" rather than "sensor warm-up" unless the EPM-E4 control isolates the device.
- **Severity**: Minor | **Evidence Anchor**: `text: §V-E line 213 "every subject's first minute of temperature reads 4--6 $^\circ$C above skin"` | **Confidence**: 4 — E4 temperature and EDA behaviour

### W9: The title and abstract overstate what was tested
"Evaluative Stress Does Not [transfer]" suggests a transfer experiment on evaluative stress. No such experiment exists. What transfers in Table II is the stress-vs-non-stress detector, including WESAD's TSST, which is the most evaluative stressor in the set. The abstract's "Wrist detectors transfer because they detect autonomic arousal" is an interpretation. Given W1–W3, the scoped claim is "stress detectors trained on protocol labels transfer, and cannot distinguish the stressors from other arousing tasks at equal HR/EDA level".
**Fix:** Retitle, e.g., "... Arousal Transfers Across Five Empatica E4 Datasets but Is Not Specific to Stress". Rephrase the last abstract sentences to match the scoped claim.
- **Severity**: Major | **Evidence Anchor**: `text: title line 15 "Arousal Transfers Across Five Empatica E4 Datasets, Evaluative Stress Does Not"` | **Confidence**: 4 — construct-validity judgement

### W10: Canonical wearable-stress and stress-psychophysiology references are missing
The paper makes a psychophysiological claim, but it cites no stress-physiology theory and neither of the two standard wearable-affect reviews. Specific gaps are listed under Missing Key References. At minimum: Schmidt et al. (2019), for the activity confound and feature inventory; Vos et al. (2023), for small datasets and weak validation; Dickerson & Kemeny (2004), for the components of social-evaluative threat; the TSST (Kirschbaum et al., 1993) and friendly-TSST literature, which show that removing evaluation leaves an HR response but abolishes cortisol. That literature is direct external support for the authors' thesis and should frame it. Also Boucsein (2012) and the SPR recommendations for EDA.
- **Severity**: Minor | **Evidence Anchor**: `absence: Related Work §II and bibliography — expected Schmidt 2019, Vos 2023, Dickerson & Kemeny 2004, TSST/f-TSST, EDA methodology references; checked §II-A–C, §VI, reference list` | **Confidence**: 4 — the author's own CLAUDE.md-linked notes also recommend Schmidt 2019 and Vos 2023

### W11: Several citations are represented imprecisely
- **Prajod et al. (2024), line 41.** "attribute transfer failure to stressor type, on chest ECG" leaves out that their failing pair also differed in stress intensity (author's evidence report: "Supported, with a design confound"). This matters because the present paper argues that stimulus potency sets the ceiling. The DOI in the bibliography (…3685741) differs from the one in the author's evidence report (…3685738). Verify it.
- **Sosa et al. (2026), line 47.** "the reviewed studies did not test classifiers on such states" is stronger than the author's note ("mostly did not"). Use "rarely".
- **Milstein & Gordon (2020), line 277.** "Wrist EDA agrees with palmar EDA at about r = 0.3" generalises a single condition (SCL r = 0.298 during conversation). State the condition, and add van Dooren et al. (2012) for site differences in skin conductance.
- **Akkaya (2026)** is cited from the abstract only. The bibliography says so, but the text (line 47) uses it as evidence without that caveat.
- **Severity**: Minor | **Evidence Anchor**: `text: §II-A line 41 "Prajod et al.~\cite{prajod2024} attribute transfer failure to stressor type, on chest ECG"` | **Confidence**: 4 — checked against the author's full-text notes

### W12: Terminology issues around cardiac indices
"Pulse rate, a weak vagal proxy" (line 33) is imprecise. HR is dually innervated. RMSSD and pNN50 are the vagally mediated indices, and on the wrist they are pulse-rate variability (PRV), not HRV (compare the cited Watanabe et al., "pulse rate variability"). SDNN from 60-s windows is ultra-short and poorly comparable with the standard 5-min measure. State this, citing Laborde et al. (2017) or Shaffer & Ginsberg (2017). Treating "arousal" as one-dimensional ignores autonomic space (Berntson et al., 1991). Hyperventilation and speech move HRV through respiration, not through sympathetic arousal.
- **Severity**: Minor | **Evidence Anchor**: `text: §I line 33 "the wrist device measures pulse rate, a weak vagal proxy"` | **Confidence**: 5 — standard psychophysiology terminology

### W13: Table I and the text disagree on PhysioNet stress tasks, and z-unit effect sizes are not comparable across datasets
Line 52 lists "opinion tasks" among PhysioNet stressors, and the descriptor labels "Real, and Opposite Opinion" as stress. Table I lists only "Stroop, TMCT, subtraction". State whether the opinion tasks are in the stress class. The effect sizes on line 101 (0.4 z vs 2.2 z) and "a fifth of the electrodermal response of the TSST" (line 271) are in per-subject z-units. Those units depend on each session's stress fraction and are not physiological magnitudes. Give µS and bpm as well.
- **Severity**: Minor | **Evidence Anchor**: `table: Table I, row "PhysioNet ... Stroop, TMCT, subtraction"` | **Confidence**: 4 — Hongn 2025 full text

### Detailed Comments

#### Literature Review
- **Coverage:** Strong on 2020–2026 ML transfer papers. Missing: stress psychophysiology (Dickerson & Kemeny; TSST/f-TSST), the two standard wearable-affect reviews (Schmidt 2019; Vos 2023), EDA methodology (Boucsein; SPR guidelines; Kleckner), and E4 validation beyond Milstein (Menghini et al., 2019; Schuurmans et al., 2020).
- **Integration quality:** §II-A synthesises transfer results well. The synthesis in §II-C is too thin for the paper's central claim.
- **Research gap argument:** Overstated. See W4, Kwon iso-HR.

#### Theoretical Framework
- **Appropriateness:** The arousal-vs-stress framing suits a wrist device, but it is used without its theory. Appraisal theory and the Dickerson & Kemeny model predict that the evaluative component shows up mainly in HPA output. A wrist null is therefore the prior expectation and should be framed that way.
- **Application depth:** "Arousal" is operationalised as HR + tonic EDA, which is the stress response itself (W3). Autonomic space and active/passive coping are not considered (W2, W12).
- **Alternative frameworks:** Biopsychosocial challenge/threat (cardiovascular pattern rather than magnitude) is the natural framework for "evaluative threat signature". The paper should say why the wrist cannot test it.

#### Academic Argument Quality
- **Factual accuracy:** UBFC-Phys control misdescribed (W1). Campanella Lego and Stress-Predict hyperventilation are stressors according to their authors (W2). PhysioNet opinion tasks inconsistent (W13).
- **Argument logic:** The matched-arousal null cannot tell "detector learned arousal" from "wrist carries only arousal" (W3). "Stressor potency" is circular (W5).
- **Terminology precision:** "non-evaluative", "warm-up", "HRV", "vagal proxy" (W1, W8, W12).

#### Contribution to the Field
- **Incremental contribution:** Five-dataset E4 LODO with audited labels; the specificity-panel concept; a quantified warm-up/time confound; exercise-negative guidance. Together these advance the understanding of what wrist stress benchmarks measure.
- **Positioning:** Needs explicit contrast with Kwon (2026) iso-HR and with Aydoğan (2026), which is already done well for exercise.
- **Overclaiming:** Title, abstract and §VI "has not learned evaluative threat" (W9, W3).

#### Missing Key References
- Schmidt, P., Reiss, A., Dürichen, R., & Van Laerhoven, K. (2019). Wearable-based affect recognition — a review. *Sensors*, 19(19), 4079. Activity confound and feature inventory.
- Vos, G., et al. (2023). Systematic review of generalisable ML for wearable stress monitoring. Citation details are in the author's evidence report (read as arXiv v3); use the version of record.
- Dickerson, S. S., & Kemeny, M. E. (2004). Acute stressors and cortisol responses: A theoretical integration and synthesis of laboratory research. *Psychological Bulletin*, 130(3), 355–391. Social-evaluative threat and uncontrollability as the active components.
- Kirschbaum, C., Pirke, K.-M., & Hellhammer, D. H. (1993). The 'Trier Social Stress Test'. *Neuropsychobiology*, 28, 76–81.
- Wiemers, U. S., Schoofs, D., & Wolf, O. T. (2013). A friendly version of the Trier Social Stress Test does not activate the HPA axis in healthy men and women. *Stress*. [UNVERIFIED volume/pages; search lead.] This is direct support for "speech without evaluation raises HR but not cortisol".
- Ringgold et al. (2025), ISPNE abstract, TSST vs f-TSST within subject (in the author's gap scan: "The TSST elicited greater cortisol and heart rate increases ... than the f-TSST"). Also the EmpkinS TSST/f-TSST dataset (OSF sh3xn), for the "decisive study" paragraph.
- Boucsein, W. (2012). *Electrodermal Activity* (2nd ed.). Springer; and Boucsein et al. (2012). Publication recommendations for electrodermal measurements. *Psychophysiology*, 49(8), 1017–1034.
- Kleckner, I. R., et al. (2018). Simple, transparent, and flexible automated quality assessment procedures for ambulatory electrodermal activity data. *IEEE Trans. Biomed. Eng.*, 65(7), 1460–1467.
- Taylor, S., et al. (2015). Automatic identification of artifacts in electrodermal activity data. *Proc. IEEE EMBC*.
- Menghini, L., et al. (2019). Stressing the accuracy: Wrist-worn wearable sensor validation over different conditions. *Psychophysiology*, 56(11), e13441. Still abstract-only in the author's notes; read in full before citing.
- Schuurmans, A. A. T., et al. (2020). Validity of the Empatica E4 wristband to measure heart rate variability (HRV) parameters: A comparison to electrocardiography (ECG). *J. Med. Syst.*, 44, 190.
- van Dooren, M., de Vries, J. J. G., & Janssen, J. H. (2012). Emotional sweating across the body: Comparing 16 different skin conductance measurement locations. *Physiol. Behav.*, 106(2), 298–304.
- Kreibig, S. D. (2010). Autonomic nervous system activity in emotion: A review. *Biol. Psychol.*, 84(3), 394–421. Anger vs fear autonomic profiles.
- Berntson, G. G., Cacioppo, J. T., & Quigley, K. S. (1991). Autonomic determinism: The modes of autonomic control, the doctrine of autonomic space, and the laws of autonomic constraint. *Psychol. Rev.*, 98(4), 459–487.
- Obrist, P. A. (1981). *Cardiovascular Psychophysiology: A Perspective*. Plenum. Active vs passive coping.
- Vinkers, C. H., et al. (2013). The effect of stress on core and peripheral body temperature in humans. *Stress*. [UNVERIFIED volume/pages; search lead.]
- Carroll, D., et al. (2017). *Neurosci. Biobehav. Rev.*, 77, 74–86 (in the author's gap scan). Blunted reactors.
- Laborde, S., Mosley, E., & Thayer, J. F. (2017). Heart rate variability and cardiac vagal tone in psychophysiological research. *Front. Psychol.*, 8, 213.

### Questions for Authors
1. How were the Campanella one-minute CV presentation and the rests between Lego tasks labelled? Were they excluded, or pooled into a class?
2. Are PhysioNet "Real" and "Opposite Opinion" tasks in the stress class (line 52 vs Table I)?
3. For exercise and EPM-E4 windows, which recording supplies the per-subject z-score statistics?
4. Do the available self-reports (UBFC-Phys anxiety scores; PhysioNet per-stage 1–10; WESAD SSSQ/STAI; EPM-E4 arousal ratings) confirm that each comparator state is less stressful than the stress class for the same subjects or group?
5. With HR-only matching, as in Kwon et al., does WESAD stress vs baseline stay separable in your pipeline? If so, how do you reconcile that with the 0.584 reference row?
6. What fraction of Lego and speech windows would a Kleckner-type EDA quality rule reject, and does the Lego AUROC below 0.5 survive screening?
7. Does EPM-E4, which has no stress anticipation, show the same first-10-min temperature and tonic-EDA rise?
8. In how many subjects per dataset is wrist EDA flat (non-responsive), and how are they handled?

### Minor Issues
- Line 33: "HRV" measured from BVP should be "PRV" at least once, with a note on the convention.
- Line 97: "An AUROC near 0.5 means the model has learned nothing beyond arousal magnitude". Scope this to "beyond the arousal index", given W3.
- Line 161: "fall by 0.05–0.10" for EPM-E4. Give CIs for the emotion rows in Table III, as done for the other rows.
- Table III: the anger row (33 windows, 1 per subject) should carry a small-cell flag like hyperventilation.
- Line 213: the sentence starting "subject-centred skin temperature rises at 0.02–0.08 °C/min ..." is one very long sentence mixing four datasets. Split it.
- Line 275: "No public E4 dataset separates speech from evaluation within subject" is correct for E4. Say that non-E4 TSST/f-TSST data exist (EmpkinS) and that Richer et al. (2024) used an f-TSST control.
- Line 277: the limitation "a null on the evaluative-threat signature may be instrument failure" is the right caveat. Move it from Limitations into the main Discussion argument.
