# Journal-Fit Review Card: IEEE JBHI

**Seat:** Journal-Fit Reviewer (Associate Editor, wearable sensing and affective computing)
**Manuscript:** "What Do Wrist-Worn Stress Detectors Detect? Arousal Transfers Across Five Empatica E4 Datasets, Evaluative Stress Does Not" (`paper/main.tex`, 315 lines; `main.pdf`, 9 pages)
**Recommendation signal:** **Major Revision**

### Reviewer Confidence
4. Journal fit, positioning and structure are within my competence. I checked a sample of numbers against the committed tables in `outputs/tables/jbhi_v2/`. I leave detailed statistics to the methodology seat.

### Calibration Status
`NOT_CALIBRATED`

### Criterion-Bound Judgements
| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Journal scope fit | JBHI aims and scope (wearable sensing, health informatics) | MEETS | main.tex L15, L25, L52 | This is wrist physiological sensing with a benchmark and an evaluation methodology, which is core JBHI material. | The link to health outcomes is thin (see W6). | no |
| Originality against prior cross-dataset work | Seat focus | PARTLY_MEETS | L41–49; bib `aydogan2026`, `kwon2026`, `mishra2020`, `sosa2026` | The specificity panel and the matched-arousal test are new as a combination. The "stress or arousal" framing is already in the title of Aydoğan 2026, and small transfer cost was already shown by Mishra 2020. | This relies on the manuscript's own literature coverage. I did not run an independent search. | yes: the novelty claim has to be repositioned |
| Significance of the "arousal not stress" thesis without a new method | Seat focus | PARTLY_MEETS | L199–201, Table IV, L271 | Negative and diagnostic contributions are publishable in JBHI if they are decisive. At present the key test is partly circular, and there is no index-only baseline (W1, W2). | none identified | yes |
| Clarity and structure | JBHI author guidance / seat focus | PARTLY_MEETS | L37 (five contributions), L241, L213, Table V | The paper is dense and written like a summary. Too many results are packed into a 9-page paper, it names CSV files in the prose, and some sentences are garbled. | none identified | yes (revision needed, not rejection) |
| Title and abstract accuracy | Seat focus | DOES_NOT_MEET | L15, L25 | "Evaluative stress does not [transfer]" misstates the finding, and the abstract misreports the hyperventilation cell. | none identified | yes |
| Internal consistency of reported numbers | Seat focus (spot check) | PARTLY_MEETS | Table III vs Table V vs L161; Table IV vs Table II | Some numbers conflict across the paper (W5). | Spot check only | no (fixable) |
| Length vs JBHI norms | JBHI regular paper limits | MEETS | main.pdf 9 pp. | The paper is inside the regular-paper limit, but check the current page-charge threshold. The density is the problem, not the page count. | Current JBHI page policy not verified here | no |

### Summary Assessment
The manuscript rebuilds a wrist-E4 stress-detection pipeline across five public datasets. It audits labels against the descriptor papers, uses subject-held-out and leave-one-dataset-out evaluation with corrected tests, and reports three things. First, cross-dataset transfer costs only 0.015–0.05 balanced accuracy with HR/HRV/EDA features. Second, a fixed detector labels many non-stress arousal states as stress, and at matched arousal it cannot separate evaluative stress from non-evaluative tasks, hyperventilation, Lego or anger clips, although it does separate exercise. Third, time in session and sensor warm-up confound baseline-vs-stress designs.

The empirical work is careful. The self-correction of an earlier leaky pipeline (L84) is exemplary. The specificity panel and the position-matched rule are practical reporting contributions that JBHI readers would use.

As an editor, I have three concerns about the paper as submitted:
- The central thesis is inferred mainly from a matched-arousal test whose index is built from the model's own inputs, without the one direct test the thesis implies: does a label-free arousal index alone match XGBoost under LODO?
- The originality claim does not engage with the psychophysiology literature on autonomic specificity, or with Aydoğan 2026, whose title already asks "stress or arousal?".
- The paper tries to be a benchmark, a confound study, an error budget and a negative-results catalogue at once, and it reads like a lab notebook rather than a journal argument.

All three can be fixed with a revision. None calls for rejection.

### Strengths
1. **Leakage-controlled, auditable evaluation**: The paper uses subject-grouped splits, LOSO and LODO on held-out subjects, and Nadeau–Bengio corrected tests with Holm correction (L94). It also openly reports the 94% to 49% collapse of an earlier record-wise pipeline (L84, `leakage_check.csv`). This is the evaluation standard JBHI should reward. [`text: L84, L94`]
2. **Label audit against descriptor papers**: Seven concrete corrections are listed (L82). The paper is honest that the fixes are a correctness contribution and not the cause of the small transfer cost (L137). That restraint strengthens credibility. [`text: L82, L137`]
3. **Missing-channel diagnosis**: The UBFC-Phys drop to 0.589 BA, with AUROC at 0.931 and an oracle threshold recovering 0.910, is shown to come from absent temperature through XGBoost's missing-value branch (L133). This is a useful, generalisable warning for multi-dataset wearable work. [`table: Table II; text: L133`]
4. **Specificity panel as a reporting norm**: Scoring one fixed detector on untrained arousal states under three training conditions (L97, Table III) is simple and portable. The analogy to assay cross-reactivity (L273) is apt and could be adopted beyond this paper. [`table: Table III; text: L273`]
5. **Time-in-session and warm-up quantification**: The time-only classifier beating physiology on PhysioNet (0.83 vs 0.78) and Stress-Predict (0.95 vs 0.70) is a striking, easily replicated finding (L211). It is well explained by the per-dataset warm-up analysis (L213), with WESAD as a natural contrast. [`figure: Fig. 3; text: L211–213`]
6. **Positive control in the matched-arousal test**: Exercise remains separable (0.892, and 0.91 or 0.82 with single-channel indices, L199). This shows the test can detect separability when it exists, so the nulls are not an artefact of the procedure alone. [`table: Table IV`]

### Weaknesses
1. **The thesis lacks its most direct test: an arousal-index-only detector**
   - What is wrong: The claim "wrist detectors transfer because they detect autonomic arousal" (L25, L271, L280) predicts that a label-free arousal index, the paper's own mean of per-subject z-scored HR and tonic EDA (L97), used alone as a score, should come close to the XGBoost LODO BA/AUROC in Table II. This comparison is not reported. The committed probe tables (`contribution_probes/arousal/partA_matched_arousal.csv`) have no such row, and I found no script that computes it.
   - Fix: Add an "arousal index only" row to Table II for all five targets (within, other-4, all-5; BA and AUROC; threshold learned on source). If it reaches within about 0.03–0.05 of XGBoost, the thesis is shown directly and the matched-arousal test becomes confirmatory. If it does not, the thesis must be narrowed to "the detector does not separate stress from arousal-matched non-stress states".
   - **Severity**: Critical | **Evidence Anchor**: `text: L25, L97, L271; table: Table II` | **Confidence**: 4. The implication is logical; the absence was checked in the repo.
2. **The matched-arousal test is partly circular, and its reference row undercuts the reading**
   - What is wrong: The index is built from two model features (L97). Stratifying on a close proxy of the score pushes within-bin AUROC toward 0.5 by construction. The "Disjoint" variant (L201) helps, but it removes only `eda_mean` plus the two index features, while correlated EDA and HR features remain. The reference row "Baseline and rest" is itself only 0.584 within bins (Table IV). Read literally, the detector also barely separates stress from rest at matched arousal. That supports "arousal only", but it also means the test cannot tell "no evaluative signature" from "within-quintile variance is too small to show any signature". Within-bin AUROCs below 0.5 (Lego 0.355, anger 0.390) also need explaining; the text interprets them as "nothing beyond arousal", but they suggest inverse ranking.
   - Fix: (a) Define the index from channels or features disjoint from the model in the main analysis, for example an index from EDA and a model from cardiac features only, and the reverse, and make that the headline. (b) Report the within-bin spread of the index relative to the between-class difference, so readers can judge power. (c) Discuss AUROC < 0.5 explicitly. (d) Leave statistical depth to the methodology seat, but state the thesis in terms of what this test can and cannot exclude.
   - **Severity**: Major | **Evidence Anchor**: `table: Table IV; text: L97, L199–201` | **Confidence**: 3. This is an editorial reading; the formal statistics belong to the methodology seat.
3. **Originality is not positioned against psychophysiology or against Aydoğan 2026**
   - What is wrong: Related Work (L39–49) treats "has the detector learned anything beyond arousal" as untested (L33, L49). But the question of whether stress and other states have distinct autonomic signatures is a long-standing debate in psychophysiology, where autonomic specificity versus general arousal has been argued for decades (e.g., Kreibig 2010, Biol. Psychol., on autonomic specificity in emotion). The social-evaluative-threat construct the paper relies on comes from the stress literature (e.g., Dickerson and Kemeny 2004, Psychol. Bull.). Neither is cited. Aydoğan and Villagra Povina 2026, titled "Stress or arousal? Exercise confounding…", already frames the question and found the PhysioNet exercise result (L44, L161). The paper's exercise finding largely replicates them, so the novelty rests on the non-exercise states. The two review papers the field standardly cites, Schmidt et al. 2019 (the wearable affect/stress review) and Vos et al. 2023, are absent from the bibliography.
   - Fix: Add a Related Work subsection "Arousal versus stress in psychophysiology" with the autonomic-specificity and social-evaluative-threat literature. State explicitly that the exercise result replicates Aydoğan 2026 and that the new contribution is the non-exercise panel plus the matched-arousal protocol. Cite Schmidt 2019 and Vos 2023 for the known issues (small datasets, activity confound, validation practice). Rewrite L49 to "not tested for wrist detectors across a panel of non-stress states" rather than "not tested".
   - **Severity**: Major | **Evidence Anchor**: `text: L33, L44, L49; bib: aydogan2026` | **Confidence**: 4
4. **Title and abstract misstate the main finding**
   - What is wrong: The title says "Evaluative Stress Does Not [transfer]". The paper does not show that evaluative stress fails to transfer. Stress detection transfers at 0.015–0.05 cost (Table II). What it shows is that the detector cannot separate evaluative stress from arousal-matched non-evaluative states. In the abstract (L25), hyperventilation is reported as "about 50%" under the "adding a state to the training negatives removes…" sentence. But Table III lists it as 0.42 (dataset absent) rising to 0.55 (added as negative), so adding it increases false alarms. The abstract also crams 15+ numbers into one paragraph, which is hard to read.
   - Fix: Retitle, for example "What Do Wrist-Worn Stress Detectors Detect? Cross-Dataset Transfer and Arousal Specificity Across Five Empatica E4 Datasets". Or keep the question and drop the second clause. Report hyperventilation correctly (0.54 all-absent to 0.55 added, per `partA_panel_compact.csv`). Cut the abstract to about 5 key numbers: transfer cost, exercise false alarms, matched-arousal range, time-only classifier, error budget.
   - **Severity**: Major | **Evidence Anchor**: `text: L15, L25; table: Table III` | **Confidence**: 5
5. **Numbers conflict across tables and text**
   - What is wrong:
     - (a) Exercise false-stress with exercise in training is 8–10% in Table III and L161, but "4–6%" in Table V (L263). These probably come from different experiments (the novelty run vs the specificity panel), but the paper does not say so.
     - (b) Methods (L97) defines three training conditions, but Table III shows only two. The middle condition ("all datasets in training, state absent") is the correct comparator for "adding the state changes nothing", and the hyperventilation cell flips interpretation depending on which column is used (0.415 / 0.538 / 0.547 in `partA_panel_compact.csv`).
     - (c) Stress-Predict within-dataset BA is 0.697 in Table II but 0.710 in Table V (budget, LOSO), and PhysioNet external is 0.746 vs the budget's 0.746. The splitting protocol behind each table needs to be stated in its caption.
     - (d) L101 reports the modality comparison on the baseline-vs-stress task of a three-dataset table with raw features, a task the paper itself calls confounded (L94, L211). As the opening result, this is confusing.
   - Fix: Show all three panel conditions in Table III. Reconcile or label the 4–6% vs 8–10% figures. State the evaluation protocol (20 splits vs LOSO) in every table caption. Move the three-dataset modality result to a supplement, or re-run it on the main task.
   - **Severity**: Major (the paper is fixable, but a reader who sees conflicting numbers loses trust) | **Evidence Anchor**: `table: Table III, Table V; text: L97, L101, L161, L263` | **Confidence**: 5 (checked against the CSV)
6. **Health-informatics significance is under-argued**
   - What is wrong: JBHI readers will ask what this means for deployed stress monitoring, for example just-in-time interventions, clinical populations and false-alarm burden. The Discussion (L271–277) is about reporting norms and a future lab study. There is no estimate of what a 42–70% false-stress rate on everyday arousal states means for daily-life alert rates, and no discussion of clinical use cases where arousal detection is itself enough.
   - Fix: Add a short "Implications for deployed monitoring" paragraph. Translate the panel rates into an expected false-alert rate for a plausible day (exercise, meetings, commuting) and say when an arousal detector is still clinically useful (for example agitation monitoring, or anxiety where arousal is the target). This turns a negative result into guidance.
   - **Severity**: Major | **Evidence Anchor**: `text: L271–277, L280` | **Confidence**: 4
7. **Structure is overloaded: five contributions and many side analyses**
   - What is wrong: The paper claims five contributions (L37) and runs about ten analyses in 9 pages. Results paragraphs such as L241 (error budget, about 350 words, around 30 numbers) and L213 (warm-up) are very dense. The negative-results table (Table V) mixes domain adaptation, few-shot, self-report, traits and exercise. The paper names 12 CSV files inline (`\texttt{...csv}`), which is not JBHI style.
   - Fix: Organise the paper around two contributions: (i) the LODO benchmark with audited labels, and (ii) specificity and matched arousal, with time and warm-up as a confound rule. Move the error budget detail, the responder sensitivity, the chest-ECG bound and most of Table V to supplementary material. Replace inline CSV names with one "Reproducibility" paragraph mapping figures and tables to files.
   - **Severity**: Major | **Evidence Anchor**: `text: L37, L213, L241; table: Table V` | **Confidence**: 5
8. **Small probe cells carry a field-level claim**
   - What is wrong: Anger has 33 windows (about 1 per subject; subject median false-stress 0.0), hyperventilation 53 windows (1.7 per subject), and the UBFC control group 8 subjects (Table I, L161). The paper admits this (L161, L277), yet the abstract and conclusion list anger and hyperventilation alongside Lego as equal evidence.
   - Fix: In the abstract and conclusion, rest the claim on the adequately sized cells (Lego, UBFC control, with caveat) and present anger and hyperventilation as exploratory. Alternatively, add the candidate E4 datasets the authors are aware of to enlarge those cells.
   - **Severity**: Minor | **Evidence Anchor**: `table: Table I, Table III; text: L25, L280` | **Confidence**: 5

### Detailed Comments

#### Journal Fit
- The paper is a good fit for JBHI's wearable and affective-sensing readership. It resembles an evaluation-methodology paper more than a method paper, and JBHI does publish those when the insight is decisive and actionable. The specificity panel and the position-matched rule are actionable. The arousal thesis is not yet decisive (W1, W2). If the authors decline to add the index-only baseline, *IEEE Transactions on Affective Computing* or *Psychophysiology* might suit the conceptual framing better. With W1–W3 addressed, JBHI is the right venue.

#### Originality
- Pieces that are new: a five-dataset E4 LODO with descriptor-audited labels; the non-exercise specificity panel; the matched-arousal protocol; quantified warm-up overlap with baseline. The exercise confound and the small transfer cost (Mishra 2020) are not new, and the paper should say so up front (W3). "Arousal not stress" is an old psychophysiology idea. The new part is demonstrating it quantitatively for deployed wrist classifiers, and the paper should claim exactly that.

#### Significance
- Without a new method, significance depends on the paper changing practice. The proposed reporting norms (L273) are the most likely route to citation. I recommend making them a numbered, checklist-style box ("Specificity reporting checklist for wearable stress detectors") so they are easy to adopt and cite.

#### Structural Coherence
- The storyline (transfer, then what transfers, then why the ceiling) is sound. The execution buries it. See W7. Section V-A (L101) opens on a confounded task and a different table, which breaks the flow into Section V-B.

#### Title & Abstract
- See W4. The keywords are fine; consider adding "evaluation methodology".

#### Conclusion
- The conclusion (L280) is aligned with the results, apart from the "evaluative stress" wording. "Honest abstention" is mentioned but never tested or defined in the paper; either remove it or add a one-line pointer. The two proposed future studies are a good close, but they repeat L275 almost verbatim; keep one.

### Questions for Authors
1. What BA and AUROC does the label-free arousal index alone achieve under LODO for each target (W1)?
2. Which experiment produced 4–6% exercise false-stress (Table V) versus 8–10% (Table III)?
3. How should readers interpret within-bin AUROC well below 0.5 (Lego 0.355, anger 0.390)? Is the model ranking these states above stress at equal arousal?
4. For the "all datasets in training, state absent" condition, what is the full Table III column, and does it change any conclusion?
5. Would the thesis survive with the index computed from a channel entirely absent from the model in the main analysis, not only in a variant?

### Minor Issues
- L18: the placeholder e-mail addresses `[email]@utsa.edu` must be filled in.
- L213: the garbled sentence "…rises at 0.02–0.08 °C/min in the first 10 min on all four datasets with a continuous clock keeps rising past 20 min on WESAD…" needs splitting. Also say which "four datasets" are meant, since there are five stress datasets plus EPM-E4.
- L161: the sentence "and the external PhysioNet drop is significant (BA 0.746 to 0.583…)" is misplaced in the panel paragraph and its referent is unclear. State that it is the LODO BA with exercise windows included as negatives.
- L302: the reference `akkaya2026` is marked "(abstract only)", yet it supports a substantive claim (L47, L268). Read the full text, or soften the claim and remove the annotation from the bibliography.
- Table I: Stress-Predict stress lists 33 subjects, but the text says 34 after exclusion. Explain the missing subject.
- Table IV: the header "Disjoint" needs a one-word gloss in the caption itself, not only in the footnote.
- L94: "one significant difference in 72 comparisons" and L101 "108-configuration" refer to different counts (72 test rows vs 108 configuration rows in `tuned_baselines_summary.csv`). Make that explicit.
- Fig. 2 and Fig. 3 are figure* spreads in a 9-page paper. Check that the panels stay legible at print size.
- British spelling ("normalisation", "personalisation") is fine, but IEEE house style is American. Harmonise before submission.
