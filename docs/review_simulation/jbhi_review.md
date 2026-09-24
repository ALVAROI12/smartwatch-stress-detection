# Editorial Decision Package: simulated IEEE JBHI review, Round 1

- **Manuscript**: "What Do Wrist-Worn Stress Detectors Detect? Arousal Transfers Across Five Empatica E4 Datasets, Evaluative Stress Does Not" (`paper/main.tex`, HEAD `3922e90`)
- **Decision date**: 2026-09-23 | **Mode**: standard synthesis | `calibration_status: NOT_CALIBRATED`
- **Sprint contract**: none was used, so the mechanical v3.6.2 steps (scoring matrix, failure-condition evaluation, `editorial_decision=` lines) were skipped. No cross-model blind check was run (`ARS_CROSS_MODEL` not set).
- **Inputs**: `review_EIC.md` (Journal-Fit), `review_R1_methods.md`, `review_R2_domain.md`, `review_R3_perspective.md`, `review_DA.md`. Item IDs below (for example R1-W6, DA-C1, EIC-W4) point into those files.

## Review panel provenance and correlated-error disclosure

No typed `review-panel-provenance/1.0` artifact exists. The axes below are what the dispatcher reported; I did not verify them.

| Axis | Status |
|---|---|
| Role-separated | true |
| Separate invocation contexts | true (reported) |
| Blind to peer outputs | true (reported) |
| Model family distinct | **false**: all five seats and this synthesiser are Claude |
| Provider distinct | false |
| Human reviewer distinct | false: no human seat |

**Correlated-error disclosure.** All five seats ran on the same model family. Agreement between seats is therefore weaker evidence than agreement between independent human referees: shared blind spots and shared false beliefs would appear as "consensus". Treat CONSENSUS labels as agreement between roles, not as independent replication. I spot-checked the load-bearing factual claims against the repository (listed in §3), and those checks are the stronger evidence.

**Manuscript version.** The reviewers' quotes match the text after `50658aa`. For example, R1-W5 quotes L97's "or tonic EDA alone where heart rate is missing", which that commit added. Most numeric points therefore still apply to HEAD. I checked each numeric point against the current `main.tex`. Only the items marked **[already addressed]** were fixed by later commits.

---

## 1. Decision: **Major Revision**

| Seat | Recommendation | Confidence |
|---|---|---|
| Journal-Fit (EIC) | Major Revision | 4 |
| R1 Methodology | Major Revision | 4 |
| R2 Domain (psychophysiology / E4) | Major Revision | 4 |
| R3 Perspective (digital health / translation) | Major Revision | 4 |
| Devil's Advocate | findings only: 3 CRITICAL, 9 MAJOR, 5 MINOR | per finding |

**Rationale.** All four scoring seats independently recommend Major Revision, and none recommends rejection. They agree on the strengths: subject-grouped LODO with paired arms, the descriptor-level label audit, the UBFC-Phys missing-temperature diagnosis, the time-only and warm-up confound, and the specificity-panel idea (EIC-S1–S5, R1-S1–S5, R2-S1–S5, R3-S1–S5, DA "Observations"). They also agree that the paper's central interpretive claim is stronger than its instruments support. The headline says "evaluative stress does not [transfer]" and "the ceiling is stressor potency". It rests on three things: (a) a matched-arousal test that barely separates even the contrast the model was trained on (reference within-bin AUROC 0.584, Table IV L185); (b) non-significance after Holm correction, read as absence of an effect; and (c) comparator states that, by their own descriptors, are partly stressors. Two DA CRITICALs are validated and one is partly valid (§3), so Accept and Minor are ruled out. Rejection is not warranted either. Every blocking item can be fixed by reanalysis of data the author already holds plus a narrower claim. No seat asks for new data collection as a condition. The descriptive and benchmark results (contribution 1, the exercise result, the time and warm-up confound) survive every attack, including the DA's. If the author declines the reanalysis, the EIC suggests *IEEE Trans. Affective Computing* or *Psychophysiology* would suit the conceptual framing better (EIC Journal Fit).

### Blocking issues

| Ref | Blocking issue | Sources | Evidence anchor | Resolved by |
|---|---|---|---|---|
| B1 | The matched-arousal test cannot tell "the model learned only arousal" from "stress shows at the wrist mainly as arousal". There is no calibrated sensitivity, and the headline "Disjoint" column is not channel-disjoint. | DA-C1; EIC-W1, W2; R1-W2, W6; R2-W3 | `table: Table IV, reference row 0.584 [0.54, 0.64] vs UBFC 0.577 [0.50, 0.67]`; `code: matched_arousal_hardening.py:135` | REV-3, REV-4 |
| B2 | Null results are presented as evidence of absence. There are no equivalence margins or power, and no CIs on the headline LODO costs. | DA-C2; R1-W1, W7, W8; R3-W4; EIC-W2(d) | `text: L25 "none different from chance after Holm correction"`; `table: Table II` (no CIs) | REV-5 |
| B3 | The title, abstract and conclusion claim more than was tested: "evaluative stress does not", "because", "stressor potency", "wrist-worn stress detectors". Part of the problem is that comparators are misdescribed. | EIC-W4; R2-W1, W2, W5, W9; R3-W3; DA-C2, C3, M1, M2, M6, M7 | `text: L15, L25, L161 "without an evaluating panel", L280` | REV-2, REV-8, REV-10 |

---

## 2. Consensus and disagreement

Consensus is counted over the four scoring seats (EIC, R1, R2, R3). The DA is reported alongside but not counted. "Silent" means the seat did not address the point; it is not opposition.

### Points of agreement

| ID | Sub-claim | Seats | Label |
|---|---|---|---|
| SC-1 | The title and abstract claim ("evaluative stress does not [transfer]", "cannot separate") goes beyond the evidence. No transfer experiment on evaluative stress exists; nulls are read as proof. | EIC-W4, R1-W1, R2-W9, R3-W4 (+DA-C2) | **CONSENSUS-4** |
| SC-2 | Within-bin AUROCs below 0.5 (Lego 0.355, anger 0.390) are called "chance" but indicate inverse ranking or an artefact, and need explaining. | EIC-W2(c)/Q3, R1-W5, R2-W6, R3-W4 (+DA-M5) | **CONSENSUS-4** |
| SC-3 | Small cells (anger 33 windows, one per subject; hyperventilation 53 windows, 1.7 per subject; UBFC control 8 subjects) carry a field-level claim. Demote them to exploratory. | EIC-W8, R1-W7, R2-W2, R3-W1 | **CONSENSUS-4** |
| SC-4 | The matched-arousal index conditions on the model's own dominant inputs. The reference row (0.584) shows the test flattens every contrast. | EIC-W2, R1-W2, R2-W3 (+DA-C1); R3 silent | **CONSENSUS-3** |
| SC-5 | The null inference needs an equivalence margin or a minimum detectable effect. | R1-W1, R3-W4, EIC-W2(d) (+DA-C2); R2 silent | **CONSENSUS-3** |
| SC-6 | The garbled warm-up sentence at L213 needs rewriting. | EIC minor, R1 minor, R2 minor; R3 silent | **CONSENSUS-3** (still present at HEAD) |
| SC-7 | Novelty is mispositioned. It needs to engage psychophysiology (autonomic specificity; Dickerson & Kemeny), Aydoğan 2026 and Kwon 2026's iso-HR test, and to cite Schmidt 2019 and Vos 2023 (only `schmidt2018wesad` is in the bibliography). | EIC-W3, R2-W4, R2-W10 (+DA-M9) | corroborated (2) |
| SC-8 | "Stressor potency" is circular or ecological (n = 5 datasets) and has no independent measure. | R1-W9, R2-W5 (+DA-C3) | corroborated (2) |
| SC-9 | Whole-session z-scoring can erase between-subject response size, and probe states are scaled against different reference sessions. | R1-W3, R2-W7 | corroborated (2) |
| SC-10 | Health-informatics consequences are missing: PPV, false alerts per day, harms, claim language. | EIC-W6, R3-W2 | corroborated (2) |
| SC-11 | The reporting norms are two sentences. A checklist is needed with minimum cell sizes, operating point, margin, and settling duration. | EIC Significance, R3-W1, R3-W6 | corroborated (2) |
| SC-12 | Numbers conflict or protocols go unlabelled: 4–6% vs 8–10% exercise; two of three panel conditions shown; Table V (LOSO) vs Table II (20 splits); EPM-E4 named in a task it contributes no subjects to. | EIC-W5a–d, R1-W14 | corroborated (2) |
| SC-13 | "Honest abstention" is recommended but never tested. | EIC Conclusion, R3-W7 | corroborated (2) |
| SC-14 | Citation precision problems: Prajod's intensity confound, Sosa "did not" should be "rarely", Milstein r ≈ 0.3 condition, Akkaya cited from its abstract only. | R2-W11, EIC minor (+DA-m5) | corroborated (2) |
| SC-15 | Scope: one research device, 132 lab subjects, no demographics, a single model family. | R3-W3 (+DA-M7) | single-reviewer |
| SC-16 | L201 misreports the disjoint variants: anger becomes separable at 0.743 (p_holm 0.003) and external exercise reverses to 0.287 (p_holm 0.0035). | R1-W6 (+DA-M3). **Verified** in `partA_disjoint.csv` | single-reviewer, factual: must fix |
| SC-17 | The UBFC-Phys control is not "the same tasks without evaluation": different topic, easier task, experimenter present. | R2-W1 (+DA-M1). **Verified**: the author's own note at `docs/literature/dataset-and-claims-verification.md:26` says "Do not call the control tasks 'non-evaluative'" | single-reviewer, factual: must fix |
| SC-18 | The Lego, hyperventilation and film comparators are stressors or passive states by their descriptors; no manipulation checks. | R2-W2, R2-W5b (+DA-M1) | single-reviewer |
| SC-19 | Cardiac missingness differs by class and moves the index definition between windows. | R1-W5 | single-reviewer |
| SC-20 | EDA processing is unspecified and has no artefact rule, yet the key comparators involve hand and arm motion. | R2-W6 | single-reviewer |
| SC-21 | The "added negative" effect is confounded with the added class's window count. | R1-W4 | single-reviewer |
| SC-22 | The position-matched rule leaves time-only BA at 0.881 on Stress-Predict and cannot be applied to Campanella or UBFC-Phys. | R1-W10. **Verified** in `time_probe_summary.csv` | single-reviewer, factual |
| SC-23 | Pooled-stress pairs mix scores from six LODO models. | R1-W11 (+DA-M5) | single-reviewer |
| SC-24 | Headline LODO costs have no CIs; R1's recomputed NB-corrected CIs reach 0.08–0.13. | R1-W8 (+DA-M6) | single-reviewer |
| SC-25 | Error budget: the Shapley values have no CIs, and F4 mixes estimands. | R1-W9 | single-reviewer |
| SC-26 | The paper tries to do too much: five contributions and twelve inline CSV names. | EIC-W7 | single-reviewer |
| SC-27 | Missing ethics and data-use statement; Table III lacks CIs for the exercise and EPM rows. | R3-W8, R2 minor | corroborated (2) |
| SC-28 | Warm-up could also be anticipation or acclimatisation. Suggested control: EPM-E4. | R2-W8 | single-reviewer (minor) |

Minor single-reviewer items (fixed 0.5 threshold R1-W12, analysis history R1-W13, feature-table release R1-W14, PRV terminology R2-W12, the "opinion tasks" inconsistency R2-W13, the email placeholder, 72 vs 108, "(69% to 8%)" range) go into REV-1 and REV-13 without arbitration.

### Points of disagreement

**D1. What the matched-arousal problem calls for** (direction disagreement)
- EIC-W1 (Critical): add an arousal-index-only detector to Table II. If it comes within about 0.03–0.05 of XGBoost, the thesis is shown directly.
- R2-W3: a wrist null is the *prior expectation*, because the discriminating marker is HPA output. Recast the result as a property of the instrument and paradigms, not of what the detector "learned".
- R1-W2 and DA-C1: report differences from the reference row and add a positive control.
- **Resolution**: these are complementary, so do all of them (REV-3, REV-4). The claim is set by the outcome. If index-only BA/AUROC matches XGBoost, the "detects arousal" wording is licensed (EIC). If not, adopt R2's narrower wording ("wrist HR/HRV/EDA carry no detectable stressor-specific information beyond sympathetic magnitude in these paradigms"). This follows the evidence-first principle. R2's framing is within the domain seat's expertise and needs no new data.

**D2. Is exercise a valid positive control?** (existence disagreement)
- EIC-S6: yes, it shows the test can detect separability.
- R1-W2: no, it differs by a gross cross-channel pattern, not a subtle signature.
- **Resolution**: defer to R1 (methodology). Exercise shows the test is not degenerate. It does not calibrate the test for subtle differences. The verified reversal of external exercise to 0.287 under the EDA-index / cardiac-model variant (SC-16) also shows exercise's behaviour depends on the variant. A graded synthetic control is required (REV-3).

**D3. How severe is the null-inference problem?** (severity disagreement)
- R1-W1: Critical. R3-W4, EIC-W2: Major. DA-C2: CRITICAL.
- **Resolution**: Critical, and blocking (B2). The title rests on it. On statistics, defer to the methodology seat.

**D4. Add material or cut it?** (direction disagreement)
- EIC-W7: cut to two contributions and move analyses to a supplement.
- R3-W1, W2, W5 and R1 (supplementary tables): add a checklist, a PPV table, interference curves and variant tables.
- **Resolution**: the main text carries two contributions (the LODO benchmark; specificity plus matched arousal with the time rule), plus a checklist box and a short deployment paragraph. Everything else goes to the supplement: all variant cells, the error-budget detail, the responder analysis, the chest-ECG bound, most of Table V, and the V3 and interference framing. This satisfies both sides.

**D5. Is "arousal, not stress" news?** (perspective difference)
- DA-M9 and R2-W3: it is textbook.
- R3-S1: it is field-changing for products.
- EIC: new only as a quantitative demonstration for deployed wrist classifiers.
- **Resolution**: adopt the EIC's formulation (REV-11). The claim is the quantitative wrist demonstration, not the idea.

---

## 3. Devil's Advocate CRITICAL adjudication

**DA-C1: the matched-arousal test cannot discriminate the two hypotheses → VALIDATED (with one qualification).**
- **Argument**: the index is built from the model's own `hr_mean` and `eda_tonic_mean`. The trained reference contrast falls to 0.584, and three of the four "null" probes sit at or below it. The "Disjoint" rerun leaves correlated EDA features in the model.
- **Corroboration**: EIC-W2, R1-W2, R2-W3 (SC-4, CONSENSUS-3).
- **Verification**: `scripts/matched_arousal_hardening.py:135` confirms the main `disjoint_hr+eda` variant drops only `hr_mean`, `eda_tonic_mean` and `eda_mean`. `eda_min`, `eda_max`, `eda_range` and the phasic features stay in, so the "Disjoint" column in Table IV (L178, footnote L196) is **not channel-disjoint**, and L201's "The conclusions are unchanged" overstates what that column shows.
- **Qualification**: the two cross-channel variants (lines 136–137: HR-only index with an EDA-only model; EDA-only index with a cardiac model minus `hr_mean`) *are* fully channel-disjoint. So C1's "circularity is not removed" holds for the headline column but not for the analysis as a whole. Those variants are the right basis for the headline. As the DA itself notes, under them the reference rises to about 0.63–0.68 while UBFC control stays at about 0.47–0.49, which partly answers C1 for UBFC. But the same variants make Lego, and anger with the state added as a negative, separable, and they reverse external exercise (SC-16). They cut both ways and must all be reported.
- **Required**: REV-3 (differences from the reference with a joint subject bootstrap; channel-disjoint variants as the headline with every cell; a graded positive control) and REV-4.

**DA-C2: the evaluative-stress claim rests on one underpowered, between-subject contrast → VALIDATED.**
- **Argument**: UBFC test vs control involves 8 control subjects and 80 windows. The CI [0.50, 0.67] contains the reference-level 0.584 that is called "significant" elsewhere. There is no equivalence margin or power analysis, and the Limitations (L277) concede "instrument failure".
- **Corroboration**: R1-W1, W7; R3-W4; EIC-W4, W8.
- **Strengthened by R2-W1** (verified against the author's own notes): the UBFC control is not evaluation-free. It is a lower dose of evaluation with an easier task. Even a tight equivalence result could not support the word "evaluative" as written.
- **Required**: REV-5 (margin, TOST, minimum detectable effect), REV-2 (correct the description) and REV-10 (retitle). The decisive test is the within-subject crossed design the paper already proposes (L275); that is future work, not a condition of this revision.

**DA-C3: "the ceiling is stressor potency" is circular or contradicted → PARTLY VALID.**
- **Circularity strand: validated.** Potency is inferred from the same HR and EDA response the detector uses, and by elimination after four filters. Corroborated by R1-W9 (ecological inference, n = 5) and R2-W5.
- **Data-gap strand: [already addressed] in data, not yet in text.** `stress_effect_size_all_datasets.csv` (commit `9cccad6`) gives EDA effect sizes for all five datasets (Stress-Predict 0.72 z, PhysioNet 0.39, WESAD 2.24), and these back L271's "a sixth (PhysioNet) to a third (Stress-Predict)" (`3922e90`). L101 still cites `stress_effect_size_by_dataset.csv`, which covers two datasets only. That is a one-line text fix.
- **"Contradicted by self-report" strand: rejected.** *Rejection rationale*: a null association is not a contradiction. PhysioNet's ρ CIs span zero on both sides ([−0.42, 0.21], [−0.30, 0.31]). The WESAD negative ρ is based on 15 units, with a within-model CI [−0.87, 0.17] that includes zero. The paper's alternative (self-report is a poor ruler for arousal, L271) is not excluded by these data. The correct reading is "no independent measure supports potency", not "potency is refuted".
- **Required**: rename to "physiological response magnitude", or add an independent measure plus the within-subject mixed model (R1-W9.4). Cite the all-dataset table. Report a leave-two-out WESAD ρ (DA-m4). All of this is REV-8.

DA MAJOR items map to roadmap items: M1 → REV-2; M2, M6, M7 → REV-10; M3 → REV-1/REV-3; M4 → REV-4 (state a falsification rule in advance); M5 → REV-3; M8 → §5; M9 → REV-11.

---

## 4. Revision roadmap (ordered)

The order is by dependency: fix facts, then rerun analyses, then rewrite the claims to match the results. Effort figures are rough single-author estimates, given for planning only. No item requires new data collection.

| # | What to do | Type | Effort | Resolves |
|---|---|---|---|---|
| **REV-1** | **Correct factual and number errors still present at HEAD**: (a) abstract L25 hyperventilation "about 50%" → the actual 0.42 → 0.55 increase (DA-m2 asks why it rises); (b) label the source of 4–6% (Table V L263) vs 8–10% (Table III L149, L161); (c) show all three panel conditions in Table III (L147 has two; values exist in `partA_panel_compact.csv`); (d) state LOSO vs 20-split in every caption (Table V L227 0.710 vs Table II L117 0.697); (e) L101: EPM-E4 contributes no subjects to the 50-subject task, and cite `stress_effect_size_all_datasets.csv`; (f) rewrite L201 to report the anger and exercise reversals; (g) Table I 33 vs L52 "34 after exclusion"; (h) PhysioNet opinion tasks, L52 vs L65; (i) split the L213 sentence; (j) the email placeholder at L18; (k) 72 tests vs 108 configurations (L94 now names the nine-class task, which is partly done); (l) abstract "(69% to 8%)" → give the range; (m) replace the naive split CI at L101 with an NB-corrected one; (n) CIs for every Table III row | text-only (c, n: reading existing outputs) | 1 day | EIC-W4, W5a–d, minors; R1-W6, W8 (last point), W14, minors; R2-W13, minors; R3-W8 (CIs); DA-M3 (part), m2, m3 |
| **REV-2** | **Describe comparators accurately**: UBFC control becomes a "lower-intensity social-evaluative condition" everywhere (L52, L161, L181, L191, L199, L271, L280); neutral probe names; add a Table I column grading each state on social-evaluative threat, time pressure, active vs passive, speech, hand motion; add a manipulation-check table from the existing self-reports (UBFC anxiety, PhysioNet 1–10, WESAD, EPM SAM); say how the Campanella CV task and inter-task rests were labelled | text + light analysis of existing self-reports | 2 days | R2-W1, W2, W5b, Q1, Q4; DA-M1; DA-C2 (part) |
| **REV-3** | **Rebuild matched-arousal inference**: probe AUROC minus reference, joint subject bootstrap; headline = channel-disjoint variants (lines 136–137), every cell in a supplementary table with raw and whole-family Holm p-values; graded synthetic positive control; complete-case and missingness-stratified runs; score cross-dataset pairs with one model or rank-normalise; common support per pair; discuss AUROC < 0.5; HR-only matching on WESAD stress vs baseline to reconcile with Kwon | analysis-only (scripts exist) | 4–5 days | DA-C1, M3, M5; EIC-W2, Q3, Q5; R1-W2, W5, W6, W11, W13; R2-W3(c), W4, Q5 |
| **REV-4** | **Arousal-index-only detector in LODO** (BA and AUROC; threshold learned on source; within / other-4 / all-5), with a falsification rule stated in the Methods before the results | analysis-only | 1 day | EIC-W1, Q1; DA-M4; drives D1 |
| **REV-5** | **Equivalence and power**: pre-stated margins (for example \|AUROC − 0.5\| < 0.10, or relative to the reference; 0.05 BA for LODO cost); TOST or CI-inclusion; per-subject paired CIs for LODO costs (R1's NB table is a start); simulated minimum detectable effect per pair; GLMM or exact CI for anger | analysis-only | 2–3 days | DA-C2, M6; R1-W1, W7, W8, Q4; R3-W4; EIC-W2(d) |
| **REV-6** | **Normalisation and weighting sensitivity**: baseline-referenced and raw features for UBFC, Lego, EPM and exercise in the panel and matched test; state each probe's reference session; raw µS/bpm effect sizes next to z-units; exercise subsampled to 33–80 windows; equal-weight added classes | analysis-only (`--normalisation baseline` exists) | 2–3 days | R1-W3, W4, Q1, Q2; R2-W7, W13(b), Q3 |
| **REV-7** | **EDA quality**: specify the decomposition and SCR criteria; apply a Kleckner-type rule; report rejection and flat-signal rates by state; rerun Tables III–IV on screened windows | analysis-only | 2–4 days | R2-W6, Q6, Q8; R1-W5 (part) |
| **REV-8** | **Potency**: rename to "physiological response magnitude", or test it within subject with a mixed model across the five datasets; bootstrap the Shapley values by subject; report F4 separately; give removed fractions per filter; leave-two-out WESAD ρ | analysis + text | 1–2 days | DA-C3, m4; R1-W9; R2-W5(c) |
| **REV-9** | **Position-matched rule**: report time-only BA under matching for every dataset, say where matching is impossible, and present the rule as "necessary, not sufficient" or tighten it | analysis-only (numbers exist) | 0.5 day | R1-W10 |
| **REV-10** | **Retitle and rewrite the abstract and conclusion** after REV-3 to REV-8. Drop "Evaluative Stress Does Not" and "because"; scope to "five laboratory E4 datasets"; cut the abstract to about 5 numbers; mark anger and hyperventilation exploratory; move the L277 instrument-failure caveat into the Discussion; frame the finding as a property of stage-label training on wrist channels | text-only | 1 day | EIC-W4, W8; R1-W1; R2-W9, minor L277; R3-W3(a); DA-C2, M2, M6, M7, m1 |
| **REV-11** | **Literature and positioning**: Kwon iso-HR (§II-C, remove "has not ... been tested" at L33); Aydoğan as the exercise precedent; an arousal-vs-stress psychophysiology paragraph (Kreibig; Dickerson & Kemeny; TSST/f-TSST); Schmidt 2019, Vos 2023; EDA methods references; citation-precision fixes; read Akkaya in full or soften. Project rule: every new claim needs a full-text quote, and R2 flags several references as unverified leads | text + reading | 2–3 days | EIC-W3; R2-W4, W10, W11, W12; DA-M9, m5 |
| **REV-12** | **Deployment implications and a reporting checklist**: PPV and false-alerts-per-hour table over assumed prevalences; harms and claim language; a checklist box (required states, minimum cells, operating point plus fixed-sensitivity rate, a priori disjoint index, margin, "not available" code); settling duration from Fig. 3; activity-gate definition; test abstention with a coverage curve or remove it; optional V3 and interference-curve framing | text + light computation | 2 days | EIC-W6, Significance, Conclusion; R3-W1, W2, W5, W6, W7, Q1, Q2; R1-W12 |
| **REV-13** | **Restructure and reproducibility**: two contributions in the main text, everything else to the supplement (D4); one Reproducibility paragraph instead of inline CSV names; release the window-level feature table and tag the commit; analysis-history statement; ethics and data-use statement | text-only | 2–3 days | EIC-W7; R1-W13, W14; R3-W8 |
| **REV-14** | **Scope arms**: an HR/HRV-only (consumer-proxy) arm in the LODO and the panel; a demographics column in Table I where the datasets provide it | analysis-only | 1–2 days | R3-W3(b, c), Q3, Q4 |
| **REV-15** | **Warm-up control**: EPM-E4 first 10–20 min (no stressor anticipation); change "sensor warm-up" to "settling" unless the control isolates the device; clarify "above skin" | analysis-only | 0.5 day | R2-W8, Q7 |
| (optional) | Enlarge the anger and hyperventilation cells with candidate E4 datasets, or run the within-subject crossed study | **new data** | weeks to months | EIC-W8 (alternative); DA-C2 (decisive version) |

Total without the optional row: about 4–5 weeks of analysis and writing.

**Already addressed at HEAD**:
- The effect-size backing for Stress-Predict and "a sixth to a third" (DA-C3 / DA-m3 data gap): `9cccad6` and `3922e90`, L271. The citation at L101 is still pending (REV-1e).
- The nine-class qualifier on the 72-comparison result (EIC minor, in part): L94.

I found nothing else among the reviewers' numeric points that later commits fixed.

---

## 5. What the author can defend or rebut rather than change

1. **Evaluation design.** Every seat and the DA treat the leakage control, paired LODO arms, audited labels and the UBFC missing-channel diagnosis as strengths. Do not reopen them.
2. **Main normalisation choice.** Whole-session z-scoring is already declared transductive, with raw and causal sensitivities (L91, L135). Defend it as the main analysis. R1-W3 and R2-W7 ask only for sensitivities on the specific probe contrasts (REV-6), not for a change of method.
3. **Exercise precedent.** The paper already cites Aydoğan's matching result (L44, L161). Answer EIC-W3 by stating explicitly that the exercise finding replicates Aydoğan and that the contribution is the non-exercise panel. No new work is needed.
4. **DA-C3 "contradicted by self-report".** Rebut this as argued in §3: the self-report nulls are uninformative, not contrary. Still accept the rename.
5. **DA-M8 (SAM-arousal transfer table).** The DA rates its own confidence 3 and says the table "predates the probes; its pipeline may differ". The author can explain that the table uses a different task and pipeline (self-reported arousal labels, not protocol stages) and is not evidence about the stage-label detector. It should be acknowledged, not silently omitted.
6. **DA-M2 ("nothing wrist-specific").** Partly defensible. The chest-ECG swap (L243) bounds how much better cardiac sensing can add (+0.04 on WESAD). Concede that it was not run through the panel, and frame the thesis as "stage-label training on wrist channels" (REV-10).
7. **DA-M9 and R2-W3 ("already known" / "prior expectation").** Defensible under the EIC's and R3's framing: the idea is old, but no one had quantified it for deployed wrist classifiers across a panel. The prior expectation is the reason to test it, since products are marketed as stress detectors. State the prior explicitly rather than argue against it.
8. **R1-W4 (class size).** Partly rebuttable from the paper's own tables. Lego (667 windows, more than fear's 394, which becomes separable) falls only from 0.43 to 0.33, so window count alone does not explain which states collapse. The subsample check in REV-6 is still cheap and settles the point.
9. **R1-W8 (NB variance factor for the external arm).** Accept R1's alternative, a per-subject paired analysis, rather than arguing over the correction factor.
10. **R3-W3 ("the E4 is no longer sold").** R3 labels this as its own recollection. Scoping the title to the E4 and adding an HR-only proxy arm is enough; the device's market status is not the author's to defend.
11. **Fear clips (DA-M4, R2-W2).** The author can keep the cardiac-threat reading as one hypothesis, alongside active vs passive coping and with a stated falsification rule. It does not have to be dropped.

