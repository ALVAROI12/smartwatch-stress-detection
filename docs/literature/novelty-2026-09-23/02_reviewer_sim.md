# Simulated JBHI review: novelty and acceptance risk

Manuscript: `paper/main.tex`, "What Do Wrist-Worn Stress Detectors Detect? Arousal Transfers Across Five Empatica E4 Datasets, Evaluative Stress Does Not" (state at commit `4ab234a`, 2026-09-23).
Scope: novelty and acceptance risk only. Context read: `CLAUDE.md`, `STATUS.md`, `docs/contribution_map.md`, `docs/literature/novelty-search.md`, `sources/novelty/kwon-2026-iso-hr-cross-corpus.txt`, and the tables under `outputs/tables/jbhi_v2/`.

---

## 1. Verdict now: MAJOR REVISION, with a real risk of reject at first round

The work is careful: audited labels, subject-held-out evaluation, Holm-corrected tests, and a feature-disjoint robustness check. That puts it above most wrist-stress submissions. As written, though, the novelty statement is exposed on three sides.

(a) The central question has already been asked, in a paper the manuscript cites for something else. Kwon et al. (2026) state: "We tested whether binary stress-versus-non-stress performance reflects stress-specific physiology rather than heart-rate (HR) arousal", and they introduce an iso-HR matched evaluation. Aydoğan and Villagra Povina (2026) are titled "Stress or arousal?". The introduction says: "Whether a detector trained on stressor labels has learned anything beyond arousal has not, to our knowledge, been tested." A reviewer who knows Kwon will read this as a misrepresentation. That reviewer may well be one of Kwon's authors, since the paper is suggested-reviewer bait.

(b) The paper's key results are nulls on small cells: UBFC-Phys control 8 subjects, hyperventilation 53 windows, anger 33 windows. The paper gives no equivalence or power argument, so "cannot separate" is read as absence of evidence.

(c) Five contributions plus a table of nine negative results reads as a thesis chapter, not a focused journal claim.

Probability estimate (subjective): reject after the first round about 35%, major revision about 55%, minor revision under 10%. Fixing (a) and (b), which needs mostly text plus one cheap analysis, moves this toward a solid major-then-accept.

## Top 5 novelty objections, as a reviewer would write them

**R1. "Arousal-versus-stress is not a new question, and the matched test is not new."**
> "The authors claim that whether a stress detector has learned anything beyond arousal 'has not been tested'. This is incorrect. Kwon et al. [kwon2026], which the authors cite only for LODO, explicitly test 'whether binary stress-versus-non-stress performance reflects stress-specific physiology rather than heart-rate arousal' using an iso-heart-rate matched evaluation on three E4 corpora, including WESAD and Stress-Predict. Aydoğan and Villagra Povina [aydogan2026] frame exercise false alarms as 'stress or arousal?'. Sosa et al. [sosa2026] review the same construct problem. The authors must state precisely what their matched-arousal test adds over iso-HR matching, and reconcile their near-chance stress-vs-rest within-bin AUROC (0.58) with Kwon's finding that WESAD's non-cardiac signal survives HR matching (AUROC 0.935)."

**R2. "The matched-arousal test is close to circular and its nulls are underpowered."**
> "The arousal index is the mean of z-scored HR and tonic EDA, which are the detector's two strongest inputs. Conditioning on them and then finding that the score adds little is expected for any model whose output is mostly a monotone function of those inputs. Indeed, the detector barely separates stress from its own training negatives inside bins (0.584). The test therefore seems unable to tell 'no evaluative signature' apart from 'the conditioning removed the signal'. Beyond that, the non-separable cells are tiny (8 UBFC control subjects; 53 hyperventilation windows from 31 subjects; 33 anger windows, one per subject). Failing to reject AUROC = 0.5 is not evidence of AUROC = 0.5. Please give equivalence bounds or a minimum detectable effect, and a positive control showing the test can detect a specific signature of plausible size in cells this small."

**R3. "The five-dataset LODO benchmark is incremental."**
> "Pooled LODO on public E4 corpora has been done (Kwon 2026, three corpora). Pairwise transfer exists (Calza-Metre 2026, four corpora), and so do multi-dataset pooling studies (AutoStress/Schreiber 2026; Vos 2023; Xiao 2025). Moving from three to five corpora and reporting that the cost is 'within the sampling error of 15–35 test subjects' is a modest increment. The authors themselves decline to claim equality. The missing-temperature threshold diagnosis is an XGBoost missing-value artefact, not a finding about physiology. What is the transferable scientific result here?"

**R4. "'The ceiling is stressor potency' is an interpretation, not a tested result."**
> "The claim rests on effect sizes computed for only two datasets (PhysioNet vs WESAD, `stress_effect_size_by_dataset.csv`) and on a residual that four filters do not explain. A residual is not evidence for any particular cause. Stressor type, posture, speech, protocol, population and lab are all confounded with dataset, so n = 5 datasets cannot identify 'potency'. The paper's own responder analysis shows a 0.09 gain on PhysioNet from subject selection, which points to subject non-response as much as to stressor potency. Either test potency directly (dose–response across stressors and subjects) or soften the claim."

**R5. "The paper is a collection of probes and negative results without a clear methodological contribution."**
> "The paper lists five contributions and a table of nine negative results (domain adaptation, few-shot, per-user thresholds, self-report, detectability trait...). Several are known: chronological few-shot gains are small (Tervonen 2026, Akkaya 2026), exercise false alarms and their fix (Aydoğan 2026, 82.6% → 6–7%), the order/sequence confound (Richer 2024), and test-inclusive normalisation inflation (Tognotti 2026). The domain-adaptation null is shown only for WESAD↔PhysioNet, yet the abstract generalises it. The proposed 'reporting standards' (specificity panel, matched-arousal test) are not validated: no evidence shows they change a published conclusion or rank models differently from BA. JBHI readers need either a method, a validated protocol, or a resource. Which one is this?"

Secondary points a careful reviewer would also raise, and which cost credibility if left in:
- **Inconsistent exercise numbers.** The panel says 8–10% with exercise in training (Table III); the negative-results table says 4–6% (Table VI). Both exist in the repo from different runs (`specificity/partA_panel_compact.csv` vs `novelty_experiments/exercise_negatives_physionet.csv`); pick one or explain the difference.
- **Robustness text understates a counter-example.** Sec. V-D says the two further disjoint variants "give the same pattern, except that Lego becomes separable". In `hardening/partA_disjoint.csv`, the EDA-only-index / cardiac-model variant with anger among the training negatives gives anger 0.743, p_holm 0.003, which is separable. This must be reported.
- **Position matching does not remove time on Stress-Predict.** `timeprobe/time_probe_summary.csv` shows time-only BA is still 0.881 after position matching on Stress-Predict (0.62 on PhysioNet, 0.60 on WESAD). The paper reports only that physiology survives matching. A reviewer will ask why the "position-matched rule" is recommended when time still predicts under it.
- **Scope of the domain-adaptation null.** DA was run only for WESAD↔PhysioNet (`domain_adaptation_*.csv`), but the abstract says "domain adaptation ... add nothing significant" without that qualifier.
- **Anger has 33 windows, one per subject.** It should not appear in the abstract on an equal footing with Lego (667 windows).

---

## 2. Evidence already in the repo, per objection

| Obj. | Repo evidence that answers it | Used in paper? |
|---|---|---|
| **R1** (vs Kwon iso-HR) | `outputs/tables/jbhi_v2/contribution_probes/hardening/partA_disjoint.csv`, variant `disjoint_hr_only_model_eda` (HR-only index, EDA-only model) is in effect **Kwon's iso-HR non-cardiac test extended to non-stress arousal states**. Under it, stress vs baseline/rest stays separable (0.683, p_holm 0.0035), which reproduces Kwon's direction on our data, while stress vs UBFC control (0.467) and hyperventilation (0.454) do not. That contrast is the novelty: iso-HR shows the EDA signal beats *rest*; our panel shows the same signal does not beat *non-evaluative arousal*. Also `docs/literature/novelty-search.md` §1–2 (Kwon's design: no per-subject normalisation, negatives are baseline/relax only, no probe states). `sources/novelty/kwon-2026-iso-hr-cross-corpus.txt` §2.5, §3.6. | **No.** Kwon appears only as a LODO reference; "iso-heart-rate" appears only in the bibliography title. The variant is mentioned in one sentence as a robustness check, not as the reconciliation. |
| **R2** (circularity, power) | Circularity: `hardening/partA_disjoint.csv` (index features and `eda_mean` removed; HR-only and EDA-only cross variants). Positive controls: exercise 0.892 and fear 0.779 (`hardening/partA_inference.csv`), which show the test *can* return a high AUROC. Exercise-minus-pair bootstrap differences exclude zero (`diff_lo` 0.18–0.36). Subject-bootstrap CIs give implicit upper bounds (UBFC ≤ 0.67, hyperventilation ≤ 0.62, Lego ≤ 0.54). No equivalence test, MDE or synthetic positive control exists (grep found none). | Partly. The disjoint variant and exercise/fear controls are in the paper. CIs are shown but not framed as bounds, and no power argument is made. |
| **R3** (LODO incremental) | `outputs/tables/jbhi_v2/novelty_experiments/kwon_setting_summary.csv`: in Kwon's own WESAD+Stress-Predict setting, our pipeline gives Stress-Predict external AUROC 0.67 vs their 0.56, and per-subject scaling moves WESAD-target AUROC from 0.834 to 0.914. `novelty_experiments/leave_one_dataset_out_summary_{raw,causal}.csv` and `normalisation_vs_session.csv`: the cost stays small without transductive scaling. `label_fix_sensitivity/resthalf_*`: second-half rests raise external PhysioNet 0.746 → 0.797 (a concrete label-definition effect on transfer). `novelty-search.md` comparison table (ours vs Kwon, Calza-Metre, Xiao, Vos, Mishra, AutoStress, Dahal). | Kwon-setting result: yes (one sentence). Normalisation variants: yes. **Second-half-rest result: no** (0.797 does not appear). Comparison table vs prior LODO work: no. |
| **R4** (potency untested) | `stress_effect_size_by_dataset.csv` (PhysioNet and WESAD only). `contribution_probes/specificity/partB_per_subject_recall.csv` (per-subject × per-stressor recall, within and external, plus baseline physiology). `hardening/partB_responder_sensitivity.csv` (subject response magnitude vs gain). `arousal/partC_onset_offset_curves.csv`. `arousal/scores.csv` (per-window within and external scores). The raw material for a stressor-level dose–response exists but has not been assembled. | Effect sizes for two datasets: yes. Per-stressor recall (e.g. Stress-Predict Stroop external recall 0.24): in `contribution_map.md` only, not the paper. No dose–response. |
| **R5** (no method) | `docs/contribution_map.md` §4 ranks 1–2 (panel as reporting standard; benchmark release). Frozen splits (`scripts/generate_frozen_splits.py`), `harmonization_table.csv`, `leakage_check.csv` (94% → 49%). The negative results are backed by `fewshot/`, `domain_adaptation_*chronological_gap30*.csv`, `mild_stress/`, `specificity/partB_miss_consistency.csv`. What is missing: any demonstration that the panel changes a conclusion, e.g. two models with equal BA but different panel profiles. | Negatives: yes (Table VI). Release: yes (one paragraph). "Panel changes model ranking": not done anywhere. |

---

## 3. Contribution statement: problems and a rewrite

### Where the current text is weak, vague or overclaims

1. **Intro ¶1, "has not, to our knowledge, been tested".** This is false as stated (Kwon iso-HR; Aydoğan). It is the single most dangerous sentence in the paper.
2. **Abstract, "domain adaptation, few-shot personalisation and per-user thresholds add nothing significant".** DA covers two datasets only. The few-shot result is correct but is known (Tervonen, Akkaya).
3. **Abstract and Conclusion, "the ceiling is stressor potency".** This is stated as a finding but supported only by a residual and a two-dataset effect-size contrast (R4).
4. **Title, "Evaluative Stress Does Not [transfer]".** The data do not show that evaluative stress fails to transfer. They show that the detector does not separate evaluated from non-evaluated arousal at matched arousal, in small between-subject cells. The title claims a positive negative.
5. **Contribution (1), "diagnosing one apparent domain gap as a missing channel".** This is a software artefact of XGBoost's missing-value branch. Keep it as a pitfall in the text, not as a contribution.
6. **Contribution (2), "separate exercise but not non-evaluative arousal".** It gives no numbers, no bounds and no credit to Aydoğan for the exercise half.
7. **Contribution (5).** A list of five negatives dilutes the paper. Two are known, and two (self-report, trait) are side questions.
8. **"We propose ... reporting standards".** Nothing validates the standards (R5).
9. **Five contributions is too many for the evidence per item.** JBHI reviewers reward three sharp ones.

### Proposed contribution bullets (all numbers already in the repo)

1. **A specificity test that extends iso-HR matching from rest to non-stress arousal states.** On five public E4 datasets, a fixed HR/HRV/EDA detector at matched autonomic arousal separates stress from baseline/rest (within-bin AUROC 0.58; 0.68 under HR-only matching with an EDA-only model, consistent with Kwon et al.). It does not separate stress from the same speech-and-arithmetic tasks performed without evaluation (0.58 [0.50, 0.67]), from hyperventilation (0.52 [0.44, 0.61]) or from manual Lego tasks (0.36 [0.24, 0.54]). Adding those states to training does not change this (p_holm ≥ 0.64), whereas exercise becomes separable (0.89 [0.84, 0.93]). *Falsifiable:* a within-subject evaluated-vs-non-evaluated contrast with AUROC > 0.67 at matched arousal would refute it.
2. **A five-dataset LODO benchmark where transfer is cheap once labels and scaling are fixed.** The transfer cost is 0.015–0.05 BA with HR/HRV/EDA features. In the Kwon et al. setting the same pipeline raises Stress-Predict external AUROC from 0.56 to 0.67, and the small cost survives raw (non-transductive) features (−0.01 to 0.05). Reported failures in the literature track low within-dataset ceilings and missing per-subject normalisation more than stressor mismatch. *Falsifiable:* a sixth E4 dataset with an external cost above 0.10 under the same protocol.
3. **Protocol position as a measured confound.** Minutes since recording start alone out-predicts wrist physiology within dataset on PhysioNet (0.83 vs 0.78) and Stress-Predict (0.95 vs 0.70) because baselines coincide with sensor warm-up (68–100% of baseline windows in the first 10 min on three datasets). A model given time collapses externally (Stress-Predict 0.66 → 0.56; WESAD 0.90 → 0.45). This yields a concrete evaluation rule: report the time-only classifier alongside any baseline-referenced result.
4. **A bounded error budget for the low-ceiling datasets.** Recovery contamination, sensor loss and onset latency together explain at most 0.06 of the 0.22–0.29 within-dataset error on PhysioNet and Stress-Predict, and a strict responder filter at most 0.09 (PhysioNet only). Motion-robust cardiac sensing is worth at most +0.04 on WESAD (chest-ECG substitution 0.916 → 0.956). The remaining ceiling is therefore not a pipeline problem. *(Upgrade to "is explained by stressor response magnitude" only after analysis B in §4.)*
5. **Released artefacts.** Descriptor-audited protocol-stage labels (seven corrections, plus hyperventilation kept separate from stress), 20 frozen subject splits and the probe scripts. The leakage demonstration (94% window-split vs 49% subject-held-out) moves to Methods as motivation, not a contribution.

Negatives (DA, few-shot, thresholds, self-report, trait) become one paragraph plus the table, framed as "consistent with the arousal account", not as a listed contribution.

Title suggestion: "What Do Wrist-Worn Stress Detectors Detect? A Matched-Arousal Specificity Test Across Five Empatica E4 Datasets". Drop "Evaluative Stress Does Not" (see point 4 above).

---

## 4. The two analyses that would most raise acceptance odds

### A. Equivalence bounds, minimum detectable effect and an explicit iso-HR reconciliation for the matched-arousal test (answers R1 and R2)

- **What:** add three outputs to the matched-arousal inference.
  1. **TOST / MDE.** For each probe cell, report the largest within-bin AUROC the subject bootstrap excludes (one-sided 95%), and the minimum detectable AUROC at 80% power under the existing subject-clustered permutation scheme. Get the latter by injecting a synthetic shift of known size into the probe windows' scores within bins, a positive control at each cell's real n.
  2. **Specificity contrast against the reference.** Bootstrap the difference (stress vs baseline/rest within-bin AUROC) minus (stress vs probe within-bin AUROC), with the reference as the comparator instead of exercise. This is the direct statement "the detector's residual signal beats rest but not non-evaluative arousal".
  3. **An iso-HR row.** Kwon's exact 5-bpm within-subject HR-bin matching for stress vs rest and stress vs each probe, with EDA-only and all-feature scores. This goes in a small table beside Kwon's WESAD numbers.
- **Why it matters:** it turns the nulls into bounded claims (e.g. "evaluation adds at most 0.17 AUROC beyond arousal") and makes Kwon a foil rather than a threat.
- **Script to extend:** `scripts/matched_arousal_hardening.py`. It already has the subject-clustered permutation, Holm correction, bootstrap and disjoint variants, and reads `contribution_probes/arousal/scores.csv`. The iso-HR matching is about 30 lines on the same per-window table.
- **Cost:** 5–8 hours (2–3 coding, about 1 compute, 2–3 writing one table and a paragraph).

### B. Stressor-potency dose–response (answers R4, and makes the headline claim falsifiable)

- **What:** use the stressor unit (WESAD TSST; PhysioNet Stroop, TMCT, opinion; Stress-Predict Stroop, interview; UBFC T2, T3; Campanella subtraction; about 9–10 units) and the subject × stressor unit (about 250 cells).
  - Compute each cell's physiological response (stressor-minus-own-baseline z for HR and tonic EDA, as in the F4 responder rule).
  - Regress within and external recall on response magnitude with a subject random effect (or subject-bootstrap the Spearman ρ).
  - Test whether *dataset* or *stressor label* adds explanatory power once response magnitude is in the model.
- **Prediction:** if "the ceiling is stressor potency" holds, response magnitude explains most between-stressor variance in recall (e.g. Stress-Predict Stroop external recall 0.24 sits at the low-response end), and dataset identity adds little.
- **Refutation:** if dataset identity still explains recall at equal response, something other than potency (label definition, protocol, pipeline) sets the ceiling. Either outcome is publishable, and either removes R4.
- **Script to extend:** `scripts/specificity_panel_probe.py` Part B, which already writes `partB_per_subject_recall.csv` per subject × stressor with within/external recall. Add the response-magnitude columns using the F4 code in `scripts/matched_arousal_hardening.py`. Extend `stress_effect_size_by_dataset.csv` to all five datasets in the same pass.
- **Cost:** 6–10 hours.

Lower priority, not recommended before submission: the frozen public PPG encoder LODO row (`contribution_map.md` rank 8; 2–3 weeks with GPU). It answers a "did you try deep learning" comment but not a novelty objection.

---

## 5. JBHI fit: recommended framing

| Framing | Fit | Why |
|---|---|---|
| Methods/benchmark paper | Weak for JBHI | The benchmark adds 2 datasets to Kwon's LODO, and the artefacts (labels, splits) are a resource, better placed in a Scientific Data / D&B companion (`contribution_map.md` rank 2). Leading with it invites R3. |
| Negative-results paper | Weak | JBHI has no negative-results track. Leading with nulls on small cells maximises R2 and R5. |
| **Critical-evaluation (validity) paper** | **Recommended** | JBHI publishes validity and evaluation-methodology papers for health wearables. The paper's strongest, least-anticipated asset is a *measurement-validity test* (matched-arousal specificity across a panel of non-stress arousal states, reconciled with iso-HR), backed by a confound quantification (time/warm-up) and a bounded error budget. That is a critical evaluation of what the field's standard pipeline measures, with transfer as the setting rather than the claim. |

Concretely: lead with the specificity/matched-arousal result (with analysis A), keep LODO as the necessary setting ("the detector transfers; here is what transfers"), and put time-in-session and the error budget (with analysis B) as the ceiling analysis. Fold the negatives into one paragraph. Credit Kwon (iso-HR, LODO), Aydoğan (exercise and the PhysioNet audit), Richer (sequence effects) and Tognotti (normalisation) up front, and state the increment over each in one sentence. Move the benchmark release to a separate resource paper.
