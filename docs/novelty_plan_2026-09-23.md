# Novelty plan for the JBHI paper (2026-09-23)

Progress: step 1 done (PR #9); step 2 done (`docs/arousal_kit.md`, `scripts/arousal_kit.py`).

This file brings together the five agent reports of 2026-09-23 (`docs/literature/novelty-2026-09-23/`), the earlier contribution map (`docs/contribution_map.md`), the novelty search (`docs/literature/novelty-search.md`) and the graphify knowledge graph of the repo (`graphify-out/`, 769 nodes, built 2026-09-23). It is the working plan for making the paper's novelty defensible. Decisions still belong to Alvaro and Dr. Pan.

Source reports:

| # | Report | Question it answers |
|---|---|---|
| 01 | `01_collision_check.md` | Has anyone already published each claim? |
| 02 | `02_reviewer_sim.md` | How would a strict JBHI reviewer judge the novelty? |
| 03 | `03_unused_findings.md` | Which repo results are missing from the paper, and where do paper and tables disagree? |
| 04 | `04_theory_framing.md` | Which theory turns "it detects arousal" into a positive contribution? |
| 05 | `05_closable_gaps.md` | Which gaps can be closed in days with the existing data? |

Full texts fetched by report 04 are in `sources/novelty-2026-09-23-fulltexts/` (git-ignored, `jbhi-revision` checkout).

## 1. Where the paper stands

- Simulated reviewer verdict: major revision, with about a 35% risk of reject at first round. Rigour is not the problem; the novelty statement is.
- No single paper scoops the whole paper. Two papers must be handled before submission:
  - **Zhou, Soleymani & Matarić (IEEE BIBM 2023, arXiv:2402.15513).** Abstract: "the evaluated models may be identifying emotional arousal instead of stress". Mostly chest ECG, within-corpus 5-fold cross-validation, transfer to arousal ratings of emotion videos (CASE), and no arousal matching. Not yet cited.
  - **Kwon et al. (2026, *Healthcare*).** Their iso-HR test matches heart rate within subjects (5 bpm bins) and finds that the WESAD stress-versus-rest signal survives in EDA and temperature. The manuscript cites Kwon only for leave-one-dataset-out.
- The graph shows the same gap: Kwon exists as two unlinked nodes, one from `novelty-search.md` and one from `paper/main.pdf`. The paper never connects to Kwon's matched test.
- The most dangerous sentence is `paper/main.tex` line 33 (intro): "has not, to our knowledge, been tested". The Related Work gap statement at line 49 ("No study has scored one fixed, subject-independent wrist detector … or asked whether …") has the same problem.

## 2. Novelty verdicts per claim (report 01)

| Claim | Verdict | What stays new |
|---|---|---|
| Label audit of five public E4 datasets | Novel | Only the PhysioNet whole-session labelling was found independently (Aydoğan 2026). |
| Time in session and E4 warm-up as a confound | Novel as a measurement | The idea exists (Richer 2024; Li et al. 2021 TPAMI for EEG block designs); nobody has measured it on public wrist stress data. |
| Matched-arousal specificity test | Partly anticipated (Kwon) | Matching on HR plus tonic EDA, against non-stress arousing states (Lego, non-evaluated speech, hyperventilation, emotion clips), with a model trained on other datasets. |
| Detectors detect arousal, not stress | Partly anticipated (Zhou 2023, Aydoğan 2026, Kaya 2026) | Using arousal to explain why the detector transfers across datasets. |
| Five-dataset E4 leave-one-dataset-out | Partly anticipated (Kwon: 3 corpora; Liu & Ning, arXiv 2026-09-18: 5 EDA datasets; Xiao: 7) | Audited labels, every dataset as target, and a small transfer cost. |
| Specificity panel | Partly anticipated for exercise (Aydoğan 2026) | The multi-state panel, and the finding that adding a state to training fixes exercise only. |

Proposed novelty sentence: "Earlier work suggested that stress models detect arousal [Zhou 2023; Aydoğan 2026], and heart-rate-matched tests found stress-versus-rest signal in EDA [Kwon 2026]. We are first to test this on one subject-independent wrist detector across five public E4 datasets with audited labels. The same property explains both of its behaviours: it transfers at a cost of 0.015–0.05 balanced accuracy, and at matched heart rate and EDA it cannot separate evaluative stress from a manual task, non-evaluated speech, hyperventilation or emotion clips (AUROC 0.36–0.58). Only exercise stays separable (0.89), and time in session alone matches or beats physiology on two datasets."

## 3. Framing (reports 02 and 04)

- **Paper type:** a critical evaluation of measurement validity, not a methods paper, a benchmark paper or a negative-results paper. The benchmark release becomes a separate resource paper.
- **Central theory: Cacioppo's psychophysiological inference.** A lab protocol estimates P(signal | stressor); a deployed detector claims P(stress | signal). The Handbook (3rd ed., ch. 1, read in full): "These conditional probabilities are equal only when the relationship … is 1:1". The specificity panel and matched-arousal test estimate the reverse direction directly for wrist detectors and show a many-to-one mapping. Wrist "stress" is a concomitant, not a marker.
- **Supporting frames (quotes in report 04):**
  - Autonomic specificity: Kreibig 2010, Siegel 2018, Quigley & Barrett 2014. With HR and EDA alone, arousal is the dominant shared dimension, so the null is what theory predicts.
  - Epel et al. 2018: treating a rise from baseline as evidence of being "stressed" is "problematic". Backs the split into stressor exposure, physiological response and appraisal.
  - V3 framework (Goldsack 2020): the paper fills the empty discriminant-validity cell of clinical validation.
  - Shortcut learning (Geirhos 2020): the arousal shortcut is present in every training dataset, so it survives leave-one-dataset-out; only a construct change at matched arousal exposes it.
  - Challenge versus threat (Seery; Blascovich): what separates them is cardiac output against peripheral resistance, which the E4 cannot measure. Predicts the null for UBFC and Lego.
- **Title:** drop "Evaluative Stress Does Not". Candidate: "What Do Wrist-Worn Stress Detectors Detect? A Matched-Arousal Specificity Test Across Five Empatica E4 Datasets".
- **Contributions:** cut from five to three to five sharp, falsifiable bullets. Report 02 §3 has a full rewrite with numbers and a refutation condition for each. Negatives (domain adaptation, few-shot, per-user thresholds, self-report, trait) become one paragraph plus the table.

## 4. Strong results not yet in the paper (reports 02, 03, 05)

1. **Reconciliation with Kwon, already computed.** `outputs/tables/jbhi_v2/contribution_probes/hardening/partA_disjoint.csv`, variant `disjoint_hr_only_model_eda`: stress versus rest stays separable (0.68, p_holm 0.0035), stress versus UBFC control (0.47) and hyperventilation (0.45) do not.
2. **Wrist HRV moves the wrong way under stress (WESAD).** Chest ECG RMSSD −11.6 ms per subject, wrist +26.5 ms; HR rise +21.3 bpm on ECG, +9.2 bpm at the wrist; agreement during stress SDNN r = 0.12, RMSSD r = 0.38. From an agent's own pass over `wrist_vs_ecg_per_window.csv`; **no committed script reproduces it yet.**
3. **Untrained arousal index as a baseline.** Done in step 2 (`docs/arousal_kit.md`): the index matches the external model on all five datasets (PhysioNet 0.804 vs 0.807; report 05's 0.662 was a join artefact).
4. **Missing PPG predicts the label.** AUROC of (1 − PPG coverage): WESAD 0.81, UBFC-Phys 0.74, PhysioNet 0.34 (exercise also loses PPG there). Same XGBoost missing-value mechanism as the UBFC temperature shift.
5. **Self-report does not track protocol labels.** Only 47–61% of PhysioNet subjects report more stress after the tasks; dropping stress windows without a self-reported rise leaves pooled balanced accuracy unchanged (0.849). Run predates the label fixes.
6. **Second-half rests.** External PhysioNet 0.746 to 0.797. CLAUDE.md says to report both; the paper does not.
7. **Time after position matching.** Minutes since start still reaches 0.88 on Stress-Predict after position matching, so the proposed position-matched rule is not enough there.
8. **Pooling versus few-shot.** From four source datasets WESAD reaches 0.884 with no target labels, and few-shot adds under 0.01. Needs one matched rerun.

## 5. Paper-versus-table mismatches to fix (reports 02 and 03)

| Place in paper | Problem | Source of truth |
|---|---|---|
| Negative-results table, mild-stress row | Both ρ values are PhysioNet. WESAD external ρ = −0.60 [−0.84, −0.08]. | mild-stress probe tables |
| Detectability claim | "abs(ρ) ≤ 0.35, CIs include 0" is contradicted by Stress-Predict baseline EDA vs external recall, ρ = 0.48 [0.18, 0.70]. | contribution probes |
| Model comparisons | "0 of 48 after Holm" vs 72 comparisons, 1 significant. | `significance_models.csv` |
| Data Availability | Promises audited label tables and 20 frozen splits on `jbhi-revision`; none is tracked in git. | `git ls-files` |
| Exercise false-stress with exercise in training | 8% in abstract and panel vs 4–6% in the negatives table; different training sets, never stated. | specificity panel, novelty experiments |
| Specificity panel | Methods defines three training conditions, the table shows two. | `specificity_panel_probe.py` output |
| Robustness text | "Same pattern except Lego", but in one variant anger becomes separable (0.74, p_holm 0.003). | `hardening/` tables |
| Subject count | "about 118" vs 132. | harmonised-classes table |
| Leakage example | 94% in the paper vs 92.1% (balanced accuracy 0.807 vs 0.355). | `leakage_check.csv` |
| Abstract, domain adaptation | Stated generally; the null covers only WESAD and PhysioNet. | `domain_adaptation.py` tables |

## 6. Plan, in order

| Step | What | Effort | Script to extend | Answers |
|---|---|---|---|---|
| 1 | Credit Zhou and Kwon; rewrite intro line 33 and Related Work; add the Kwon reconciliation (§4 item 1); fix every row of §5; cut contributions to the report-02 rewrite; retitle. | 1 day | none | Reviewer objections 1 and 3; all mismatches |
| 2 | Arousal-controlled evaluation kit: arousal-index baseline with CIs (dedupe the join first), matched-arousal AUROC as one reusable function, equivalence bounds and minimum detectable effect with a synthetic positive control, each probe state against rest, an explicit iso-HR table rerunning Kwon's test on our data, and the stress score split into arousal plus residual. | 2–4 days | `scripts/matched_arousal_hardening.py` | Objections 1 and 2 (circularity, small cells) |
| 3 | Commit a script for the wrist-versus-chest HRV direction result (§4 item 2), per subject with CIs. | about 1 day | `scripts/validate_cardiac_against_ecg.py` | New mechanism: HR level and EDA carry the detector, not vagal tone |
| 4 | Stressor-potency dose–response: per subject and stressor (about 10 stressors), relate recall to physiological response size; test whether dataset identity still explains recall. | 6–10 h | `scripts/specificity_panel_probe.py` (Part B) plus the responder code in the hardening script | Objection 4; makes "the ceiling is stressor potency" falsifiable |
| 5 | Missing-PPG shortcut: a coverage flag in LODO and abstention under low coverage. | 2–3 days | `scripts/leave_one_dataset_out.py`, `scripts/threshold_transfer_probe.py` | Second artefact of the same missing-value mechanism |
| 6 (optional) | Frozen PaPaGei or Pulse-PPG encoder under LODO and matched arousal. Watch for the speech-motion and missing-PPG shortcuts. | 4–7 days | `scripts/leave_one_dataset_out.py` | "Would a foundation model escape the arousal ceiling?" |
| 7 (optional) | Theory-driven checks: felt arousal versus valence on EPM-E4 and WESAD self-assessment; still-window pulse amplitude as a vascular channel; a V3-style validity table with positive predictive value under a stated daily mix (about 9% with report 04's assumed mix). | 1–3 days each | self-report and feature scripts | Strengthens the Cacioppo framing |
| 8 | Deposit labels, splits and probe scripts on Zenodo so Data Availability is true. | 2–3 days | `scripts/generate_frozen_splits.py` | Mismatch in §5; the resource paper later |

Steps 1–4 are the minimum for a credible submission. Step 1 needs no new experiments.

## 7. Risks and open checks

- **Self-report risk.** Report 05 found the external model's score barely tracks self-rated arousal on EPM-E4 emotion clips (ρ 0.08, not clustered by subject), and report 04 found |r| ≤ 0.23 within subject. Either do a subject-level analysis, or state the claim as *autonomic* arousal (physiological mobilisation), not felt arousal.
- **Quotes still to verify in full text before citing:** Uphill, Bosch 2026, Kaya 2026 (read through a summariser); Cronbach & Meehl, Campbell & Fiske, McEwen, Mauss and pulse-arrival-time effect sizes (from memory). Two items in report 01 returned HTTP 403.
- **Items computed by agents, not by committed scripts:** the HRV-direction result, the arousal-index baseline and the missing-PPG AUROCs. Each needs a committed script before it enters the paper.
- **Liu & Ning (arXiv, 2026-09-18)** is close to the benchmark niche. Deposit early if the resource paper goes ahead.

## 8. Decisions for Alvaro and Dr. Pan

1. Adopt the critical-evaluation framing and the new title?
2. Cut the contributions to the report-02 rewrite?
3. Which of steps 5–8 to fund before submission?
4. Autonomic arousal only, or run the self-report analysis first?
