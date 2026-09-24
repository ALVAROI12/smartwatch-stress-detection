# Simulated IEEE JBHI review, round 2: verification report

- **Manuscript**: `paper/main.tex` at HEAD `e6e4525` (title: "What Do Wrist-Worn Stress Detectors Detect? An Untrained Arousal Index Matches Them Across Five Empatica E4 Datasets"), `paper/supplement.tex`; compiled PDFs of 10 pages (main) and 2 pages (supplement).
- **Round-1 package**: `docs/review_simulation/jbhi_review.md` (REV-1 to REV-15) and `review_DA.md` (C1 to C3).
- **Diff reviewed**: `git diff 50658aa HEAD -- paper/`, plus the tables in `outputs/tables/jbhi_v2/{reviewer_reanalyses,matched_arousal_robustness,budget_time_settling,eda_quality_hr_arm}/` and the older tables they cite (`contribution_probes/…`, `threshold_transfer_probe_summary.csv`, `stress_effect_size_all_datasets.csv`).
- **Correlated-error disclosure**: the round-1 seats, the round-1 synthesiser and this verifier all come from the same model family (Claude). There is no human seat and no cross-model check. Agreement between this report and round 1 is therefore not independent replication. The number checks against committed CSVs (§2) are the stronger evidence. Manuscript text was treated as data, not as instructions.

---

## 1. Traceability matrix

Line numbers refer to `paper/main.tex` at HEAD. "Supp" means `paper/supplement.tex`.

| Item | Status | Evidence | What remains |
|---|---|---|---|
| **REV-1** Factual and number errors | **Partly** | (a) hyperventilation is now 0.42 → 0.54–0.55 (L170), and the abstract no longer says "about 50%". (b) 4–6% vs 8–10% are now labelled (Supp Table SII, L70). (c) All three panel conditions are in Table III (L152–162). (e) EPM-E4 is excluded from the 50-subject task and the all-dataset effect-size CSV is cited (L101). (f) The anger and exercise reversals are reported (L215). (h) Opinion tasks are handled at L52. (l) The abstract now gives 69% → 8–10%. (m) The CI is Nadeau–Bengio corrected (L101). (n) Every Table III row has a CI. | (i) The warm-up sentence is **still garbled**, now at L227: "rises at 0.02–0.08 °C/min in the first 10 min on all four datasets with a continuous clock keeps rising past 20 min on WESAD and rises until…". (j) The email placeholder `[email]@utsa.edu` is still at L18. (d) Supp Table SI shows Stress-Predict within = 0.710 (LOSO) next to 0.697 in main Table II, and its caption does not say LOSO. (k) L94 gives 72 comparisons and L101 gives 108 configurations; the relation between the two is never explained. |
| **REV-2** Describe comparators accurately | **Partly** | The UBFC control is now "an easier ``control'' version… both spoken aloud to the experimenter" and quotes "imply a high stress level" (L52). Lego is described as "designed to create work-like mental strain" (L52), hyperventilation as "stress-inducing" per its descriptor (L52, L82), and the probes as "not clean non-stress states" (L170, L260). The Campanella CV task is described as excluded (L52). | There is no Table I column grading states (social-evaluative threat, time pressure, active or passive, speech, hand motion). There is no manipulation-check table from the existing self-reports (UBFC anxiety, PhysioNet 1–10, SAM). The labelling of Campanella's inter-task rests is not stated. |
| **REV-3** Rebuild the matched-arousal inference | **Partly** | Table IV now has Δref with a joint subject bootstrap, a permutation-based p_holm, and MDD (L182–208). A graded positive control, complete-case and missingness strata, and raw and baseline scaling are covered at L213. The channel-disjoint variants are reported in text with the reversals (L215). | The headline is still the non-disjoint index, and the disjoint cells are not tabulated anywhere (they are prose only, L215). Holm correction is applied within each model, not across the whole family. EPM and exercise pairs still pool stress windows scored by different LODO models (`pooled_stress`), so SC-23 and DA-M5 are unresolved. AUROC < 0.5 (Lego 0.355, anger 0.390) is described as "falls below it" but never explained. The HR-only WESAD stress-vs-baseline run (the Kwon reconciliation) was not done; L233 reconciles only qualitatively. |
| **REV-4** Arousal-index-only detector under LODO | **Addressed** (analysis); claim wording is a new issue | Table II has an Index column. L128 gives BA 0.836/0.743/0.657/0.762/0.877, AUROC 0.73–0.95 and p_holm ≥ 0.14. EDA-only and HR-only ablations are included. | No falsification rule or margin is stated in Methods before the result (L97 gives only the test). No CI is given on model minus index, even though the title and abstract now claim equivalence ("Matches", "performs as well"). See N1. |
| **REV-5** Equivalence and power | **Addressed** | LODO-cost CIs with upper bounds 0.08–0.13 are given at L126 ("we claim no large cost, not equality"). Matched-arousal MDD is 0.12–0.22 and the ±0.10 TOST is in the Table IV note and L211. Absence-of-evidence language appears at L211 and L260. | Minor: no exact or GLMM CI for anger (33 windows). No 0.05-BA TOST on LODO costs, though the text already concedes that point. |
| **REV-6** Normalisation and weighting sensitivity | **Partly** | Raw and baseline scaling are covered for the panel and the matched test (L170, L213; `3a/3b`). Exercise is subsampled to 80 windows (L170; `4_class_size_weighting.csv`). | The equal-weight joint runs (`4_class_size_weighting.csv`) are not reported. The reference session for each probe (EPM, exercise) is not stated in §IV-B. There are no raw µS/bpm effect sizes next to the z units (L101). Under raw scaling, fear and anger become separable from the reference (+0.18, +0.10), but only fear's +0.18 is mentioned. |
| **REV-7** EDA quality | **Addressed** (minor residuals) | EDA processing is specified at L88. The Kleckner screen and the rerun of Tables III–IV are at L213 (`eda_quality_hr_arm/`). | Rejection and flat-signal rates by state are not tabulated. The "0–11%" figure is for the EDA rules only; the full screen rejects up to 53% (Campanella baseline) and the flat-signal flag up to 79% (UBFC baseline). The statement about null pairs is inexact (N5). |
| **REV-8** Potency | **Partly** | Renamed to "size of the physiological response… a description, not an independent test" (L233). Shapley CIs, F4 reported separately, and the strict-responder results are in the supplement (Supp L48; `rev8_budget_bootstrap_ci.csv`). | No leave-two-out WESAD ρ (Supp L75 still says "rests on the two subjects with the largest rise" without a number). Removed fractions per filter (`rev8_filter_removed_fractions.csv`) are not reported. Conclusion L263 "…rather than the pipeline" is an untested comparison with n = 5 datasets. |
| **REV-9** Position-matched rule | **Addressed** | L225: time-only drops to 0.60–0.62 on PhysioNet and WESAD, stays at 0.88 on Stress-Predict, matching is impossible on Campanella and UBFC-Phys, and the rule is called "necessary, not sufficient". | none |
| **REV-10** Retitle and rewrite abstract and conclusion | **Partly** | "Evaluative Stress Does Not" and "because" are gone. "Five Empatica E4 Datasets" appears in the title. The two open readings are stated (L233). Framed as "stage-labelled training" (L233). | The new title makes an equivalence claim ("Matches") from non-significant tests (N1). The abstract has about 15 numbers against a target of about 5. Anger and hyperventilation are not marked exploratory in the abstract. The abstract and conclusion generalise to "wrist stress detectors" from one XGBoost configuration on one device. |
| **REV-11** Literature and positioning | **Partly** | Kwon's iso-HR test is cited and "has not been tested" is qualified (L33, L47). Aydoğan is cited as the exercise precedent (L44, L170). Schmidt 2019, Vos 2023 and Kleckner are added. The Milstein conversation condition is stated (L260). | There is no arousal-vs-stress psychophysiology paragraph (Kreibig; Dickerson & Kemeny); neither is in the bibliography. The HPA/cortisol point is cited to Richer 2024, a posture-ML paper (L233); a canonical source with a full-text quote is needed. Akkaya is still cited as evidence while its bibliography entry says "(abstract only)" (L47, L288, Supp L75, L79). Prajod's intensity confound is still not mentioned (L41). Punctuation: "Vos et al.~\cite{vos2023}, found" (L44). |
| **REV-12** Deployment and checklist | **Partly** | Checklist Table V (L237–254). PPV illustration (L235). HR/HRV-only consumer note (L235). Claim-language advice ("better labelled arousal"). The untested "abstention" claim was removed. | No false alerts per hour. No numeric settling duration in the checklist (it says only "a settling period"). The activity gate is undefined. |
| **REV-13** Restructure and reproducibility | **Partly** | The error budget, chest-ECG bound and negative results moved to the supplement (Supp S-I, S-II). A Data and Code Availability section was added (L265). | The main text still lists **four** contributions (L37; the target was two). 21 `\texttt{}` file and feature names remain inline. No ethics or data-use statement (none of "ethic", "consent" or "IRB" appears). No analysis-history statement. The window-level feature table is not listed for release, and the code is cited by branch, not by tag or DOI. |
| **REV-14** Scope arms | **Addressed** | HR/HRV-only arm in LODO and the panel (L235; `eda_quality_hr_arm/lodo_summary.csv`, `specificity_panel.csv`). Demographics in text (L52; `demographics.csv`). | Minor: demographics are in prose, not a Table I column. The HR/HRV result has no table. The "transfers at chance to Campanella" wording hides that within-dataset BA is also at chance (0.478). |
| **REV-15** Warm-up control | **Addressed** (with caveat) | EPM-E4 control and "settling" wording at L227 (`rev15_settling_control.csv`). WESAD first-minute temperature 4–6 °C above (median 5.66 in the CSV). | Caveat (N10): EPM-E4 contains fear and anger clips, so it is not a stimulus-free control. Its tonic-EDA slope over minutes 1–10 is −0.035 [−0.19, 0.11] with 45% of subjects rising; the "drifts upward" claim rests only on the 15–19 vs 1–4 min difference. |
| **DA-C1** Circular matched-arousal test | **Addressed** (by concession plus analyses) | Δref, positive control and disjoint variants (L208–215). The Discussion concedes "Two readings remain open and the present data cannot separate them" (L233). The thesis now rests mainly on the index-only LODO result. | Tabulate the disjoint cells. The positive control shows the small pairs (UBFC n = 80) cannot detect a 1-SD feature shift; that is stated ("Only the large pairs can therefore support a null", L213) and should be carried into the abstract's null list. |
| **DA-C2** Evaluative claim rests on one underpowered contrast | **Addressed** | "Evaluative" removed from the title and claims. The UBFC control is described as lower-demand. MDD and TOST reported. "These are absence-of-evidence results" (L211). | none blocking |
| **DA-C3** "Stressor potency" circular | **Addressed** (minor residuals) | Renamed and declared descriptive (L233). The self-report nulls are not treated as contradictions (Supp L75). | Leave-two-out WESAD ρ. Remove "rather than the pipeline" (L263). |

**Counts.** REV items: 6 Addressed (4, 5, 7, 9, 14, 15), 9 Partly (1, 2, 3, 6, 8, 10, 11, 12, 13), 0 Not addressed. DA CRITICAL: 3 of 3 Addressed (C1 by concession).

---

## 2. Number verification (spot-check of numbers added in revision)

I checked 58 numbers against the committed CSVs. None were substantive mismatches. There are 3 rounding or wording discrepancies and one cross-table inconsistency:

| # | Manuscript | Value | CSV | CSV value | OK? |
|---|---|---|---|---|---|
| 1–5 | L126 LODO costs | 0.03/0.03/0.05/0.04/0.015 | D_lodo_cost_corrected | 0.0316/0.0302/0.0517/0.0394/0.0148 | ✓ |
| 6 | L126 Stress-Predict CI | 0.052 [0.003, 0.101] | D | 0.0517 [0.0028, 0.1006] | ✓ |
| 7 | L126, abstract: upper bounds | 0.08–0.13 | D hi | 0.076–0.132 | ✓ |
| 8 | L126 "p_holm ≥ 0.20" | 0.20 | D p_holm (Stress-Predict) | **0.1962** | ≈ (should read ≥ 0.19 or ≈ 0.20) |
| 9–13 | L128 index external BA | 0.836/0.743/0.657/0.762/0.877 | C ba_splits | 0.8362/0.7434/0.6569/0.7625/0.8766 | ✓ |
| 14 | L128 index AUROC | 0.73–0.95 | C auroc_splits | 0.731–0.951 | ✓ |
| 15–16 | L128 model − index | +0.048 WESAD, +0.075 UBFC | C | 0.048, 0.0748 | ✓ |
| 17 | L128 p_holm | ≥ 0.14 | C | 0.1376 | ✓ |
| 18 | L128 EDA-only PhysioNet | −0.15, significant | C | 0.1472, p_holm 0.0007 | ✓ |
| 19–30 | Table IV Δref, all 12 rows | e.g. +0.17 [+0.08, +0.25] … −0.29 [−0.38, −0.12] | A_reference_relative | all match to 2 dp | ✓ |
| 31–42 | Table IV p_holm and MDD | e.g. 0.41, 0.62, 0.13; MDD 0.07–0.22 | A p_perm_holm; B mdd_80pct | all match | ✓ |
| 43 | Table IV note: hyperventilation equivalent | 90% CI [0.45, 0.60] | B | 0.4492–0.5957, equiv_0.10 True | ✓ |
| 44 | L211 "0.15–0.29 below, CIs excl. 0" | | A cond3 | 0.155–0.292, all hi < 0 | ✓ |
| 45 | L211 exercise HR/EDA-only index | 0.91 / 0.82 (ext 0.65 / 0.54) | partA_matched_arousal (same_dataset) | 0.912 / 0.818 (0.654 / 0.538) | ✓ |
| 46 | L211 exercise-minus-pair lower bounds | 0.18–0.36 | partA_inference diff_lo | 0.179–0.361 (pooled over both models) | ✓ (ambiguous which model) |
| 47 | L213 exercise Δref raw / baseline | −0.37 / −0.22 | 3b | −0.366 / −0.224 | ✓ |
| 48 | L213 fear Δref complete-case / raw | 0.00 / +0.18 | 2_missingness; 3b | −0.0005 / +0.1788 | ✓ |
| 49 | L213 eda_slope injection on UBFC | Δ = 0.04, CI crosses 0 | 1_positive_control | 0.0409 [−0.014, 0.096] | ✓ |
| 50 | L213 Lego external under screen | 0.32 | eda_quality matched_arousal (eda_rules_only) | 0.315 | ✓ |
| 51 | L215 disjoint Lego / anger / exercise | 0.73, 0.84 / 0.74 / 0.29 | F_channel_disjoint | 0.7264, 0.8389 / 0.7425 / 0.2873 | ✓ |
| 52 | L215 hyperventilation and UBFC in every disjoint cell | 0.45–0.54 | F | 0.4537–0.539 | ✓ |
| 53 | L215 WESAD stress without pulse | 63% vs 10% | E_hr_missingness | 0.633 vs 0.102 | ✓ |
| 54 | L170 exercise subsampled to 80; raw/baseline | 19%; 81–83% → 19–30% | 4_class_size; 3a | 0.1865; 0.814/0.831 → 0.297/0.194 | ✓ |
| 55 | L225 position-matched time-only | 0.60–0.62; Stress-Predict 0.88 (phys 0.67) | rev9_position_matched_summary | 0.602/0.62; 0.881 (0.67) | ✓ |
| 56 | L227 EPM temperature / EDA; stress datasets | −0.02 °C, +0.69 µS; +0.39–0.87, +0.33–0.59 | rev15_settling_control | −0.0228, 0.6929; 0.393–0.873, 0.326–0.588 | ✓ |
| 57 | L235 HR/HRV-only | Campanella 0.50, Stress-Predict 0.56; exercise 96–97% → 45% | eda_quality lodo_summary; specificity_panel | 0.501, 0.559; 0.957/0.970 → 0.446/0.450 | ✓ |
| 58 | L52 demographics | WESAD 27.5, 12/15 male; Stress-Predict 32, 10/35; Campanella 21/29; UBFC 21.8 (n = 56) | demographics.csv | same | ✓ |
| — | Supp L48 budget | label-free ≤ 0.056 (hi 0.09); F4 ≤ 0.022; residual 0.17–0.23, lower CI 0.11–0.28 | rev8_budget_bootstrap_ci | 0.0564 [0.019, 0.092]; 0.0218; 0.175–0.229, lo 0.108–0.281 | ✓ |
| — | L101 EDA effect sizes | 0.4 / 0.7 vs 1.7–2.2 z | stress_effect_size_all_datasets | 0.39 / 0.72 vs 1.72–2.24 | ✓ |

**Discrepancies:**
- **Cross-table inconsistency (new).** The "added as negative" false-stress rates for the same configuration (HR/HRV/EDA, no screen) differ between `contribution_probes/specificity/partA_panel_compact.csv` (used in Table III) and `eda_quality_hr_arm/specificity_panel.csv`:

  | State | Table III (compact) | eda_quality table |
  |---|---|---|
  | Anger | 0.35 | 0.27 |
  | Aerobic | 0.099 | 0.083 |
  | Anaerobic | 0.081 | 0.071 |
  | UBFC control | 0.563 | 0.525 |
  | Fear | 0.252 | 0.244 |
  | Happiness | 0.183 | 0.172 |
  | Sadness | 0.301 | 0.313 |

  The external rates match exactly, so the added-negative training differs between the two scripts (pooling, seed or split). Reconcile or document.
- **Kleckner wording.** L213 says the screen "leaves every null pair at or below the reference". In fact the UBFC control is slightly above the reference under both screens: Δ +0.023 [−0.07, 0.13] (Kleckner) and +0.016 [−0.09, 0.13] (EDA rules).
- **Units.** Table IV's n is the probe-window count (the reference n of 3839 is baseline plus rest windows). L213 calls 5445 (the pair total) the "reference pair" and 80 (probe windows only) the "UBFC-Phys pair". Define n in the caption and use one unit.

---

## 3. New issues introduced by the revision

**Substantive**
- **N1. The title and abstract claim equivalence from non-significance, which repeats round-1 blocker B2.** "An Untrained Arousal Index **Matches** Them" and "performs as well" rest on no model-minus-index difference surviving Holm correction at 20 splits. The UBFC-Phys gap is 0.075 BA (raw p = 0.028) and WESAD's is 0.048. No CI or equivalence margin is given, although L128 correctly hedges with "Up to the power of these tests". Fix: report Nadeau–Bengio CIs on model minus index and a ±0.05 TOST (0.5 day, the data exist), or retitle ("…Performs Close to Them" / "…Is Hard to Beat").
- **N2. The abstract is over-specific and over-general at once.** It has about 15 numbers against the roadmap's target of about 5, lists the small exploratory cells (anger with 1 window per subject, hyperventilation with 1.7) as evidence without flagging them, and concludes about "wrist stress detectors" from one fixed XGBoost configuration on one device.
- **N3. The EPM-E4 settling control is not stimulus-free.** Its clips include fear and anger, and over minutes 1–10 its tonic-EDA slope is negative. The inference at L227 ("EDA drift… looks like electrode and skin settling, whereas the temperature rise goes with the stress protocols") needs hedging.
- **N4. The Kleckner screen is described only in part.** "Reject 0–11%" is the EDA-rules subset. The full Kleckner screen and the flat-signal flag remove up to 53% and 79% of some classes, and under the full screen Campanella's within-dataset BA falls from 0.882 to 0.735 (`lodo_summary.csv`, not reported). Report the rejection and flat rates by state in the supplement.
- **N5. The robustness paragraph is selective in two small ways.** It omits the HR-missing stratum, where the UBFC control exceeds the reference (+0.18, n = 17, p_holm 0.065). It also reports only the feature-level positive control; the score-level control is detected even on the 80-window UBFC pair.

**Presentation and IEEE style**
- **N6.** A dangling reference: Supp L70 `Table~\ref{tab:panel}` renders as "Table ??" in `supplement.pdf`, because the label lives in main.tex. Hard-code "Table III of the main paper" or use `xr`.
- **N7.** L227 has a garbled run-on sentence (SC-6 persists in a new place).
- **N8.** The email placeholder is still at L18. There is no ethics or data-use statement, no funding line and no conflict-of-interest statement.
- **N9.** 21 `\texttt{}` CSV and feature names in body text; IEEE style would move them to the Data Availability section. Also the stray comma at L44 and the "(abstract only)" tag inside the Akkaya bibliography entry (L288, Supp L79).
- **N10.** The Fig. 2 caption and L211 call the reference "0.58 externally", and L233 says "0.784 within dataset". The within-dataset reference pools per-dataset LOSO models over stress windows from all datasets (the SC-23 mixing). Say so, or compute it per dataset.
- **N11.** L235: "transfers at chance to Campanella" (HR/HRV-only) should say that within-dataset BA is also at chance (0.478), so this is a ceiling, not a transfer failure.
- **N12. Length.** The main text is exactly 10 pages; body text ends on page 9 and the references fill page 10. Check the current JBHI author guidelines: to my knowledge pages beyond 8 incur over-length charges and 10 is the hard cap. The abstract has 237 words, which fits a 250-word limit if that still applies. Any additions (a psychophysiology paragraph, an ethics statement) must be offset by cuts, for example the contributions paragraph (L37) and the inline file names.

---

## 4. Updated decision: **Minor Revision** (round 1 was Major Revision)

**Rationale.**
- **B1 (C1) is resolved.** The thesis no longer depends on the circular matched-arousal test: the revision adds an index-only LODO test, Δref, a positive control and disjoint variants, and concedes that the two readings cannot be separated.
- **B2 (C2) is resolved for the matched-arousal and LODO-cost claims.** MDD, TOST, CIs and "absence-of-evidence" wording are now in place.
- **B3 is largely resolved.** The evaluative claim and "because" are gone, the comparators are described accurately, and potency has been renamed.
- **All numbers are accurate.** Every one of the 58 checked numbers matches a committed table.
- **What remains** is one reintroduced overclaim (N1, the "Matches" title, which needs a half-day analysis or a retitle), a set of text-level omissions (ethics statement, psychophysiology positioning, placeholders, a dangling reference, a garbled sentence), and several secondary analysis items that a JBHI referee would likely accept as limitations.

None requires new data. The correlated-error caveat applies: a panel from a different model family, or a human reviewer, might weigh the scope limits (one device, one model family, 132 lab subjects) more heavily and hold to Major Revision.

### Must do before submission (ordered)

| # | Item | Type | Effort |
|---|---|---|---|
| 1 | N1: Nadeau–Bengio CIs and a ±0.05 TOST on model minus index (all five targets); then retitle and reword the abstract to match the result ("matches" only if TOST passes) | analysis + text | 0.5–1 day |
| 2 | Fix the supplement "Table ??" (N6), the garbled L227 sentence (N7) and the email placeholder (N8) | text-only | 1 h |
| 3 | Add an ethics and data-use statement (original approvals, licences), plus funding and conflict-of-interest lines | text-only | 1 h |
| 4 | Reconcile the two added-negative panel tables (§2), or state which configuration Table III uses and why the eda_quality rerun differs | analysis (rerun or document) | 0.5 day |
| 5 | Correct L213: UBFC is slightly above the reference under the screen; give full-screen and flat rejection rates (supplement table); mention the HR-missing stratum and the score-level positive control (N4, N5) | text + existing CSVs | 0.5 day |
| 6 | REV-11 remainder: a short psychophysiology paragraph (Kreibig 2010; Dickerson & Kemeny 2004) with full-text quotes per the project rule; replace the Richer citation for HPA and cortisol; read Akkaya in full or drop it; Prajod's intensity confound | text + reading | 1–2 days |
| 7 | Abstract: at most about 6 numbers; flag anger and hyperventilation as exploratory; scope "wrist stress detectors" to "stage-labelled HR/HRV/EDA detectors on these datasets" (N2) | text-only | 2 h |
| 8 | Hedge the EPM-E4 settling inference (N3); report the Campanella slope CI crossing zero | text-only | 1 h |
| 9 | Define n in Table IV and use one unit in L213 (§2); change p_holm ≥ 0.20 to ≥ 0.19 | text-only | 0.5 h |
| 10 | Page budget: cut L37 to two contributions and move the inline CSV names to the Data Availability section, to make room for items 3 and 6 | text-only | 2 h |

### Should do (a referee may ask; can be deferred to the response letter)

| Item | Type | Effort |
|---|---|---|
| Tabulate all disjoint-variant cells with whole-family Holm (supplement) | analysis (numbers exist) | 0.5 day |
| Score EPM pairs with one model, or rank-normalise per model (SC-23) | analysis | 1 day |
| Leave-two-out WESAD ρ; per-filter removed fractions in the supplement | analysis (numbers partly exist) | 0.5 day |
| Manipulation-check table from existing self-reports; state-grading column in Table I | light analysis + text | 1–2 days |
| False alerts per hour at an assumed prevalence; numeric settling duration and activity-gate definition in the checklist | text + light computation | 0.5 day |
| Tag a release (or Zenodo DOI) and add the window-level feature table; one-line analysis-history statement | text + repo | 0.5 day |
| Explain AUROC < 0.5 (inverse ranking within bins) | text | 1 h |
| Report the equal-weight joint runs; state each probe's reference session | text (numbers exist) | 2 h |

**Optional (new data):** enlarge the anger, hyperventilation and UBFC-control cells, or run the within-subject crossed study. This is not a condition for this revision.
