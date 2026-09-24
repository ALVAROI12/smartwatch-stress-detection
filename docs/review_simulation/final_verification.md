# Final pre-submission verification: paper/main.tex, supplement.tex (HEAD 43e520b)

## 1. Verdict

**Ready after the fixes listed below.** Every number in the new per-dataset matched-arousal material (Table IV, V-D, abstract, contributions, discussion, conclusion, checklist, Fig. 2) and in the two-feature LR material matches the committed tables. The Dickerson & Kemeny quote and its study count check out word for word. No result is wrong in a way that changes a conclusion. What needs fixing: one leftover overclaim about the fear clips (section 3), one rounding error in Table IV, the bibliography numbering (IEEE order of first citation), two terminology and consistency slips, the Siegel quote (it cannot be checked against the local text), and the author placeholders.

## 2. Wrong or unverifiable items

| main.tex line | Claim | Source value | Fix (exact replacement) |
|---|---|---|---|
| 196 (Table IV, external Lego) | `\m0.25 [\m0.44, \m0.08]` | delta_same_dataset.csv: diff_lo = −0.4349 | `\m0.25 [\m0.43, \m0.08]` |
| 232 (Discussion) | "Exercise and fear clips separate from stress at matched arousal only under session scaling" | 3b_normalisation_matched.csv: fear under raw scaling is 0.856, Δref +0.18 [0.11, 0.23], p_holm < 0.001. The fear margin disappears only on complete-case windows (2_missingness_strata: Δ −0.0005) | "Exercise separates from stress at matched arousal only under session scaling, and the fear-clip margin vanishes on windows with usable pulse, so neither separability is a robust property of the wrist signal;" |
| 175 (Fig. 2 caption) | "(not robust to other scalings, Section~V-D)" | Same as above: fear is robust to raw scaling | "(exercise only under session scaling, fear clips not on windows with usable pulse; Section~V-D)" |
| 128 (V-B) | "with near-equal learned weights for the two channels" | two_feature_lr/summary.csv: coef_eda/coef_hr = 1.09, 1.54, 1.45, 1.09, 0.93 | "with weights of similar size for the two channels (EDA-to-heart-rate ratio 0.9--1.5)" |
| 212 (Robustness) | "...leaves every null pair within 0.03 of the pooled reference or below it and pushes Lego further below (external within-bin AUROC 0.32)" | eda_quality_hr_arm/matched_arousal.csv: 0.315 is the EDA-rules-only screen. Under the full Kleckner screen Lego is 0.411 (Δ −0.186, *less* far below than the unscreened −0.229) | "...and, under the EDA rules, pushes Lego further below (external within-bin AUROC 0.32)" |
| 210 (V-D) | "Exercise reaches 0.892 within bins once the model has seen it ($p_{\mathrm{holm}}=0.003$)" | 0.003 is the permutation test against 0.5 (partA_inference.csv). Table IV's p_holm column for the same row reads <0.001 (Δref test), so readers see two p-values for one row | "...once the model has seen it (against 0.5, $p_{\mathrm{holm}}=0.003$)" |
| 224, 220 (V-E text, Fig. 3 caption) | "physiology 0.78 / 0.70", "physiology model", "against the physiology model" | time_probe_summary std_within: PhysioNet 0.776, Stress-Predict 0.698. These are the HR/HRV/EDA values (Table II); the "physiology" set (with temperature) gives 0.760 / 0.712. Section IV-A defines "physiology" as the temperature set | Replace "physiology" with "HR/HRV/EDA" in V-E (L224: "(HR/HRV/EDA 0.78; …)", "(HR/HRV/EDA 0.70; …)", "The HR/HRV/EDA model survives position-matched negatives", "(HR/HRV/EDA 0.67)") and in the Fig. 3 caption ("against the HR/HRV/EDA model"). Same in L229 if it is meant: "does not beat physiology" → "does not beat the HR/HRV/EDA model" (L226) |
| 232 (Discussion, Siegel) | "a meta-analysis of 202 emotion-induction studies in which the pattern of autonomic effect sizes ``did not clearly distinguish one emotion category from another''" | **UNVERIFIABLE locally.** sources/novelty/siegel-2018-emotion-fingerprints.txt contains no abstract. The quote string is absent, "202" is absent, and the Methods say "leaving 204 unique studies from 195 full-text articles". Commit 8cb0aa7 says the quote was checked against PMC5876074 | Paste the PMC abstract into the source file and re-check. If the abstract does not say 202, use "a meta-analysis of over 200 emotion-induction studies" |
| 47 (Related Work, Sosa) | two verbatim quotes | No full text in sources/. The quotes match the verbatim record in docs/literature/contribution-gap-scan.md (marked FT) | No text change. Add the full text to sources/novelty/ for the audit trail |
| 47 (Related Work, Watanabe) | "found the Trier Social Stress Test (TSST) to be the worst condition for E4 beat detection on WESAD" | watanabe-2025 text: the WESAD stress task (and the Stress-free arithmetic and cold-pressor tasks) were "more sensitive to low-cut frequencies than those collected during rest". No "worst" or TSST ranking found | "found PPG beat detection during the WESAD stress task more sensitive to filter settings than at rest" (or cite the exact table if one ranks conditions) |
| 88 (Features) | "windows with fewer than 50\% usable beats carry missing cardiac values" | extract_features.py: MIN_CARDIAC_COVERAGE = 0.5 is the fraction of the window *covered* by accepted beats | "windows less than half covered by accepted beats carry missing cardiac values" |
| 101 (V-A) | Nadeau–Bengio CIs 0.79–0.88, 0.76–0.86, 0.68–0.82 and the LOSO accelerometer value 0.75 | repeated_subject_splits.csv stores only the uncorrected CI (0.81–0.85); loso_summary.csv has no acc_only row for this task. Means 0.83/0.81/0.75 and LOSO 0.85/0.81 match | UNVERIFIABLE from committed tables. Commit the corrected-CI table, or cite the script that prints the values |
| 18, 267 | e-mail placeholder `[email]@utsa.edu`; TODO for Funding and Conflict of Interest | — | Fill in before submission (JBHI requires both statements) |
| supplement 75 | "Stroop (mean rise 0.7 of 10) … arithmetic (2.5)" | mild_stress_by_task.csv: these are window-weighted (0.667, 2.495). The subject means are 0.53 and 2.24–2.38, and main text L170 says Stroop 0.5 and TMCT 2.4 | "Stroop (mean rise 0.5 of 10 per subject) is detected as well as the TMCT (2.4)". Alternatively keep the values and add "window-weighted" |

## 3. Consistency problems, with exact fixes

1. **Bibliography order (IEEE numbers references by first citation).** main.tex bibitems 13–32 are out of order: siegel2018, dickerson2004 and kleckner2018 sit in the middle, and iqbal2022 comes after garcia. Reorder the `\bibitem`s after `saeb2017` to: bahameish2024, schmidt2019, vos2023, iqbal2022, stewart2020, han2024, tervonen2026, akkaya2026, milstein2020, watanabe2025, sosa2026, schmidt2018wesad, hongn2025data, meziati2023, campanella2024, garcia2020epm, kleckner2018, nadeau2003, siegel2018, dickerson2004. In supplement.tex, reorder to kwon2026, tervonen2026, akkaya2026. All 32 keys are cited and all cited keys are present, and there are no undefined `\ref`s.
2. **The fear-clip robustness overclaim** (Discussion L232, Fig. 2 caption L175) contradicts the Robustness paragraph (L212). Fixes are in section 2.
3. **Table IV `n` for exercise (3585) differs from Table III (2143 + 1620 = 3763)** without explanation. Add to the Table IV footnote: "$n$: probe windows with a defined index (178 exercise windows lack both heart rate and EDA)."
4. **Error budget, main L229:** "buy 0.05--0.06 BA on PhysioNet and Stress-Predict" omits that the external Stress-Predict model gains only 0.02. The supplement states it. Replace with "buy 0.05--0.06 BA on PhysioNet and Stress-Predict (0.02 for the external model on Stress-Predict)".
5. **Deployment, L234:** "false-stress rates of 0.10, 0.33 and 0.69, the range of Table III". Table III spans 0.08–0.70. Replace with "spanning Table~\ref{tab:panel}".
6. **Fig. 1:** the "no temperature channel in target" annotation overlaps the UBFC-Phys bars in the right panel (PDF p. 5). Move the xytext in make_paper_figures.py L37 (for example to (1.2, 1.02)) or drop the annotation, since the caption already says it.
7. **Checked with no problem found:**
   - Title "Nearly Matches" matches "within 0.075".
   - Abstract, contributions, Discussion and Conclusion all state 0.95 for WESAD and 0.57–0.70 for the other four. No leftover "only weakly", pooled 0.58 or 0.78 used as the reference (grep clean). The pooled 0.58 appears only as a labelled pooled bar, as the pooling artefact, and in the Robustness paragraph, where the text states it is pooled.
   - Section cross-references (V-B, V-D, V-E, II–VI) are correct.
   - Abstract length: 232 words, under the 250 limit.
   - The main PDF is 10 pages and the supplement 2. The rendered Table IV matches the .tex.

## 4. Counts

- **Priority 1 (changes since 584fbe2 and the listed text):** about 235 items checked.
  - Table IV: 72 cells.
  - Table II: 5 LR, 5 index and 30 model values.
  - About 60 numbers in the V-B, V-D, Robustness, EDA-screen and channel-disjoint text.
  - About 25 numbers in the abstract, contributions, Discussion and Conclusion, including the Kwon reconciliation values (0.98, 0.93–0.95).
  - 11 manipulation-check values and 12 demographic values.
  - EDA methods, compared line by line with extract_features.py.
  - 2 Dickerson items (quote and 208 studies), both verbatim.
  - 2 Siegel items, not verifiable locally.
  - Result: 2 wrong numbers (Table IV CI; the Lego 0.32 attribution), 1 overclaim (fear robustness), 2 wording inaccuracies (LR weights, cardiac coverage), 1 terminology slip (physiology vs HR/HRV/EDA), 1 unverifiable quote (Siegel).
- **Priority 2:** 32 bibliography keys, 5 labels and refs, 4 figure captions checked against the rendered PDF pages, every section reference, abstract length, and a grep for stale pooled-reference claims. Result: 6 consistency fixes.
- **Priority 3 (spot checks of older numbers, 18 done):** Table I counts, wrist vs ECG (2.4 bpm, r = 0.94, RMSSD r = 0.53, +34 ms, 37%, −7 bpm), leakage (92%, 49%, 0.36), threshold probe (0.589, 0.931, 0.910, 0.18, 0.31–0.42, oracle 0.02–0.05, 0.867→0.732), EDA effect sizes (0.4, 0.7, 1.7–2.2; a sixth to a third), causal and raw normalisation (0.845/0.841, 0.774/0.744, 0.688/0.665, −0.01 to 0.05, 60 tests, none significant), label fixes (0.790→0.837, +0.0075, p = 0.021), Kwon setting (0.67, 0.834→0.914, ≤0.02), time probe (0.83, 0.95, 16/20, 20/20, 0.66→0.56, 0.90→0.45, position-matched ≤0.03/0.04, 0.60–0.62, 0.88), warm-up (0.02–0.08 °C/min, +1.8/+0.9, 13/15, +6.5, 16–26 min, 17–49 min, 68/90/97/100%), settling control (−0.02, +0.39 to +0.87, +0.69, +0.33 to +0.59), Table III (all 27 cells plus references), HR/HRV-only arm (0.50, 0.56, 96–97%, 45%), supplement Table S-I (25 cells), chest-ECG bound (0.916→0.956, 0.918, 0.891, 0.886→0.962), tuning (108 configurations, 1 of 72 significant), modality means. Result: all correct except the unverifiable Nadeau–Bengio CIs.
- **Older quotes re-checked verbatim:**
  - Correct: Prajod, Schmidt 2019, Vos/Iqbal, UBFC-Phys ("imply a high stress level", steps of 17 and 10), Campanella ("mental strain"), Stress-Predict ("stress-inducing tasks"), Kwon numbers, Milstein 0.42, Tognotti 3–13, Tervonen 1.7, Aydoğan (82.6%, 71.0%, 0.062/0.070), Richer (sequence effects, friendly TSST).
  - Wording overstated: Watanabe.
