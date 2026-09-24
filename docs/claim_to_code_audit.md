# Claim-to-code audit of `paper/main.tex` (branch jbhi-revision, HEAD 4ab234a)

Read-only audit, 2026-09-23. Every table cited below is committed (`git ls-files outputs/tables`). Line numbers refer to `paper/main.tex`. Paths are relative to the repo root: `T/` = `outputs/tables/jbhi_v2/`, `CP/` = `outputs/tables/jbhi_v2/contribution_probes/`, `NE/` = `outputs/tables/jbhi_v2/novelty_experiments/`.

Caveat on method: partway through, the Bash tool was blocked by a worktree-isolation hook. The script-to-table mapping was built with grep before the block (every committed table basename grepped across `scripts/*.py`; f-string names resolved by hand). All value checks were done by reading each CSV in full. Row 125 (split counts) was counted by hand from the per-split CSV.

## Summary counts

| Class | Rows |
|---|---|
| BACKED | 135 (132 claims + 3 figures) |
| MISMATCH | 27 (26 claims + 1 figure caption) |
| TABLE ONLY | 0 (none of the cited values; 7 uncommitted-writer tables exist but are not cited, see Problems §D) |
| UNBACKED | 10 |
| CITATION | 9 groups (about 25 claims about other papers) |
| **Total rows** | **181** |

Serious mismatches (they change a stated result or method): #47 (0 of 48 vs 1 of 72 significant), #48 (15% vs 20% held out), #72 (all-five training does not match within-dataset on UBFC-Phys), #99 (recall is not "unchanged or higher"), #146 (strict-rule residuals), #148 (PhysioNet recall after 3 min), #157 (one baseline-EDA correlation excludes 0), #167 (118 subjects), #30 (Campanella end is estimated, not logged), #116 (exercise single-index AUROCs come from a different model). The rest are small range or wording errors.

## Claim table

| # | main.tex line | Claim (short) | Value in paper | Table | Value in table | Script (writer) | Class |
|---|---|---|---|---|---|---|---|
| 1 | 25, 37, 280 | LODO transfer cost, HR/HRV/EDA | 0.015–0.05 | T/leave_one_dataset_out_summary.csv | 0.015–0.051 | leave_one_dataset_out.py:119-122 | BACKED |
| 2 | 25, 37 | DA, few-shot and per-user threshold add nothing significant | none | T/domain_adaptation_physiology.csv; T/fewshot/fewshot_summary_session_chronological_gap30.csv | no gain significant (threshold significantly hurts PN) | domain_adaptation.py:217-218; fewshot_unseen_dataset.py:137 | BACKED |
| 3 | 25 | Exercise false alarms | 69% → 8% | CP/specificity/partA_panel_compact.csv | 0.692/0.695 → 0.099/0.081 | specificity_panel_probe.py | BACKED |
| 4 | 25 | Speech+arithmetic without evaluation | 55% → 56% | same | 0.550 → 0.563 | specificity_panel_probe.py | BACKED |
| 5 | 25 | Hyperventilation | about 50% | same | 0.415 / 0.538 / 0.547 | specificity_panel_probe.py | BACKED (approx.) |
| 6 | 25 | Manual tasks | 43% → 33% | same | 0.427 → 0.328 | specificity_panel_probe.py | BACKED |
| 7 | 25 | Matched arousal, non-exercise pairs, none ≠ chance after Holm | AUROC 0.36–0.58 | CP/hardening/partA_inference.csv | 0.355–0.577; p_holm 0.126–0.619 | matched_arousal_hardening.py | BACKED |
| 8 | 25 | Exercise separable | 0.89 | same | 0.892 | matched_arousal_hardening.py | BACKED |
| 9 | 25 | Time beats physiology on two hardest datasets | qualitative | CP/timeprobe/time_probe_summary.csv | PN 0.826 vs 0.776; SP 0.947 vs 0.698 | time_in_session_probe.py:98-100 | BACKED |
| 10 | 25, 37 | Error budget explains at most 0.06 of 0.22–0.29 error | 0.06; 0.22–0.29 | CP/arousal/partB_error_budget.csv | four-filter gain 0.018–0.064; within 1−BA 0.224/0.290 | matched_arousal_probe.py:231 | BACKED |
| 11 | 25, 37, 271 | Strict responder filter | up to 0.09 | CP/hardening/partB_responder_sensitivity.csv | 0.094 | matched_arousal_hardening.py | BACKED |
| 12 | 52 | Subjects per dataset | 15/35/34/19/29; EPM-E4 33 | T/harmonization_table.csv | same | run_jbhi_experiments.py:105-111, 219 | BACKED |
| 13 | 52 | E4 sampling rates | 64/4/4/32 Hz | code | extract_features.py:34 | — | BACKED |
| 14 | 63-64 | WESAD windows | 311; 566/356/165 | T/harmonization_table.csv | same | run_jbhi_experiments.py | BACKED |
| 15 | 65 | PN stress | 35 / 232 | same | 39+1+192 = 232; 35 subj | run_jbhi_experiments.py | BACKED |
| 16 | 66 | PN baseline; rests | 174; 1193 | same | 174; 621+572 | run_jbhi_experiments.py | BACKED |
| 17 | 67 | Aerobic; anaerobic | 30/2143; 31/1620 | same | same | run_jbhi_experiments.py | BACKED |
| 18 | 68 | SP stress subjects / windows | 34 / 851 | T/harmonization_table.csv; CP/specificity/partA_specificity_panel.csv | 851 windows, but 33 subjects have stress windows (Stroop 31, TSST 32; panel n_subjects = 33) | run_jbhi_experiments.py; specificity_panel_probe.py | MISMATCH |
| 19 | 69 | SP baseline; relax | 34 / 616; 1079 | T/harmonization_table.csv | same | run_jbhi_experiments.py | BACKED |
| 20 | 70 | Hyperventilation | 31 / 53 | same | same | run_jbhi_experiments.py | BACKED |
| 21 | 71-73 | UBFC stress/baseline/control | 11/110; 19/95; 8/80 | same | same | run_jbhi_experiments.py | BACKED |
| 22 | 74-76 | Campanella | 29/102; 116; 667 | same | 102; 116; 58+174+435 | run_jbhi_experiments.py | BACKED |
| 23 | 77 | EPM-E4 | 33; 33/394/297/131 | same | same | run_jbhi_experiments.py | BACKED |
| 24 | 82 | WESAD raw E4 starts earlier | 16–26 min | CP/wesad_warmup/wesad_offsets.csv | 16.0–25.8 | warmup_since_donning.py | BACKED |
| 25 | 82 | PN stress windows under old whole-session labels | 248 of 2,933 | none | not found in any committed table (v1 harmonization table has other counts) | — | UNBACKED |
| 26 | 82 | PN f13 excluded | — | code | extract_features.py:40-41, 173 | — | BACKED |
| 27 | 82 (66) | PN baseline = 3 min before first tag | 180 s | code | relabel_windows.py:33-35, 72-75 (v2 sessions only; v1 has marked baselines, lines 39) | — | BACKED (note) |
| 28 | 82 | SP S01 dropped | — | code | extract_features.py:210-212, 259 | — | BACKED |
| 29 | 82 | SP boundaries > 180 s from a tag dropped | 180 s | code | extract_features.py:213-248 | — | BACKED |
| 30 | 82 | Campanella subtraction ends at each subject's logged end | logged per subject | code | extract_features.py:292-306, 333: no per-subject markers exist; end = recording end − 300 s tail − 15 s guard (capped at 1800 s) | — | MISMATCH |
| 31 | 82 | Hyperventilation kept as separate probe | — | code | extract_features.py:207; excluded from LODO task, leave_one_dataset_out.py:33, 88 | — | BACKED |
| 32 | 82 | Three participants recorded in two files merged | 3 | code only | merge in extract_features.py:183, relabel_windows.py:118; count of 3 not tabulated | — | UNBACKED (count) |
| 33 | 84 | Earlier pipeline reported | 94% | T/leakage_check.csv | reproduction gives 0.921 (random 15% windows); 94% (README 94.53%) is in no table | run_jbhi_experiments.py:114-125 | UNBACKED |
| 34 | 84 | Same model, 15% subjects held out | 49% (BA 0.36) | T/leakage_check.csv | 0.494 (BA 0.355) | run_jbhi_experiments.py:114-125, 220 | BACKED |
| 35 | 88 | 60 s windows, 30 s step, inside one stage | — | code | extract_features.py:33, 116-133 | — | BACKED |
| 36 | 88 | NeuroKit2 beats after band-pass and artefact rejection | — | code | extract_features.py:44-53 (nk.ppg_clean + ppg_findpeaks; IBI 330–1500 ms and <20% off rolling median) | — | BACKED |
| 37 | 88 | <50% usable beats → missing cardiac | 50% | code | extract_features.py:35, 56-63 (coverage of window by accepted IBIs < 0.5, or < 20 beats) | — | BACKED (wording) |
| 38 | 88 | Wrist vs chest ECG, WESAD | 1,398 win; MAE 2.4 bpm; r 0.94 | T/wrist_vs_ecg_summary.csv | 1398; 2.38; 0.94 | validate_cardiac_against_ecg.py | BACKED |
| 39 | 88 | RMSSD | r 0.53; bias +34 ms | same | 0.53; 33.87 | validate_cardiac_against_ecg.py | BACKED |
| 40 | 88 | TSST usable PPG; HR bias | 37%; −7 bpm | same (Stress row) | 36.66%; −7.19 | validate_cardiac_against_ecg.py | BACKED |
| 41 | 88 | EDA feature list | — | code | extract_features.py:85-93 | — | BACKED |
| 42 | 88 | Temperature only in "physiology" set | — | code | leave_one_dataset_out.py:81-82 | — | BACKED |
| 43 | 91 | Whole-session per-subject z-scoring, label-free | — | code | leave_one_dataset_out.py:43-53, 83-87 | — | BACKED |
| 44 | 91 | Raw and causal variants; first four windows share statistics | 4 | code | leave_one_dataset_out.py:35, 45-46, 54-62 | — | BACKED |
| 45 | 94 | Task = stress vs baseline, rest, meditation, amusement | — | code | leave_one_dataset_out.py:33, 88 | — | BACKED |
| 46 | 94 | XGBoost, fixed hyperparameters | — | code | probe_normalisation.py:32-33 (LODO and probes); run_jbhi_experiments.py:69-74 | — | BACKED |
| 47 | 94 | Nested tuning: model comparisons significant after Holm | 0 of 48 | T/significance_models.csv | 1 of 72 significant (affective_only_no_exercise, subject_zscore, physiology_only, mlp vs xgboost, p_holm 0.038) | compare_models.py:66-72 | MISMATCH |
| 48 | 94 | Subject-grouped splits hold out | 15% of subjects | code | run_jbhi_experiments.py:82 `test_size=0.2`; used by leave_one_dataset_out.py:104 and all probes; compare_models.py:21 assumes 0.2/0.8. Only leakage_check uses 0.15 | — | MISMATCH |
| 49 | 94 | LOSO where per-subject scores are needed | — | code | run_jbhi_experiments.py:141-157; matched_arousal_probe.py:35-41 | — | BACKED |
| 50 | 94 | LODO schemes (a)(b)(c) | — | code | leave_one_dataset_out.py:101-116 | — | BACKED |
| 51 | 94 | Nadeau–Bengio corrected t + Holm | — | code | compare_models.py:25-37 (also label_fix_significance.py, novelty_experiments_summary.py) | — | BACKED |
| 52 | 94 | Subject-bootstrap 95% CIs on per-subject rates | — | CP/specificity/partA_panel_compact.csv (ci columns) | present | specificity_panel_probe.py | BACKED (line not pinned) |
| 53 | 94 | Threshold 0.5 throughout | 0.5 | code | leave_one_dataset_out.py:113; matched_arousal_probe.py:190 | — | BACKED |
| 54 | 97 | Specificity panel, three training conditions | 3 | CP/specificity/partA_specificity_panel.csv | 1_external / 2_all_absent / 3_added_negative | specificity_panel_probe.py | BACKED |
| 55 | 97 | Arousal index = mean of z-scored HR and tonic EDA | — | code | matched_arousal_probe.py:63-64; matched_arousal_hardening.py:47-48. Falls back to EDA alone where HR is missing (not stated) | — | BACKED (note) |
| 56 | 97 | Quintile bins, AUROC averaged by bin size | — | code | matched_arousal_probe.py:67-97; matched_arousal_hardening.py:50-60 | — | BACKED |
| 57 | 97 | "A logistic regression of the stress score on arousal and class" | logistic | code | matched_arousal_probe.py:99-115 is ordinary least squares on logit(score) | — | MISMATCH (method wording) |
| 58 | 97 | Position-matched negatives, 5th–95th pct | — | code | time_in_session_probe.py:69, 81 | — | BACKED |
| 59 | 97 | Test-side filters, Shapley over all orders | — | code | matched_arousal_probe.py:180-225 | — | BACKED |
| 60 | 101 | Pooled baseline-vs-stress on the three-dataset table; 50 subjects | 50 | T/loso_summary.csv | n_subjects 50; the task pools WESAD + PhysioNet only (run_jbhi_experiments.py:48) | run_jbhi_experiments.py | BACKED (note) |
| 61 | 101 | All modalities, 20 splits | 0.83 (0.81–0.85) | T/repeated_subject_splits.csv (normalisation = none) | 0.831 (0.813–0.850) | run_jbhi_experiments.py:224 | BACKED (note: unnormalised) |
| 62 | 101 | Physiology only | 0.81 (0.79–0.83) | same | 0.810 (0.788–0.832) | run_jbhi_experiments.py | BACKED |
| 63 | 101 | Accelerometer only | 0.75 (0.72–0.78) | same | 0.749 (0.719–0.779) | run_jbhi_experiments.py | BACKED |
| 64 | 101 | LOSO all / physiology | 0.85, 0.81 | T/loso_summary.csv | 0.849, 0.814 | run_jbhi_experiments.py:226 | BACKED |
| 65 | 101 | LOSO accelerometer only | 0.75 | none | LOSO runs only all_modalities and physiology_only (run_jbhi_experiments.py:141-142) | — | UNBACKED |
| 66 | 101 | No modality difference survives Holm | p_holm = 1.0 | T/significance_modalities.csv | tuned XGBoost, none: 1.0, 1.0, 1.0 (these are tuned runs, not the fixed-model numbers quoted in #61-63) | compare_models.py:71 | BACKED (note) |
| 67 | 101 | Complete nested tuning run | 96 configurations | T/tuned_baselines_summary.csv | 108 configurations (6 tasks × 2 norm × 3 modality × 3 model); 72 excluding cross-dataset | tune_baselines.py:86-96, 147 | MISMATCH |
| 68 | 101 | Within-dataset BA (HR/HRV/EDA) | 0.916/0.882/0.877/0.776/0.697 | T/leave_one_dataset_out_summary.csv | same | leave_one_dataset_out.py | BACKED |
| 69 | 101 | Effect sizes PN vs WESAD | EDA 0.4 vs 2.2 z; HR 1.7 vs 2.2 | T/stress_effect_size_by_dataset.csv | 0.39 vs 2.24; 1.72 vs 2.16 | transfer_gap.py:68 | BACKED |
| 70 | 115-119 | Table II, all 30 cells | — | T/leave_one_dataset_out_summary.csv | all match | leave_one_dataset_out.py | BACKED |
| 71 | 124 | Per-dataset costs | 0.03/0.03/0.05/0.04/0.015 | same | 0.032/0.030/0.051/0.040/0.015 | leave_one_dataset_out.py | BACKED |
| 72 | 124 | Training on all five matches or exceeds within on every target | every target | same | UBFC-Phys 0.839 vs 0.877; Stress-Predict 0.689 vs 0.697 | leave_one_dataset_out.py | MISMATCH |
| 73 | 124 | Costs within error for "15–35 test subjects" | 15–35 | code | each split holds out 20% of a dataset (3–7 subjects); 15–35 are dataset sizes | run_jbhi_experiments.py:82-88 | MISMATCH (wording) |
| 74 | 129 | Fig. 1 caption: UBFC with temperature | 0.589; AUROC 0.931 | T/leave_one_dataset_out_summary.csv | 0.589; 0.931 | leave_one_dataset_out.py | BACKED |
| 75 | 133 | Oracle threshold UBFC (with temperature) | 0.910 | T/threshold_transfer_probe_summary.csv | 0.91 | threshold_transfer_probe.py | BACKED |
| 76 | 133 | UBFC has no temperature in any window | 100% | code | extract_features.py:272-287 loads only BVP and EDA for UBFC | — | BACKED (code) |
| 77 | 133 | Mean stress score UBFC vs others | 0.18 vs 0.31–0.42 | T/threshold_transfer_probe_summary.csv | 0.18; 0.306–0.419 | threshold_transfer_probe.py | BACKED |
| 78 | 133 | Dropping temperature lifts UBFC to | 0.837 | same | 0.837 | threshold_transfer_probe.py | BACKED |
| 79 | 133 | Oracle recovers (HR/HRV/EDA) | 0.02–0.05 | same | 0.022–0.053 | threshold_transfer_probe.py | BACKED |
| 80 | 133 | Label-free prevalence cut, Campanella | 0.867 → 0.732 | same | 0.867 → 0.732 | threshold_transfer_probe.py | BACKED |
| 81 | 135 | Raw-feature external cost | −0.01 to 0.05 | NE/leave_one_dataset_out_summary_raw.csv | −0.010 to 0.046 | leave_one_dataset_out.py (--normalisation raw --tag _raw) | BACKED |
| 82 | 135 | Causal: WESAD, PN, SP within/external | 0.845/0.841; 0.774/0.744; 0.688/0.665 | NE/leave_one_dataset_out_summary_causal.csv | same | leave_one_dataset_out.py (--tag _causal) | BACKED |
| 83 | 135, 277 | Causal loses ~0.10 externally on UBFC, Campanella | ~0.10 | same vs main | −0.095; −0.102 | leave_one_dataset_out.py | BACKED |
| 84 | 135 | Short recordings | 7–11 windows/subject | T/harmonization_table.csv (derived) | Campanella 218/29 = 7.5; UBFC 205/19 = 10.8 | run_jbhi_experiments.py | BACKED (derived) |
| 85 | 135 | No external difference survives Holm across 60 tests | 0 of 60 | NE/normalisation_vs_session.csv | min external p_holm 0.087 | novelty_experiments_summary.py | BACKED |
| 86 | 137 | Every external score rose after the corrections | all 5 datasets | T/label_fix_sensitivity/before_after_significance.csv | covers WESAD and UBFC only; no committed pre-fix LODO for PN, SP, Campanella | label_fix_significance.py | UNBACKED (partial) |
| 87 | 137 | UBFC external before/after | 0.790 → 0.837 | same | 0.7902 → 0.8373 | label_fix_significance.py | BACKED |
| 88 | 137 | 1 of 16 survives Holm (WESAD ext. AUROC) | +0.0075, p_holm 0.021 | same | 16 non-within rows, 1 True; +0.0075, 0.0208 | label_fix_significance.py | BACKED |
| 89 | 137 | Kwon setting: fixes change transfer by at most | 0.02 | NE/kwon_setting_wesad_label_fix_test.csv; NE/kwon_setting_summary.csv | max diff 0.0203 | novelty_experiments_summary.py | BACKED |
| 90 | 137 | Our external AUROC on SP | 0.67 | NE/kwon_setting_summary.csv | 0.669 | novelty_experiments_summary.py | BACKED |
| 91 | 137 | Per-subject scaling, WESAD as target | 0.834 → 0.914 AUROC | same | raw 0.834; session 0.914 | novelty_experiments_summary.py | BACKED |
| 92 | 149 | Exercise rows | 0.69/0.70 → 0.10/0.08; 30–31/3763 | CP/specificity/partA_panel_compact.csv | 0.692/0.695 → 0.099/0.081 | specificity_panel_probe.py | BACKED |
| 93 | 150 | UBFC control | 0.55 [0.46,0.66] → 0.56 [0.45,0.67] | same | 0.550 [0.462,0.662] → 0.563 [0.448,0.673] | specificity_panel_probe.py | BACKED |
| 94 | 151 | Hyperventilation | 0.42 [0.26,0.59] → 0.55 [0.36,0.72] | same | 0.415 [0.255,0.593] → 0.547 [0.362,0.723] | specificity_panel_probe.py | BACKED |
| 95 | 152 | Lego | 0.43 [0.38,0.48] → 0.33 [0.27,0.39] | same | 0.427 [0.376,0.477] → 0.328 [0.274,0.387] | specificity_panel_probe.py | BACKED |
| 96 | 153 | EPM anger/fear/sadness/happiness | 0.42/0.35/0.37/0.24 → 0.35/0.25/0.30/0.18 | same | 0.424/0.353/0.374/0.236 → 0.350/0.252/0.301/0.183 | specificity_panel_probe.py | BACKED |
| 97 | 155 | In-task non-stress, dataset absent | 0.06–0.29 | CP/specificity/partA_specificity_panel.csv (1_external, false_stress_core) | 0.084–0.291 (0.06 appears only under 2_all_absent) | specificity_panel_probe.py | MISMATCH (minor) |
| 98 | 156 | Stress recall by dataset | 0.53–0.91 | same (1_external, stress_recall) | 0.529–0.907 | specificity_panel_probe.py | BACKED |
| 99 | 156, 161 | Stress recall unchanged or higher when any state is added | unchanged or higher | same (3_added_negative vs 2_all_absent) | drops: UBFC 0.744 → 0.698 (Lego), Campanella 0.917 → 0.886 (EPM), PN 0.735 → 0.709 (EPM) | specificity_panel_probe.py | MISMATCH |
| 100 | 161 | Exercise never seen | 69–70% | CP/specificity/partA_panel_compact.csv | 0.692/0.695 | specificity_panel_probe.py | BACKED |
| 101 | 161 | Exercise as negative | 8–10% | same | 0.081/0.099 | specificity_panel_probe.py | BACKED |
| 102 | 161 | External PN drop with exercise | 0.746 → 0.583, p_holm 0.011 | NE/exercise_negatives_physionet.csv | 0.746 → 0.583, 0.0106 | novelty_experiments_summary.py | BACKED |
| 103 | 161 | EPM clips fall by | 0.05–0.09 | CP/specificity/partA_panel_compact.csv | 0.053–0.101 (fear 0.353 → 0.252) | specificity_panel_probe.py | MISMATCH (minor) |
| 104 | 161 | Exercise negatives raise recall | SP 0.63 → 0.70; UBFC 0.74 → 0.79 | CP/specificity/partA_specificity_panel.csv | 0.629 → 0.708/0.697; 0.744 → 0.798/0.788 | specificity_panel_probe.py | BACKED |
| 105 | 161 | Hyperventilation per subject; UBFC control subjects | 1.7; 8 | T/harmonization_table.csv | 53/31 = 1.71; 8 | run_jbhi_experiments.py | BACKED (derived) |
| 106 | 180-185 | Table IV external block: within-bin AUROC, CI, p_holm | e.g. Lego 0.355 [0.24,0.54] 0.13 | CP/hardening/partA_inference.csv | all match (0.3551 [0.2428,0.5445] 0.1259, etc.) | matched_arousal_hardening.py | BACKED |
| 107 | 180-193 | Table IV "Pooled" column | 0.754 … 0.699 | CP/arousal/partA_matched_arousal.csv (auc_pooled, hr+eda) | all 12 match | matched_arousal_probe.py:136 | BACKED |
| 108 | 180-193 | Table IV "Disjoint" column | 0.44 … 0.52 | CP/hardening/partA_disjoint.csv (disjoint_hr+eda) | all 12 match (UBFC 0.515 → 0.52) | matched_arousal_hardening.py | BACKED |
| 109 | 180-193 | Table IV n | 667/80/53/33/394/3839/3763 | CP/arousal/partA_matched_arousal.csv (n_probe) | same (3839 = 5445 − 1606) | matched_arousal_probe.py | BACKED |
| 110 | 188-193 | Table IV added-negative block | 0.892, 0.779, 0.496, 0.568, 0.560, 0.431 | CP/hardening/partA_inference.csv (cond3) | same, CIs and p_holm match | matched_arousal_hardening.py | BACKED |
| 111 | 196 | 2000 bootstrap and 2000 permutation draws; Holm within model type | 2000 | code | matched_arousal_hardening.py:62-83, 109 | — | BACKED |
| 112 | 199 | External within-bin values; p_holm range | 0.577/0.523/0.355/0.390; 0.13–0.62 | CP/hardening/partA_inference.csv | same | matched_arousal_hardening.py | BACKED |
| 113 | 199 | Added as negative | 0.43–0.57, p_holm ≥ 0.64 | same | 0.431–0.568; ≥ 0.636 | matched_arousal_hardening.py | BACKED |
| 114 | 199 | Exercise; fear | 0.892 (p_holm 0.003); 0.779 | same | 0.892 (0.003); 0.7787 | matched_arousal_hardening.py | BACKED |
| 115 | 199 | Exercise minus each pair, bootstrap lower bounds | 0.18–0.36 | same (diff_lo) | 0.179–0.361 | matched_arousal_hardening.py | BACKED |
| 116 | 199 | Exercise with HR-only / EDA-only index | 0.65 / 0.54 | CP/arousal/partA_matched_arousal.csv | 0.654 / 0.538 exist but from the **external** model (hr+eda 0.759). For the added-negative model that gives 0.892: HR-only 0.912, EDA-only 0.818 | matched_arousal_probe.py | MISMATCH (context) |
| 117 | 201 | Disjoint: UBFC, hyperv., Lego, anger | 0.41–0.52; exercise 0.88; fear 0.78 | CP/hardening/partA_disjoint.csv | external 0.412–0.522; added-negative 0.501–0.539 (hyperv. 0.54); 0.877; 0.777 | matched_arousal_hardening.py | MISMATCH (minor) |
| 118 | 201 | Lego separable under HR-only index / EDA model | 0.73–0.84 | same (disjoint_hr_only_model_eda) | 0.726, 0.839 | matched_arousal_hardening.py | BACKED |
| 119 | 201 | "259 Lego windows with usable pulse" | 259 Lego | same | n = 259 is the whole pair (stress + Lego) | matched_arousal_hardening.py | MISMATCH (wording) |
| 120 | 201 | Campanella windows with no pulse | 63% | none | not tabulated | — | UNBACKED |
| 121 | 201 | Exercise vs stress index medians; bins | 4.8 vs 0.8; lowest three quintiles | CP/arousal/partA_matched_arousal.csv | 4.781 vs 0.794; bins "0.81 0.64 0.83 na na" | matched_arousal_probe.py | BACKED |
| 122 | 207, 213 | Baseline within 10 min of recording start | Camp 100%, SP 97%, PN 68% (90% in 15) | CP/wesad_warmup/warmup_since_donning.csv | 1.0, 0.966, 0.684 (0.897) | warmup_since_donning.py | BACKED |
| 123 | 207, 213 | WESAD protocol clock offset | 16–26 min; median 21 | CP/wesad_warmup/wesad_offsets.csv | 16.0–25.8; median 21.0 | warmup_since_donning.py | BACKED |
| 124 | 211 | Time-only vs physiology | PN 0.83 vs 0.78; SP 0.95 vs 0.70; WESAD 0.90 vs 0.92; Campanella perfect | CP/timeprobe/time_probe_summary.csv | 0.826/0.776; 0.947/0.698; 0.900/0.916; 1.0 | time_in_session_probe.py:100 | BACKED |
| 125 | 211 | Time-only better per split | PN 16/20; SP 20/20 | CP/timeprobe/time_probe_per_split.csv | 16/20 (loses seeds 0, 16, 18, 19); 20/20 | time_in_session_probe.py:98 | BACKED |
| 126 | 211 | Minutes as extra feature | SP 0.70 → 0.95; ext 0.66 → 0.56; WESAD 0.90 → 0.45 | CP/timeprobe/time_probe_summary.csv | 0.698 → 0.949; 0.656 → 0.560; 0.896 → 0.452 | time_in_session_probe.py | BACKED |
| 127 | 211 | Position-matched change | ≤ −0.03 within, ≤ −0.04 external | same (ba_matched) | within −0.028…+0.013; external −0.036…+0.015 | time_in_session_probe.py | BACKED |
| 128 | 211 | Baseline-referenced set calls later non-stress stress | 50–83% | T/order_confound_check.csv | 49.96–83.49 | probe_normalisation.py:94 | BACKED |
| 129 | 213 | Temp slope first 10 min, four datasets | 0.02–0.08 °C/min | CP/wesad_warmup/warmup_since_donning.csv | 0.021–0.082 | warmup_since_donning.py | BACKED |
| 130 | 213 | Temperature keeps rising past 20 min on WESAD and Campanella | both | same (temp_slope_20_40) | WESAD +0.055; Campanella −0.014 | warmup_since_donning.py | MISMATCH |
| 131 | 213 | Minutes 1–5 vs 30–40 | +1.8 and +0.9 °C | same | 1.79; 0.85 | warmup_since_donning.py | BACKED |
| 132 | 213 | WESAD per-subject gain over 30 min | +0.3 to +6.5 °C | none | per-subject values not tabulated | — | UNBACKED |
| 133 | 213 | Tonic EDA rises first 10 min on all four | all > 0 | same | 0.026–0.151 | warmup_since_donning.py | BACKED |
| 134 | 213 | Stress arrives at median | 7–28 min | CP/timeprobe/warmup_drift.csv (stress_median_min) | 6.7 / 23.4 / 27.75 | time_in_session_probe.py:46 | BACKED |
| 135 | 213 | WESAD baseline since switch-on; none within 15 min | 17–49 min, median 33; 0% | CP/wesad_warmup/wesad_minutes_since_donning_by_class.csv; warmup_since_donning.csv | 17.3–49.3, median 33.3; 0.0 | warmup_since_donning.py | BACKED |
| 136 | 213 | Every WESAD subject's first minute reads 4–6 °C above skin | per subject | CP/wesad_warmup/warmup_curves_1min.csv | only the mean is tabulated (bin 0 +4.4 °C subject-centred) | warmup_since_donning.py | UNBACKED |
| 137 | 225-229 | Table V, all 30 cells | — | CP/arousal/partB_error_budget.csv | all match (residual = 1 − ba_all_filters) | matched_arousal_probe.py:231 | BACKED |
| 138 | 218, 241 | Filters F1–F4 and 24-order Shapley | 5 min / 0.5 / 60 s / 0.5 SD | code | matched_arousal_probe.py:169-185, 217-225 | — | BACKED |
| 139 | 241 | Four filters together buy on PN and SP | 0.05–0.06 BA | CP/arousal/partB_error_budget.csv | PN 0.049/0.064; SP within 0.061; SP external 0.018 | matched_arousal_probe.py | MISMATCH (SP external) |
| 140 | 241 | CIs excluding zero: F1, F3 on SP; F3 external on PN | — | same | consistent (PN-within and SP-within F4 lower bound 0.000) | matched_arousal_probe.py | BACKED |
| 141 | 241 | Residual error | 0.17–0.23 (0.33 SP ext.) | same | 0.175/0.190/0.229; 0.329 | matched_arousal_probe.py | BACKED |
| 142 | 241 | Lenient rules: same subjects; F4 gain | 32/35, 32/34; 0.01–0.02 | CP/hardening/partB_responder_sensitivity.csv | 32, 32; 0.005–0.019 | matched_arousal_hardening.py | BACKED |
| 143 | 241 | 1.0 SD both channels on PN; top half | 17/35; +0.094 [0.05,0.15]; ext 0.065 [0.01,0.13]; 0.062 [0.01,0.13] | same | 17; 0.094 [0.046,0.152]; 0.065 [0.012,0.131]; 0.062 [0.013,0.131] | matched_arousal_hardening.py | BACKED |
| 144 | 241 | SP strict rules | 12–17 of 34; +0.03 within; −0.02 external | same | 12/17; 0.032/0.025; external −0.020 (1.0sd_both) but +0.026 (top_half) | matched_arousal_hardening.py | MISMATCH (partial) |
| 145 | 241 | Non-response ≈ half of four-filter total on PN (strict) | about half | same (F4_shapley_share) | 0.46–0.77 | matched_arousal_hardening.py | BACKED (approx.) |
| 146 | 241 | Residual under strict rules | PN 0.13–0.19; SP 0.23–0.31 | same (1 − ba_all_filters) | PN 0.095–0.142; SP 0.206–0.348 | matched_arousal_hardening.py | MISMATCH |
| 147 | 241 | SP recall first minute vs after 3 min | 0.47 vs 0.77 | CP/arousal/partC_onset_offset_curves.csv | 0.468 vs 0.765 (within) | matched_arousal_probe.py | BACKED |
| 148 | 241 | PN recall first minute vs after 3 min | 0.63 vs 0.77 | same | within 0–1 min 0.626; 3+ min 0.586 (0.765 is the 2–3 min bin; external 3+ is 0.931) | matched_arousal_probe.py | MISMATCH |
| 149 | 241 | PN rest false alarms after offset | 0.26 vs 0.06–0.09 | same | 0.262; 0.056–0.094 | matched_arousal_probe.py | BACKED |
| 150 | 243 | Chest-ECG bound on WESAD | 0.916 → 0.956; AUROC 0.977 → 0.993; ECG alone 0.918; low coverage 0.886 → 0.962; EDA 0.891 | CP/timeprobe/wesad_ecg_ceiling_summary.csv | same | time_in_session_probe.py | BACKED |
| 151 | 255 | UDA vs source-only, physiology, z-scored | 0.54–0.70 vs 0.54–0.70 | T/domain_adaptation_physiology.csv | DA 0.519–0.698 (subject_dann 0.519); source 0.544–0.698 | domain_adaptation.py:218 | MISMATCH (minor) |
| 152 | 256, 268 | Few-shot refit gain, first-in-time | +0.00 to +0.04, none significant | T/fewshot/fewshot_summary_session_chronological_gap30.csv | −0.042 (UBFC k=1) to +0.043; all p_holm ≥ 0.56 | fewshot_unseen_dataset.py:137 | MISMATCH (minor) |
| 153 | 257 | Random windows: only significant gain | WESAD k=5 +0.03 | T/fewshot/fewshot_summary_session_random_gap30.csv | +0.029, p_holm 0.0 | fewshot_unseen_dataset.py | BACKED |
| 154 | 258 | Per-user threshold, PN k=3 | −0.08, p_holm < 0.05 | T/fewshot/fewshot_summary_session_chronological_gap30.csv | −0.086, p_holm 0.0 | fewshot_unseen_dataset.py | BACKED |
| 155 | 259 | Self-rated rise vs recall "(PhysioNet 54 units, WESAD 15)" | ρ −0.08 [−0.42,0.21] and 0.02 [−0.30,0.31] | T/mild_stress/mild_stress_summary.csv | both are PhysioNet (within, external); WESAD is −0.49 [−0.87,0.17] and −0.60 [−0.84,−0.08] | mild_stress_probe.py | MISMATCH (labelling) |
| 156 | 260 | Detectability across stressors | SP −0.14 [−0.51,0.24]; PN 0.44 [−0.04,0.78] | CP/specificity/partB_miss_consistency.csv | −0.138 [−0.512,0.239]; 0.444 [−0.042,0.784] | specificity_panel_probe.py | BACKED |
| 157 | 261 | Recall vs baseline HR, RMSSD, EDA, PPG coverage | \|ρ\| ≤ 0.35, all CIs include 0 | same | SP recall_ext vs base_eda_mean ρ 0.483 [0.184,0.701] | specificity_panel_probe.py | MISMATCH |
| 158 | 262 | Worst 20% subjects' share of misses | 37–61% | same | 0.369–0.611 | specificity_panel_probe.py | BACKED |
| 159 | 263 | Exercise absent / present in training | 69.5%; 4–6% | NE/leave_one_dataset_out_summary_session_exercise.csv | 0.695; 0.043/0.064 | leave_one_dataset_out.py:114-115 (--exercise-negatives) | BACKED |
| 160 | 268 | First k windows per class; 30 s gap | — | code | fewshot_unseen_dataset.py (split_support_query) | — | BACKED (line not pinned) |
| 161 | 268 | Stroop vs arithmetic mean rise | 0.7 vs 2.5 | none (docs/novelty_experiments.md:87 gives 0.67 / 2.49) | derivable from T/mild_stress/mild_stress_window_predictions.csv, not tabulated | mild_stress_probe.py | UNBACKED |
| 162 | 268 | PN within vs external agree on who is hard | ρ 0.64 [0.37,0.82] | CP/specificity/partB_miss_consistency.csv | 0.639 [0.368,0.822] | specificity_panel_probe.py | BACKED |
| 163 | 271 | Low-ceiling datasets' stressors give a fifth of the TSST EDA response | ~1/5 | T/stress_effect_size_by_dataset.csv | PN 0.39 vs WESAD 2.24 (0.17); Stress-Predict not in the table | transfer_gap.py | UNBACKED (SP part) |
| 164 | 271 | 0.06 / 0.09 on PN / nothing on SP | — | partB tables | consistent | matched_arousal_probe.py; matched_arousal_hardening.py | BACKED |
| 165 | 277 | Raw and causal keep cost on long, lose ~0.10 on short | — | NE LODO summaries | as #81-83 | leave_one_dataset_out.py | BACKED |
| 166 | 277 | Disjoint HR-only variant "halves" Lego and hyperventilation windows | half | CP/hardening/partA_disjoint.csv (n) | Lego pair 769 → 259 (−66%); hyperv. 904 → 552 (−39%) | matched_arousal_hardening.py | MISMATCH (minor) |
| 167 | 277 | Pooled subject count | about 118 | T/harmonization_table.csv | 15+35+34+19+29 = 132 (165 with EPM-E4) | run_jbhi_experiments.py | MISMATCH |
| 168 | 277 | Costs within sampling error of 15–35 test subjects | 15–35 | code | as #73 (3–7 test subjects per split) | run_jbhi_experiments.py:82-88 | MISMATCH (wording) |
| F1 | 128 (Fig. 1) | fig_lodo.pdf | — | T/leave_one_dataset_out_summary.csv | make_paper_figures.py:25 reads it; annotation 0.589 matches | make_paper_figures.py | BACKED |
| F2 | 165 (Fig. 2) | fig_panel.pdf | — | (a) CP/specificity/partA_panel_compact.csv; (b) CP/arousal/partA_matched_arousal.csv (hr+eda) | point estimates match Table IV; (b) error bars are the 500-draw CIs of matched_arousal_probe.py:67, not Table IV's 2000-draw CIs (e.g. Lego [0.25,0.54] vs [0.24,0.54]) | make_paper_figures.py:46, 71 | BACKED (note) |
| F3 | 206 (Fig. 3) | fig_time.pdf | — | (a) CP/timeprobe/time_probe_summary.csv; (b,c) CP/timeprobe/warmup_trajectory_2min_bins.csv | caption percentages come from warmup_since_donning.csv, which the script does not read; values match | make_paper_figures.py:103-104 | BACKED |
| F4 | 236-237 (Fig. 4) | "Each bar is 1 − BA" | 1 − BA | CP/arousal/partB_error_budget.csv | make_paper_figures.py:135 clips negative Shapley values to 0, so the SP-external bar totals 0.361, while 1 − BA = 0.347 (F2 = −0.015) | make_paper_figures.py | MISMATCH (caption) |
| C1 | 33 | Prajod; Calza-Metre; Xiao 0.88 → 0.53–0.69 across seven datasets; Mishra ~0.01; Kwon WESAD vs Stress-Predict | — | — | — | — | CITATION |
| C2 | 41 | Kwon AUROC ~0.90 (WESAD), 0.52–0.57, within SP 0.63, cost 0.06–0.08; Calza-Metre 0.87 → 0.63; Xiao 0.53–0.69; Mishra median ~0.01; Schreiber & Maleshkova; Prajod chest ECG | — | — | — | — | CITATION |
| C3 | 44 | Kwon, Singh (label noise); Mishra (post-stress rest); Aydoğan (PN sessions mix blocks; 82.6%; 6–7%); Richer; Tognotti 3–13 points; Fecke & Rehof; Saeb; Bahameish | — | — | — | — | CITATION |
| C4 | 47 | Stewart, Han, Tervonen (+1.7 points, 1–3 windows), Akkaya (abstract only); Milstein RMSSD r = 0.42; Watanabe (TSST worst); Sosa verbatim quotes | — | — | — | — | CITATION |
| C5 | 52 | Dataset descriptors (Schmidt, Hongn, Iqbal, Meziati Sabour, Campanella, Garcia-Moreno) and protocol descriptions | — | — | — | — | CITATION |
| C6 | 137 | Kwon Stress-Predict external AUROC 0.56 | — | — | — | — | CITATION |
| C7 | 161 | Aydoğan & Villagra Povina 71–83% | — | — | — | — | CITATION |
| C8 | 213, 268 | Richer (sequence effect); Tervonen, Akkaya (few-shot) | — | — | — | — | CITATION |
| C9 | 277 | Milstein: wrist vs palmar EDA r ≈ 0.3 | — | — | — | — | CITATION |

## Problems

### A. MISMATCH (value or method differs from the committed table or code)

1. **#47, line 94: "0 of 48 comparisons after Holm (significance_models.csv)".** The table has 72 model comparisons and 1 is significant: affective_only_no_exercise / subject_zscore / physiology_only, mlp vs xgboost, p_holm = 0.038. Fix the text: "1 of 72 comparisons, on the nine-class affective task only; none on the stress tasks". CLAUDE.md already says this.
2. **#48, line 94: "20 subject-grouped splits with 15% of subjects held out".** `grouped_split` defaults to `test_size=0.2` (run_jbhi_experiments.py:82). LODO, the probes and tuning all use it, and the Nadeau–Bengio correction assumes 0.2/0.8 (compare_models.py:21). Only the leakage check uses 15%. Change the text to 20%.
3. **#67, line 101: "complete 96-configuration nested tuning run".** tuned_baselines_summary.csv has 108 configurations (6 tasks × 2 normalisations × 3 modalities × 3 models), 72 without the two cross-dataset tasks. Change to 108, or to "72 within-split configurations".
4. **#72, line 124: "Training on all five datasets matches or exceeds within-dataset training on every target".** It does not on UBFC-Phys (0.839 vs 0.877) or Stress-Predict (0.689 vs 0.697). Reword to "on three of five targets, and is within 0.04 on the others", or drop the sentence.
5. **#73 and #168, lines 124 and 277: "15–35 test subjects".** Each split holds out 20% of the target dataset, which is 3–7 test subjects. 15–35 are the dataset sizes. Reword to "3–7 held-out subjects per split (15–35 per dataset)".
6. **#30, line 82: Campanella "the subtraction span ends at each subject's logged end".** No per-subject log exists. The code estimates the end as recording end − 300 s − 15 s guard, capped at 1800 s (extract_features.py:302-306, 333; its own comment says "no per-subject markers exist"). Reword to "ends at a per-subject end estimated from the recording length", or implement the ACC-based end that the code comment suggests.
7. **#18, line 68 (Table I): Stress-Predict stress "34 / 851".** 33 subjects have stress windows (Stroop 31 and TSST 32 in harmonization_table.csv; n_subjects = 33 in partA_specificity_panel.csv and partC). Change to 33 / 851. Consider adding a subject-union column to `harmonization_table()` so the union is tabulated.
8. **#99, lines 156 and 161: "Stress recall is unchanged or higher when any probed state is added".** Compared with condition 2 in partA_specificity_panel.csv, recall drops: UBFC-Phys 0.744 → 0.698 (Lego added), Campanella 0.917 → 0.886 and PhysioNet 0.735 → 0.709 (EPM added). Reword to "changes by at most −0.05 and rises when exercise is added". Update the Table III reference row the same way.
9. **#146, line 241: residual under the strict rules "0.13–0.19 on PhysioNet and 0.23–0.31 on Stress-Predict".** From partB_responder_sensitivity.csv (1 − ba_all_filters): PhysioNet 0.095–0.142, Stress-Predict 0.206–0.348. Replace with these ranges.
10. **#148, line 241: PhysioNet recall "0.63 against 0.77 after 3 min".** Within-LOSO recall is 0.586 after 3 min. 0.765 is the 2–3 min bin, and the external 3+ min value is 0.931. Use "0.63 in the first minute against 0.77 in the third", or state the 3+ min value.
11. **#157, line 261 (Table VI): "Recall vs baseline HR, RMSSD, EDA, PPG coverage: |ρ| ≤ 0.35, CIs include 0".** partB_miss_consistency.csv has Stress-Predict external recall vs baseline EDA ρ = 0.483 [0.184, 0.701]. Report it as the exception, or restrict the claim to within-dataset recall.
12. **#116, line 199: "a heart-rate-only index gives 0.65 and an EDA-only index 0.54" for exercise.** These are the external-model values (hr+eda 0.759). For the model that separates exercise at 0.892 (exercise as a negative), the values are 0.912 (HR-only) and 0.818 (EDA-only). Either say the numbers are for the external model, or use the added-negative values, which weaken the "cardiac signature" reading.
13. **#139, line 241: "Together they buy 0.05–0.06 BA on PhysioNet and Stress-Predict".** Stress-Predict external gains only 0.018. Say "0.05–0.06 except Stress-Predict external (0.02)".
14. **#144, line 241: Stress-Predict strict rules gain "−0.02 externally".** True for the 1.0-SD-both rule. The top-half rule gives +0.026. Say "−0.02 to +0.03".
15. **#130, line 213: temperature "keeps rising past 20 min on WESAD and Campanella".** The 20–40 min slope is −0.014 °C/min on Campanella (warmup_since_donning.csv). The +0.9 °C difference is reached by minute 20. Reword for Campanella.
16. **#155, line 259 (Table VI): the ρ values labelled as PhysioNet and WESAD are both PhysioNet (within and external).** WESAD gives −0.49 [−0.87, 0.17] within and −0.60 [−0.84, −0.08] external; the external CI excludes zero, and docs/novelty_experiments.md:88 explains that it rests on 2 subjects. Relabel the cell, and mention WESAD with that caveat.
17. **#57, line 97: "A logistic regression of the stress score on arousal and class".** The code fits ordinary least squares on logit(score) (matched_arousal_probe.py:99-115). Call it "a linear regression of the logit stress score". The coefficient is not reported anywhere in the paper, so the sentence could also be deleted.
18. **#117, line 201: disjoint values "stay at 0.41–0.52".** The added-negative hyperventilation value is 0.539. Say "0.41–0.54".
19. **#119, line 201: "the 259 Lego windows with usable pulse".** n = 259 is the whole pair (stress + Lego windows). Say "259 windows (stress and Lego) with usable pulse", or tabulate the Lego count.
20. **#166, line 277: the disjoint HR-only variant "halves" the Lego and hyperventilation windows.** Lego pair 769 → 259 (−66%), hyperventilation 904 → 552 (−39%). Say "cuts by 40–66%".
21. **#167, line 277: "pooled subject count (about 118 before overlap checks)".** Table I gives 15+35+34+19+29 = 132 in the five stress datasets (165 with EPM-E4). Use 132, or explain where 118 comes from.
22. **#97, line 155 (Table III): in-task non-stress false-stress rate, dataset absent, "0.06–0.29".** The dataset-absent condition gives 0.084–0.291. 0.06 appears only in condition 2. Use 0.08–0.29.
23. **#103, line 161: EPM clips "fall by 0.05–0.09".** Fear falls 0.353 → 0.252 (0.10). Use 0.05–0.10.
24. **#151, line 255 (Table VI): UDA "0.54–0.70".** Subject-adversarial DANN gives 0.519 (PhysioNet→WESAD, z-scored). Use 0.52–0.70.
25. **#152, lines 256 and 268: first-in-time refit gain "+0.00 to +0.04".** UBFC-Phys k = 1 gives −0.042. Use "−0.04 to +0.04".
26. **F4, line 237: Fig. 4 caption "Each bar is 1 − BA".** make_paper_figures.py:135 clips negative Shapley values to 0, so the Stress-Predict external bar totals 0.361 against 1 − BA = 0.347. Either draw negative segments, or amend the caption ("negative contributions are shown as zero").

### B. UNBACKED (no committed table or code produces the value)

1. **#65, line 101: LOSO accelerometer-only 0.75.** LOSO is run only for all_modalities and physiology_only (run_jbhi_experiments.py:141-142). Either add acc_only to the LOSO loop and rerun `run_jbhi_experiments.py` on `harmonized_windows_v2_three_datasets_fixed.csv` (per CLAUDE.md), or remove "0.75" from "(0.85, 0.81, 0.75)".
2. **#25, line 82: "248 of 2,933 windows in the earlier whole-session labelling".** No committed table has these counts (the v1 harmonization table has different ones). Commit the table they came from (probably the pre-fix `harmonized_windows_v2_before_label_fixes.csv` summary in the thesis worktree, or a `relabel_windows.py` run), or drop the numbers.
3. **#33, line 84: "reported 94% accuracy".** This is the historical README figure (94.53%). The committed reproduction (leakage_check.csv) gives 92.1% on random windows. Write "the earlier pipeline reported 94% (our reproduction: 92%)", or cite the thesis.
4. **#86, line 137: "Every external score rose after the seven corrections".** Only WESAD and UBFC-Phys have committed before/after numbers. Run `leave_one_dataset_out.py --input harmonized_windows_v2_before_label_fixes.csv --tag _before_fixes` and commit the result, or limit the claim to the two datasets tested.
5. **#120, line 201: "63% of Campanella windows have none [usable pulse]".** Not tabulated. Add the cardiac-coverage fraction per dataset to a committed table (for example a column in harmonization_table or warmup_drift), or drop the number.
6. **#132, line 213: "WESAD subjects gain +0.3 to +6.5 °C over 30 min".** Per-subject values are not in any table. Add a per-subject output to `warmup_since_donning.py` and commit it.
7. **#136, line 213: "every subject's first minute of temperature reads 4–6 °C above skin".** Only the mean (bin 0: +4.4 °C, subject-centred) is in warmup_curves_1min.csv. Add a per-subject first-minute table in `warmup_since_donning.py`, or reword to the mean.
8. **#161, line 268: "Stroop (mean rise 0.7 of 10) … arithmetic (2.5)".** Only in docs/novelty_experiments.md:87. It can be derived from mild_stress_window_predictions.csv, but no table holds it. Add a per-task mean rise to `mild_stress_probe.py`'s mild_stress_summary.csv.
9. **#163, line 271: "stressors produce a fifth of the electrodermal response of the TSST" for the low-ceiling datasets.** Backed for PhysioNet only (0.39 vs 2.24 z). Stress-Predict is not in stress_effect_size_by_dataset.csv. Extend `transfer_gap.py`'s effect-size table to all five datasets, or say "PhysioNet's stressors".
10. **#32, line 82: "The three participants recorded in two files".** The merge is in code (extract_features.py:183; relabel_windows.py:118), but the count of three is in no table. List the merged subjects in a committed table (for example a harmonization audit output), or drop the number.

### C. TABLE ONLY

None of the values cited in the paper fall in this class.

### D. Reproducibility issues found along the way (no single claim affected)

1. **The input feature table is not in this repo.** matched_arousal_probe.py:14, matched_arousal_hardening.py:13 and time_in_session_probe.py:18 read `/Users/octa/Projects/smartwatch-stress-detection/data/processed/combined/harmonized_windows_v2.csv` (thesis worktree) by absolute path. The other scripts default to `data/processed/...`, which is not committed. The "code released" statement (line 283) cannot reproduce Tables III–V or Figs. 2–4 without that file. Fix: publish the feature table, or a script that rebuilds it and states where the raw data goes, and replace the absolute paths with a repo-relative default.
2. **One script call does not produce all its committed tables.** `run_jbhi_experiments.py` writes harmonization_table.csv together with the split, LOSO and leakage tables. Per CLAUDE.md, the committed split/LOSO tables come from the three-dataset table and harmonization_table.csv from a separate six-dataset run, so no single call reproduces all of them. Document both calls (or add a `--harmonization-only` flag).
3. **Line 101 mixes two runs.** The quoted BAs (0.83/0.81/0.75) are the fixed-hyperparameter XGBoost with normalisation = none (repeated_subject_splits.csv). The cited Holm test (significance_modalities.csv) is on the tuned models (0.817/0.798/0.746). Methods (line 91) says features are z-scored per subject, yet these numbers are unnormalised; the z-scored values are 0.823 (all) and 0.794 (physiology) under LOSO (loso_summary.csv), and 0.828/0.780/0.727 for tuned XGBoost over the 20 splits (significance_modalities.csv). State which normalisation and which run each number comes from.
4. **Uncommitted-writer tables (not cited in the paper, but committed).** `label_fix_sensitivity/{base120,base240,resthalf}_{leave_one_dataset_out,transfer_gap}_summary.csv` have prefixed names that no script writes (leave_one_dataset_out.py only appends a suffix tag; transfer_gap.py writes a fixed name), so they were renamed by hand. `source_only_fixes_probe.csv` has no writer in `scripts/`. If the paper ever cites them, document the renaming or commit the writer.
5. **Fig. 2(b) and Table IV use different CIs** (500-draw bootstrap from matched_arousal_probe.py vs 2000-draw from matched_arousal_hardening.py). Point to partA_inference.csv in `fig2_panel()` so the figure matches the table.
6. **The arousal index silently falls back to EDA alone where heart rate is missing** (matched_arousal_probe.py:64). Say so in Methods (line 97).
