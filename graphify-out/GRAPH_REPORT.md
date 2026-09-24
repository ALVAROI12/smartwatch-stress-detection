# Graph Report - novelty-plan  (2026-09-23)

## Corpus Check
- 7 files · ~469,588 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 881 nodes · 1694 edges · 55 communities (52 shown, 3 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 250 edges (avg confidence: 0.84)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Novelty Collision & Reviewer Sim
- Feature Extraction (HR/HRV/EDA)
- Legacy API & Splits
- Legacy Detector & Tuning
- Novelty Search Papers
- Contribution Report & Gap Scan
- Dataset Label Fixes
- Arousal Thesis & Warm-up
- Domain Adaptation & Few-Shot
- Harmonization Audit
- Legacy Robustness & Active Learning
- Biomarker Evidence Report
- Paper Figure Builder
- Model Comparison Stats
- LODO & Missing-PPG Shortcut
- Frozen Split Evaluation
- Legacy Six Classes
- Contribution Map
- Legacy SHAP Importance
- README Datasets & Methods
- Stress Construct Critique
- Novelty Candidates Brief
- LODO Normalise & Mild Stress
- Normalisation Probe
- Novelty Experiments
- Legacy Model Enhancements
- Few-Shot Unseen Dataset
- Project Memory (CLAUDE.md)
- Matched Arousal Probe
- Matched Arousal Hardening
- Stressor Potency Dose-Response
- Theory Framing Reframe
- Closable Gaps (Benchmark, Encoders)
- LODO Prior Work (Kwon, AutoStress)
- Cacioppo Inference & PPV
- Legacy Anomaly Detection
- Legacy Confusion & Conformal
- Autonomic Specificity & Analyses A/B
- Legacy Feature Correlation
- Self-Report Labels
- Session Z-Scoring
- Arousal-Controlled Evaluation Kit
- Window-Split Leakage
- Legacy Leakage Symptoms
- Wrist vs Chest ECG & Deps
- Harmonization LaTeX Export
- LODO Normalise Tests
- Exercise False Alarms
- Stress Constructs (Epel, Allostasis)
- Shortcut Learning & Transfer Cost
- Legacy Hyperparameter Plots
- Legacy Deployment & Drift
- Legacy Learning Curves
- Legacy Subject Clustering
- Legacy Deep Learning

## God Nodes (most connected - your core abstractions)
1. `What Do Wrist-Worn Stress Detectors Detect? (JBHI manuscript)` - 66 edges
2. `Novelty Search for the JBHI Rewrite` - 48 edges
3. `Dataset and Claims Verification (full text)` - 26 edges
4. `Novelty plan for the JBHI paper (2026-09-23)` - 25 edges
5. `Contribution Gap Scan (literature scout)` - 24 edges
6. `grouped_split()` - 21 edges
7. `04 Theory Framing: measurement-validity reframe` - 20 edges
8. `Novelty-collision check (2026-09-23)` - 18 edges
9. `Wearable Stress Biomarkers Evidence Report` - 16 edges
10. `Leave-One-Dataset-Out Evaluation` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Stress-Predict (reading list)` --semantically_similar_to--> `Stress-Predict Dataset`  [INFERRED] [semantically similar]
  docs/literature/wearable-stress-biomarkers-reading-list.md → paper/main.pdf
- `Matched-Arousal Test` --semantically_similar_to--> `Arousal-controlled evaluation kit (rank 1)`  [INFERRED] [semantically similar]
  paper/main.pdf → docs/literature/novelty-2026-09-23/05_closable_gaps.md
- `WESAD (reading list)` --semantically_similar_to--> `WESAD`  [INFERRED] [semantically similar]
  docs/literature/wearable-stress-biomarkers-reading-list.md → paper/main.pdf
- `Prajod et al. 2024 (ICMI)` --semantically_similar_to--> `Prajod et al. 2024 (Stressor Type Matters)`  [INFERRED] [semantically similar]
  docs/literature/wearable-stress-biomarkers-evidence-report.md → paper/main.pdf
- `Kwon et al. 2026` --semantically_similar_to--> `Kwon et al. 2026`  [INFERRED] [semantically similar]
  docs/literature/novelty-search.md → paper/main.pdf

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Four Evaluation Rules Supporting Arousal-Not-Stress Thesis** — paper_main_specificity_panel, paper_main_matched_arousal_test, paper_main_position_matched_negatives, paper_main_test_side_error_budget, paper_main_arousal_transfers_stress_does_not_thesis [EXTRACTED 1.00]
- **Five E4 datasets in leave-one-dataset-out** — readme_wesad, readme_physionet_wearable_stress_exercise_dataset, readme_stress_predict, readme_ubfc_phys, readme_campanella_2024_dataset, readme_leave_one_dataset_out_transfer_cost [EXTRACTED 1.00]
- **Five E4 Datasets in LODO Benchmark** — paper_main_wesad, paper_main_physionet_hongn_2025_stress_and_exercise_dataset, paper_main_stress_predict_dataset, paper_main_ubfc_phys_dataset, paper_main_campanella_2024_dataset, paper_main_leave_one_dataset_out_evaluation [EXTRACTED 1.00]
- **Legacy per-class SHAP beeswarm plots** — legacy_outputs_figures_shap_class_aerobic, legacy_outputs_figures_shap_class_amusement, legacy_outputs_figures_shap_class_anaerobic, legacy_outputs_figures_shap_class_baseline, legacy_outputs_figures_shap_class_emotion, shap_feature_attribution [EXTRACTED 1.00]
- **Legacy per-class SHAP waterfall explanations (39 features, XGBoost)** — legacy_outputs_figures_shap_waterfall_baseline, legacy_outputs_figures_shap_waterfall_emotion, legacy_outputs_figures_shap_waterfall_stress, legacy_outputs_figures_shap_waterfall_baseline_accelerometer_dominance [EXTRACTED 1.00]
- **PhysioNet Label Fixes (f13, second-half rests, baseline length)** — docs_literature_dataset_and_claims_verification_fix_5_exclude_physionet_f13, docs_literature_dataset_and_claims_verification_fix_6_physionet_second_half_rests, docs_literature_dataset_and_claims_verification_fix_7_physionet_v2_baseline_length, paper_main_physionet_hongn_2025_stress_and_exercise_dataset [EXTRACTED 1.00]
- **Legacy deployment and monitoring analyses** — legacy_outputs_figures_deployment_metrics, legacy_outputs_figures_drift_detection, legacy_outputs_figures_conformal_prediction [INFERRED 0.65]
- **Evidence of inflated legacy evaluation** — legacy_outputs_figures_validation_comparison_validation_gap, legacy_outputs_figures_temporal_patterns_state_persistence, legacy_outputs_figures_stress_regression_degenerate_target [INFERRED 0.75]
- **Legacy figures showing subject-leakage inflation** — legacy_outputs_figures_confusion_matrices_all_models_confusion_perfect_diagonal, legacy_outputs_figures_cv_vs_loso_comparison_leakage_gap, legacy_outputs_figures_conformal_prediction_set_size_anomaly, legacy_outputs_figures_deep_learning_training [INFERRED 0.75]
- **Legacy XGBoost hyperparameter tuning** — legacy_outputs_figures_hyperparam_sensitivity, legacy_outputs_figures_optuna_optimization, legacy_outputs_figures_model_comparison_final [INFERRED 0.75]
- **Evidence that wrist detectors detect arousal not stress** — docs_contribution_map_matched_arousal_test, docs_contribution_map_specificity_panel, docs_novelty_experiments_exercise_as_non_stress_negatives, docs_contribution_map_stressor_potency_ceiling [INFERRED 0.85]
- **Leakage-controlled negative method results** — readme_few_shot_personalisation, docs_novelty_experiments_few_shot_calibration_on_unseen_dataset, readme_domain_adaptation_coral_mmd_dann, docs_novelty_experiments_mild_stress_pilot, docs_contribution_map_subject_detectability_trait [INFERRED 0.85]
- **Legacy anomaly/outlier analysis figures** — legacy_outputs_figures_anomaly_detection, legacy_outputs_figures_anomaly_distribution, legacy_outputs_figures_anomaly_signal_distributions [INFERRED 0.85]
- **Legacy feature analysis: importance, correlation, interactions** — legacy_outputs_figures_feature_importance_rf, legacy_outputs_figures_feature_importance_comparison, legacy_outputs_figures_feature_correlation_heatmap, legacy_outputs_figures_feature_interactions [INFERRED 0.85]
- **Legacy global feature-importance views showing accelerometer dominance** — legacy_outputs_figures_shap_summary_bar, legacy_outputs_figures_shap_heatmap_by_class, legacy_outputs_figures_shap_vs_rf_importance, legacy_outputs_figures_shap_class_stress, legacy_outputs_figures_accelerometer_dominated_shap_importance [INFERRED 0.85]
- **Legacy ~94.5% results from window-level split (subject leakage, inflated)** — legacy_outputs_figures_best_model_confusion, legacy_outputs_figures_bootstrap_ci, legacy_outputs_figures_adversarial_robustness, legacy_readme [INFERRED 0.85]
- **Legacy window-split results inflated vs subject-held-out** — legacy_outputs_figures_model_comparison_final_xgboost_best_cv, legacy_outputs_figures_optimization_comparison_perfect_test_accuracy, legacy_outputs_figures_loso_accuracy_distribution_loso_far_below_cv, legacy_outputs_figures_learning_curves_persistent_overfit_gap [INFERRED 0.85]
- **Legacy results inflated by window-level split (subject leakage)** — legacy_outputs_figures_holdout_confusion_matrix, legacy_outputs_figures_error_analysis, legacy_outputs_figures_enhancement_summary_optuna_xgboost, legacy_outputs_figures_final_validation_dashboard_validation_gap [INFERRED 0.85]
- **Legacy per-class SHAP waterfall examples** — legacy_outputs_figures_shap_waterfall_aerobic, legacy_outputs_figures_shap_waterfall_amusement, legacy_outputs_figures_shap_waterfall_anaerobic, legacy_outputs_figures_shap_waterfall_stress [INFERRED 0.85]
- **Legacy six-class outputs built on leaky window-level split** — legacy_outputs_figures_rejection_option, legacy_outputs_figures_rfe_analysis, legacy_outputs_figures_shap_analysis, window_level_split_leakage [INFERRED 0.85]
- **Prior work anticipating the arousal-not-stress claim** — docs_literature_novelty_2026_09_23_01_collision_check_zhou_2023, docs_literature_novelty_2026_09_23_01_collision_check_kwon_2026, docs_literature_novelty_2026_09_23_01_collision_check_aydogan_2026, docs_literature_novelty_2026_09_23_01_collision_check_kaya_2026, docs_literature_novelty_2026_09_23_01_collision_check_sosa_2026 [EXTRACTED 1.00]
- **Steps 1-4: minimum for a credible submission** — docs_novelty_plan_2026_09_23_step1_credit_rewrite, docs_novelty_plan_2026_09_23_step2_arousal_kit, docs_novelty_plan_2026_09_23_step3_hrv_direction_script, docs_novelty_plan_2026_09_23_step4_potency_dose_response [EXTRACTED 1.00]
- **Theory frames supporting critical-evaluation framing** — docs_novelty_plan_2026_09_23_cacioppo_psychophysiological_inference, docs_novelty_plan_2026_09_23_shortcut_learning_geirhos, docs_novelty_plan_2026_09_23_v3_framework_goldsack, docs_novelty_plan_2026_09_23_autonomic_specificity, docs_novelty_plan_2026_09_23_epel_2018_stress_constructs, docs_novelty_plan_2026_09_23_challenge_vs_threat [EXTRACTED 1.00]
- **Emotion psychophysiology evidence that HR/EDA carry mainly arousal** — docs_literature_novelty_2026_09_23_04_theory_framing_kreibig_2010, docs_literature_novelty_2026_09_23_04_theory_framing_siegel_2018, docs_literature_novelty_2026_09_23_04_theory_framing_quigley_barrett_2014, docs_literature_novelty_2026_09_23_04_theory_framing_autonomic_specificity_debate [EXTRACTED 1.00]
- **Measurement-validity reframe of the arousal result** — docs_literature_novelty_2026_09_23_04_theory_framing_psychophysiological_inference_problem, docs_literature_novelty_2026_09_23_04_theory_framing_discriminant_validity, docs_literature_novelty_2026_09_23_04_theory_framing_construct_shared_shortcut, docs_literature_novelty_2026_09_23_04_theory_framing_challenge_threat_model, docs_literature_novelty_2026_09_23_04_theory_framing_many_to_one_mapping [INFERRED 0.85]
- **Arousal-controlled reporting standard components** — docs_literature_novelty_2026_09_23_05_closable_gaps_arousal_index_baseline, docs_literature_novelty_2026_09_23_05_closable_gaps_arousal_controlled_auroc, docs_literature_novelty_2026_09_23_05_closable_gaps_kwon_2026_iso_hr, docs_literature_novelty_2026_09_23_05_closable_gaps_arousal_residual_decomposition, docs_literature_novelty_2026_09_23_04_theory_framing_analysis_c_ppv [INFERRED 0.75]

## Communities (55 total, 3 thin omitted)

### Community 0 - "Novelty Collision & Reviewer Sim"
Cohesion: 0.06
Nodes (62): Novelty-collision check (2026-09-23), Aydogan & Villagra Povina 2026 (Med Eng Phys) Stress or arousal?, Bosch et al. 2026 (Int J Psychophysiol), Claim: detectors detect arousal, not stress, Claim: five-dataset E4 leave-one-dataset-out, Claim: label audit of five public E4 datasets, Claim: matched-arousal specificity test (HR + tonic EDA index), Claim: specificity panel of non-stress states (+54 more)

### Community 1 - "Feature Extraction (HR/HRV/EDA)"
Cohesion: 0.07
Nodes (53): functools, neurokit2, pickle, cardiac_features(), clean_beats(), extract_campanella(), extract_epm(), extract_physionet() (+45 more)

### Community 2 - "Legacy API & Splits"
Cohesion: 0.07
Nodes (44): BaseModel, csv, fastapi, get, hashlib, json, get_model_info(), health_check() (+36 more)

### Community 3 - "Legacy Detector & Tuning"
Cohesion: 0.08
Nodes (39): FixedTrial, joblib, Extract features from raw signals, Predict stress class from features, StressDetector, optuna, cross_dataset(), cross_dataset_arousal() (+31 more)

### Community 4 - "Novelty Search Papers"
Cohesion: 0.07
Nodes (36): Novelty Search for the JBHI Rewrite, Akkaya 2026, Albaladejo-Gonzalez et al. 2023, Amin et al. 2025, Bahameish et al. 2024, Bent et al. 2020, Calza-Metre and Borzi 2026, Can Benouis and Andre 2026 (+28 more)

### Community 5 - "Contribution Report & Gap Scan"
Cohesion: 0.09
Nodes (33): Contribution Map Report for the JBHI Revision (IEEE), Audited Five-Dataset Benchmark Release, Foundation-Model Check, Contribution Gap Scan (literature scout), Abdel-Ghaffar et al. 2025 (Fitbit Body Response), Alchieri et al. 2026 (UME), Brandebusemeyer et al. 2026, EmoWork Dataset (2025) (+25 more)

### Community 6 - "Dataset Label Fixes"
Cohesion: 0.08
Nodes (33): Iqbal et al. 2022 (Stress-Predict), Q6 Recovery-Aware Labels (MIL, PU), Dataset and Claims Verification (full text), Author Label Deviations, BIOSTRESS Dataset, EmpathicSchool Dataset (Hosseini 2025), Fix 1 Stress-Predict Boundary Tag Snapping, Fix 2 Drop Stress-Predict S01 (+25 more)

### Community 7 - "Arousal Thesis & Warm-up"
Cohesion: 0.09
Nodes (33): Q4 Sensor Warm-up After Donning, Ranking vs Threshold Decomposition, WESAD (reading list), Paper Fig: Time in Session and Sensor Warm-up, What Do Wrist-Worn Stress Detectors Detect? (JBHI manuscript), Akkaya 2026, Bahameish et al. 2024, Baseline-Order / Time-in-Session Confound (+25 more)

### Community 8 - "Domain Adaptation & Few-Shot"
Cohesion: 0.13
Nodes (23): copy, Fig6: Few-Shot Personalisation (PhysioNet->WESAD), PhysioNet-to-WESAD: source-only 0.70, random 5-shot 0.92, first-in-time 5-shot + 30 s gap 0.86, coral(), evaluate(), finetune_k(), GradReverse, main() (+15 more)

### Community 9 - "Harmonization Audit"
Cohesion: 0.17
Nodes (23): build_dataset_label_matrix(), build_harmonization_table(), build_label_coverage_table(), build_overlap_tables(), build_shared_labels_table(), build_subject_label_table(), coerce_timestamp_series(), file_sha256() (+15 more)

### Community 10 - "Legacy Robustness & Active Learning"
Cohesion: 0.11
Nodes (25): Active Learning: Accuracy vs Labeled Samples (legacy figure), Uncertainty sampling active learning (0.81 at 1050 labels vs random 0.77, entropy 0.78), Adversarial Robustness Analysis (legacy figure), Perturbation robustness test (random noise, FGSM, feature dropout; baseline ~0.945, noise drops to 0.49 at eps 0.5), Confusion Matrix - XGBoost (legacy figure), Perfect six-class XGBoost confusion matrix (zero off-diagonal errors, 2103 windows), Bootstrap accuracy and F1 distributions (legacy figure), Bootstrap 95% CI: accuracy mean 0.9456 (0.935-0.955), F1 mean 0.9453 (+17 more)

### Community 11 - "Biomarker Evidence Report"
Cohesion: 0.10
Nodes (23): Wearable Stress Biomarkers Evidence Report, Basaran et al. 2024, Cheng et al. 2022, Claim Verdicts (8 supported, 7 overstated, 3 misattributed, 2 contradicted), Cukic et al. 2023, Four-Class Label Taxonomy Plan, Hosseini et al. 2022 (Nurse Dataset), Kim et al. 2023 (+15 more)

### Community 12 - "Paper Figure Builder"
Cohesion: 0.16
Nodes (17): matplotlib, matplotlib_pyplot, ci(), fig2_panel(), Figures for the JBHI manuscript. Reads committed tables only; writes PDFs next…, dot_rows(), exercise(), few_shot() (+9 more)

### Community 13 - "Model Comparison Stats"
Cohesion: 0.17
Nodes (18): io, itertools, scipy, compare(), corrected_ttest(), holm(), main(), DataFrame (+10 more)

### Community 14 - "LODO & Missing-PPG Shortcut"
Cohesion: 0.26
Nodes (13): argparse, Informative PPG missingness as cross-dataset shortcut (rank 2), numpy, pandas, pathlib, Leave-one-dataset-out stress detection across every dataset that has stress and…, Which personal normalisation helps stress detection transfer, and is the gain…, Subject-independent evaluation suite for the JBHI revision. Runs on the output… (+5 more)

### Community 15 - "Frozen Split Evaluation"
Cohesion: 0.20
Nodes (16): build_skipped_summary(), confidence_interval(), evaluate_assignments(), evaluate_single_split(), get_feature_sets(), load_xgb_classifier(), main(), parse_args() (+8 more)

### Community 16 - "Legacy Six Classes"
Cohesion: 0.18
Nodes (18): Accelerometer Feature Dominance in Legacy Model, Aerobic exercise class, Amusement class, Anaerobic exercise class, Baseline class, Emotion class, Rejection Option Figure (accuracy-coverage trade-off), Rejection Option (confidence-threshold selective prediction) (+10 more)

### Community 17 - "Contribution Map"
Cohesion: 0.14
Nodes (17): Akkaya 2026, Counterbalanced UTSA lab study, docs/literature/contribution-gap-scan.md, Error budget (Shapley-averaged test-side filters), Farahani et al. 2026, Five-dataset benchmark release, Hardening probes, Iqbal et al. 2022 (+9 more)

### Community 18 - "Legacy SHAP Importance"
Cohesion: 0.19
Nodes (14): Accelerometer features dominate legacy six-class SHAP importance, Legacy XGBoost six-class classifier (window-level split), SHAP beeswarm - Stress class (legacy), SHAP dependence plots - Stress class (legacy), SHAP importance heatmap by class, top 20 (legacy), XGBoost SHAP feature importance, stacked by class (legacy), Random Forest vs SHAP feature importance (legacy), SHAP waterfall - Aerobic example (legacy) (+6 more)

### Community 19 - "README Datasets & Methods"
Cohesion: 0.23
Nodes (13): Campanella 2024 dataset, docs/literature/dataset-and-claims-verification.md, Domain adaptation (CORAL, MMD, DANN), Empatica E4, EPM-E4, HR/HRV + EDA main feature set, Label audit, Missing-temperature threshold shift (+5 more)

### Community 20 - "Stress Construct Critique"
Cohesion: 0.18
Nodes (12): PULSE (Zhao et al. 2025), Specificity panel, Subject detectability trait, Dickerson & Kemeny 2004, Milstein & Gordon 2020, Theory (b) social-evaluative EDA signature, Theory (c) trait reactivity, Theory (d) temporal edge error (+4 more)

### Community 21 - "Novelty Candidates Brief"
Cohesion: 0.15
Nodes (13): Novelty Search Brief, Candidate: Few-Shot Personalisation, Candidate: Five-Dataset LODO, Candidate: Label Quality Drives Transfer Failure, Candidate: Pitfalls with Controlled Evidence, Candidate: Wrist HR/HRV vs Chest ECG, Fig5: Wrist PPG vs Chest ECG (WESAD), HR error vs chest ECG 7.7 bpm under stress (1.4-1.9 otherwise) (+5 more)

### Community 22 - "LODO Normalise & Mild Stress"
Cohesion: 0.21
Nodes (12): fit_predict(), main(), normalise(), DataFrame, ndarray, Per-subject scaling. Exercise sessions never contribute statistics; they borrow…, ext_scores(), bootstrap_rho() (+4 more)

### Community 23 - "Normalisation Probe"
Cohesion: 0.18
Nodes (13): cond3_scores(), fit_predict(), main(), DataFrame, ndarray, Window minus the median of the same recording's strictly earlier, non-…, trailing_delta(), grouped_split() (+5 more)

### Community 24 - "Novelty Experiments"
Cohesion: 0.20
Nodes (11): Aydogan & Villagra Povina 2026, Exercise as non-stress negatives, Few-shot calibration on unseen dataset, Kwon et al. setting (WESAD + Stress-Predict), Mild-stress pilot, Normalisation variants (session, raw, baseline, causal), AutoStress 2026, docs/literature/novelty-search.md (+3 more)

### Community 25 - "Legacy Model Enhancements"
Cohesion: 0.20
Nodes (12): Enhancement Summary Figure (legacy), Legacy enhancements: calibration, RFE-30, 3x augmentation, transformer, lightweight, multi-task, Optuna-optimized XGBoost (test acc ~0.945, 3.4 MB), Error Analysis Figure (legacy), Per-class error: Amusement 0.19, Baseline 0.14 highest; Emotion 0.02, Stress 0.03, Final Validation Dashboard (legacy), Legacy cross-dataset accuracy: WESAD ~0.43, EPM-E4 ~0, PhysioNet ~0.11, Legacy dataset composition: PhysioNet 63.7%, EPM-E4 23.9%, WESAD 12.4% (+4 more)

### Community 26 - "Few-Shot Unseen Dataset"
Cohesion: 0.24
Nodes (11): bootstrap(), centroid_score(), class_weights(), main(), DataFrame, ndarray, Per-user few-shot calibration of a model trained on OTHER datasets…, Distance to the non-stress centroid minus distance to the stress centroid,… (+3 more)

### Community 27 - "Project Memory (CLAUDE.md)"
Cohesion: 0.20
Nodes (10): Baseline tuning (tune_baselines.py), docs/advisor_correction_sheet.md, Four pipeline defects, Khan et al. 2025, Prajod et al. 2024, Schmidt et al. 2019, Vos et al. 2023, Working rules (+2 more)

### Community 28 - "Matched Arousal Probe"
Cohesion: 0.18
Nodes (7): math, os, binned_auc(), class_coef(), Probe 3: matched-arousal separability (A), ceiling error budget (B),…, logit(score) ~ arousal + class(stress=1); subject-cluster bootstrap CI on class…, Quintile bins on pooled pair; within-bin AUROC weighted by bin size; subject…

### Community 29 - "Matched Arousal Hardening"
Cohesion: 0.27
Nodes (8): binned(), bins_of(), holm(), pair_masks(), Hardening fixes 1, 2, 5 for the matched-arousal probe. Reuses committed…, Within-quintile weighted AUROC, subject bootstrap CI, block permutation p…, run_partA(), wauc()

### Community 30 - "Stressor Potency Dose-Response"
Cohesion: 0.22
Nodes (6): Analysis B: stressor-potency dose-response, Step 4: stressor-potency dose-response, scipy_stats, boot_rate(), Specificity panel (A) and subject-level miss consistency (B). Matches LODO…, Pooled window rate, CI by subject bootstrap, per-subject median/IQR.

### Community 31 - "Theory Framing Reframe"
Cohesion: 0.27
Nodes (10): 04 Theory Framing: measurement-validity reframe, Campbell & Ehlert (2012) Psychoneuroendocrinology, Can, Arnrich & Ersoy (2019) J Biomed Inform, Biopsychosocial challenge vs threat model, Kaya et al. (2026) arXiv 2604.12671 TSST vs cycling with cortisol, Porter & Goolkasian (2019) Front Psychol, Seery (2011) Neurosci Biobehav Rev, Sharma et al. (2026) JMIR scoping review of naturalistic stress studies (+2 more)

### Community 32 - "Closable Gaps (Benchmark, Encoders)"
Cohesion: 0.22
Nodes (10): 05 Closable gaps with existing data, Alchieri et al. (2026) UME EDA foundation model, Benchmark artefact release on Zenodo (rank 5), Farahani, Cao & Rahmani (2026) arXiv 2608.18397 conformal gate, Frozen public PPG encoder under LODO (rank 4), Geenjaar et al. (2026) multimodal-guided PPG FM, Liu & Ning (2026) arXiv 2609.22622 EDA survey and benchmark, PaPaGei PPG foundation model (ICLR 2025) (+2 more)

### Community 33 - "LODO Prior Work (Kwon, AutoStress)"
Cohesion: 0.24
Nodes (10): Kwon et al. 2026, Schreiber and Maleshkova 2026 (AutoStress), Verdict: LODO Partly Anticipated, Fig2: Leave-One-Dataset-Out (HR/HRV/EDA), Same-dataset vs other-datasets-only vs all-datasets training (HR/HRV/EDA, 20 splits), Stress-Predict lowest balanced accuracy (~0.65 external, ~0.70 within), Paper Fig: LODO Bar Chart (with/without temperature), Kwon et al. 2026 (+2 more)

### Community 34 - "Cacioppo Inference & PPV"
Cohesion: 0.25
Nodes (9): Analysis C: deployment PPV via Bayes on specificity-panel rates, Cacioppo, Tassinary & Berntson (2007) Handbook of Psychophysiology ch. 1, Cacioppo & Tassinary (1990) Am Psychol, Discriminant clinical validation of wrist stress detectors, Goldsack et al. (2020) V3 framework, npj Digit Med, Many-to-one psychophysiological mapping (wrist stress is a concomitant, not a marker), Outcome/marker/concomitant/invariant taxonomy, Psychophysiological inference problem: P(physiology|state) vs P(state|physiology) (+1 more)

### Community 35 - "Legacy Anomaly Detection"
Cohesion: 0.25
Nodes (9): Isolation Forest anomaly detection (legacy figure), Isolation Forest anomaly scoring (anomaly threshold ~0.50; highest rates in Aerobic 10% and Anaerobic 8%), Anomaly level distribution and outlier-method agreement (legacy figure), Multi-method outlier agreement (levels 0-4; ~6500 windows level 0; ~200 flagged by all 3 methods), Signal feature distributions by anomaly level (legacy figure), Physiologically implausible feature values (temp_mean ~200-240, acc_mag_mean ~140, eda_mean up to 70), Calibration curves: base, Platt, isotonic (legacy figure), Per-class probability calibration (isotonic closest to diagonal; Amusement and Baseline noisiest) (+1 more)

### Community 36 - "Legacy Confusion & Conformal"
Cohesion: 0.22
Nodes (9): Conformal Prediction Figure (legacy), Conformal set size mean 0.92 at 90% coverage (sub-singleton sets), Six-class Confusion Matrices, 8 Models (legacy), Perfect diagonal for Decision Tree, Random Forest, XGBoost, MLP, Cross-Dataset Analysis Figure (legacy), Class labels aliased with dataset identity, CV vs LOSO Performance Figure (legacy), Window-level CV vs LOSO gap (0.95 vs 0.72) (+1 more)

### Community 37 - "Autonomic Specificity & Analyses A/B"
Cohesion: 0.25
Nodes (8): Analysis A: SAM felt arousal vs valence against detector score, Analysis B: PPG pulse amplitude and WESAD pulse arrival time as vascular channel, Autonomic specificity debate, Kreibig (2010) Biol Psychol: ANS specificity in emotion, Quigley & Barrett (2014) Biol Psychol: first discriminant function is arousal, Siegel et al. (2018) Psychol Bull: 202-study meta-analysis, no ANS fingerprints, Smets et al. (2018) SWEET, npj Digit Med, Self-report construct validity: SAM arousal vs valence (rank 6)

### Community 38 - "Legacy Feature Correlation"
Cohesion: 0.32
Nodes (8): Feature Correlation Heatmap, Top 20 (legacy), Redundant feature pairs (acc_energy~acc_mag_mean 1.00, temp_mean~temp_min 0.99, eda_mean~eda_max 0.99), Feature Importance Comparison RF vs XGBoost (legacy), Top 20 Feature Importance, Random Forest (legacy), Accelerometer features dominate importance (acc_y_mean top; no BVP/HR in RF top 20), Feature Interactions and Mutual Information Figure (legacy), Legacy selected features, Accelerometer-dominated feature selection

### Community 39 - "Self-Report Labels"
Cohesion: 0.50
Nodes (7): epm_reports(), main(), physionet_reports(), DataFrame, Path, Attach participants' own ratings to the harmonized windows. Protocol labels say…, wesad_reports()

### Community 40 - "Session Z-Scoring"
Cohesion: 0.29
Nodes (7): Tognotti et al. 2026, Fecke and Rehof 2026, Tognotti et al. 2026, Fig3: Normalisation Variants, Causal scaling drops UBFC-Phys (~0.84 to 0.74) and Campanella (~0.87 to 0.77), Whole-session z-score vs past-windows-only vs no per-subject scaling, Per-Subject Whole-Session Z-Scoring

### Community 41 - "Arousal-Controlled Evaluation Kit"
Cohesion: 0.29
Nodes (7): arousal_controlled_auroc(scores, index, labels, subjects), Arousal-controlled evaluation kit (rank 1), Label-free arousal index baseline (z HR + z tonic EDA), Arousal + residual decomposition of stress score (rank 3), Kwon et al. (2026) Healthcare: iso-HR matched test and HR-leakage index, Matched-arousal novelty claim overstated (main.tex line 33), Central claim: wrist detectors detect autonomic arousal; ceiling is stressor potency

### Community 42 - "Window-Split Leakage"
Cohesion: 0.33
Nodes (6): Campanella et al. 2023 (Sensors), Fig1: Window vs Subject Split, Random window split: 0.92 accuracy / 0.81 balanced accuracy (11-class), Subject-grouped split: 0.49 accuracy / 0.36 balanced accuracy (11-class), Saeb et al. 2017, Window-Level Split Leakage

### Community 43 - "Legacy Leakage Symptoms"
Cohesion: 0.33
Nodes (6): Stress regression plot (R2 = 1.000, MAE = 0.000), Degenerate stress regression: all targets = 1.5, Temporal patterns: state transition matrix and state durations, Near-deterministic state persistence (self-transition 0.95-1.00), XGBoost accuracy across validation strategies, Validation gap: 5-fold CV 95.00%, holdout 93.79%, LOSO 72.00%, cross-dataset 17.94%

### Community 44 - "Wrist vs Chest ECG & Deps"
Cohesion: 0.33
Nodes (4): Wrist PPG vs chest ECG validation, neurokit2, optuna, xgboost

### Community 45 - "Harmonization LaTeX Export"
Cohesion: 0.47
Nodes (5): latex_escape(), main(), parse_args(), Namespace, Export a paper-ready LaTeX harmonization table from harmonization_table.csv.

### Community 46 - "LODO Normalise Tests"
Cohesion: 0.47
Nodes (5): sys, DataFrame, recording(), test_causal_uses_only_earlier_non_overlapping_windows(), test_session_and_baseline_scale_exercise_with_protocol_statistics()

### Community 47 - "Exercise False Alarms"
Cohesion: 0.50
Nodes (5): Aydogan and Villagra Povina 2026, Fig4: Exercise False Alarms, 69.5% of PhysioNet exercise windows called stress without exercise in training (4.3%/6.4% with), Aydogan and Villagra Povina 2026, Exercise False-Stress Alarms

### Community 48 - "Stress Constructs (Epel, Allostasis)"
Cohesion: 0.50
Nodes (4): Allostasis: detector as metabolic mobilisation index, Dickerson & Kemeny (2004) Psychol Bull: social-evaluative threat, Epel et al. (2018) Front Neuroendocrinol 'More than a feeling', Stressor exposure vs physiological response vs appraisal

### Community 49 - "Shortcut Learning & Transfer Cost"
Cohesion: 0.50
Nodes (4): Construct-shared shortcut (arousal) survives leave-one-dataset-out, Geirhos et al. (2020) Shortcut learning, Nat Mach Intell, Morgan's Canon, LODO transfer cost 0.015-0.05 BA (HR/HRV/EDA)

### Community 50 - "Legacy Hyperparameter Plots"
Cohesion: 0.50
Nodes (4): XGBoost hyperparameter sensitivity plot, Learning rate is most sensitive XGBoost hyperparameter (14.8% range), Optuna XGBoost optimization history and hyperparameter importance, Gamma dominates Optuna hyperparameter importance (~0.80)

### Community 51 - "Legacy Deployment & Drift"
Cohesion: 0.67
Nodes (3): Deployment Metrics Figure (legacy), Drift Detection Figure (legacy), KS-test feature drift monitor

## Knowledge Gaps
- **142 isolated node(s):** `Redundant feature pairs (acc_energy~acc_mag_mean 1.00, temp_mean~temp_min 0.99, eda_mean~eda_max 0.99)`, `Rejection Option (confidence-threshold selective prediction)`, `Recursive Feature Elimination (RFE)`, `SHAP dependence plots - Stress class (legacy)`, `SHAP waterfall - Amusement example (legacy)` (+137 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 266 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `What Do Wrist-Worn Stress Detectors Detect? (JBHI manuscript)` connect `Arousal Thesis & Warm-up` to `Novelty Collision & Reviewer Sim`, `LODO Prior Work (Kwon, AutoStress)`, `Contribution Report & Gap Scan`, `Dataset Label Fixes`, `Session Z-Scoring`, `Arousal-Controlled Evaluation Kit`, `Window-Split Leakage`, `Biomarker Evidence Report`, `Exercise False Alarms`, `Novelty Candidates Brief`, `Theory Framing Reframe`?**
  _High betweenness centrality (0.170) - this node is a cross-community bridge._
- **Why does `Novelty Search for the JBHI Rewrite` connect `Novelty Search Papers` to `LODO Prior Work (Kwon, AutoStress)`, `Feature Extraction (HR/HRV/EDA)`, `Dataset Label Fixes`, `Arousal Thesis & Warm-up`, `Session Z-Scoring`, `Domain Adaptation & Few-Shot`, `Exercise False Alarms`, `Novelty Candidates Brief`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `Novelty plan for the JBHI paper (2026-09-23)` connect `Novelty Collision & Reviewer Sim` to `Closable Gaps (Benchmark, Encoders)`, `Arousal-Controlled Evaluation Kit`, `Stressor Potency Dose-Response`, `Theory Framing Reframe`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Dataset and Claims Verification (full text)` (e.g. with `Hyperventilation Block (Stress-Predict)` and `Per-Subject Whole-Session Z-Scoring`) actually correct?**
  _`Dataset and Claims Verification (full text)` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Redundant feature pairs (acc_energy~acc_mag_mean 1.00, temp_mean~temp_min 0.99, eda_mean~eda_max 0.99)`, `Rejection Option (confidence-threshold selective prediction)`, `Recursive Feature Elimination (RFE)` to the rest of the system?**
  _142 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Novelty Collision & Reviewer Sim` be split into smaller, more focused modules?**
  _Cohesion score 0.06240084611316764 - nodes in this community are weakly interconnected._
- **Should `Feature Extraction (HR/HRV/EDA)` be split into smaller, more focused modules?**
  _Cohesion score 0.06662770309760374 - nodes in this community are weakly interconnected._