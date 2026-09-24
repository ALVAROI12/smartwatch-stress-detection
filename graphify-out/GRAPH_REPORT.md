# Graph Report - novelty-plan  (2026-09-23)

## Corpus Check
- 7 files · ~471,763 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 857 nodes · 1546 edges · 55 communities (42 shown, 13 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 176 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Feature Extraction (HR/HRV/EDA)
- Project Memory & Central Claim
- Contribution Map
- Theory Framing (Cacioppo, V3)
- Legacy API & Splits
- Novelty Collision & Reviewer Sim
- Baseline Tuning & Experiments
- Novelty Search Papers
- Domain Adaptation & Few-Shot
- Contribution Gap Scan
- Harmonization Audit
- Model Comparison Stats
- Normalisation Probe
- Biomarker Evidence Report
- Arousal-Controlled Evaluation Kit
- Legacy Robustness & Active Learning
- Paper Figure Builder
- Dataset Label Fixes
- Frozen Split Evaluation
- Legacy Six Classes
- Novelty Experiments & Missing PPG
- Matched Arousal Hardening
- Legacy SHAP Importance
- Few-Shot Unseen Dataset
- Legacy Feature Correlation
- Legacy Model Enhancements
- Significance Tests
- Legacy Anomaly Detection
- Legacy Confusion & Conformal
- Specificity Panel Probe
- Self-Report Labels
- Novelty Candidates Brief
- Legacy Leakage Symptoms
- Harmonization LaTeX Export
- LODO Normalise Tests
- Legacy Hyperparameter Plots
- LODO Figure
- Wrist vs Chest ECG Figure
- Window Leakage Prior Work
- Legacy Deployment & Drift
- Window-Split Leakage Figure
- Normalisation Figure
- Binned AUC Helpers
- Arousal Logit Model
- Order Confound Prior Work
- Few-Shot Prior Work
- Missing Channel Prior Work
- Wrist HRV Prior Work
- Legacy Learning Curves
- Legacy Subject Clustering
- Exercise False Alarms Figure
- Legacy Deep Learning
- Error Budget Figure
- Specificity Figure
- Time-in-Session Figure

## God Nodes (most connected - your core abstractions)
1. `Novelty Search for the JBHI Rewrite` - 48 edges
2. `What Do Wrist-Worn Stress Detectors Detect? (manuscript)` - 26 edges
3. `Contribution Gap Scan (literature scout)` - 24 edges
4. `Dataset and Claims Verification (full text)` - 24 edges
5. `grouped_split()` - 21 edges
6. `04 Theory Framing: measurement-validity reframe` - 19 edges
7. `Novelty-collision check (2026-09-23)` - 17 edges
8. `Wearable Stress Biomarkers Evidence Report` - 16 edges
9. `Novelty plan for the JBHI paper (2026-09-23)` - 16 edges
10. `normalise()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Legacy model comparison figure` --conceptually_related_to--> `Window-level split leakage`  [INFERRED]
  legacy/outputs/figures/model_comparison_final.pdf → README.md
- `Few-shot calibration on unseen dataset` --semantically_similar_to--> `Few-shot personalisation`  [INFERRED] [semantically similar]
  docs/novelty_experiments.md → README.md
- `Matched-arousal novelty claim overstated (main.tex line 33)` --references--> `What Do Wrist-Worn Stress Detectors Detect? (manuscript)`  [EXTRACTED]
  docs/literature/novelty-2026-09-23/05_closable_gaps.md → paper/main.pdf
- `arousal_controlled_auroc()` --implements--> `Matched-arousal specificity test`  [INFERRED]
  scripts/arousal_kit.py → CLAUDE.md
- `Paper direction: What Do Wrist-Worn Stress Detectors Detect?` --semantically_similar_to--> `Detector detects arousal; ceiling is stressor potency`  [INFERRED] [semantically similar]
  STATUS.md → CLAUDE.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Emotion psychophysiology evidence that HR/EDA carry mainly arousal** — docs_literature_novelty_2026_09_23_04_theory_framing_kreibig_2010, docs_literature_novelty_2026_09_23_04_theory_framing_siegel_2018, docs_literature_novelty_2026_09_23_04_theory_framing_quigley_barrett_2014, docs_literature_novelty_2026_09_23_04_theory_framing_autonomic_specificity_debate [EXTRACTED 1.00]
- **Five E4 datasets in leave-one-dataset-out** — readme_wesad, readme_physionet_wearable_stress_exercise_dataset, readme_stress_predict, readme_ubfc_phys, readme_campanella_2024_dataset, readme_leave_one_dataset_out_transfer_cost [EXTRACTED 1.00]
- **Legacy per-class SHAP beeswarm plots** — legacy_outputs_figures_shap_class_aerobic, legacy_outputs_figures_shap_class_amusement, legacy_outputs_figures_shap_class_anaerobic, legacy_outputs_figures_shap_class_baseline, legacy_outputs_figures_shap_class_emotion, shap_feature_attribution [EXTRACTED 1.00]
- **Legacy per-class SHAP waterfall explanations (39 features, XGBoost)** — legacy_outputs_figures_shap_waterfall_baseline, legacy_outputs_figures_shap_waterfall_emotion, legacy_outputs_figures_shap_waterfall_stress, legacy_outputs_figures_shap_waterfall_baseline_accelerometer_dominance [EXTRACTED 1.00]
- **PhysioNet Label Fixes (f13, second-half rests, baseline length)** — docs_literature_dataset_and_claims_verification_fix_5_exclude_physionet_f13, docs_literature_dataset_and_claims_verification_fix_6_physionet_second_half_rests, docs_literature_dataset_and_claims_verification_fix_7_physionet_v2_baseline_length [EXTRACTED 1.00]
- **Prior work anticipating the arousal-not-stress claim** — docs_literature_novelty_2026_09_23_01_collision_check_zhou_2023, docs_literature_novelty_2026_09_23_01_collision_check_kwon_2026, docs_literature_novelty_2026_09_23_01_collision_check_aydogan_2026, docs_literature_novelty_2026_09_23_01_collision_check_kaya_2026, docs_literature_novelty_2026_09_23_01_collision_check_sosa_2026 [EXTRACTED 1.00]
- **Legacy deployment and monitoring analyses** — legacy_outputs_figures_deployment_metrics, legacy_outputs_figures_drift_detection, legacy_outputs_figures_conformal_prediction [INFERRED 0.65]
- **Arousal-controlled reporting standard components** — docs_literature_novelty_2026_09_23_05_closable_gaps_arousal_index_baseline, docs_literature_novelty_2026_09_23_05_closable_gaps_arousal_controlled_auroc, docs_literature_novelty_2026_09_23_05_closable_gaps_kwon_2026_iso_hr, docs_literature_novelty_2026_09_23_05_closable_gaps_arousal_residual_decomposition, docs_literature_novelty_2026_09_23_04_theory_framing_analysis_c_ppv [INFERRED 0.75]
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
- **Measurement-validity reframe of the arousal result** — docs_literature_novelty_2026_09_23_04_theory_framing_psychophysiological_inference_problem, docs_literature_novelty_2026_09_23_04_theory_framing_discriminant_validity, docs_literature_novelty_2026_09_23_04_theory_framing_construct_shared_shortcut, docs_literature_novelty_2026_09_23_04_theory_framing_challenge_threat_model, docs_literature_novelty_2026_09_23_04_theory_framing_many_to_one_mapping [INFERRED 0.85]
- **Five public E4 datasets in LODO** — paper_main_schmidt_wesad, paper_main_hongn_physionet, paper_main_stress_predict, paper_main_ubfc_phys, paper_main_campanella_2024 [EXTRACTED 1.00]
- **Arousal-controlled evaluation kit analyses** — docs_arousal_kit_untrained_arousal_index, docs_arousal_kit_equivalence_mde, docs_arousal_kit_probe_vs_rest, docs_arousal_kit_iso_hr_test, docs_arousal_kit_arousal_residual [EXTRACTED 1.00]
- **Theory frames for the arousal claim** — docs_novelty_plan_2026_09_23_cacioppo_psychophysiological_inference, docs_novelty_plan_2026_09_23_autonomic_specificity, docs_novelty_plan_2026_09_23_v3_framework, docs_novelty_plan_2026_09_23_shortcut_learning, docs_novelty_plan_2026_09_23_challenge_threat [EXTRACTED 1.00]

## Communities (55 total, 13 thin omitted)

### Community 0 - "Feature Extraction (HR/HRV/EDA)"
Cohesion: 0.07
Nodes (53): functools, neurokit2, pickle, cardiac_features(), clean_beats(), extract_campanella(), extract_epm(), extract_physionet() (+45 more)

### Community 1 - "Project Memory & Central Claim"
Cohesion: 0.05
Nodes (56): Detector detects arousal; ceiling is stressor potency, Empatica E4 wrist device, Seven label fixes (commit 6070b6b), Leave-one-dataset-out over five E4 datasets, Matched-arousal specificity test, Per-subject z-scoring (transductive), CLAUDE.md project instructions (JBHI revision), Subject-grouped splits with Nadeau-Bengio corrected tests and Holm (+48 more)

### Community 2 - "Contribution Map"
Cohesion: 0.05
Nodes (49): Akkaya 2026, Counterbalanced UTSA lab study, docs/literature/contribution-gap-scan.md, Error budget (Shapley-averaged test-side filters), Farahani et al. 2026, Five-dataset benchmark release, Hardening probes, Iqbal et al. 2022 (+41 more)

### Community 3 - "Theory Framing (Cacioppo, V3)"
Cohesion: 0.05
Nodes (50): 04 Theory Framing: measurement-validity reframe, Allostasis: detector as metabolic mobilisation index, Analysis A: SAM felt arousal vs valence against detector score, Analysis B: PPG pulse amplitude and WESAD pulse arrival time as vascular channel, Analysis C: deployment PPV via Bayes on specificity-panel rates, Autonomic specificity debate, Cacioppo, Tassinary & Berntson (2007) Handbook of Psychophysiology ch. 1, Cacioppo & Tassinary (1990) Am Psychol (+42 more)

### Community 4 - "Legacy API & Splits"
Cohesion: 0.07
Nodes (40): BaseModel, csv, fastapi, get, hashlib, joblib, json, get_model_info() (+32 more)

### Community 5 - "Novelty Collision & Reviewer Sim"
Cohesion: 0.09
Nodes (43): Novelty-collision check (2026-09-23), Aydogan & Villagra Povina 2026 (Med Eng Phys) Stress or arousal?, Bosch et al. 2026 (Int J Psychophysiol), Claim: detectors detect arousal, not stress, Claim: five-dataset E4 leave-one-dataset-out, Claim: label audit of five public E4 datasets, Claim: matched-arousal specificity test (HR + tonic EDA index), Claim: specificity panel of non-stress states (+35 more)

### Community 6 - "Baseline Tuning & Experiments"
Cohesion: 0.11
Nodes (35): FixedTrial, optuna, cross_dataset(), cross_dataset_arousal(), fit_predict(), harmonization_table(), leakage_check(), main() (+27 more)

### Community 7 - "Novelty Search Papers"
Cohesion: 0.06
Nodes (33): Novelty Search for the JBHI Rewrite, Akkaya 2026, Albaladejo-Gonzalez et al. 2023, Amin et al. 2025, Aydogan and Villagra Povina 2026, Bent et al. 2020, Calza-Metre and Borzi 2026, Can Benouis and Andre 2026 (+25 more)

### Community 8 - "Domain Adaptation & Few-Shot"
Cohesion: 0.13
Nodes (23): copy, Fig6: Few-Shot Personalisation (PhysioNet->WESAD), PhysioNet-to-WESAD: source-only 0.70, random 5-shot 0.92, first-in-time 5-shot + 30 s gap 0.86, coral(), evaluate(), finetune_k(), GradReverse, main() (+15 more)

### Community 9 - "Contribution Gap Scan"
Cohesion: 0.08
Nodes (27): Audited Five-Dataset Benchmark Release, Foundation-Model Check, Contribution Gap Scan (literature scout), Abdel-Ghaffar et al. 2025 (Fitbit Body Response), Alchieri et al. 2026 (UME), Brandebusemeyer et al. 2026, EmoWork Dataset (2025), EmpkinS PEPbench (+19 more)

### Community 10 - "Harmonization Audit"
Cohesion: 0.18
Nodes (22): build_dataset_label_matrix(), build_harmonization_table(), build_label_coverage_table(), build_overlap_tables(), build_shared_labels_table(), build_subject_label_table(), coerce_timestamp_series(), file_sha256() (+14 more)

### Community 11 - "Model Comparison Stats"
Cohesion: 0.12
Nodes (23): io, compare(), corrected_ttest(), holm(), main(), DataFrame, ndarray, Series (+15 more)

### Community 12 - "Normalisation Probe"
Cohesion: 0.17
Nodes (19): argparse, fit_predict(), main(), DataFrame, ndarray, Which personal normalisation helps stress detection transfer, and is the gain…, Window minus the median of the same recording's strictly earlier, non-…, trailing_delta() (+11 more)

### Community 13 - "Biomarker Evidence Report"
Cohesion: 0.10
Nodes (23): Wearable Stress Biomarkers Evidence Report, Basaran et al. 2024, Cheng et al. 2022, Claim Verdicts (8 supported, 7 overstated, 3 misattributed, 2 contradicted), Cukic et al. 2023, Four-Class Label Taxonomy Plan, Hosseini et al. 2022 (Nurse Dataset), Kim et al. 2023 (+15 more)

### Community 14 - "Arousal-Controlled Evaluation Kit"
Cohesion: 0.15
Nodes (16): Arousal plus residual decomposition, numpy, os, pathlib, arousal_controlled_auroc(), _bins(), iso_hr_subsample(), main() (+8 more)

### Community 15 - "Legacy Robustness & Active Learning"
Cohesion: 0.13
Nodes (21): Active Learning: Accuracy vs Labeled Samples (legacy figure), Uncertainty sampling active learning (0.81 at 1050 labels vs random 0.77, entropy 0.78), Adversarial Robustness Analysis (legacy figure), Perturbation robustness test (random noise, FGSM, feature dropout; baseline ~0.945, noise drops to 0.49 at eps 0.5), Confusion Matrix - XGBoost (legacy figure), Perfect six-class XGBoost confusion matrix (zero off-diagonal errors, 2103 windows), Bootstrap accuracy and F1 distributions (legacy figure), Bootstrap 95% CI: accuracy mean 0.9456 (0.935-0.955), F1 mean 0.9453 (+13 more)

### Community 16 - "Paper Figure Builder"
Cohesion: 0.16
Nodes (17): matplotlib, matplotlib_pyplot, ci(), fig2_panel(), Figures for the JBHI manuscript. Reads committed tables only; writes PDFs next…, dot_rows(), exercise(), few_shot() (+9 more)

### Community 17 - "Dataset Label Fixes"
Cohesion: 0.10
Nodes (21): Dataset and Claims Verification (full text), Author Label Deviations, BIOSTRESS Dataset, Campanella et al. 2023 (Sensors), EmpathicSchool Dataset (Hosseini 2025), Fix 1 Stress-Predict Boundary Tag Snapping, Fix 2 Drop Stress-Predict S01, Fix 3 Campanella Subtraction Span per Subject (+13 more)

### Community 18 - "Frozen Split Evaluation"
Cohesion: 0.19
Nodes (17): build_skipped_summary(), confidence_interval(), evaluate_assignments(), evaluate_single_split(), get_feature_sets(), load_xgb_classifier(), main(), parse_args() (+9 more)

### Community 19 - "Legacy Six Classes"
Cohesion: 0.18
Nodes (18): Accelerometer Feature Dominance in Legacy Model, Aerobic exercise class, Amusement class, Anaerobic exercise class, Baseline class, Emotion class, Rejection Option Figure (accuracy-coverage trade-off), Rejection Option (confidence-threshold selective prediction) (+10 more)

### Community 20 - "Novelty Experiments & Missing PPG"
Cohesion: 0.16
Nodes (15): Informative PPG missingness as cross-dataset shortcut (rank 2), Kwon et al. setting (WESAD + Stress-Predict), Mild-stress pilot, Normalisation variants (session, raw, baseline, causal), scipy_stats, main(), normalise(), DataFrame (+7 more)

### Community 21 - "Matched Arousal Hardening"
Cohesion: 0.20
Nodes (12): fit_predict(), ndarray, binned(), bins_of(), cond3_scores(), ext_scores(), holm(), pair_masks() (+4 more)

### Community 22 - "Legacy SHAP Importance"
Cohesion: 0.19
Nodes (14): Accelerometer features dominate legacy six-class SHAP importance, Legacy XGBoost six-class classifier (window-level split), SHAP beeswarm - Stress class (legacy), SHAP dependence plots - Stress class (legacy), SHAP importance heatmap by class, top 20 (legacy), XGBoost SHAP feature importance, stacked by class (legacy), Random Forest vs SHAP feature importance (legacy), SHAP waterfall - Aerobic example (legacy) (+6 more)

### Community 23 - "Few-Shot Unseen Dataset"
Cohesion: 0.22
Nodes (12): Few-shot calibration on unseen dataset, bootstrap(), centroid_score(), class_weights(), main(), DataFrame, ndarray, Per-user few-shot calibration of a model trained on OTHER datasets… (+4 more)

### Community 24 - "Legacy Feature Correlation"
Cohesion: 0.18
Nodes (13): Feature Correlation Heatmap, Top 20 (legacy), Redundant feature pairs (acc_energy~acc_mag_mean 1.00, temp_mean~temp_min 0.99, eda_mean~eda_max 0.99), Feature Importance Comparison RF vs XGBoost (legacy), Top 20 Feature Importance, Random Forest (legacy), Accelerometer features dominate importance (acc_y_mean top; no BVP/HR in RF top 20), Feature Interactions and Mutual Information Figure (legacy), Legacy inclusion criteria, Legacy missing value imputation report (+5 more)

### Community 25 - "Legacy Model Enhancements"
Cohesion: 0.20
Nodes (12): Enhancement Summary Figure (legacy), Legacy enhancements: calibration, RFE-30, 3x augmentation, transformer, lightweight, multi-task, Optuna-optimized XGBoost (test acc ~0.945, 3.4 MB), Error Analysis Figure (legacy), Per-class error: Amusement 0.19, Baseline 0.14 highest; Emotion 0.02, Stress 0.03, Final Validation Dashboard (legacy), Legacy cross-dataset accuracy: WESAD ~0.43, EPM-E4 ~0, PhysioNet ~0.11, Legacy dataset composition: PhysioNet 63.7%, EPM-E4 23.9%, WESAD 12.4% (+4 more)

### Community 26 - "Significance Tests"
Cohesion: 0.25
Nodes (6): itertools, math, pandas, scipy, Paired significance tests on the tuned baselines. Repeated subject-grouped…, Probe 3: matched-arousal separability (A), ceiling error budget (B),…

### Community 27 - "Legacy Anomaly Detection"
Cohesion: 0.25
Nodes (9): Isolation Forest anomaly detection (legacy figure), Isolation Forest anomaly scoring (anomaly threshold ~0.50; highest rates in Aerobic 10% and Anaerobic 8%), Anomaly level distribution and outlier-method agreement (legacy figure), Multi-method outlier agreement (levels 0-4; ~6500 windows level 0; ~200 flagged by all 3 methods), Signal feature distributions by anomaly level (legacy figure), Physiologically implausible feature values (temp_mean ~200-240, acc_mag_mean ~140, eda_mean up to 70), Calibration curves: base, Platt, isotonic (legacy figure), Per-class probability calibration (isotonic closest to diagonal; Amusement and Baseline noisiest) (+1 more)

### Community 28 - "Legacy Confusion & Conformal"
Cohesion: 0.22
Nodes (9): Conformal Prediction Figure (legacy), Conformal set size mean 0.92 at 90% coverage (sub-singleton sets), Six-class Confusion Matrices, 8 Models (legacy), Perfect diagonal for Decision Tree, Random Forest, XGBoost, MLP, Cross-Dataset Analysis Figure (legacy), Class labels aliased with dataset identity, CV vs LOSO Performance Figure (legacy), Window-level CV vs LOSO gap (0.95 vs 0.72) (+1 more)

### Community 29 - "Specificity Panel Probe"
Cohesion: 0.25
Nodes (4): Contribution Map Report for the JBHI Revision (IEEE), boot_rate(), Specificity panel (A) and subject-level miss consistency (B). Matches LODO…, Pooled window rate, CI by subject bootstrap, per-subject median/IQR.

### Community 30 - "Self-Report Labels"
Cohesion: 0.50
Nodes (7): epm_reports(), main(), physionet_reports(), DataFrame, Path, Attach participants' own ratings to the harmonized windows. Protocol labels say…, wesad_reports()

### Community 31 - "Novelty Candidates Brief"
Cohesion: 0.33
Nodes (6): Novelty Search Brief, Candidate: Few-Shot Personalisation, Candidate: Five-Dataset LODO, Candidate: Label Quality Drives Transfer Failure, Candidate: Pitfalls with Controlled Evidence, Candidate: Wrist HR/HRV vs Chest ECG

### Community 32 - "Legacy Leakage Symptoms"
Cohesion: 0.33
Nodes (6): Stress regression plot (R2 = 1.000, MAE = 0.000), Degenerate stress regression: all targets = 1.5, Temporal patterns: state transition matrix and state durations, Near-deterministic state persistence (self-transition 0.95-1.00), XGBoost accuracy across validation strategies, Validation gap: 5-fold CV 95.00%, holdout 93.79%, LOSO 72.00%, cross-dataset 17.94%

### Community 33 - "Harmonization LaTeX Export"
Cohesion: 0.47
Nodes (5): latex_escape(), main(), parse_args(), Namespace, Export a paper-ready LaTeX harmonization table from harmonization_table.csv.

### Community 34 - "LODO Normalise Tests"
Cohesion: 0.60
Nodes (4): DataFrame, recording(), test_causal_uses_only_earlier_non_overlapping_windows(), test_session_and_baseline_scale_exercise_with_protocol_statistics()

### Community 35 - "Legacy Hyperparameter Plots"
Cohesion: 0.50
Nodes (4): XGBoost hyperparameter sensitivity plot, Learning rate is most sensitive XGBoost hyperparameter (14.8% range), Optuna XGBoost optimization history and hyperparameter importance, Gamma dominates Optuna hyperparameter importance (~0.80)

### Community 36 - "LODO Figure"
Cohesion: 0.50
Nodes (4): Fig2: Leave-One-Dataset-Out (HR/HRV/EDA), Same-dataset vs other-datasets-only vs all-datasets training (HR/HRV/EDA, 20 splits), Stress-Predict lowest balanced accuracy (~0.65 external, ~0.70 within), Paper Fig: LODO Bar Chart (with/without temperature)

### Community 37 - "Wrist vs Chest ECG Figure"
Cohesion: 0.50
Nodes (4): Fig5: Wrist PPG vs Chest ECG (WESAD), HR error vs chest ECG 7.7 bpm under stress (1.4-1.9 otherwise), RMSSD correlation with chest ECG r = 0.38 under stress (0.53-0.66 otherwise), Usable wrist PPG: baseline 90%, amusement 74%, meditation 78%, stress 37%

### Community 38 - "Window Leakage Prior Work"
Cohesion: 0.67
Nodes (3): Bahameish et al. 2024, Saeb et al. 2017, Verdict: Window Leakage Already Published

### Community 39 - "Legacy Deployment & Drift"
Cohesion: 0.67
Nodes (3): Deployment Metrics Figure (legacy), Drift Detection Figure (legacy), KS-test feature drift monitor

### Community 40 - "Window-Split Leakage Figure"
Cohesion: 0.67
Nodes (3): Fig1: Window vs Subject Split, Random window split: 0.92 accuracy / 0.81 balanced accuracy (11-class), Subject-grouped split: 0.49 accuracy / 0.36 balanced accuracy (11-class)

### Community 41 - "Normalisation Figure"
Cohesion: 0.67
Nodes (3): Fig3: Normalisation Variants, Causal scaling drops UBFC-Phys (~0.84 to 0.74) and Campanella (~0.87 to 0.77), Whole-session z-score vs past-windows-only vs no per-subject scaling

## Knowledge Gaps
- **159 isolated node(s):** `Uncertainty sampling active learning (0.81 at 1050 labels vs random 0.77, entropy 0.78)`, `KNN imputation k=5`, `Legacy safety considerations`, `Stress-Predict (reading list)`, `Basaran et al. 2024` (+154 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 308 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Novelty Search for the JBHI Rewrite` connect `Novelty Search Papers` to `Feature Extraction (HR/HRV/EDA)`, `Window Leakage Prior Work`, `Domain Adaptation & Few-Shot`, `Contribution Gap Scan`, `Order Confound Prior Work`, `Few-Shot Prior Work`, `Missing Channel Prior Work`, `Wrist HRV Prior Work`, `Dataset Label Fixes`, `Novelty Candidates Brief`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Why does `What Do Wrist-Worn Stress Detectors Detect? (manuscript)` connect `Project Memory & Central Claim` to `Theory Framing (Cacioppo, V3)`, `Specificity Panel Probe`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Why does `Contribution Map Report for the JBHI Revision (IEEE)` connect `Specificity Panel Probe` to `Contribution Gap Scan`, `Significance Tests`, `Project Memory & Central Claim`, `Arousal-Controlled Evaluation Kit`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `What Do Wrist-Worn Stress Detectors Detect? (manuscript)` (e.g. with `Empatica E4 wrist device` and `Contribution Map Report for the JBHI Revision (IEEE)`) actually correct?**
  _`What Do Wrist-Worn Stress Detectors Detect? (manuscript)` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Uncertainty sampling active learning (0.81 at 1050 labels vs random 0.77, entropy 0.78)`, `KNN imputation k=5`, `Legacy safety considerations` to the rest of the system?**
  _159 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Feature Extraction (HR/HRV/EDA)` be split into smaller, more focused modules?**
  _Cohesion score 0.06662770309760374 - nodes in this community are weakly interconnected._
- **Should `Project Memory & Central Claim` be split into smaller, more focused modules?**
  _Cohesion score 0.052597402597402594 - nodes in this community are weakly interconnected._