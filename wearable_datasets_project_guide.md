# Machine Learning-Based Anomaly Detection for Stress Monitoring
## Complete Project Roadmap

---

## PROJECT RULES
- Concise and explicit outputs only
- No unnecessary explanations, guides, or README files
- Jupyter notebook organization from start
- No emojis
- No code/material changes without explicit permission

---

## DATASET OVERVIEW

### Dataset 1: WESAD (Wearable Stress and Affect Detection)
- **Subjects**: 15
- **Devices**: RespiBAN (chest) + Empatica E4 (wrist)
- **Sensors**: ECG, EDA, EMG, Respiration, Temperature, BVP, 3-axis Acceleration
- **Sampling Rates**: RespiBAN (700Hz), E4 (BVP: 64Hz, EDA: 4Hz, Temp: 4Hz, ACC: 32Hz)
- **Classes**: Baseline, Amusement, Stress (TSST)
- **E4 Only Sensors**: BVP, EDA, Temperature, ACC
- **Source**: GitHub (Schmidt et al., 2018)

### Dataset 2: EPM-E4 (EmoPulse Moments E4)
- **Subjects**: 53
- **Device**: Empatica E4 only
- **Sensors**: HR, EDA, Temperature, 3D Accelerometer, BVP
- **Classes**: Baseline, Anger, Sadness, Happiness, Fear
- **Clip Durations**: 43-395 seconds
- **Key Moments**: Specific timestamps of maximum emotional intensity
- **Source**: University of Granada (Ethics: 2100/CEIH/2021)

### Dataset 3: PhysioNet Wearable (Hongn et al., 2025)
- **Subjects**: 36 (stress), 30 (aerobic), 31 (anaerobic)
- **Device**: Empatica E4 only
- **Sensors**: BVP (64Hz), EDA (4Hz), Temperature (4Hz), 3-axis ACC (32Hz), HR, IBI
- **Classes**: Stress, Aerobic Exercise, Anaerobic Exercise
- **Stress Protocol**: Math tasks, opinion tasks, subtraction (self-report 1-10)
- **Aerobic**: Storer-Davis adaptation, 35min cycling, 60-110 rpm
- **Anaerobic**: Wingate adaptation, 3-4 max sprints (30-45s)
- **Versions**: V1 (18 subjects - Sxx), V2 (18 subjects - fxx)
- **Source**: PhysioNet (DOI: 10.13026/he0v-tf17)

### Common E4 Sensors Across All Datasets
- BVP (Blood Volume Pulse)
- EDA (Electrodermal Activity)
- Temperature (Skin Temperature)
- ACC (3-axis Accelerometer)
- HR (Heart Rate - derived from BVP)
- IBI (Inter-Beat Interval - derived from BVP)

### Final Class Labels (6-7 classes)
1. Baseline
2. Stress
3. Amusement
4. Anger
5. Sadness
6. Happiness
7. Fear
8. Aerobic Exercise
9. Anaerobic Exercise

---

## STEP 1: DATASET INSPECTION

### 1.1 Structure Analysis
**WESAD:**
- File format: pickle files (.pkl)
- Structure: subject folders containing signal data + labels
- E4 signals to extract: BVP, EDA, TEMP, ACC (ignore chest RespiBAN)
- Label mapping: baseline=1, amusement=2, stress=3

**EPM-E4:**
- File format: CSV files per sensor
- Structure: subject folders with individual CSV per signal type
- Signals: BVP.csv, EDA.csv, TEMP.csv, ACC.csv, HR.csv
- Key moments: timestamps in documentation
- Labels: Baseline, Anger, Sadness, Happiness, Fear

**PhysioNet:**
- File format: CSV files per sensor
- Structure: 3 main folders (STRESS, AEROBIC, ANAEROBIC) / subject subfolders
- Files: BVP.csv, EDA.csv, TEMP.csv, ACC.csv, IBI.csv, HR.csv, tags.csv
- CSV format: Row 1 = UTC timestamp, Row 2 = sampling rate (Hz), Row 3+ = data
- Tags: event markers for protocol segmentation
- Metadata: subject-info.csv, Stress_level_v1.csv, Stress_level_v2.csv
- Constraints: data_constraints.txt (issues log)

### 1.2 Data Location Mapping
- Create directory tree for each dataset
- Map all subject IDs
- Identify missing/corrupted files from constraints
- Document sampling rates per signal per dataset

### 1.3 Output
- Dataset structure report (subjects, files, signals available)
- Missing data log per subject
- Sampling rate table

---

## STEP 2: DATA PREPROCESSING

### 2.1 Subject Profile Inspection
**Per Dataset:**
- Load all subject demographic data
- Extract: age, gender, conditions (if available)
- Identify subjects with complete E4 data only
- Flag subjects with data issues (from constraints files)
- Create subject inclusion/exclusion criteria

**Output:**
- subject_profiles.csv (all datasets combined)
- inclusion_criteria.txt
- excluded_subjects.txt with reasons

### 2.2 Signal Quality Check
- Check for NaN, infinite values
- Check for flat signals (stuck sensors)
- Check for extreme outliers (sensor malfunction)
- Verify sampling rates match documentation
- Segment data by tags/labels

### 2.3 Feature Extraction

**Window Parameters:**
- Window size: TBD (test 30s, 60s, 120s)
- Overlap: 50%
- Only use valid labeled segments

**Features per Signal:**

**BVP/HR Features:**
- Mean HR
- Std HR
- Min/Max HR
- HR variability (RMSSD, SDNN)
- pNN50
- LF/HF ratio (frequency domain)

**EDA Features:**
- Mean EDA
- Std EDA
- Number of SCR peaks
- Mean SCR amplitude
- SCR rise time
- Tonic component (cvxEDA if available)
- Phasic component

**Temperature Features:**
- Mean temperature
- Std temperature
- Temperature slope (rate of change)
- Min/Max temperature

**ACC Features:**
- Mean magnitude
- Std magnitude
- Mean per axis (x, y, z)
- Signal Magnitude Area (SMA)
- Energy per axis
- Entropy

**Derived Features:**
- Movement intensity
- Activity classification (stationary, low, moderate, high)

### 2.4 Feature Importance (Random Forest)
- Train RF classifier on all extracted features
- Get feature importance scores
- Filter features available in smartwatch deployment:
  - Keep: BVP-derived, EDA, TEMP, ACC
  - Remove: Any chest-only sensors
- Threshold: Keep features with importance > X percentile

**Output:**
- feature_importance_scores.csv
- selected_features.txt
- RF model performance metrics

### 2.5 Windowed Feature Tables
**Process:**
- Apply sliding window to each subject's signals
- Extract features per window
- Assign label to each window
- Handle transition windows (discard or label separately)

**Output per Dataset:**
- WESAD_windowed_features.csv
- EPM4_windowed_features.csv
- PhysioNet_windowed_features.csv

**Columns:**
- subject_id
- dataset_source
- window_id
- timestamp_start
- timestamp_end
- [all selected features]
- label (class name)

---

## STEP 3: COMBINATION AND FILLING GAPS

### 3.1 Feature Normalization
**Identify Duplicate Features:**
- Map features across datasets with different names
- Example: "heart_rate" vs "HR" vs "mean_hr"
- Create unified feature naming convention

**Naming Convention:**
- hr_mean, hr_std, hr_min, hr_max
- eda_mean, eda_std, eda_scr_peaks, eda_scr_amp
- temp_mean, temp_std, temp_slope
- acc_magnitude_mean, acc_magnitude_std, acc_sma, acc_energy

**Output:**
- feature_mapping_table.csv (old_name, new_name, dataset)
- Rename all features in the 3 tables

### 3.2 Label Standardization
**Unified Labels:**
1. Baseline
2. Stress
3. Amusement
4. Anger
5. Sadness
6. Happiness
7. Fear
8. Aerobic
9. Anaerobic

**Mapping:**
- WESAD: baseline→Baseline, stress→Stress, amusement→Amusement
- EPM-E4: map directly
- PhysioNet: map directly

### 3.3 Dataset Combination
- Concatenate 3 windowed feature tables
- Add dataset_source column
- Verify feature alignment
- Check for label distribution imbalance

**Output:**
- combined_dataset.csv (all subjects, all features, all windows)
- label_distribution.csv (count per class)

### 3.4 Missing Value Imputation (KNN)
**Process:**
- Identify features with missing values
- Apply KNN imputation (k=5, test 3-7)
- Use all available features for similarity
- Stratify by label (optional: test both approaches)

**Validation:**
- Create synthetic missing values in complete data
- Apply KNN imputation
- Calculate imputation accuracy (MAE, RMSE)
- Report per feature

**Output:**
- imputation_accuracy_report.csv
- combined_dataset_filled.csv
- missing_value_analysis.txt (before/after counts)

---

## STEP 4: TRAINING AND VALIDATION

### 4.1 Data Preparation
- Split features (X) and labels (y)
- Normalize/standardize features (MinMaxScaler or StandardScaler)
- Check class balance (if imbalanced, apply SMOTE or class weights)

### 4.2 Model Training (Multiple Algorithms)

**Models to Train:**
1. Random Forest (RF)
2. XGBoost
3. Support Vector Machine (SVM)
4. Gradient Boosting
5. Multi-Layer Perceptron (MLP)
6. K-Nearest Neighbors (KNN)
7. Logistic Regression (baseline)
8. Decision Tree (baseline)

**Training Strategy:**
- Standard k-fold cross-validation (k=5 or 10)
- Stratified splits to maintain class distribution
- Hyperparameter tuning (GridSearchCV or RandomizedSearchCV)

**Metrics:**
- Accuracy
- Precision, Recall, F1-score (per class and weighted)
- Confusion matrix
- ROC-AUC (if applicable)
- Training time

**Output:**
- model_comparison_results.csv
- confusion_matrices/ (folder with plots)
- best_hyperparameters.json
- trained_models/ (saved models)

### 4.3 Feature Analysis Post-Training
- Get feature importance from tree-based models
- Get coefficients from linear models
- Compare with RF feature selection from Step 2.4
- Identify most discriminative features per class

### 4.4 Leave-One-Subject-Out (LOSO) Validation
**Process:**
- For each subject: train on all others, test on held-out subject
- Repeat for all subjects
- Test all trained models
- This simulates real-world deployment (new user)

**Metrics:**
- Per-subject accuracy
- Overall LOSO accuracy
- Per-class performance
- Subject-specific performance variance

**Output:**
- loso_results.csv (subject_id, model, accuracy, precision, recall, f1)
- loso_confusion_matrices/
- loso_performance_summary.txt
- worst_performing_subjects.txt (for analysis)

---

## STEP 5: ANOMALY DETECTION

### 5.1 Outlier Identification
**Methods:**
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Autoencoder reconstruction error

**Process:**
- Apply to combined dataset
- Flag windows that don't fit any trained class well
- Compare prediction confidence scores
- Identify physiological signal ranges outside normal bounds per class

**Signal Range Analysis:**
- Define normal ranges per class for each signal:
  - HR: min, max, mean, std
  - EDA: min, max, mean, std
  - Temperature: min, max, mean, std
  - ACC magnitude: min, max, mean, std

**Output:**
- normal_ranges_per_class.csv
- outlier_windows.csv (windows flagged as anomalies)
- outlier_detection_comparison.csv (performance of different methods)

### 5.2 Anomaly Classification (0-4 Severity Levels)

**Level Definitions:**
- **Level 0**: Very low signal activity
  - Low temperature, minimal movement, low HR, low EDA
- **Level 1**: Mild deviation
  - Slightly outside normal ranges
- **Level 2**: Moderate anomaly
  - Clear deviation in 2+ signals
- **Level 3**: High anomaly
  - Significant deviation in multiple signals
- **Level 4**: Severe/Critical anomaly
  - Extreme values, aggressive patterns, multiple signals critically out of range

**Classification Criteria:**
- Distance from normal range (z-score based)
- Number of signals deviating
- Magnitude of deviation
- Pattern consistency

**Process:**
- Calculate deviation scores per signal per window
- Aggregate into overall anomaly score
- Map score to 0-4 levels
- Assign level to each anomalous window

**Output:**
- anomaly_levels.csv (window_id, level, signal_values, deviations)
- anomaly_level_distribution.csv
- anomaly_signal_ranges.csv (actual values per level)
- anomaly_examples.csv (representative samples per level for literature comparison)

### 5.3 Validation with Literature
- Compare detected anomaly ranges with published studies
- Document physiological plausibility
- Identify potential false positives

---

## STEP 6: DOCUMENTATION

### 6.1 Thesis Document (thesis.md)
**Structure:**
1. Introduction
2. Dataset Description
3. Preprocessing Methods
4. Feature Extraction Results
5. Feature Importance Analysis
6. Model Training Results
7. LOSO Validation Results
8. Anomaly Detection Methodology
9. Anomaly Classification Results
10. Discussion
11. Conclusions

**Update After Each Step:**
- Add results as they are generated
- Include key findings
- Reference figures and tables

### 6.2 Visualizations

**Required Plots:**

**Step 1-2:**
- Dataset size comparison (bar chart)
- Signal sampling rates comparison
- Missing data heatmap per subject

**Step 2:**
- Feature importance bar plot (top 20)
- Feature correlation heatmap
- Class distribution pie/bar chart

**Step 3:**
- Missing value patterns (before/after)
- KNN imputation accuracy per feature
- Feature distribution before/after normalization

**Step 4:**
- Model comparison bar chart (accuracy, F1)
- Confusion matrices (all models)
- ROC curves (if applicable)
- Feature importance comparison across models
- LOSO performance box plots
- Per-subject accuracy heatmap

**Step 5:**
- Outlier detection scatter plots (PCA/t-SNE)
- Anomaly level distribution
- Signal ranges per anomaly level (box plots)
- Anomaly severity heatmap

**All Plots:**
- High resolution (300 DPI minimum)
- Clear labels, legends, titles
- Consistent color scheme
- Save as PNG and PDF

**Output Folders:**
- figures/
- tables/
- results/

### 6.3 Tables
- All CSV outputs mentioned above
- LaTeX formatted tables for key results
- Summary statistics tables

---

## STEP 7: MODEL OPTIMIZATION (ADDED)

### 7.1 Ensemble Methods
- Voting classifier (best 3-5 models)
- Stacking ensemble
- Compare with individual models

### 7.2 Deep Learning Approaches
- 1D CNN on raw signals (if needed)
- LSTM for temporal dependencies
- Hybrid CNN-LSTM
- Compare with traditional ML

### 7.3 Model Compression
- Quantization for deployment
- Pruning
- Knowledge distillation
- Size vs accuracy tradeoff analysis

---

## STEP 8: INTERPRETABILITY (ADDED)

### 8.1 SHAP Values
- Calculate SHAP values for best model
- Feature contribution analysis per class
- Individual prediction explanations

### 8.2 Attention Mechanisms
- If using DL: visualize attention weights
- Identify which signal segments are most important

---

## STEP 9: DEPLOYMENT READINESS (ADDED)

### 9.1 Smartwatch Constraints
- Feature computation time analysis
- Memory requirements
- Battery impact estimation
- Model inference time

### 9.2 Real-Time Feasibility
- Latency analysis
- Streaming window implementation
- Edge computing requirements

---

## STEP 10: FINAL VALIDATION (ADDED)

### 10.1 Holdout Test Set
- Set aside 10-15% of data at the start (untouched)
- Final evaluation on best model
- Compare with LOSO results

### 10.2 Cross-Dataset Validation
- Train on 2 datasets, test on 3rd
- Assess generalization across studies
- Identify dataset-specific biases

### 10.3 Clinical Validation Plan
- Protocol for real-world testing
- Success metrics definition
- Safety considerations

---

## NOTEBOOK ORGANIZATION

```
notebooks/
├── 01_dataset_inspection.ipynb
├── 02_subject_profiles.ipynb
├── 03_signal_quality_check.ipynb
├── 04_feature_extraction.ipynb
├── 05_feature_importance.ipynb
├── 06_feature_normalization.ipynb
├── 07_dataset_combination.ipynb
├── 08_missing_value_imputation.ipynb
├── 09_model_training.ipynb
├── 10_loso_validation.ipynb
├── 11_anomaly_detection.ipynb
├── 12_anomaly_classification.ipynb
├── 13_model_optimization.ipynb
├── 14_interpretability.ipynb
└── 15_final_validation.ipynb

data/
├── raw/
│   ├── WESAD/
│   ├── EPM-E4/
│   └── PhysioNet/
├── processed/
│   ├── windowed_features/
│   ├── combined/
│   └── imputed/
└── results/

models/
├── trained/
├── optimized/
└── final/

outputs/
├── figures/
├── tables/
├── thesis.md
└── results/
```

---

## KEY DELIVERABLES

1. Combined dataset with 6-9 classes
2. Trained ML/DL models
3. LOSO validation results
4. Anomaly detection system (0-4 levels)
5. Anomaly signal range tables
6. Complete thesis.md document
7. All visualizations and tables
8. Deployable model for smartwatch

---

## CRITICAL NOTES

- Only use Empatica E4 sensors (wrist-worn, smartwatch compatible)
- Exclude all chest-worn sensors (RespiBAN from WESAD)
- Focus on non-invasive, deployable features
- Maintain subject-level separation for LOSO
- Document all data exclusions and reasons
- Keep track of computational requirements for deployment