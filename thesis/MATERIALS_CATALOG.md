# THESIS MATERIALS CATALOG

Generated: 2025-10-04 19:37:42

Project: Machine Learning Pipeline for Stress Detection

Author: Your Name

---

## SUMMARY

- **Total Figures**: 4
- **Total Tables**: 3
- **Organization Date**: 2025-10-04

---

## FIGURES

### 1. Chapter4_Fig5_OverallResults
- **Original File:** `wesad_ml_results.png`
- **Location:** `thesis/figures/Chapter4_Fig5_OverallResults.png`
- **Size:** 449.7 KB
- **Description:** Comprehensive results visualization including performance metrics and class distribution

### 2. Chapter5_Fig1_ClinicalValidation
- **Original File:** `quick_validation_plots.png`
- **Location:** `thesis/figures/Chapter5_Fig1_ClinicalValidation.png`
- **Size:** 335.1 KB
- **Description:** Clinical stress index validation showing perfect separation between baseline and stress

### 3. Chapter4_Fig7_CrossSubjectValidation
- **Original File:** `cross_subject_analysis.png`
- **Location:** `thesis/figures/Chapter4_Fig7_CrossSubjectValidation.png`
- **Size:** 396.7 KB
- **Description:** Leave-One-Subject-Out cross-validation performance across all subjects

### 4. Chapter4_Fig6_FeatureCorrelation
- **Original File:** `feature_correlation_analysis.png`
- **Location:** `thesis/figures/Chapter4_Fig6_FeatureCorrelation.png`
- **Size:** 685.6 KB
- **Description:** Correlation heatmap of top physiological features


## TABLES

### 1. Table 4.1: Random Forest Performance Metrics
- **Location:** `thesis/tables/Table4_1_RandomForestPerformance.csv`
- **Dimensions:** 5 rows × 3 columns
- **Description:** Leave-One-Subject-Out cross-validation results for Random Forest classifier

### 2. Table 4.2: Model Comparison
- **Location:** `thesis/tables/Table4_2_ModelComparison.csv`
- **Dimensions:** 2 rows × 6 columns
- **Description:** Comparative performance of Random Forest and XGBoost on WESAD dataset

### 3. Table 4.3: Top 10 Features by Importance
- **Location:** `thesis/tables/Table4_3_TopFeatures.csv`
- **Dimensions:** 10 rows × 2 columns
- **Description:** Most important physiological features for stress classification


## USAGE INSTRUCTIONS

### For LaTeX:
```latex
% Include figure
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{thesis/figures/Chapter4_Fig1_ConfusionMatrix.png}
\caption{Your caption here}
\label{fig:confusion_matrix}
\end{figure}
```

### For Markdown:
```markdown
![Figure Caption](thesis/figures/Chapter4_Fig1_ConfusionMatrix.png)
```
