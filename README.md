# Smartwatch Stress Detection

Machine learning-based stress detection using wearable sensor data from multiple datasets.

## 📊 Project Overview

This thesis project implements anomaly detection and classification models for stress monitoring using physiological signals from smartwatches and wearable devices.

### Key Results
- **Best Model**: Optuna-optimized XGBoost with **94.53% accuracy**
- **95% Confidence Interval**: [93.5%, 95.5%]
- **Datasets**: WESAD, EPM-E4, PhysioNet (10,511 samples, 96 subjects)
- **Features**: 39 physiological features from HR, HRV, EDA, temperature, and accelerometer

## 📁 Project Structure

```
├── notebooks/                    # Jupyter notebooks (Steps 1-14)
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_subject_profiles.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── ...
│   ├── 13_enhancements.ipynb     # Model improvements & optimizations
│   └── 14_advanced_analysis.ipynb # Statistical analysis & deployment
├── outputs/
│   ├── models/                   # Trained models (.pkl, .keras)
│   ├── figures/                  # Visualizations
│   └── tables/                   # Results tables
├── api.py                        # FastAPI deployment endpoint
├── wearable_datasets_project_guide.md
└── README.md
```

## 🔬 Methods & Techniques

### Data Processing
- Multi-dataset fusion (WESAD, EPM-E4, PhysioNet)
- 10-second sliding windows
- Feature engineering from physiological signals

### Models Implemented
- XGBoost (best performer)
- Random Forest
- Gradient Boosting
- Neural Networks (MLP, CNN, Transformer)
- Ensemble methods (Stacking, Voting)

### Advanced Analysis
- Bootstrap confidence intervals
- McNemar's statistical tests
- SHAP explainability
- Adversarial robustness testing
- Active learning simulation
- Conformal prediction (uncertainty quantification)

## 🚀 Quick Start

1. **Install dependencies**:
```bash
pip install pandas numpy scikit-learn xgboost tensorflow shap optuna
```

2. **Run notebooks in order** (01-14)

3. **Deploy API**:
```bash
pip install fastapi uvicorn
uvicorn api:app --reload --port 8000
```

## 📈 Key Findings

| Metric | Value |
|--------|-------|
| Test Accuracy | 94.53% |
| F1-Score | 94.53% |
| Inference Speed | 1,168 pred/sec |
| Latency | <1ms |

## 📄 License

Research use only.

## 👤 Author

Alvaro Ibarra - Thesis Project 2026
