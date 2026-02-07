# Smartwatch Stress Detection Project

This project implements machine learning models for stress detection using smartwatch sensor data.

## Project Structure

```
smartwatch-stress-detection/
├── README.md                   # Project overview and setup instructions
├── config/                     # Configuration files
├── data/                       # All datasets organized by processing stage
│   ├── raw_datasets/           # Original, unprocessed data
│   │   ├── datasets/           # Source datasets (WESAD, EPM-E4, etc.)
│   │   └── raw/               # Raw sensor readings
│   ├── processed_datasets/     # Intermediate processed data
│   ├── final_datasets/         # Final, analysis-ready datasets
│   │   ├── combined/           # Combined datasets
│   │   └── unified/           # Unified feature datasets
│   └── experimental/           # Experimental data processing outputs
├── docs/                       # Project documentation
├── figures/                    # Project-level figures and plots
├── models/                     # Trained machine learning models
├── notebooks/                  # Jupyter notebooks organized by workflow
│   ├── 01_data_processing/     # Data loading and preprocessing
│   ├── 02_feature_engineering/ # Feature extraction and selection
│   ├── 03_anomaly_detection/   # Anomaly detection analysis
│   ├── 04_model_development/   # Model training and validation
│   ├── 05_evaluation/          # Model evaluation and testing
│   └── archive/               # Old or experimental notebooks
├── results/                    # All results organized by analysis type
│   ├── anomaly_detection/      # Anomaly detection results
│   ├── model_training/         # Training results and logs
│   ├── evaluations/            # Model evaluation results
│   ├── plots_figures/          # Generated plots and visualizations
│   ├── reports_summaries/      # Analysis reports and summaries
│   └── archived_results/       # Historical results
├── archive/                    # Archived files and outdated materials
└── temp/                      # Temporary files (ignored by git)
```

## Getting Started

1. **Data Processing**: Start with notebooks in `01_data_processing/`
2. **Feature Engineering**: Continue with `02_feature_engineering/`
3. **Anomaly Detection**: Run analysis in `03_anomaly_detection/`
4. **Model Development**: Train models using `04_model_development/`
5. **Evaluation**: Assess performance with `05_evaluation/`

## Data Flow

1. Raw sensor data → `data/raw_datasets/`
2. Preprocessed data → `data/processed_datasets/`
3. Final datasets → `data/final_datasets/`
4. Analysis results → `results/` (organized by type)

## Key Files

- **Configuration**: `config/config.yaml`
- **Main Dataset**: `data/final_datasets/unified/final_unified_dataset.csv`
- **Documentation**: `docs/README.md`, `docs/DEVELOPMENT_LOG.md`
- **Results**: Organized in `results/` by analysis type

## Project Status

- ✅ Data processing pipeline complete
- ✅ Anomaly detection analysis complete
- ✅ Model training pipeline established
- 🔄 Model evaluation in progress
- 📋 Final deployment preparation pending

## Dependencies

See individual notebook requirements or use the project configuration files.