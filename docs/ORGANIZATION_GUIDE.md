# Project Organization Guidelines

## Directory Structure Standards

### Data Organization
- **raw_datasets/**: Never modify these files
- **processed_datasets/**: Intermediate processing results
- **final_datasets/**: Analysis-ready data only
- **experimental/**: Temporary or experimental data processing

### Results Organization
- **anomaly_detection/**: All anomaly detection outputs
- **model_training/**: Training logs, metrics, model checkpoints
- **evaluations/**: Final model evaluation results
- **plots_figures/**: All visualizations and charts
- **reports_summaries/**: Written reports and analysis summaries
- **archived_results/**: Historical results for reference

### Notebook Workflow
1. **01_data_processing**: Load and preprocess raw data
2. **02_feature_engineering**: Extract and select features
3. **03_anomaly_detection**: Detect physiological anomalies
4. **04_model_development**: Train and tune models
5. **05_evaluation**: Evaluate model performance

## File Naming Conventions

### Notebooks
- Use descriptive names: `01_wesad_preprocessing.ipynb`
- Include sequence numbers for workflow order
- Use underscores, not spaces

### Data Files
- Include processing date when relevant
- Use descriptive names: `final_unified_dataset.csv`
- Separate metadata from data files

### Results Files
- Include timestamp for dated results
- Use consistent naming: `anomaly_detection_results_YYYYMMDD.csv`
- Organize by analysis type

## Best Practices

### Data Management
- Keep raw data immutable
- Document all processing steps
- Use relative paths in code
- Maintain data lineage documentation

### Code Organization
- One analysis per notebook
- Clear markdown documentation
- Reproducible results
- Error handling and validation

### Results Management
- Save intermediate results
- Include metadata with all outputs
- Use version control for code, not large data files
- Archive old results regularly

## Cleanup Guidelines

### Regular Maintenance
- Remove temporary files monthly
- Archive old results quarterly
- Clean up duplicate files
- Update documentation regularly

### File Types to Remove
- `.DS_Store` files (Mac system files)
- `*copia*` or `*copy*` duplicate files
- Temporary files (`*.tmp`, `*.cache`)
- Empty directories
- Obsolete notebooks or scripts