# Smartwatch Stress Detection

A comprehensive machine learning pipeline for real-time stress detection using smartwatch sensor data (heart rate, HRV, skin conductance, skin temperature).

## Features

- **Clinical Stress Index (CSI)**: Evidence-based stress scoring system
- **ML Pipeline**: Optimized random forest classifier for stress detection
- **Smartwatch Support**: Works with Empatica E4 and compatible wearables
- **Validation**: Leave-One-Subject-Out (LOSO) cross-validation
- **TensorFlow Lite**: Model optimization for edge deployment

## Project Structure

```
├── src/
│   └── core/              # Core stress detection modules
│       ├── clinical_stress_index.py
│       ├── csi_validation.py
│       ├── smartwatch_ml_pipeline.py
│       └── production_rf_classifier.py
├── scripts/               # Analysis and visualization utilities
├── models/                # Pre-trained models
├── data/                  # Data storage
├── tests/                 # Unit tests
├── config/                # Configuration files
└── docs/                  # Documentation
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main pipeline:
   ```bash
   python main.py
   ```

3. Generate validation reports:
   ```bash
   python scripts/generate_thesis_figures.py
   ```

## Requirements

- Python 3.8+
- NumPy, Pandas, SciPy
- Scikit-learn
- TensorFlow/Keras

See `requirements.txt` for complete dependencies.

## Documentation

- [Analysis Package](docs/README.md)
- [Configuration](config/config.yaml)

## License

This project is part of a thesis on smartwatch-based stress detection.
