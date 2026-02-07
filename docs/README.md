# SmartWatch Stress Detection

A machine learning system for detecting physiological stress using PPG (photoplethysmography) and accelerometer data from consumer smartwatches.

##  Project Overview

This project aims to develop a stress detection system compatible with commercial smartwatches (Apple Watch, Samsung Galaxy Watch, etc.) for early detection of stress-related mental health conditions.

### Key Features
- **PPG-based stress detection** using Heart Rate Variability (HRV)
- **Smartwatch compatibility** - works with sensors available in consumer devices
- **Real-time processing** capability for continuous monitoring
- **Cross-dataset validation** for robust performance
- **Mental health applications** for anxiety and stress prevention

##  Datasets

### WESAD (Wearable Stress and Affect Detection)
- **Participants**: 15 subjects
- **Sensors**: PPG, EDA, Accelerometer, Temperature
- **Classes**: Baseline, Stress, Amusement
- **Environment**: Laboratory controlled

### EmpaticaE4Stress
- **Participants**: 29 subjects  
- **Sensors**: PPG, EDA, Accelerometer, Temperature
- **Classes**: Rest, Stress
- **Environment**: Simulated work environment

##  Project Structure

```
smartwatch-stress-detection/
 data/                          # Data storage
    raw/                       # Raw dataset files
    processed/                 # Cleaned and processed data
    wesad/                     # WESAD dataset
    empatica_e4_stress/       # EmpaticaE4Stress dataset
 src/                          # Source code
    preprocessing/            # Signal processing
    features/                 # Feature extraction
    models/                   # ML models
    utils/                    # Utilities
    visualization/            # Plotting functions
 notebooks/                    # Jupyter notebooks
 config/                       # Configuration files
 results/                      # Model outputs and reports
 tests/                        # Unit tests
 requirements.txt              # Python dependencies
```

##  Quick Start

### 1. Environment Setup

```
# Clone repository
git clone <repository-url>
cd smartwatch-stress-detection

# Create virtual environment
python3 -m venv stress_detection_env

# Activate virtual environment
# Windows:
stress_detection_env\Scripts\activate
# macOS/Linux:
source stress_detection_env/bin/activate

# Upgrade pip and install pinned dependencies via Makefile helper
pip install --upgrade pip
make install-deps

```

### 2. Data Preparation
```
# Download WESAD dataset
# Place in data/wesad/

# Download EmpaticaE4Stress dataset
# URL: https://data.mendeley.com/datasets/kb42z77m2g/2
# Place in data/empatica_e4_stress/
```

### 3. Run Analysis

```
# Start with data exploration
jupyter notebook notebooks/01_data_exploration.ipynb

# Or run complete pipeline
python main.py

# Run lint and unit test suite before committing changes
make check
```

##  Developer Workflow

```
# Format source files in-place
make format

# Static formatting check without modifying files
make lint

# Execute automated tests
make test
```

Make targets default to the active Python interpreter, so activate the virtual environment first. All commands finish with a non-zero exit code if an issue is detected, which makes them CI-friendly.

##  Methodology

### Signal Processing
- **PPG Filtering**: Chebyshev II filter (0.5-5 Hz)
- **Artifact Detection**: 20% threshold with motion-based cleaning
- **RR Interval Extraction**: Peak detection with validation
- **Windowing**: 5-minute windows with 50% overlap

### Feature Engineering
**Heart Rate Variability (HRV) Features:**
- Time domain: SDNN, RMSSD, pNN50, Mean HR
- Frequency domain: LF, HF, LF/HF ratio
- Geometric: HRV triangular index, TINN

**Accelerometer Features:**
- Motion magnitude and variability
- Energy spectrum analysis
- Activity context classification

### Machine Learning
- **Primary**: Random Forest (100 estimators)
- **Secondary**: XGBoost with hyperparameter optimization
- **Validation**: Leave-One-Subject-Out cross-validation
- **Metrics**: Accuracy, Precision, Recall, F1-score

##  Expected Performance

| Configuration | Accuracy | Use Case |
|--------------|----------|----------|
| PPG + Accelerometer | 85-88% | Universal smartwatch compatibility |
| PPG + EDA + Accelerometer | 90-95% | Research-grade devices |
| Person-specific models | 95-98% | Individual calibration |

##  Hardware Compatibility

###  Fully Compatible
- Apple Watch (all series)
- Samsung Galaxy Watch series
- Fitbit Versa/Sense series
- Garmin smartwatches

###  Partially Compatible
- Basic fitness trackers (PPG only)
- Older smartwatch models

##  Configuration

Key settings in `config/config.yaml`:
- Sampling rates and filtering parameters
- Feature extraction settings
- Model hyperparameters
- Cross-validation strategy
- Logging level, console/file handlers, and log rotation

##  Testing

```
# Run unit tests
make test

# Run with coverage
pytest --cov=src tests/
```

##  Results

Results are saved in the `results/` directory:
- **Models**: Trained model files
- **Figures**: Performance plots and visualizations  
- **Reports**: Detailed analysis and metrics

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  References

1. Schmidt, P., et al. "Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection." ICMI 2018.
2. Campanella, S., et al. "PPG and EDA dataset collected with Empatica E4 for stress assessment." Data in Brief 2024.
3. Can, Y.S., et al. "Continuous stress detection using wearable sensors in real life." Sensors 2019.

##  Ethical Considerations

This system is intended for research and wellness applications. It is not a medical device and should not be used for clinical diagnosis. Always consult healthcare professionals for medical concerns.

---

**Contact**: [Your contact information]
**Last Updated**: 2025-11-21