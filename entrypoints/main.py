#!/usr/bin/env python3
"""
SmartWatch Stress Detection - Main Execution Script
===================================================

This is the main entry point for the stress detection pipeline.
It orchestrates the complete workflow from data processing to model evaluation.

Usage:
    python main.py [--mode] [--config]

Modes:
    - process: Process raw WESAD data
    - train: Train machine learning models
    - evaluate: Evaluate model performance
    - all: Run complete pipeline (default)

Author: [Your Name]
Date: 2025
"""

import argparse
import sys
from pathlib import Path
import json
import time
import logging
from copy import deepcopy

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import our modules
from preprocessing.wesad_processor import WESADProcessor
from models.stress_classifier import StressClassifier
from utils.logging_utils import setup_logging


logger = logging.getLogger("smartwatch.main")

def setup_directories() -> None:
    """Create necessary directories if they don't exist."""

    directories = [
        "data/raw",
        "data/processed",
        "data/wesad",
        "results",
        "results/figures",
        "results/models",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logger.debug("Verified directory structure: %s", directories)

def process_data(config):
    """Process raw WESAD data"""

    logger.info("Step 1: Data processing started")

    processor = WESADProcessor(
        data_path=config.get('data_path', 'data/wesad'),
        output_path=config.get('output_path', 'data/processed')
    )

    windows = processor.process_all_subjects(
        window_size_sec=config.get('window_size_sec', 180),
        overlap_sec=config.get('overlap_sec', 90)
    )

    if len(windows) > 0:
        logger.info("Data processing completed: %d windows extracted", len(windows))
        return True
    else:
        logger.error("Data processing failed: no windows extracted")
        return False

def train_models(config):
    """Train and evaluate machine learning models"""

    logger.info("Step 2: Model training and evaluation started")

    features_file = Path(config.get('features_file', 'data/processed/wesad_features.json'))

    if not features_file.exists():
        logger.error("Features file not found: %s", features_file)
        logger.error("Please run data processing before training")
        return False

    classifier = StressClassifier(
        data_path=str(features_file),
        results_path=config.get('results_path', 'results')
    )

    results = classifier.run_complete_analysis()

    if results:
        logger.info("Model training completed successfully")
        return True
    else:
        logger.error("Model training failed")
        return False

def run_complete_pipeline(config):
    """Run the complete stress detection pipeline"""

    logger.info("SmartWatch stress detection pipeline started")

    start_time = time.time()

    if not process_data(config):
        logger.error("Pipeline aborted during data processing")
        return False

    if not train_models(config):
        logger.error("Pipeline aborted during model training")
        return False

    end_time = time.time()
    duration = end_time - start_time

    results_path = config.get('results_path', 'results')
    logger.info(
        "Pipeline completed successfully in %.1f minutes",
        duration / 60.0,
    )
    logger.info("Results saved in: %s", results_path)
    logger.info("Detailed metrics available in results directory")

    return True

def load_config(config_file: str = "config/config.yaml") -> dict:
    """Load configuration from file or use defaults."""

    config_path = Path(config_file)

    default_config = {
        'data_path': 'data/wesad',
        'output_path': 'data/processed',
        'results_path': 'results',
        'features_file': 'data/processed/wesad_features.json',
        'window_size_sec': 180,
        'overlap_sec': 90,
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'class_weight': 'balanced',
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
            'console': {
                'enabled': True,
                'level': 'INFO',
            },
            'file': {
                'enabled': True,
                'path': 'logs/pipeline.log',
                'max_bytes': 1048576,
                'backup_count': 5,
                'level': 'INFO',
            },
        },
    }

    merged_config = deepcopy(default_config)

    def _deep_merge(base: dict, updates: dict) -> dict:
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                _deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    if config_path.exists():
        try:
            import yaml

            with open(config_path, 'r', encoding='utf-8') as handle:
                file_config = yaml.safe_load(handle) or {}

            if not isinstance(file_config, dict):
                logger.warning(
                    "Configuration file %s did not contain a mapping; using defaults",
                    config_path,
                )
                file_config = {}

            merged_config = _deep_merge(merged_config, file_config)
            logger.info("Configuration loaded from %s", config_path)

        except ImportError:
            logger.warning("PyYAML not installed; using default configuration")
        except Exception as exc:
            logger.error("Error loading config file %s: %s", config_path, exc)
            logger.info("Reverting to default configuration")
    else:
        logger.info("Configuration file not found at %s; using defaults", config_path)

    return merged_config

def main():
    """Main execution function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="SmartWatch Stress Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run complete pipeline
  python main.py --mode process     # Only process data
  python main.py --mode train       # Only train models
  python main.py --config custom.yaml  # Use custom config
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['process', 'train', 'evaluate', 'all'],
        default='all',
        help='Pipeline mode to run (default: all)'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    override_level = 'DEBUG' if args.verbose else None
    setup_logging(config.get('logging', {}), override_level=override_level)

    logger.debug("CLI arguments: %s", args)

    setup_directories()

    logger.info("Selected pipeline mode: %s", args.mode)
    
    # Run based on mode
    success = False
    
    if args.mode == 'process':
        success = process_data(config)
    
    elif args.mode == 'train' or args.mode == 'evaluate':
        success = train_models(config)
    
    elif args.mode == 'all':
        success = run_complete_pipeline(config)
    
    # Exit with appropriate code
    if success:
        logger.info("Operation completed successfully")
        sys.exit(0)
    else:
        logger.error("Operation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()