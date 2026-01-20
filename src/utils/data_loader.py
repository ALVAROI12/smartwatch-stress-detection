"""
WESAD Dataset Loader Module

This module provides functions for loading and parsing the WESAD (Wearable Stress and Affect Detection) 
dataset. It handles the extraction of physiological signals and labels from the pickled data files.

Author: Research Team
Date: 2024
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WESAD dataset constants
WESAD_SAMPLING_RATES = {
    'chest': {
        'ACC': 700,
        'ECG': 700,
        'EDA': 700,
        'EMG': 700,
        'Resp': 700,
        'Temp': 700
    },
    'wrist': {
        'ACC': 32,
        'BVP': 64,  # PPG signal
        'EDA': 4,
        'TEMP': 4
    }
}

WESAD_LABEL_MAPPING = {
    0: 'Not defined',
    1: 'Baseline',
    2: 'Stress',
    3: 'Amusement',
    4: 'Meditation',
    5: 'Other',
    6: 'Other',
    7: 'Other'
}

# Labels of interest for stress detection
STRESS_LABELS = {
    'baseline': 1,
    'stress': 2,
    'amusement': 3
}


class WESADDataLoader:
    """
    A class for loading and preprocessing WESAD dataset files.
    
    This class provides methods to load individual subject data, extract specific
    sensor modalities, and prepare data for machine learning analysis.
    """
    
    def __init__(self, dataset_path: Union[str, Path]):
        """
        Initialize the WESAD data loader.
        
        Parameters
        ----------
        dataset_path : str or Path
            Path to the WESAD dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.subject_dirs = self._discover_subjects()
        
        logger.info(f"Initialized WESAD loader with {len(self.subject_dirs)} subjects")
    
    def _discover_subjects(self) -> List[Path]:
        """
        Discover all subject directories in the dataset.
        
        Returns
        -------
        List[Path]
            List of paths to subject directories
        """
        if not self.dataset_path.exists():
            logger.error(f"Dataset path {self.dataset_path} does not exist")
            return []
        
        subject_dirs = [
            path for path in self.dataset_path.iterdir() 
            if path.is_dir() and path.name.startswith('S')
        ]
        
        return sorted(subject_dirs)
    
    def load_subject(self, subject_id: str) -> Optional[Dict]:
        """
        Load data for a specific subject.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier (e.g., 'S2', 'S3')
            
        Returns
        -------
        Dict or None
            Loaded subject data or None if loading fails
        """
        subject_path = self.dataset_path / subject_id
        
        if not subject_path.exists():
            logger.error(f"Subject directory {subject_path} not found")
            return None
        
        # Find the primary data file
        pickle_files = list(subject_path.glob('*.pkl'))
        
        if not pickle_files:
            logger.error(f"No pickle files found in {subject_path}")
            return None
        
        # Select the appropriate data file
        primary_file = self._select_primary_file(pickle_files, subject_id)
        
        try:
            logger.info(f"Loading data for {subject_id} from {primary_file.name}")
            with open(primary_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            logger.info(f"Successfully loaded {subject_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load {primary_file}: {str(e)}")
            return None
    
    def _select_primary_file(self, pickle_files: List[Path], subject_id: str) -> Path:
        """
        Select the primary data file from available pickle files.
        
        Parameters
        ----------
        pickle_files : List[Path]
            List of available pickle files
        subject_id : str
            Subject identifier
            
        Returns
        -------
        Path
            Path to the primary data file
        """
        # Look for file matching subject ID pattern
        for pkl_file in pickle_files:
            if subject_id.lower() in pkl_file.stem.lower():
                return pkl_file
        
        # If no match, return the largest file
        return max(pickle_files, key=lambda x: x.stat().st_size)
    
    def extract_signals(self, subject_data: Dict, modality: str = 'wrist') -> Dict[str, np.ndarray]:
        """
        Extract physiological signals from loaded subject data.
        
        Parameters
        ----------
        subject_data : Dict
            Loaded subject data dictionary
        modality : str, default 'wrist'
            Sensor modality ('wrist' or 'chest')
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing extracted signals
        """
        signals = {}
        
        if 'signal' not in subject_data:
            logger.warning("No signal data found in subject data")
            return signals
        
        signal_data = subject_data['signal']
        
        if modality not in signal_data:
            logger.warning(f"Modality '{modality}' not found in signal data")
            return signals
        
        modality_data = signal_data[modality]
        
        # Extract each signal type
        for signal_name, signal_values in modality_data.items():
            if hasattr(signal_values, 'flatten'):
                signals[signal_name] = signal_values.flatten()
            else:
                signals[signal_name] = np.array(signal_values)
        
        logger.info(f"Extracted {len(signals)} signals from {modality} modality")
        return signals
    
    def extract_labels(self, subject_data: Dict) -> np.ndarray:
        """
        Extract experimental labels from subject data.
        
        Parameters
        ----------
        subject_data : Dict
            Loaded subject data dictionary
            
        Returns
        -------
        np.ndarray
            Array of experimental labels
        """
        if 'label' not in subject_data:
            logger.warning("No label data found in subject data")
            return np.array([])
        
        labels = subject_data['label']
        
        if hasattr(labels, 'flatten'):
            return labels.flatten()
        else:
            return np.array(labels)
    
    def get_signal_info(self, subject_data: Dict) -> Dict:
        """
        Get information about available signals and their properties.
        
        Parameters
        ----------
        subject_data : Dict
            Loaded subject data dictionary
            
        Returns
        -------
        Dict
            Information about available signals
        """
        signal_info = {}
        
        if 'signal' not in subject_data:
            return signal_info
        
        for modality, modality_data in subject_data['signal'].items():
            signal_info[modality] = {}
            
            for signal_name, signal_values in modality_data.items():
                if hasattr(signal_values, 'shape'):
                    signal_info[modality][signal_name] = {
                        'shape': signal_values.shape,
                        'dtype': str(signal_values.dtype),
                        'length': len(signal_values),
                        'sampling_rate': WESAD_SAMPLING_RATES.get(modality, {}).get(signal_name, 'Unknown')
                    }
        
        return signal_info
    
    def load_all_subjects(self, modality: str = 'wrist') -> Dict[str, Dict]:
        """
        Load data for all available subjects.
        
        Parameters
        ----------
        modality : str, default 'wrist'
            Sensor modality to extract
            
        Returns
        -------
        Dict[str, Dict]
            Dictionary containing data for all subjects
        """
        all_data = {}
        
        for subject_dir in self.subject_dirs:
            subject_id = subject_dir.name
            
            subject_data = self.load_subject(subject_id)
            if subject_data is not None:
                all_data[subject_id] = {
                    'signals': self.extract_signals(subject_data, modality),
                    'labels': self.extract_labels(subject_data),
                    'raw_data': subject_data
                }
        
        logger.info(f"Loaded data for {len(all_data)} subjects")
        return all_data
    
    def get_stress_detection_data(self, subject_data: Dict, target_labels: List[int] = None) -> Tuple[Dict, np.ndarray]:
        """
        Prepare data specifically for stress detection tasks.
        
        Parameters
        ----------
        subject_data : Dict
            Loaded subject data
        target_labels : List[int], optional
            List of target labels to include (default: [1, 2] for baseline and stress)
            
        Returns
        -------
        Tuple[Dict, np.ndarray]
            Filtered signals and corresponding labels
        """
        if target_labels is None:
            target_labels = [STRESS_LABELS['baseline'], STRESS_LABELS['stress']]
        
        # Extract signals and labels
        signals = self.extract_signals(subject_data, 'wrist')
        labels = self.extract_labels(subject_data)
        
        if len(signals) == 0 or len(labels) == 0:
            logger.warning("No signals or labels found")
            return {}, np.array([])
        
        # Filter data to include only target labels
        mask = np.isin(labels, target_labels)
        
        filtered_signals = {}
        for signal_name, signal_values in signals.items():
            if len(signal_values) == len(labels):
                filtered_signals[signal_name] = signal_values[mask]
            else:
                logger.warning(f"Signal {signal_name} length mismatch with labels")
        
        filtered_labels = labels[mask]
        
        logger.info(f"Filtered data: {len(filtered_labels)} samples with target labels")
        return filtered_signals, filtered_labels


def validate_wesad_structure(dataset_path: Union[str, Path]) -> bool:
    """
    Validate the structure of a WESAD dataset directory.
    
    Parameters
    ----------
    dataset_path : str or Path
        Path to the dataset directory
        
    Returns
    -------
    bool
        True if structure is valid, False otherwise
    """
    path = Path(dataset_path)
    
    if not path.exists():
        logger.error(f"Dataset path {path} does not exist")
        return False
    
    # Check for subject directories
    subject_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith('S')]
    
    if len(subject_dirs) == 0:
        logger.error("No subject directories found")
        return False
    
    # Validate a sample subject directory
    sample_subject = subject_dirs[0]
    pickle_files = list(sample_subject.glob('*.pkl'))
    
    if len(pickle_files) == 0:
        logger.error(f"No pickle files found in {sample_subject}")
        return False
    
    logger.info(f"Dataset structure validation passed: {len(subject_dirs)} subjects found")
    return True


def get_dataset_summary(dataset_path: Union[str, Path]) -> Dict:
    """
    Generate a summary of the WESAD dataset.
    
    Parameters
    ----------
    dataset_path : str or Path
        Path to the dataset directory
        
    Returns
    -------
    Dict
        Dataset summary information
    """
    loader = WESADDataLoader(dataset_path)
    
    summary = {
        'total_subjects': len(loader.subject_dirs),
        'subject_ids': [d.name for d in loader.subject_dirs],
        'dataset_path': str(loader.dataset_path),
        'sampling_rates': WESAD_SAMPLING_RATES,
        'label_mapping': WESAD_LABEL_MAPPING
    }
    
    # Try to get signal information from first subject
    if loader.subject_dirs:
        first_subject = loader.subject_dirs[0].name
        subject_data = loader.load_subject(first_subject)
        
        if subject_data:
            summary['signal_info'] = loader.get_signal_info(subject_data)
            summary['available_modalities'] = list(subject_data.get('signal', {}).keys())
    
    return summary


if __name__ == "__main__":
    # Example usage and testing
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    
    # Test data loading
    dataset_path = project_root / "data" / "wesad"
    
    if validate_wesad_structure(dataset_path):
        loader = WESADDataLoader(dataset_path)
        summary = get_dataset_summary(dataset_path)
        
        print("Dataset Summary:")
        print(f"Total subjects: {summary['total_subjects']}")
        print(f"Subject IDs: {summary['subject_ids']}")
        
        # Test loading first subject
        if loader.subject_dirs:
            first_subject = loader.subject_dirs[0].name
            data = loader.load_subject(first_subject)
            
            if data:
                signals = loader.extract_signals(data, 'wrist')
                labels = loader.extract_labels(data)
                
                print(f"\nFirst subject ({first_subject}) data:")
                print(f"Available signals: {list(signals.keys())}")
                print(f"Labels shape: {labels.shape}")
                print(f"Unique labels: {np.unique(labels)}")
    else:
        print("Dataset structure validation failed")