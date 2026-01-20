#!/usr/bin/env python3
"""
Smartwatch Stress Detection - LOSO Cross-Validation
==================================================

Proper Leave-One-Subject-Out (LOSO) cross-validation for realistic
accuracy assessment of stress detection across different subjects.

This addresses the critical methodological issue: training and testing
on the same subjects gives inflated accuracy. LOSO tests true 
generalization to unseen individuals.

Author: Smartwatch Stress Detection Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, filtfilt, butter
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class SmartwatchFeatures:
    """Simplified feature vector for smartwatch-compatible stress detection"""
    # PPG & Heart Rate Features (4 features)
    hr_mean: float = 0.0
    hr_std: float = 0.0
    hr_min: float = 0.0
    hr_max: float = 0.0
    
    # HRV Features (4 features)
    rmssd: float = 0.0
    pnn50: float = 0.0
    sdnn: float = 0.0
    lf_hf_ratio: float = 0.0
    
    # Accelerometer Features (8 features)
    acc_magnitude_mean: float = 0.0
    acc_magnitude_std: float = 0.0
    acc_x_energy: float = 0.0
    acc_y_energy: float = 0.0
    acc_z_energy: float = 0.0
    acc_activity_level: float = 0.0
    acc_dominant_frequency: float = 0.0
    acc_entropy: float = 0.0
    
    # Temperature Features (3 features)
    temp_mean: float = 0.0
    temp_std: float = 0.0
    temp_trend: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models (19 features total)"""
        return np.array([
            # PPG & HR (4)
            self.hr_mean, self.hr_std, self.hr_min, self.hr_max,
            # HRV (4) 
            self.rmssd, self.pnn50, self.sdnn, self.lf_hf_ratio,
            # Accelerometer (8)
            self.acc_magnitude_mean, self.acc_magnitude_std,
            self.acc_x_energy, self.acc_y_energy, self.acc_z_energy,
            self.acc_activity_level, self.acc_dominant_frequency, self.acc_entropy,
            # Temperature (3)
            self.temp_mean, self.temp_std, self.temp_trend
        ])
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get feature names for model interpretation"""
        return [
            # PPG & HR (4)
            'hr_mean', 'hr_std', 'hr_min', 'hr_max',
            # HRV (4)
            'rmssd', 'pnn50', 'sdnn', 'lf_hf_ratio',
            # Accelerometer (8) 
            'acc_magnitude_mean', 'acc_magnitude_std',
            'acc_x_energy', 'acc_y_energy', 'acc_z_energy',
            'acc_activity_level', 'acc_dominant_frequency', 'acc_entropy',
            # Temperature (3)
            'temp_mean', 'temp_std', 'temp_trend'
        ]


class SmartwatchFeatureExtractor:
    """Extract features compatible with consumer smartwatch sensors"""
    
    def __init__(self):
        # WESAD sampling rates
        self.bvp_fs = 64.0    # BVP sampling rate
        self.acc_fs = 32.0    # Accelerometer sampling rate 
        self.temp_fs = 4.0    # Temperature sampling rate
        
    def extract_ppg_heart_rate_features(self, bvp_signal: np.ndarray) -> Dict[str, float]:
        """Extract heart rate features from PPG/BVP signal"""
        try:
            if len(bvp_signal) < 64:  # Need at least 1 second of data
                return self._get_default_hr_features()
            
            # Basic filtering for PPG
            nyq = self.bvp_fs / 2
            low = 0.5 / nyq
            high = 8.0 / nyq
            b, a = butter(2, [low, high], btype='band')  # Reduced order
            
            if len(bvp_signal) < 2*2*3:  # padlen check for filtfilt
                filtered_bvp = bvp_signal  # Skip filtering for short signals
            else:
                filtered_bvp = filtfilt(b, a, bvp_signal)
            
            # Find peaks (R-peaks equivalent in PPG)
            height_threshold = np.std(filtered_bvp) * 0.3
            min_distance = int(0.5 * self.bvp_fs)  # Min 0.5s between peaks
            
            peaks, _ = find_peaks(filtered_bvp, 
                                height=height_threshold,
                                distance=min_distance)
            
            if len(peaks) < 2:
                return self._get_default_hr_features()
                
            # Calculate inter-beat intervals (IBI) in milliseconds
            ibi_samples = np.diff(peaks)
            ibi_ms = (ibi_samples / self.bvp_fs) * 1000
            
            # Filter physiologically plausible IBIs (400-2000ms = 30-150 BPM)
            valid_ibi = ibi_ms[(ibi_ms >= 400) & (ibi_ms <= 2000)]
            
            if len(valid_ibi) < 2:
                return self._get_default_hr_features()
            
            # Convert to heart rate (BPM)
            heart_rates = 60000 / valid_ibi  # 60000 ms/min
            
            return {
                'hr_mean': float(np.mean(heart_rates)),
                'hr_std': float(np.std(heart_rates)),
                'hr_min': float(np.min(heart_rates)),
                'hr_max': float(np.max(heart_rates)),
                'ibi_data': valid_ibi  # Keep for HRV calculation
            }
            
        except Exception as e:
            return self._get_default_hr_features()
    
    def extract_hrv_features(self, ibi_data: np.ndarray) -> Dict[str, float]:
        """Extract Heart Rate Variability features"""
        try:
            if len(ibi_data) < 3:
                return self._get_default_hrv_features()
            
            # Time domain features
            rmssd = float(np.sqrt(np.mean(np.diff(ibi_data)**2)))
            sdnn = float(np.std(ibi_data))
            
            # pNN50: percentage of successive differences > 50ms
            successive_diffs = np.abs(np.diff(ibi_data))
            pnn50 = float(np.sum(successive_diffs > 50) / len(successive_diffs) * 100)
            
            # Simple frequency domain approximation
            lf_hf_ratio = float(sdnn / rmssd if rmssd > 0 else 2.0)
            
            return {
                'rmssd': rmssd,
                'pnn50': pnn50,
                'sdnn': sdnn,
                'lf_hf_ratio': lf_hf_ratio
            }
            
        except Exception:
            return self._get_default_hrv_features()
    
    def extract_accelerometer_features(self, acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive accelerometer features"""
        try:
            if len(acc_x) == 0 or len(acc_y) == 0 or len(acc_z) == 0:
                return self._get_default_acc_features()
                
            # Calculate magnitude
            acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            
            # Remove gravity component
            acc_magnitude_no_gravity = acc_magnitude - np.median(acc_magnitude)
            
            # Basic statistics
            mag_mean = float(np.mean(acc_magnitude))
            mag_std = float(np.std(acc_magnitude))
            
            # Energy in each axis
            x_energy = float(np.sum(acc_x**2) / len(acc_x))
            y_energy = float(np.sum(acc_y**2) / len(acc_y))
            z_energy = float(np.sum(acc_z**2) / len(acc_z))
            
            # Activity level classification
            activity_level = self._classify_activity_level(acc_magnitude_no_gravity)
            
            # Frequency analysis (if enough data)
            if len(acc_magnitude_no_gravity) > 32:  # At least 1 second
                freqs, psd = signal.welch(acc_magnitude_no_gravity, fs=self.acc_fs, nperseg=min(32, len(acc_magnitude_no_gravity)))
                dominant_freq = float(freqs[np.argmax(psd)])
            else:
                dominant_freq = 0.0
            
            # Entropy (complexity measure) 
            entropy = self._calculate_entropy(acc_magnitude_no_gravity)
            
            return {
                'acc_magnitude_mean': mag_mean,
                'acc_magnitude_std': mag_std,
                'acc_x_energy': x_energy,
                'acc_y_energy': y_energy,
                'acc_z_energy': z_energy,
                'acc_activity_level': activity_level,
                'acc_dominant_frequency': dominant_freq,
                'acc_entropy': entropy
            }
            
        except Exception:
            return self._get_default_acc_features()
    
    def _classify_activity_level(self, acc_magnitude: np.ndarray) -> float:
        """Classify activity level based on acceleration magnitude"""
        if len(acc_magnitude) == 0:
            return 0.0
            
        activity_intensity = np.std(acc_magnitude)
        
        if activity_intensity < 0.5:
            return 0.0  # Rest/stationary
        elif activity_intensity < 2.0:
            return 1.0  # Light activity
        elif activity_intensity < 5.0:
            return 2.0  # Moderate activity
        else:
            return 3.0  # Vigorous activity
    
    def _calculate_entropy(self, signal: np.ndarray) -> float:
        """Calculate Shannon entropy of signal"""
        try:
            if len(signal) == 0:
                return 0.0
            # Quantize signal to calculate entropy
            hist, _ = np.histogram(signal, bins=min(20, len(signal)//2), density=True)
            hist = hist[hist > 0]  # Remove zeros
            if len(hist) == 0:
                return 0.0
            entropy = -np.sum(hist * np.log2(hist))
            return float(entropy)
        except Exception:
            return 0.0
    
    def extract_temperature_features(self, temp_signal: np.ndarray) -> Dict[str, float]:
        """Extract temperature features"""
        try:
            if len(temp_signal) == 0:
                return self._get_default_temp_features()
                
            temp_mean = float(np.mean(temp_signal))
            temp_std = float(np.std(temp_signal))
            
            # Temperature trend (linear regression slope)
            if len(temp_signal) > 2:
                x = np.arange(len(temp_signal))
                trend = float(np.polyfit(x, temp_signal, 1)[0])
            else:
                trend = 0.0
            
            return {
                'temp_mean': temp_mean,
                'temp_std': temp_std,
                'temp_trend': trend
            }
            
        except Exception:
            return self._get_default_temp_features()
    
    def extract_all_features(self, bvp_signal: np.ndarray, acc_x: np.ndarray, 
                           acc_y: np.ndarray, acc_z: np.ndarray, temp_signal: np.ndarray) -> SmartwatchFeatures:
        """Extract all smartwatch-compatible features"""
        
        # PPG/Heart Rate features
        hr_features = self.extract_ppg_heart_rate_features(bvp_signal)
        ibi_data = hr_features.pop('ibi_data', np.array([]))
        
        # HRV features
        hrv_features = self.extract_hrv_features(ibi_data)
        
        # Accelerometer features  
        acc_features = self.extract_accelerometer_features(acc_x, acc_y, acc_z)
        
        # Temperature features
        temp_features = self.extract_temperature_features(temp_signal)
        
        # Combine into SmartwatchFeatures object
        return SmartwatchFeatures(
            # PPG & HR
            hr_mean=hr_features['hr_mean'],
            hr_std=hr_features['hr_std'],
            hr_min=hr_features['hr_min'],
            hr_max=hr_features['hr_max'],
            
            # HRV
            rmssd=hrv_features['rmssd'],
            pnn50=hrv_features['pnn50'],
            sdnn=hrv_features['sdnn'],
            lf_hf_ratio=hrv_features['lf_hf_ratio'],
            
            # Accelerometer
            acc_magnitude_mean=acc_features['acc_magnitude_mean'],
            acc_magnitude_std=acc_features['acc_magnitude_std'],
            acc_x_energy=acc_features['acc_x_energy'],
            acc_y_energy=acc_features['acc_y_energy'],
            acc_z_energy=acc_features['acc_z_energy'],
            acc_activity_level=acc_features['acc_activity_level'],
            acc_dominant_frequency=acc_features['acc_dominant_frequency'],
            acc_entropy=acc_features['acc_entropy'],
            
            # Temperature
            temp_mean=temp_features['temp_mean'],
            temp_std=temp_features['temp_std'], 
            temp_trend=temp_features['temp_trend']
        )
    
    # Default feature methods
    def _get_default_hr_features(self) -> Dict[str, float]:
        return {
            'hr_mean': 75.0,
            'hr_std': 5.0, 
            'hr_min': 70.0,
            'hr_max': 80.0,
            'ibi_data': np.array([800])  # 75 BPM default
        }
    
    def _get_default_hrv_features(self) -> Dict[str, float]:
        return {
            'rmssd': 25.0,
            'pnn50': 10.0,
            'sdnn': 35.0,
            'lf_hf_ratio': 1.5
        }
    
    def _get_default_acc_features(self) -> Dict[str, float]:
        return {
            'acc_magnitude_mean': 9.81,
            'acc_magnitude_std': 0.5,
            'acc_x_energy': 1.0,
            'acc_y_energy': 1.0,
            'acc_z_energy': 8.0,
            'acc_activity_level': 0.0,
            'acc_dominant_frequency': 0.0,
            'acc_entropy': 2.0
        }
    
    def _get_default_temp_features(self) -> Dict[str, float]:
        return {
            'temp_mean': 32.0,  # Default skin temperature
            'temp_std': 0.0,
            'temp_trend': 0.0
        }


class LOSOSmartWatchPipeline:
    """Leave-One-Subject-Out validation for smartwatch stress detection"""
    
    def __init__(self, data_path: str = "data/wesad", use_temperature: bool = True):
        # use_temperature toggles whether TEMP features are kept (PPG+ACC only when False)
        self.data_path = Path(data_path)
        self.feature_extractor = SmartwatchFeatureExtractor()
        self.use_temperature = use_temperature
        self.loso_results = {}
        
    def extract_features_by_subject(self, window_size_sec: int = 60, overlap_sec: int = 30) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Extract features organized by subject for LOSO validation"""
        print(f"\n🔧 Extracting features by subject for LOSO validation...")
        print(f"   Window size: {window_size_sec}s, Overlap: {overlap_sec}s")
        
        subjects_data = {}
        
        # Get all subject files
        subject_files = list(self.data_path.glob("S*/S*.pkl"))
        print(f"Found {len(subject_files)} subject files")
        
        # Window sizes in samples for each sensor
        bvp_window = int(window_size_sec * self.feature_extractor.bvp_fs)
        acc_window = int(window_size_sec * self.feature_extractor.acc_fs)
        temp_window = int(window_size_sec * self.feature_extractor.temp_fs)
        
        # Step sizes for sliding window
        bvp_step = int((window_size_sec - overlap_sec) * self.feature_extractor.bvp_fs)
        acc_step = int((window_size_sec - overlap_sec) * self.feature_extractor.acc_fs)
        temp_step = int((window_size_sec - overlap_sec) * self.feature_extractor.temp_fs)
        
        for subject_file in sorted(subject_files):
            subject_id = subject_file.parent.name
            print(f"\nProcessing {subject_id}...")
            
            X_subject = []
            y_subject = []
            
            try:
                # Load subject data
                with open(subject_file, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                
                # Extract signals and labels
                labels = data['label'].flatten() if hasattr(data['label'], 'flatten') else data['label']
                wrist_signals = data['signal']['wrist']
                
                # Extract wrist signals (smartwatch compatible)
                bvp = wrist_signals['BVP'].flatten()
                temp = wrist_signals['TEMP'].flatten()
                acc = wrist_signals['ACC']  # 3D accelerometer
                acc_x, acc_y, acc_z = acc[:, 0], acc[:, 1], acc[:, 2]
                
                print(f"   Signals: BVP({len(bvp)}), TEMP({len(temp)}), ACC({acc.shape})")
                
                # Process conditions: baseline (1) and stress (2)
                for condition_label in [1, 2]:  # 1=baseline, 2=stress
                    condition_indices = np.where(labels == condition_label)[0]
                    
                    if len(condition_indices) == 0:
                        continue
                    
                    condition_name = "Baseline" if condition_label == 1 else "Stress"
                    
                    # Find continuous segments
                    segments = self._find_continuous_segments(condition_indices)
                    
                    windows_extracted = 0
                    for start_idx, end_idx in segments:
                        segment_length = end_idx - start_idx + 1
                        segment_duration_sec = segment_length / 700  # WESAD chest sampling rate for labels
                        
                        if segment_duration_sec >= window_size_sec:
                            # Map chest indices to wrist indices (different sampling rates)
                            bvp_start = int(start_idx * 64 / 700)
                            bvp_end = int(end_idx * 64 / 700)
                            
                            acc_start = int(start_idx * 32 / 700) 
                            acc_end = int(end_idx * 32 / 700)
                            
                            temp_start = int(start_idx * 4 / 700)
                            temp_end = int(end_idx * 4 / 700)
                            
                            # Extract windows with sliding approach
                            num_windows = min(
                                (bvp_end - bvp_start - bvp_window) // bvp_step + 1,
                                (acc_end - acc_start - acc_window) // acc_step + 1,
                                (temp_end - temp_start - temp_window) // temp_step + 1
                            )
                            
                            for i in range(max(0, num_windows)):
                                # Calculate window boundaries for each sensor
                                bvp_w_start = bvp_start + i * bvp_step
                                bvp_w_end = bvp_w_start + bvp_window
                                
                                acc_w_start = acc_start + i * acc_step
                                acc_w_end = acc_w_start + acc_window
                                
                                temp_w_start = temp_start + i * temp_step
                                temp_w_end = temp_w_start + temp_window
                                
                                # Check bounds
                                if (bvp_w_end <= len(bvp) and 
                                    acc_w_end <= len(acc_x) and 
                                    temp_w_end <= len(temp)):
                                    
                                    # Extract signals for this window
                                    bvp_window_data = bvp[bvp_w_start:bvp_w_end]
                                    temp_window_data = temp[temp_w_start:temp_w_end]
                                    acc_x_window = acc_x[acc_w_start:acc_w_end]
                                    acc_y_window = acc_y[acc_w_start:acc_w_end]
                                    acc_z_window = acc_z[acc_w_start:acc_w_end]
                                    
                                    # Extract smartwatch features
                                    features = self.feature_extractor.extract_all_features(
                                        bvp_window_data, acc_x_window, acc_y_window, acc_z_window, temp_window_data
                                    )
                                    
                                    feature_array = features.to_array()
                                    if not self.use_temperature:
                                        # Drop TEMP features (last 3 positions) for PPG+ACC-only setting
                                        feature_array = feature_array[:16]
                                    
                                    X_subject.append(feature_array)
                                    y_subject.append(0 if condition_label == 1 else 1)  # 0=baseline, 1=stress
                                    
                                    windows_extracted += 1
                                    
                                    # Limit windows to ensure balanced computation
                                    if windows_extracted >= 20:  # Max 20 windows per condition per subject
                                        break
                            
                            if windows_extracted >= 20:
                                break
                    
                    print(f"   {condition_name}: {windows_extracted} windows extracted")
                
                if len(X_subject) > 0:
                    subjects_data[subject_id] = (np.array(X_subject), np.array(y_subject))
                    print(f"   Total windows for {subject_id}: {len(X_subject)}")
                    print(f"   Class distribution: {np.bincount(y_subject)}")
                    
            except Exception as e:
                print(f"Error processing {subject_file.name}: {e}")
                continue
        
        print(f"\n✅ Feature extraction complete!")
        print(f"   Subjects with data: {len(subjects_data)}")
        
        return subjects_data
    
    def _find_continuous_segments(self, indices: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous segments in array of indices"""
        if len(indices) == 0:
            return []
        
        segments = []
        start = indices[0]
        
        for i in range(1, len(indices)):
            if indices[i] != indices[i-1] + 1:
                segments.append((start, indices[i-1]))
                start = indices[i]
        
        segments.append((start, indices[-1]))
        return segments
    
    def run_loso_validation(self, subjects_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """Run Leave-One-Subject-Out cross-validation"""
        print("\n🔄 Running Leave-One-Subject-Out (LOSO) Cross-Validation...")
        print("=" * 70)
        
        subject_ids = list(subjects_data.keys())
        n_subjects = len(subject_ids)
        
        print(f"Total subjects: {n_subjects}")
        
        # Models to evaluate
        models_config = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42,
                class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'SVM': SVC(
                C=1.0,
                gamma='scale',
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        # Initialize results storage
        for model_name in models_config.keys():
            self.loso_results[model_name] = {
                'fold_accuracies': [],
                'fold_aucs': [],
                'fold_subjects': [],
                'predictions': [],
                'true_labels': [],
                'feature_importances': []
            }
        
        # LOSO Cross-validation loop
        for i, test_subject in enumerate(subject_ids):
            print(f"\nFold {i+1}/{n_subjects}: Testing on {test_subject}")
            
            # Prepare train and test sets
            X_train_list = []
            y_train_list = []
            
            # Collect training data from all other subjects
            for subject_id in subject_ids:
                if subject_id != test_subject:
                    X_subj, y_subj = subjects_data[subject_id]
                    X_train_list.append(X_subj)
                    y_train_list.append(y_subj)
            
            if len(X_train_list) == 0:
                print("   ❌ No training data available")
                continue
            
            # Combine training data
            X_train = np.vstack(X_train_list)
            y_train = np.hstack(y_train_list)
            
            # Test data
            X_test, y_test = subjects_data[test_subject]

            # Optionally remove temperature features (keep PPG + ACC only)
            if not self.use_temperature:
                X_train = X_train[:, :16]
                X_test = X_test[:, :16]
            
            print(f"   Training: {len(X_train)} samples from {len(X_train_list)} subjects")
            print(f"   Testing:  {len(X_test)} samples from {test_subject}")
            print(f"   Train distribution: {np.bincount(y_train)}")
            print(f"   Test distribution:  {np.bincount(y_test)}")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train and evaluate each model
            for model_name, model in models_config.items():
                try:
                    # Use scaled data for SVM, original for tree-based
                    if model_name in ['SVM']:
                        X_train_model = X_train_scaled
                        X_test_model = X_test_scaled
                    else:
                        X_train_model = X_train
                        X_test_model = X_test
                    
                    # Train model
                    model.fit(X_train_model, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test_model)
                    y_proba = model.predict_proba(X_test_model)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                    
                    # Metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    try:
                        auc = roc_auc_score(y_test, y_proba)
                    except:
                        auc = 0.5  # Default for edge cases
                    
                    # Store results
                    self.loso_results[model_name]['fold_accuracies'].append(accuracy)
                    self.loso_results[model_name]['fold_aucs'].append(auc)
                    self.loso_results[model_name]['fold_subjects'].append(test_subject)
                    self.loso_results[model_name]['predictions'].extend(y_pred)
                    self.loso_results[model_name]['true_labels'].extend(y_test)
                    
                    # Feature importance (for tree-based models)
                    if hasattr(model, 'feature_importances_'):
                        self.loso_results[model_name]['feature_importances'].append(model.feature_importances_)
                    
                    print(f"      {model_name}: Accuracy={accuracy:.3f}, AUC={auc:.3f}")
                    
                except Exception as e:
                    print(f"      {model_name}: Error - {e}")
                    continue
        
        # Calculate final statistics
        self._calculate_loso_statistics()
    
    def _calculate_loso_statistics(self) -> None:
        """Calculate final LOSO statistics"""
        print("\n📊 LOSO Cross-Validation Results")
        print("=" * 70)
        
        results_summary = []
        
        for model_name, results in self.loso_results.items():
            if len(results['fold_accuracies']) == 0:
                continue
            
            # Calculate statistics
            mean_acc = np.mean(results['fold_accuracies'])
            std_acc = np.std(results['fold_accuracies'])
            mean_auc = np.mean(results['fold_aucs'])
            std_auc = np.std(results['fold_aucs'])
            
            # Overall metrics
            overall_acc = accuracy_score(results['true_labels'], results['predictions'])
            try:
                overall_auc = roc_auc_score(results['true_labels'], results['predictions'])
            except:
                overall_auc = 0.5
            
            print(f"\n{model_name}:")
            print(f"   Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
            print(f"   Mean AUC:      {mean_auc:.3f} ± {std_auc:.3f}")
            print(f"   Overall Accuracy: {overall_acc:.3f}")
            print(f"   Overall AUC:      {overall_auc:.3f}")
            print(f"   Completed folds: {len(results['fold_accuracies'])}")
            
            results_summary.append({
                'Model': model_name,
                'Mean_Accuracy': mean_acc,
                'Std_Accuracy': std_acc,
                'Mean_AUC': mean_auc,
                'Std_AUC': std_auc,
                'Overall_Accuracy': overall_acc,
                'Overall_AUC': overall_auc,
                'N_Folds': len(results['fold_accuracies'])
            })
        
        # Save results
        results_df = pd.DataFrame(results_summary)
        results_df.to_csv('results/smartwatch_loso_results.csv', index=False)
        
        # Save detailed results
        with open('results/smartwatch_loso_detailed.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for model_name, results in self.loso_results.items():
                json_results[model_name] = {
                    'fold_accuracies': [float(x) for x in results['fold_accuracies']],
                    'fold_aucs': [float(x) for x in results['fold_aucs']],
                    'fold_subjects': results['fold_subjects'],
                    'mean_accuracy': float(np.mean(results['fold_accuracies'])) if results['fold_accuracies'] else 0,
                    'std_accuracy': float(np.std(results['fold_accuracies'])) if results['fold_accuracies'] else 0,
                    'mean_auc': float(np.mean(results['fold_aucs'])) if results['fold_aucs'] else 0,
                    'std_auc': float(np.std(results['fold_aucs'])) if results['fold_aucs'] else 0
                }
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"   • results/smartwatch_loso_results.csv")
        print(f"   • results/smartwatch_loso_detailed.json")
    
    def create_loso_visualizations(self) -> None:
        """Create visualizations for LOSO results"""
        print("\n📊 Creating LOSO visualizations...")
        
        # Create results directory
        output_dir = Path('results/advanced_figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # LOSO Accuracy per fold visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Accuracy per fold
        models = list(self.loso_results.keys())
        fold_data = []
        
        for model_name in models:
            if len(self.loso_results[model_name]['fold_accuracies']) > 0:
                for i, acc in enumerate(self.loso_results[model_name]['fold_accuracies']):
                    fold_data.append({
                        'Model': model_name,
                        'Fold': i+1,
                        'Subject': self.loso_results[model_name]['fold_subjects'][i],
                        'Accuracy': acc
                    })
        
        if fold_data:
            fold_df = pd.DataFrame(fold_data)
            
            # Box plot of accuracies
            sns.boxplot(data=fold_df, x='Model', y='Accuracy', ax=ax1)
            ax1.set_title('LOSO Cross-Validation Accuracy Distribution')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1.1)
            ax1.grid(True, alpha=0.3)
            
            # Add mean accuracy labels
            for i, model in enumerate(models):
                if model in [item['Model'] for item in fold_data]:
                    model_accs = [item['Accuracy'] for item in fold_data if item['Model'] == model]
                    mean_acc = np.mean(model_accs)
                    ax1.text(i, 1.05, f'{mean_acc:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Accuracy per subject
        if fold_data:
            # Pivot to show each model's performance per subject
            pivot_df = fold_df.pivot(index='Subject', columns='Model', values='Accuracy')
            
            pivot_df.plot(kind='bar', ax=ax2, width=0.8, alpha=0.8)
            ax2.set_title('LOSO Accuracy per Subject')
            ax2.set_xlabel('Test Subject')
            ax2.set_ylabel('Accuracy')
            ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.set_ylim(0, 1.1)
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'smartwatch_loso_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: {output_dir / 'smartwatch_loso_validation.png'}")
    
    def print_realistic_summary(self) -> None:
        """Print realistic summary based on LOSO validation"""
        print("\n" + "="*80)
        print("🎯 REALISTIC SMARTWATCH STRESS DETECTION PERFORMANCE (LOSO)")
        print("="*80)
        
        print("\n📊 LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION RESULTS:")
        print("   (Tests generalization to completely new subjects)")
        
        best_model = None
        best_accuracy = 0
        
        for model_name, results in self.loso_results.items():
            if len(results['fold_accuracies']) == 0:
                continue
            
            mean_acc = np.mean(results['fold_accuracies'])
            if mean_acc > best_accuracy:
                best_accuracy = mean_acc
                best_model = model_name
            
            print(f"\n{model_name}:")
            print(f"   • Mean Accuracy:    {mean_acc:.1%} ± {np.std(results['fold_accuracies']):.1%}")
            print(f"   • Mean AUC:         {np.mean(results['fold_aucs']):.3f} ± {np.std(results['fold_aucs']):.3f}")
            print(f"   • Best Fold:        {np.max(results['fold_accuracies']):.1%}")
            print(f"   • Worst Fold:       {np.min(results['fold_accuracies']):.1%}")
            print(f"   • Subjects Tested:  {len(results['fold_accuracies'])}")
        
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model} ({best_accuracy:.1%})")
        
        print(f"\n🔬 CLINICAL IMPLICATIONS:")
        if best_accuracy > 0.8:
            print(f"   ✅ Accuracy > 80%: Clinically meaningful for screening")
        else:
            print(f"   ⚠️  Accuracy < 80%: May need improvement for clinical use")
        
        if best_accuracy > 0.7:
            print(f"   ✅ Suitable for research and monitoring applications")
        else:
            print(f"   ❌ May need more data or better features")
        
        print(f"\n⚖️  COMPARISON WITH LITERATURE:")
        print(f"   • WESAD Original (all sensors):     ~95% (person-dependent)")
        print(f"   • Our Smartwatch (LOSO):           ~{best_accuracy:.0%} (person-independent)")
        print(f"   • Trade-off for generalization:    ~{95-best_accuracy*100:.0f} percentage points")
        
        print(f"\n📱 DEPLOYMENT CONSIDERATIONS:")
        print(f"   • Sensor availability: ✅ All sensors in consumer smartwatches")
        print(f"   • Cross-person generalization: {'✅' if best_accuracy > 0.7 else '⚠️'}")
        print(f"   • Real-world applicability: {'High' if best_accuracy > 0.8 else 'Moderate' if best_accuracy > 0.7 else 'Limited'}")
        
        print("\n" + "="*80)
    
    def run_complete_loso_pipeline(self) -> None:
        """Run the complete LOSO validation pipeline"""
        print("🔄 Starting LOSO Smartwatch Stress Detection Validation")
        print("=" * 60)
        
        # Step 1: Extract features by subject
        subjects_data = self.extract_features_by_subject()
        
        if len(subjects_data) < 2:
            print("❌ Need at least 2 subjects for LOSO validation")
            return
        
        # Step 2: Run LOSO validation
        self.run_loso_validation(subjects_data)
        
        # Step 3: Create visualizations
        self.create_loso_visualizations()
        
        # Step 4: Print realistic summary
        self.print_realistic_summary()
        
        print("\n✅ LOSO validation complete!")


def main():
    """Main execution function"""
    # Set use_temperature=False to benchmark PPG + ACC only (no TEMP features)
    pipeline = LOSOSmartWatchPipeline(use_temperature=False)
    pipeline.run_complete_loso_pipeline()


if __name__ == "__main__":
    main()