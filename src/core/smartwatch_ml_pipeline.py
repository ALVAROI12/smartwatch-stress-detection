#!/usr/bin/env python3
"""
SmartWatch-Focused ML Pipeline for Stress Detection
==================================================

This pipeline focuses specifically on sensors commonly available in consumer smartwatches:
- PPG (optical heart rate signal)  ✓
- Heart Rate (HR)                  ✓  
- Inter-Beat Interval (IBI)        ✓
- Heart Rate Variability (HRV)     ✓
- Accelerometer (3-axis)           ✓
- Gyroscope                        ❌ (not in WESAD - will simulate/approximate)
- Skin/Wrist Temperature           ✓
- Respiratory Rate (derived)       ✓ (from PPG/HR)
- Step Count                       ✓ (derived from accelerometer)
- Activity Level/Type              ✓ (derived from accelerometer)
- Sleep State                      ❌ (not applicable for stress detection)

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

# ML libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class SmartwatchFeatures:
    """Feature vector for smartwatch-compatible stress detection"""
    # PPG & Heart Rate Features (10 features)
    hr_mean: float = 0.0
    hr_std: float = 0.0
    hr_min: float = 0.0
    hr_max: float = 0.0
    hr_range: float = 0.0
    
    # HRV Features (8 features)
    rmssd: float = 0.0
    pnn50: float = 0.0
    sdnn: float = 0.0
    triangular_index: float = 0.0
    lf_power: float = 0.0
    hf_power: float = 0.0
    lf_hf_ratio: float = 0.0
    total_power: float = 0.0
    
    # Respiratory Features (3 features) - derived from PPG
    breathing_rate: float = 0.0
    breathing_variability: float = 0.0
    respiratory_sinus_arrhythmia: float = 0.0
    
    # Accelerometer Features (12 features)
    acc_magnitude_mean: float = 0.0
    acc_magnitude_std: float = 0.0
    acc_magnitude_range: float = 0.0
    acc_x_energy: float = 0.0
    acc_y_energy: float = 0.0
    acc_z_energy: float = 0.0
    acc_activity_level: float = 0.0  # 0=rest, 1=light, 2=moderate, 3=vigorous
    acc_step_count: float = 0.0      # estimated steps in window
    acc_dominant_frequency: float = 0.0
    acc_entropy: float = 0.0
    acc_zero_crossing_rate: float = 0.0
    acc_posture_stability: float = 0.0
    
    # Temperature Features (4 features)
    temp_mean: float = 0.0
    temp_std: float = 0.0
    temp_trend: float = 0.0
    temp_variability: float = 0.0
    
    # Derived Activity Features (3 features)
    activity_type: float = 0.0       # 0=stationary, 1=walking, 2=running, 3=other
    movement_intensity: float = 0.0
    posture_changes: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models"""
        return np.array([
            # PPG & HR (5)
            self.hr_mean, self.hr_std, self.hr_min, self.hr_max, self.hr_range,
            # HRV (8)
            self.rmssd, self.pnn50, self.sdnn, self.triangular_index,
            self.lf_power, self.hf_power, self.lf_hf_ratio, self.total_power,
            # Respiratory (3)
            self.breathing_rate, self.breathing_variability, self.respiratory_sinus_arrhythmia,
            # Accelerometer (12)
            self.acc_magnitude_mean, self.acc_magnitude_std, self.acc_magnitude_range,
            self.acc_x_energy, self.acc_y_energy, self.acc_z_energy,
            self.acc_activity_level, self.acc_step_count, self.acc_dominant_frequency,
            self.acc_entropy, self.acc_zero_crossing_rate, self.acc_posture_stability,
            # Temperature (4)
            self.temp_mean, self.temp_std, self.temp_trend, self.temp_variability,
            # Activity (3)
            self.activity_type, self.movement_intensity, self.posture_changes
        ])
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get feature names for model interpretation"""
        return [
            # PPG & HR (5)
            'hr_mean', 'hr_std', 'hr_min', 'hr_max', 'hr_range',
            # HRV (8)
            'rmssd', 'pnn50', 'sdnn', 'triangular_index',
            'lf_power', 'hf_power', 'lf_hf_ratio', 'total_power',
            # Respiratory (3)
            'breathing_rate', 'breathing_variability', 'respiratory_sinus_arrhythmia',
            # Accelerometer (12)
            'acc_magnitude_mean', 'acc_magnitude_std', 'acc_magnitude_range',
            'acc_x_energy', 'acc_y_energy', 'acc_z_energy',
            'acc_activity_level', 'acc_step_count', 'acc_dominant_frequency',
            'acc_entropy', 'acc_zero_crossing_rate', 'acc_posture_stability',
            # Temperature (4)
            'temp_mean', 'temp_std', 'temp_trend', 'temp_variability',
            # Activity (3)
            'activity_type', 'movement_intensity', 'posture_changes'
        ]


class SmartwatchFeatureExtractor:
    """Extract features compatible with consumer smartwatch sensors"""
    
    def __init__(self, sampling_rate: float = 700.0):
        self.fs = sampling_rate
        
    def extract_ppg_heart_rate_features(self, bvp_signal: np.ndarray) -> Dict[str, float]:
        """Extract heart rate features from PPG/BVP signal"""
        try:
            # Basic filtering for PPG
            nyq = self.fs / 2
            low = 0.5 / nyq
            high = 8.0 / nyq
            b, a = butter(4, [low, high], btype='band')
            filtered_bvp = filtfilt(b, a, bvp_signal)
            
            # Find peaks (R-peaks equivalent in PPG)
            # Use adaptive height threshold
            height_threshold = np.std(filtered_bvp) * 0.3
            peaks, _ = find_peaks(filtered_bvp, 
                                height=height_threshold,
                                distance=int(0.4 * self.fs))  # Min 0.4s between peaks (150 BPM max)
            
            if len(peaks) < 2:
                return self._get_default_hr_features()
                
            # Calculate inter-beat intervals (IBI) in milliseconds
            ibi_samples = np.diff(peaks)
            ibi_ms = (ibi_samples / self.fs) * 1000
            
            # Filter physiologically plausible IBIs (300-2000ms = 30-200 BPM)
            valid_ibi = ibi_ms[(ibi_ms >= 300) & (ibi_ms <= 2000)]
            
            if len(valid_ibi) < 2:
                return self._get_default_hr_features()
            
            # Convert to heart rate (BPM)
            heart_rates = 60000 / valid_ibi  # 60000 ms/min
            
            return {
                'hr_mean': float(np.mean(heart_rates)),
                'hr_std': float(np.std(heart_rates)),
                'hr_min': float(np.min(heart_rates)),
                'hr_max': float(np.max(heart_rates)),
                'hr_range': float(np.max(heart_rates) - np.min(heart_rates)),
                'ibi_data': valid_ibi  # Keep for HRV calculation
            }
            
        except Exception as e:
            print(f"Heart rate extraction failed: {e}")
            return self._get_default_hr_features()
    
    def extract_hrv_features(self, ibi_data: np.ndarray) -> Dict[str, float]:
        """Extract Heart Rate Variability features"""
        try:
            if len(ibi_data) < 5:
                return self._get_default_hrv_features()
            
            # Time domain features
            rmssd = float(np.sqrt(np.mean(np.diff(ibi_data)**2)))
            sdnn = float(np.std(ibi_data))
            
            # pNN50: percentage of successive differences > 50ms
            successive_diffs = np.abs(np.diff(ibi_data))
            pnn50 = float(np.sum(successive_diffs > 50) / len(successive_diffs) * 100)
            
            # Triangular index approximation
            triangular_index = float(len(ibi_data) / (2 * np.std(ibi_data)) if np.std(ibi_data) > 0 else 0)
            
            # Frequency domain features
            freq_features = self._calculate_frequency_domain_hrv(ibi_data)
            
            return {
                'rmssd': rmssd,
                'pnn50': pnn50,
                'sdnn': sdnn,
                'triangular_index': triangular_index,
                **freq_features
            }
            
        except Exception as e:
            print(f"HRV extraction failed: {e}")
            return self._get_default_hrv_features()
    
    def _calculate_frequency_domain_hrv(self, ibi_data: np.ndarray) -> Dict[str, float]:
        """Calculate frequency domain HRV features"""
        try:
            # Resample IBIs to regular time series (4 Hz)
            time_original = np.cumsum(ibi_data) / 1000  # Convert to seconds
            time_regular = np.arange(0, time_original[-1], 0.25)  # 4 Hz
            
            # Interpolate
            ibi_regular = np.interp(time_regular, time_original, ibi_data)
            
            # Remove trend
            ibi_detrended = signal.detrend(ibi_regular)
            
            # Apply window
            windowed = ibi_detrended * np.hanning(len(ibi_detrended))
            
            # Calculate PSD
            freqs, psd = signal.welch(windowed, fs=4.0, nperseg=min(256, len(windowed)))
            
            # Define frequency bands (Hz)
            vlf_band = (freqs >= 0.003) & (freqs < 0.04)
            lf_band = (freqs >= 0.04) & (freqs < 0.15)
            hf_band = (freqs >= 0.15) & (freqs < 0.4)
            
            # Calculate power in each band
            vlf_power = float(np.sum(psd[vlf_band]))
            lf_power = float(np.sum(psd[lf_band]))
            hf_power = float(np.sum(psd[hf_band]))
            total_power = float(vlf_power + lf_power + hf_power)
            
            # LF/HF ratio
            lf_hf_ratio = float(lf_power / hf_power if hf_power > 0 else 0)
            
            return {
                'lf_power': lf_power,
                'hf_power': hf_power,
                'lf_hf_ratio': lf_hf_ratio,
                'total_power': total_power
            }
            
        except Exception:
            return {
                'lf_power': 0.0,
                'hf_power': 0.0, 
                'lf_hf_ratio': 0.0,
                'total_power': 0.0
            }
    
    def extract_respiratory_features(self, bvp_signal: np.ndarray, ibi_data: np.ndarray) -> Dict[str, float]:
        """Extract respiratory features from PPG signal"""
        try:
            # Method 1: Respiratory sinus arrhythmia from IBI variation
            if len(ibi_data) > 10:
                # High-frequency variation in IBI indicates respiratory influence
                ibi_diff = np.diff(ibi_data)
                rsa = float(np.std(ibi_diff))  # Respiratory sinus arrhythmia
            else:
                rsa = 0.0
            
            # Method 2: Extract respiratory rate from PPG baseline wander
            # Filter to isolate respiratory component (0.1-0.5 Hz)
            nyq = self.fs / 2
            low = 0.1 / nyq  
            high = 0.5 / nyq
            b, a = butter(4, [low, high], btype='band')
            resp_component = filtfilt(b, a, bvp_signal)
            
            # Find peaks in respiratory component
            resp_peaks, _ = find_peaks(resp_component, distance=int(2 * self.fs))  # Min 2s between breaths
            
            if len(resp_peaks) > 1:
                breath_intervals = np.diff(resp_peaks) / self.fs  # seconds
                breathing_rate = float(60 / np.mean(breath_intervals))  # breaths per minute
                breathing_variability = float(np.std(breath_intervals))
            else:
                breathing_rate = 15.0  # Default ~15 breaths/min
                breathing_variability = 0.0
            
            return {
                'breathing_rate': min(max(breathing_rate, 8), 30),  # Clamp to realistic range
                'breathing_variability': breathing_variability,
                'respiratory_sinus_arrhythmia': rsa
            }
            
        except Exception:
            return {
                'breathing_rate': 15.0,
                'breathing_variability': 0.0,
                'respiratory_sinus_arrhythmia': 0.0
            }
    
    def extract_accelerometer_features(self, acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive accelerometer features"""
        try:
            # Calculate magnitude
            acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            
            # Remove gravity (assuming 1g = 9.81 m/s^2)
            acc_magnitude_no_gravity = acc_magnitude - np.median(acc_magnitude)
            
            # Basic statistics
            mag_mean = float(np.mean(acc_magnitude))
            mag_std = float(np.std(acc_magnitude))
            mag_range = float(np.max(acc_magnitude) - np.min(acc_magnitude))
            
            # Energy in each axis
            x_energy = float(np.sum(acc_x**2) / len(acc_x))
            y_energy = float(np.sum(acc_y**2) / len(acc_y))
            z_energy = float(np.sum(acc_z**2) / len(acc_z))
            
            # Activity level classification
            activity_level = self._classify_activity_level(acc_magnitude_no_gravity)
            
            # Step counting (simple peak detection)
            step_count = self._estimate_step_count(acc_magnitude_no_gravity)
            
            # Frequency analysis
            freqs, psd = signal.welch(acc_magnitude_no_gravity, fs=self.fs)
            dominant_freq = float(freqs[np.argmax(psd)])
            
            # Entropy (complexity measure)
            entropy = self._calculate_entropy(acc_magnitude_no_gravity)
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(acc_magnitude_no_gravity - np.mean(acc_magnitude_no_gravity))))[0]
            zero_crossing_rate = float(len(zero_crossings) / len(acc_magnitude_no_gravity))
            
            # Posture stability (low-frequency power)
            low_freq_power = np.sum(psd[freqs < 1.0])
            posture_stability = float(low_freq_power / np.sum(psd))
            
            return {
                'acc_magnitude_mean': mag_mean,
                'acc_magnitude_std': mag_std,
                'acc_magnitude_range': mag_range,
                'acc_x_energy': x_energy,
                'acc_y_energy': y_energy,
                'acc_z_energy': z_energy,
                'acc_activity_level': activity_level,
                'acc_step_count': step_count,
                'acc_dominant_frequency': dominant_freq,
                'acc_entropy': entropy,
                'acc_zero_crossing_rate': zero_crossing_rate,
                'acc_posture_stability': posture_stability
            }
            
        except Exception as e:
            print(f"Accelerometer feature extraction failed: {e}")
            return self._get_default_acc_features()
    
    def _classify_activity_level(self, acc_magnitude: np.ndarray) -> float:
        """Classify activity level based on acceleration magnitude"""
        activity_intensity = np.std(acc_magnitude)
        
        if activity_intensity < 0.5:
            return 0.0  # Rest/stationary
        elif activity_intensity < 2.0:
            return 1.0  # Light activity
        elif activity_intensity < 5.0:
            return 2.0  # Moderate activity
        else:
            return 3.0  # Vigorous activity
    
    def _estimate_step_count(self, acc_magnitude: np.ndarray) -> float:
        """Simple step counting from acceleration magnitude"""
        try:
            # Find peaks that could represent steps
            height_threshold = np.std(acc_magnitude) * 1.5
            min_distance = int(0.3 * self.fs)  # Minimum 0.3s between steps
            
            peaks, _ = find_peaks(acc_magnitude, 
                                height=height_threshold,
                                distance=min_distance)
            
            # Estimate steps per minute
            window_duration_min = len(acc_magnitude) / self.fs / 60
            steps_per_min = len(peaks) / window_duration_min if window_duration_min > 0 else 0
            
            return float(min(steps_per_min, 200))  # Cap at 200 steps/min
            
        except Exception:
            return 0.0
    
    def _calculate_entropy(self, signal: np.ndarray) -> float:
        """Calculate Shannon entropy of signal"""
        try:
            # Quantize signal to calculate entropy
            hist, _ = np.histogram(signal, bins=50, density=True)
            hist = hist[hist > 0]  # Remove zeros
            entropy = -np.sum(hist * np.log2(hist))
            return float(entropy)
        except Exception:
            return 0.0
    
    def extract_temperature_features(self, temp_signal: np.ndarray) -> Dict[str, float]:
        """Extract temperature features"""
        try:
            temp_mean = float(np.mean(temp_signal))
            temp_std = float(np.std(temp_signal))
            
            # Temperature trend (linear regression slope)
            x = np.arange(len(temp_signal))
            trend = float(np.polyfit(x, temp_signal, 1)[0])
            
            # Temperature variability (coefficient of variation)
            variability = float(temp_std / temp_mean if temp_mean != 0 else 0)
            
            return {
                'temp_mean': temp_mean,
                'temp_std': temp_std,
                'temp_trend': trend,
                'temp_variability': variability
            }
            
        except Exception:
            return {
                'temp_mean': 32.0,  # Default skin temperature
                'temp_std': 0.0,
                'temp_trend': 0.0,
                'temp_variability': 0.0
            }
    
    def extract_activity_features(self, acc_features: Dict[str, float]) -> Dict[str, float]:
        """Extract high-level activity features from accelerometer data"""
        try:
            # Activity type classification (simplified)
            activity_level = acc_features['acc_activity_level']
            step_rate = acc_features['acc_step_count']
            
            # Classify activity type based on step rate and intensity
            if step_rate < 10:  # < 10 steps/min
                activity_type = 0.0  # Stationary
            elif step_rate < 80:  # 10-80 steps/min
                activity_type = 1.0  # Walking
            elif step_rate < 120:  # 80-120 steps/min  
                activity_type = 2.0  # Running
            else:
                activity_type = 3.0  # Other intensive activity
            
            # Movement intensity
            movement_intensity = min(acc_features['acc_magnitude_std'] / 10.0, 1.0)  # Normalize to 0-1
            
            # Posture changes (inverse of stability)
            posture_changes = 1.0 - acc_features['acc_posture_stability']
            
            return {
                'activity_type': activity_type,
                'movement_intensity': float(movement_intensity),
                'posture_changes': float(posture_changes)
            }
            
        except Exception:
            return {
                'activity_type': 0.0,
                'movement_intensity': 0.0,
                'posture_changes': 0.0
            }
    
    def extract_all_features(self, bvp_signal: np.ndarray, acc_x: np.ndarray, 
                           acc_y: np.ndarray, acc_z: np.ndarray, temp_signal: np.ndarray) -> SmartwatchFeatures:
        """Extract all smartwatch-compatible features"""
        
        # PPG/Heart Rate features
        hr_features = self.extract_ppg_heart_rate_features(bvp_signal)
        ibi_data = hr_features.pop('ibi_data', np.array([]))
        
        # HRV features
        hrv_features = self.extract_hrv_features(ibi_data)
        
        # Respiratory features
        resp_features = self.extract_respiratory_features(bvp_signal, ibi_data)
        
        # Accelerometer features  
        acc_features = self.extract_accelerometer_features(acc_x, acc_y, acc_z)
        
        # Temperature features
        temp_features = self.extract_temperature_features(temp_signal)
        
        # Activity features
        activity_features = self.extract_activity_features(acc_features)
        
        # Combine into SmartwatchFeatures object
        return SmartwatchFeatures(
            # PPG & HR
            hr_mean=hr_features['hr_mean'],
            hr_std=hr_features['hr_std'],
            hr_min=hr_features['hr_min'],
            hr_max=hr_features['hr_max'],
            hr_range=hr_features['hr_range'],
            
            # HRV
            rmssd=hrv_features['rmssd'],
            pnn50=hrv_features['pnn50'],
            sdnn=hrv_features['sdnn'],
            triangular_index=hrv_features['triangular_index'],
            lf_power=hrv_features['lf_power'],
            hf_power=hrv_features['hf_power'],
            lf_hf_ratio=hrv_features['lf_hf_ratio'],
            total_power=hrv_features['total_power'],
            
            # Respiratory
            breathing_rate=resp_features['breathing_rate'],
            breathing_variability=resp_features['breathing_variability'],
            respiratory_sinus_arrhythmia=resp_features['respiratory_sinus_arrhythmia'],
            
            # Accelerometer
            acc_magnitude_mean=acc_features['acc_magnitude_mean'],
            acc_magnitude_std=acc_features['acc_magnitude_std'],
            acc_magnitude_range=acc_features['acc_magnitude_range'],
            acc_x_energy=acc_features['acc_x_energy'],
            acc_y_energy=acc_features['acc_y_energy'],
            acc_z_energy=acc_features['acc_z_energy'],
            acc_activity_level=acc_features['acc_activity_level'],
            acc_step_count=acc_features['acc_step_count'],
            acc_dominant_frequency=acc_features['acc_dominant_frequency'],
            acc_entropy=acc_features['acc_entropy'],
            acc_zero_crossing_rate=acc_features['acc_zero_crossing_rate'],
            acc_posture_stability=acc_features['acc_posture_stability'],
            
            # Temperature
            temp_mean=temp_features['temp_mean'],
            temp_std=temp_features['temp_std'], 
            temp_trend=temp_features['temp_trend'],
            temp_variability=temp_features['temp_variability'],
            
            # Activity
            activity_type=activity_features['activity_type'],
            movement_intensity=activity_features['movement_intensity'],
            posture_changes=activity_features['posture_changes']
        )
    
    # Default feature methods
    def _get_default_hr_features(self) -> Dict[str, float]:
        return {
            'hr_mean': 75.0,
            'hr_std': 5.0, 
            'hr_min': 70.0,
            'hr_max': 80.0,
            'hr_range': 10.0,
            'ibi_data': np.array([800])  # 75 BPM default
        }
    
    def _get_default_hrv_features(self) -> Dict[str, float]:
        return {
            'rmssd': 25.0,
            'pnn50': 10.0,
            'sdnn': 35.0,
            'triangular_index': 15.0,
            'lf_power': 500.0,
            'hf_power': 300.0,
            'lf_hf_ratio': 1.67,
            'total_power': 1000.0
        }
    
    def _get_default_acc_features(self) -> Dict[str, float]:
        return {
            'acc_magnitude_mean': 9.81,
            'acc_magnitude_std': 0.5,
            'acc_magnitude_range': 3.0,
            'acc_x_energy': 1.0,
            'acc_y_energy': 1.0,
            'acc_z_energy': 8.0,
            'acc_activity_level': 0.0,
            'acc_step_count': 0.0,
            'acc_dominant_frequency': 0.0,
            'acc_entropy': 2.0,
            'acc_zero_crossing_rate': 0.1,
            'acc_posture_stability': 0.8
        }


class SmartwatchMLPipeline:
    """Complete ML pipeline for smartwatch stress detection"""
    
    def __init__(self, data_path: str = "data/wesad"):
        self.data_path = Path(data_path)
        self.feature_extractor = SmartwatchFeatureExtractor()
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def process_wesad_data(self, window_size_sec: int = 180, overlap_sec: int = 90) -> Tuple[np.ndarray, np.ndarray]:
        """Process WESAD data and extract smartwatch features"""
        print("\n🔧 Processing WESAD data for smartwatch compatibility...")
        
        X_features = []
        y_labels = []
        
        # Get all subject files
        subject_files = list(self.data_path.glob("S*/S*.pkl"))
        print(f"Found {len(subject_files)} subject files")
        
        fs = 700  # WESAD sampling rate
        window_size = int(window_size_sec * fs)
        step_size = int((window_size_sec - overlap_sec) * fs)
        
        for subject_file in sorted(subject_files):
            print(f"\nProcessing {subject_file.name}...")
            
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
                    print(f"   Processing {condition_name}: {len(condition_indices)} samples")
                    
                    # Find continuous segments
                    segments = self._find_continuous_segments(condition_indices)
                    
                    for start_idx, end_idx in segments:
                        segment_length = end_idx - start_idx + 1
                        
                        if segment_length >= window_size:
                            # Extract windows with sliding approach
                            for window_start in range(start_idx, end_idx - window_size + 1, step_size):
                                window_end = window_start + window_size
                                
                                # Check window purity (>95% same label)
                                window_labels = labels[window_start:window_end]
                                purity = np.sum(window_labels == condition_label) / len(window_labels)
                                
                                if purity > 0.95:
                                    # Extract signals for this window
                                    bvp_window = bvp[window_start:window_end]
                                    temp_window = temp[window_start:window_end]
                                    acc_x_window = acc_x[window_start:window_end]
                                    acc_y_window = acc_y[window_start:window_end]
                                    acc_z_window = acc_z[window_start:window_end]
                                    
                                    # Extract smartwatch features
                                    features = self.feature_extractor.extract_all_features(
                                        bvp_window, acc_x_window, acc_y_window, acc_z_window, temp_window
                                    )
                                    
                                    X_features.append(features.to_array())
                                    y_labels.append(0 if condition_label == 1 else 1)  # 0=baseline, 1=stress
                                    
            except Exception as e:
                print(f"Error processing {subject_file.name}: {e}")
                continue
        
        X = np.array(X_features)
        y = np.array(y_labels)
        
        print(f"\n✅ Feature extraction complete!")
        print(f"   Total windows: {len(X)}")
        print(f"   Feature dimensions: {X.shape[1]} features")
        print(f"   Class distribution: {np.bincount(y)}")
        
        return X, y
    
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
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train multiple ML models for smartwatch stress detection"""
        print("\n🤖 Training smartwatch ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Define models optimized for smartwatch features
        models_config = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
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
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=1000,
                alpha=0.01,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Train and evaluate each model
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for SVM and MLP, original for tree-based
            if name in ['SVM', 'MLP']:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                X_train_model = X_train
                X_test_model = X_test
            
            # Train model
            start_time = time.time()
            model.fit(X_train_model, y_train)
            train_time = time.time() - start_time
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_model, y_train, cv=cv, 
                scoring='accuracy', n_jobs=-1
            )
            
            # Test set evaluation
            y_pred = model.predict(X_test_model)
            y_proba = model.predict_proba(X_test_model)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Metrics
            test_accuracy = np.mean(y_pred == y_test)
            auc_score = roc_auc_score(y_test, y_proba)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'test_accuracy': test_accuracy,
                'auc_score': auc_score,
                'train_time': train_time,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"   CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            print(f"   Test Accuracy: {test_accuracy:.3f}")
            print(f"   AUC: {auc_score:.3f}")
            print(f"   Training time: {train_time:.2f}s")
        
        # Feature importance analysis
        self._analyze_feature_importance()
    
    def _analyze_feature_importance(self) -> None:
        """Analyze feature importance from tree-based models"""
        print("\n📊 Feature Importance Analysis...")
        
        feature_names = SmartwatchFeatures.get_feature_names()
        
        # Analyze Random Forest feature importance
        if 'RandomForest' in self.models:
            rf_importance = self.models['RandomForest'].feature_importances_
            rf_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_importance
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features (Random Forest):")
            for i, (_, row) in enumerate(rf_importance_df.head(10).iterrows()):
                print(f"   {i+1:2d}. {row['feature']:25s} : {row['importance']:.3f}")
            
            # Save feature importance
            rf_importance_df.to_csv('results/smartwatch_rf_feature_importance.csv', index=False)
        
        # Analyze XGBoost feature importance
        if 'XGBoost' in self.models:
            xgb_importance = self.models['XGBoost'].feature_importances_
            xgb_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': xgb_importance
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features (XGBoost):")
            for i, (_, row) in enumerate(xgb_importance_df.head(10).iterrows()):
                print(f"   {i+1:2d}. {row['feature']:25s} : {row['importance']:.3f}")
            
            # Save feature importance
            xgb_importance_df.to_csv('results/smartwatch_xgb_feature_importance.csv', index=False)
    
    def evaluate_models(self) -> None:
        """Print comprehensive model evaluation"""
        print("\n📈 Smartwatch Stress Detection Model Performance")
        print("=" * 70)
        
        results_df = []
        
        for name, results in self.results.items():
            results_df.append({
                'Model': name,
                'CV_Accuracy': f"{results['cv_mean']:.3f} ± {results['cv_std']:.3f}",
                'Test_Accuracy': f"{results['test_accuracy']:.3f}",
                'AUC': f"{results['auc_score']:.3f}",
                'Train_Time': f"{results['train_time']:.2f}s"
            })
            
            # Detailed classification report
            print(f"\n{name} Detailed Results:")
            report = results['classification_report']
            print(f"  Baseline Precision: {report['0']['precision']:.3f}")
            print(f"  Baseline Recall:    {report['0']['recall']:.3f}")
            print(f"  Stress Precision:   {report['1']['precision']:.3f}")
            print(f"  Stress Recall:      {report['1']['recall']:.3f}")
            print(f"  F1-Score Macro:     {report['macro avg']['f1-score']:.3f}")
        
        # Summary table
        results_df = pd.DataFrame(results_df)
        print(f"\n📊 Performance Summary:")
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv('results/smartwatch_ml_results.csv', index=False)
        
        # Save detailed results
        with open('results/smartwatch_ml_detailed_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def save_models(self) -> None:
        """Save trained models and scalers"""
        print("\n💾 Saving trained models...")
        
        models_dir = Path('results/models')
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_file = models_dir / f'smartwatch_{name.lower()}_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"   Saved: {model_file}")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_file = models_dir / f'smartwatch_{name}_scaler.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"   Saved: {scaler_file}")
    
    def run_complete_pipeline(self) -> None:
        """Run the complete smartwatch ML pipeline"""
        print("🏃‍♀️ Starting Smartwatch Stress Detection Pipeline")
        print("=" * 60)
        
        # Step 1: Process data
        X, y = self.process_wesad_data()
        
        # Step 2: Train models
        self.train_models(X, y)
        
        # Step 3: Evaluate models
        self.evaluate_models()
        
        # Step 4: Save everything
        self.save_models()
        
        print("\n✅ Pipeline complete! Check 'results/' for outputs.")
        
        # Print compatibility notes
        print("\n📱 Smartwatch Compatibility Notes:")
        print("   ✅ PPG/Heart Rate: Available on all smartwatches")
        print("   ✅ HRV: Derived from PPG signal")
        print("   ✅ Accelerometer: Standard 3-axis sensor")
        print("   ✅ Temperature: Skin temperature sensor")
        print("   ✅ Activity Detection: Derived from accelerometer")
        print("   ✅ Respiratory Rate: Derived from PPG variability")
        print("   ❌ Gyroscope: Not in WESAD, would improve activity classification")
        print("   ❌ Direct EDA: Not available, but compensated by HRV features")


def main():
    """Main execution function"""
    pipeline = SmartwatchMLPipeline()
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()