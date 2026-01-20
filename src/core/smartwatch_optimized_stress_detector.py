#!/usr/bin/env python3
"""
Smartwatch Stress Detection with Optimized Sigma-Based Threshold Formula
========================================================================

This implementation creates a stress threshold formula using only the sensors
available on consumer smartwatches (PPG and Accelerometer), using sigma 
normalization to convert values to 0-1 range with optimized feature weights.

Available sensors on consumer smartwatches:
- PPG sensor (for heart rate and HRV)  
- Accelerometer (for motion/activity)

Author: Alvaro Ibarra
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal, stats
from scipy.signal import find_peaks
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


def _default_baseline_stats() -> Dict[str, Dict[str, float]]:
    return {
        'hr_mean': {'mean': 72.0, 'std': 12.0},
        'hr_std': {'mean': 8.0, 'std': 4.0},
        'hrv_rmssd': {'mean': 35.0, 'std': 15.0},
        'hrv_pnn50': {'mean': 18.0, 'std': 12.0},
        'acc_mean': {'mean': 0.5, 'std': 0.3},
        'acc_std': {'mean': 0.2, 'std': 0.15},
        'acc_energy': {'mean': 0.3, 'std': 0.4},
        'activity_level': {'mean': 0.3, 'std': 0.3}
    }

@dataclass
class SmartWatchSensorData:
    """Data structure for consumer smartwatch sensors only"""
    # PPG-derived features (from heart rate analysis)
    hr_mean: float = 0.0        # Average heart rate
    hr_std: float = 0.0         # Heart rate variability indicator
    hrv_rmssd: float = 0.0      # HRV - Root Mean Square of Successive Differences
    hrv_pnn50: float = 0.0      # HRV - Percentage of NN50 intervals
    
    # Accelerometer features
    acc_mean: float = 0.0       # Average acceleration magnitude
    acc_std: float = 0.0        # Movement variability (fidgeting indicator)
    acc_energy: float = 0.0     # Total movement energy
    activity_level: float = 0.0 # Overall activity classification

    def to_array(self) -> np.ndarray:
        """Convert to array for processing"""
        return np.array([
            self.hr_mean, self.hr_std, self.hrv_rmssd, self.hrv_pnn50,
            self.acc_mean, self.acc_std, self.acc_energy, self.activity_level
        ])

class SmartWatchStressDetector:
    """Optimized stress detection using only smartwatch sensors"""
    
    def __init__(self):
        # Aggregated weights from LOSO RF importances (PPG+ACC, no TEMP)
        # Axis-level energies are summed; normalized to 1.0
        self.feature_weights = np.array([
            0.038,  # hr_mean
            0.266,  # hr_std (dominant)
            0.134,  # hrv_rmssd (lower = stress)
            0.114,  # hrv_pnn50 (lower = stress)
            0.066,  # acc_mean
            0.084,  # acc_std
            0.263,  # acc_energy (x+y+z combined)
            0.035   # activity_level
        ])

        self.feature_weights = self.feature_weights / np.sum(self.feature_weights)

        # Stress thresholds on 0-1 scale
        self.stress_thresholds = {
            'relaxed': 0.25,
            'mild': 0.45,
            'moderate': 0.65,
            'high': 0.85,
            'severe': 1.0
        }

        # Clinical framing for the same numeric bands
        self.clinical_labels = {
            'relaxed': 'monitor for hypoarousal if sustained',
            'mild': 'elevated but not crisis',
            'moderate': 'heightened stress, consider intervention',
            'high': 'high stress, intervene',
            'severe': 'crisis intervention recommended'
        }
    
    def extract_ppg_features(self, ppg_signal: np.ndarray, sampling_rate: float = 64.0) -> Dict[str, float]:
        """Extract heart rate and HRV features from PPG signal"""
        if len(ppg_signal) == 0:
            return {'hr_mean': 70.0, 'hr_std': 5.0, 'hrv_rmssd': 30.0, 'hrv_pnn50': 15.0}
        
        # Basic preprocessing
        # Bandpass filter for PPG (0.5-8 Hz)
        nyquist = sampling_rate / 2
        low = 0.5 / nyquist
        high = 8.0 / nyquist
        b, a = signal.butter(3, [low, high], btype='band')
        ppg_filtered = signal.filtfilt(b, a, ppg_signal)
        
        # Peak detection for heart beats
        # Adaptive threshold based on signal characteristics
        peaks, properties = find_peaks(
            ppg_filtered, 
            height=np.mean(ppg_filtered) + 0.3 * np.std(ppg_filtered),
            distance=int(sampling_rate * 0.4)  # Minimum 0.4s between beats (150 BPM max)
        )
        
        if len(peaks) < 3:
            return {'hr_mean': 70.0, 'hr_std': 5.0, 'hrv_rmssd': 30.0, 'hrv_pnn50': 15.0}
        
        # Calculate RR intervals (time between beats)
        rr_intervals = np.diff(peaks) / sampling_rate * 1000  # Convert to milliseconds
        
        # Remove outliers (RR intervals outside 300-2000ms)
        rr_intervals = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]
        
        if len(rr_intervals) < 2:
            return {'hr_mean': 70.0, 'hr_std': 5.0, 'hrv_rmssd': 30.0, 'hrv_pnn50': 15.0}
        
        # Heart rate statistics
        hr_instantaneous = 60000 / rr_intervals  # Convert to BPM
        hr_mean = np.mean(hr_instantaneous)
        hr_std = np.std(hr_instantaneous)
        
        # HRV features
        # RMSSD - Root Mean Square of Successive Differences
        rr_diffs = np.diff(rr_intervals)
        hrv_rmssd = np.sqrt(np.mean(rr_diffs ** 2))
        
        # pNN50 - Percentage of successive RR intervals differing by > 50ms
        nn50_count = np.sum(np.abs(rr_diffs) > 50)
        hrv_pnn50 = (nn50_count / len(rr_diffs)) * 100 if len(rr_diffs) > 0 else 0.0
        
        return {
            'hr_mean': hr_mean,
            'hr_std': hr_std,
            'hrv_rmssd': hrv_rmssd,
            'hrv_pnn50': hrv_pnn50
        }
    
    def extract_accelerometer_features(self, acc_x: np.ndarray, acc_y: np.ndarray, 
                                     acc_z: np.ndarray, sampling_rate: float = 32.0) -> Dict[str, float]:
        """Extract movement features from 3-axis accelerometer data"""
        if len(acc_x) == 0 or len(acc_y) == 0 or len(acc_z) == 0:
            return {'acc_mean': 1.0, 'acc_std': 0.1, 'acc_energy': 0.1, 'activity_level': 0.0}
        
        # Calculate acceleration magnitude
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Remove gravity bias (typical Earth gravity ~9.8 m/s²)
        acc_magnitude = np.abs(acc_magnitude - np.median(acc_magnitude))
        
        # Basic statistics
        acc_mean = np.mean(acc_magnitude)
        acc_std = np.std(acc_magnitude)  # Important for fidgeting detection
        
        # Energy in movement (sum of squared deviations)
        acc_energy = np.sum(acc_magnitude**2) / len(acc_magnitude)
        
        # Activity level classification
        # Based on movement intensity thresholds
        low_activity_threshold = 0.1   # m/s²
        high_activity_threshold = 2.0  # m/s²
        
        if acc_mean < low_activity_threshold:
            activity_level = 0.0  # Sedentary
        elif acc_mean < high_activity_threshold:
            activity_level = 0.5  # Light activity
        else:
            activity_level = 1.0  # Active
        
        return {
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'acc_energy': acc_energy,
            'activity_level': activity_level
        }
    
    def _z_score(self, value: float, mean: float, std: float) -> float:
        """Plain z-score with guard against zero std."""
        if std == 0:
            return 0.0
        return (value - mean) / std

    def _double_sigmoid(self, u: float, k: float = 1.8, b: float = 0.5) -> float:
        """Symmetric mapping around baseline; sharp near ±2σ."""
        u = np.clip(u, -3.0, 3.0)
        if u >= 0:
            p = 1.0 / (1.0 + np.exp(-k * (u - b)))
        else:
            p = 1.0 - 1.0 / (1.0 + np.exp(-k * (-u - b)))
        return float(np.clip(p, 0.0, 1.0))
    
    def calculate_stress_index(self, sensor_data: SmartWatchSensorData, 
                             baseline_stats: Optional[Dict] = None) -> float:
        """
        Calculate stress index using sigma normalization and feature weights
        Returns value between 0.0 (no stress) and 1.0 (maximum stress)
        """
        
        # Default baseline statistics (population averages)
        if baseline_stats is None:
            baseline_stats = _default_baseline_stats()
        
        # Get raw feature values
        features = sensor_data.to_array()
        feature_names = ['hr_mean', 'hr_std', 'hrv_rmssd', 'hrv_pnn50',
                        'acc_mean', 'acc_std', 'acc_energy', 'activity_level']
        
        # z-scores with directional sign
        z_scores = np.zeros_like(features)
        invert_features = {'hrv_rmssd', 'hrv_pnn50'}
        for i, (feature_val, feature_name) in enumerate(zip(features, feature_names)):
            baseline = baseline_stats[feature_name]
            z_val = self._z_score(feature_val, baseline['mean'], baseline['std'])
            if feature_name in invert_features:
                z_val = -z_val  # lower HRV = more stress
            z_scores[i] = z_val

        # Weighted sum of signed z-scores
        weighted_sum = float(np.dot(z_scores, self.feature_weights))

        # Double-sigmoid mapping to 0-1
        final_stress_index = self._double_sigmoid(weighted_sum)

        return final_stress_index
    
    def classify_stress_level(self, stress_index: float) -> str:
        """Convert stress index to categorical stress level"""
        if stress_index < self.stress_thresholds['relaxed']:
            return 'relaxed'
        elif stress_index < self.stress_thresholds['mild']:
            return 'mild'
        elif stress_index < self.stress_thresholds['moderate']:
            return 'moderate'
        elif stress_index < self.stress_thresholds['high']:
            return 'high'
        else:
            return 'severe'
    
    def get_stress_report(self, sensor_data: SmartWatchSensorData, 
                         baseline_stats: Optional[Dict] = None) -> Dict:
        """Generate comprehensive stress assessment report"""
        resolved_baseline = baseline_stats or _default_baseline_stats()

        # Compute z-scores with direction for contributions and confidence
        features = sensor_data.to_array()
        feature_names = ['hr_mean', 'hr_std', 'hrv_rmssd', 'hrv_pnn50',
                        'acc_mean', 'acc_std', 'acc_energy', 'activity_level']
        invert_features = {'hrv_rmssd', 'hrv_pnn50'}
        z_scores = []
        for value, name in zip(features, feature_names):
            base = resolved_baseline[name]
            z_val = self._z_score(value, base['mean'], base['std'])
            if name in invert_features:
                z_val = -z_val
            z_scores.append(z_val)
        weighted_sum = float(np.dot(z_scores, self.feature_weights))

        # Reuse main stress calculation
        stress_index = self.calculate_stress_index(sensor_data, resolved_baseline)
        stress_level = self.classify_stress_level(stress_index)
        
        feature_contributions = {}
        for name, weight, z_val, value in zip(feature_names, self.feature_weights, z_scores, features):
            feature_contributions[name] = {
                'value': value,
                'z_score': z_val,
                'weight': weight,
                'contribution': weight * z_val
            }
        
        return {
            'stress_index': stress_index,
            'stress_level': stress_level,
            'clinical_label': self.clinical_labels.get(stress_level, ''),
            'confidence': min(1.0, 0.5 + min(1.5, abs(weighted_sum)) / 3.0),
            'features': feature_contributions,
            'recommendations': self._get_recommendations(stress_level),
            'sensor_availability': {
                'ppg_sensor': True,
                'accelerometer': True,
                'eda_sensor': False,
                'temperature_sensor': False,
                'total_sensors_used': 2
            }
        }
    
    def _get_recommendations(self, stress_level: str) -> List[str]:
        """Get actionable recommendations based on stress level"""
        recommendations = {
            'relaxed': [
                "Maintain current activities",
                "Consider light exercise to maintain wellness",
                "Continue good sleep habits"
            ],
            'mild': [
                "Take deep breaths (4-7-8 breathing technique)",
                "Consider a short walk or stretching",
                "Stay hydrated and avoid excessive caffeine"
            ],
            'moderate': [
                "Practice mindfulness or meditation (5-10 minutes)",
                "Take a longer break from current activities",
                "Consider talking to someone about what's bothering you",
                "Limit stimulants like caffeine"
            ],
            'high': [
                "Stop current stressful activities if possible",
                "Practice progressive muscle relaxation",
                "Consider speaking with a healthcare provider",
                "Ensure adequate sleep tonight"
            ],
            'severe': [
                "Seek immediate support from friends, family, or professionals",
                "Remove yourself from stressful situations if safe to do so",
                "Practice grounding techniques (5-4-3-2-1 method)",
                "Consider professional mental health support"
            ]
        }
        
        return recommendations.get(stress_level, [])

def demo_smartwatch_stress_detection():
    """Demonstration of smartwatch stress detection system"""
    
    print("🎯 SmartWatch Stress Detection System Demo")
    print("=" * 60)
    print("Using ONLY sensors available on consumer smartwatches:")
    print("• PPG sensor (Heart Rate & HRV)")
    print("• 3-axis Accelerometer (Motion & Fidgeting)")
    print()
    
    # Initialize detector
    detector = SmartWatchStressDetector()
    
    # Demo scenarios
    scenarios = [
        {
            'name': 'Relaxed State',
            'description': 'Person sitting calmly, reading',
            'data': SmartWatchSensorData(
                hr_mean=68.0, hr_std=6.0, hrv_rmssd=42.0, hrv_pnn50=22.0,
                acc_mean=0.2, acc_std=0.1, acc_energy=0.05, activity_level=0.1
            )
        },
        {
            'name': 'Mild Stress',
            'description': 'Light work pressure, minor deadline',
            'data': SmartWatchSensorData(
                hr_mean=78.0, hr_std=9.0, hrv_rmssd=28.0, hrv_pnn50=12.0,
                acc_mean=0.4, acc_std=0.3, acc_energy=0.2, activity_level=0.3
            )
        },
        {
            'name': 'High Stress',
            'description': 'Important presentation, high pressure',
            'data': SmartWatchSensorData(
                hr_mean=95.0, hr_std=15.0, hrv_rmssd=18.0, hrv_pnn50=5.0,
                acc_mean=0.8, acc_std=0.7, acc_energy=0.8, activity_level=0.6
            )
        },
        {
            'name': 'Severe Stress',
            'description': 'Panic response, extreme anxiety',
            'data': SmartWatchSensorData(
                hr_mean=110.0, hr_std=20.0, hrv_rmssd=12.0, hrv_pnn50=2.0,
                acc_mean=1.2, acc_std=1.0, acc_energy=1.5, activity_level=0.8
            )
        }
    ]
    
    # Test each scenario
    for scenario in scenarios:
        print(f"📊 Scenario: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        
        report = detector.get_stress_report(scenario['data'])
        
        print(f"   Stress Index: {report['stress_index']:.3f}")
        print(f"   Stress Level: {report['stress_level'].upper()}")
        print(f"   Confidence: {report['confidence']:.2f}")
        
        # Show top contributing features
        top_features = sorted(report['features'].items(), 
                            key=lambda x: x[1]['contribution'], reverse=True)[:3]
        print("   Top Contributing Features:")
        for feature, data in top_features:
            print(f"     • {feature}: {data['value']:.2f} (weight: {data['weight']:.2f})")
        
        print(f"   Recommendations:")
        for rec in report['recommendations'][:2]:  # Show top 2
            print(f"     • {rec}")
        print()
    
    # Show feature weights
    print("⚖️  Feature Importance Weights:")
    feature_names = ['hr_mean', 'hr_std', 'hrv_rmssd', 'hrv_pnn50',
                    'acc_mean', 'acc_std', 'acc_energy', 'activity_level']
    
    print("   PPG/Heart Rate Features (60% total):")
    for i in range(4):
        print(f"     • {feature_names[i]}: {detector.feature_weights[i]:.3f}")
    
    print("   Accelerometer Features (40% total):")
    for i in range(4, 8):
        print(f"     • {feature_names[i]}: {detector.feature_weights[i]:.3f}")
    
    print()
    print("✨ Key Advantages of This Approach:")
    print("   • Works with ANY consumer smartwatch")
    print("   • Real-time processing capability")
    print("   • Sigma normalization handles individual differences")
    print("   • Evidence-based feature weights")
    print("   • Actionable stress level recommendations")
    print("   • Privacy-preserving (no cloud processing needed)")

if __name__ == "__main__":
    demo_smartwatch_stress_detection()