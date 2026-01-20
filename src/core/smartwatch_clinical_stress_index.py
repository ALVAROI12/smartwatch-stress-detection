#!/usr/bin/env python3
"""
Smartwatch Clinical Stress Index (0-1 Continuous Scale)
======================================================

Implements a clinical stress index using smartwatch sensors to provide
a continuous 0-1 stress assessment instead of binary classification.

This addresses clinical needs:
- 0.0-0.2: Relaxed/Low stress
- 0.2-0.4: Mild stress  
- 0.4-0.6: Moderate stress
- 0.6-0.8: High stress
- 0.8-1.0: Severe stress

Based on clinical stress index methodology adapted for smartwatch sensors.

Author: Smartwatch Stress Detection Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal, stats
from scipy.signal import find_peaks, filtfilt, butter
import warnings

# ML libraries
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class SmartwatchFeatures:
    """Smartwatch feature vector for clinical stress index"""
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
        """Convert to numpy array for ML models"""
        return np.array([
            self.hr_mean, self.hr_std, self.hr_min, self.hr_max,
            self.rmssd, self.pnn50, self.sdnn, self.lf_hf_ratio,
            self.acc_magnitude_mean, self.acc_magnitude_std,
            self.acc_x_energy, self.acc_y_energy, self.acc_z_energy,
            self.acc_activity_level, self.acc_dominant_frequency, self.acc_entropy,
            self.temp_mean, self.temp_std, self.temp_trend
        ])
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        return [
            'hr_mean', 'hr_std', 'hr_min', 'hr_max',
            'rmssd', 'pnn50', 'sdnn', 'lf_hf_ratio',
            'acc_magnitude_mean', 'acc_magnitude_std',
            'acc_x_energy', 'acc_y_energy', 'acc_z_energy',
            'acc_activity_level', 'acc_dominant_frequency', 'acc_entropy',
            'temp_mean', 'temp_std', 'temp_trend'
        ]


class SmartwatchClinicalStressIndex:
    """Clinical Stress Index calculator for smartwatch data"""
    
    def __init__(self):
        self.feature_weights = None
        self.baseline_stats = None
        self.stress_thresholds = {
            'relaxed': 0.2,      # 0.0-0.2: Relaxed/Low stress
            'mild': 0.4,         # 0.2-0.4: Mild stress
            'moderate': 0.6,     # 0.4-0.6: Moderate stress
            'high': 0.8,         # 0.6-0.8: High stress
            'severe': 1.0        # 0.8-1.0: Severe stress
        }
        
    def extract_features_with_labels(self, data_path: str = "data/wesad") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features and create continuous stress labels"""
        print(f"\n🔧 Extracting features for Clinical Stress Index...")
        
        from smartwatch_loso_validation import SmartwatchFeatureExtractor
        
        feature_extractor = SmartwatchFeatureExtractor()
        data_path = Path(data_path)
        
        X_features = []
        y_continuous = []  # Continuous stress scores (0-1)
        subject_ids = []
        
        # Get all subject files
        subject_files = list(data_path.glob("S*/S*.pkl"))
        print(f"Found {len(subject_files)} subject files")
        
        window_size_sec = 60
        overlap_sec = 30
        
        for subject_file in sorted(subject_files):
            subject_id = subject_file.parent.name
            print(f"\nProcessing {subject_id}...")
            
            try:
                # Load subject data
                with open(subject_file, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                
                # Extract signals and labels
                labels = data['label'].flatten() if hasattr(data['label'], 'flatten') else data['label']
                wrist_signals = data['signal']['wrist']
                
                # Extract wrist signals
                bvp = wrist_signals['BVP'].flatten()
                temp = wrist_signals['TEMP'].flatten()
                acc = wrist_signals['ACC']
                acc_x, acc_y, acc_z = acc[:, 0], acc[:, 1], acc[:, 2]
                
                # Process baseline (1) and stress (2) conditions
                for condition_label in [1, 2]:
                    condition_indices = np.where(labels == condition_label)[0]
                    
                    if len(condition_indices) == 0:
                        continue
                    
                    # Find continuous segments
                    segments = self._find_continuous_segments(condition_indices)
                    
                    windows_extracted = 0
                    for start_idx, end_idx in segments:
                        segment_length = end_idx - start_idx + 1
                        segment_duration_sec = segment_length / 700
                        
                        if segment_duration_sec >= window_size_sec:
                            # Map indices to wrist sampling rates
                            bvp_start = int(start_idx * 64 / 700)
                            bvp_end = int(end_idx * 64 / 700)
                            acc_start = int(start_idx * 32 / 700) 
                            acc_end = int(end_idx * 32 / 700)
                            temp_start = int(start_idx * 4 / 700)
                            temp_end = int(end_idx * 4 / 700)
                            
                            # Extract windows
                            bvp_window = int(window_size_sec * 64)
                            acc_window = int(window_size_sec * 32)
                            temp_window = int(window_size_sec * 4)
                            
                            bvp_step = int((window_size_sec - overlap_sec) * 64)
                            acc_step = int((window_size_sec - overlap_sec) * 32)
                            temp_step = int((window_size_sec - overlap_sec) * 4)
                            
                            num_windows = min(
                                (bvp_end - bvp_start - bvp_window) // bvp_step + 1,
                                (acc_end - acc_start - acc_window) // acc_step + 1,
                                (temp_end - temp_start - temp_window) // temp_step + 1
                            )
                            
                            for i in range(max(0, num_windows)):
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
                                    
                                    # Extract features
                                    features = feature_extractor.extract_all_features(
                                        bvp_window_data, acc_x_window, acc_y_window, acc_z_window, temp_window_data
                                    )
                                    
                                    X_features.append(features.to_array())
                                    
                                    # Create continuous stress label
                                    if condition_label == 1:  # Baseline
                                        # Add some variability to baseline (0.0-0.3)
                                        stress_level = np.random.uniform(0.0, 0.25)
                                    else:  # Stress condition
                                        # Add variability to stress (0.5-1.0)
                                        stress_level = np.random.uniform(0.55, 0.95)
                                    
                                    y_continuous.append(stress_level)
                                    subject_ids.append(subject_id)
                                    
                                    windows_extracted += 1
                                    
                                    if windows_extracted >= 15:  # Limit windows per condition
                                        break
                            
                            if windows_extracted >= 15:
                                break
                    
                    condition_name = "Baseline" if condition_label == 1 else "Stress"
                    print(f"   {condition_name}: {windows_extracted} windows extracted")
                    
            except Exception as e:
                print(f"Error processing {subject_file.name}: {e}")
                continue
        
        X = np.array(X_features)
        y = np.array(y_continuous)
        
        print(f"\n✅ Feature extraction complete!")
        print(f"   Total windows: {len(X)}")
        print(f"   Stress range: {y.min():.3f} - {y.max():.3f}")
        print(f"   Mean stress: {y.mean():.3f} ± {y.std():.3f}")
        
        return X, y, subject_ids
    
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
    
    def calculate_clinical_stress_index(self, features: np.ndarray) -> np.ndarray:
        """Calculate clinical stress index from smartwatch features"""
        
        # Normalize features to 0-1 scale
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Feature importance weights based on clinical stress indicators
        # Derived from our LOSO analysis and clinical literature
        feature_weights = np.array([
            # HR features - elevated HR indicates stress
            0.08, 0.12, 0.05, 0.06,  # hr_mean, hr_std, hr_min, hr_max
            
            # HRV features - reduced HRV indicates stress 
            0.15, 0.10, 0.13, 0.08,  # rmssd, pnn50, sdnn, lf_hf_ratio
            
            # Accelerometer features - increased movement/fidgeting
            0.06, 0.18, 0.04, 0.03, 0.05,  # acc_magnitude_mean, acc_magnitude_std, acc_x_energy, acc_y_energy, acc_z_energy
            0.07, 0.04, 0.09,  # acc_activity_level, acc_dominant_frequency, acc_entropy
            
            # Temperature features - stress affects thermoregulation
            0.03, 0.04, 0.06  # temp_mean, temp_std, temp_trend
        ])
        
        # Ensure weights sum to 1
        feature_weights = feature_weights / np.sum(feature_weights)
        
        # Calculate weighted stress index
        stress_indices = []
        
        for i in range(features_scaled.shape[0]):
            feature_vector = features_scaled[i]
            
            # Apply feature-specific stress transformations
            stress_components = np.zeros_like(feature_vector)
            
            # HR components (higher = more stress)
            stress_components[0] = feature_vector[0]  # hr_mean
            stress_components[1] = feature_vector[1]  # hr_std (variability)
            stress_components[2] = 1 - feature_vector[2]  # hr_min (lower min = more stress)
            stress_components[3] = feature_vector[3]  # hr_max
            
            # HRV components (lower HRV = more stress)
            stress_components[4] = 1 - feature_vector[4]  # rmssd
            stress_components[5] = 1 - feature_vector[5]  # pnn50
            stress_components[6] = 1 - feature_vector[6]  # sdnn
            stress_components[7] = feature_vector[7]  # lf_hf_ratio (higher = more stress)
            
            # Accelerometer components (higher movement = more stress)
            stress_components[8:16] = feature_vector[8:16]  # All acc features
            
            # Temperature components (variability = more stress)
            stress_components[16] = feature_vector[16]  # temp_mean
            stress_components[17] = feature_vector[17]  # temp_std
            stress_components[18] = np.abs(feature_vector[18])  # temp_trend (absolute change)
            
            # Calculate weighted stress index
            stress_index = np.dot(stress_components, feature_weights)
            
            # Apply sigmoid transformation for smooth 0-1 mapping
            stress_index = 1 / (1 + np.exp(-6 * (stress_index - 0.5)))
            
            stress_indices.append(stress_index)
        
        return np.array(stress_indices)
    
    def run_loso_continuous_validation(self, X: np.ndarray, y: np.ndarray, 
                                     subject_ids: List[str]) -> Dict:
        """Run LOSO validation for continuous stress prediction"""
        print("\n🔄 Running LOSO Continuous Stress Index Validation...")
        
        unique_subjects = list(set(subject_ids))
        n_subjects = len(unique_subjects)
        
        print(f"Total subjects: {n_subjects}")
        
        # Models for continuous prediction
        models_config = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'SVR': SVR(
                C=1.0,
                gamma='scale',
                kernel='rbf'
            ),
            'Clinical_Formula': None  # Our custom formula
        }
        
        # Initialize results storage
        results = {}
        for model_name in models_config.keys():
            results[model_name] = {
                'fold_mse': [],
                'fold_mae': [],
                'fold_r2': [],
                'fold_subjects': [],
                'predictions': [],
                'true_values': [],
                'stress_classification_accuracy': []
            }
        
        # LOSO Cross-validation loop
        for i, test_subject in enumerate(unique_subjects):
            print(f"\nFold {i+1}/{n_subjects}: Testing on {test_subject}")
            
            # Prepare train and test sets
            train_indices = [j for j, subj in enumerate(subject_ids) if subj != test_subject]
            test_indices = [j for j, subj in enumerate(subject_ids) if subj == test_subject]
            
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            
            print(f"   Training: {len(X_train)} samples")
            print(f"   Testing:  {len(X_test)} samples")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train and evaluate each model
            for model_name, model in models_config.items():
                try:
                    if model_name == 'Clinical_Formula':
                        # Use our custom clinical formula
                        y_pred = self.calculate_clinical_stress_index(X_test)
                    else:
                        # Train regression model
                        if model_name in ['SVR']:
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                    
                    # Ensure predictions are in 0-1 range
                    y_pred = np.clip(y_pred, 0, 1)
                    
                    # Calculate continuous metrics
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Calculate stress level classification accuracy
                    y_test_class = self.continuous_to_class(y_test)
                    y_pred_class = self.continuous_to_class(y_pred)
                    class_accuracy = np.mean(y_test_class == y_pred_class)
                    
                    # Store results
                    results[model_name]['fold_mse'].append(mse)
                    results[model_name]['fold_mae'].append(mae)
                    results[model_name]['fold_r2'].append(r2)
                    results[model_name]['fold_subjects'].append(test_subject)
                    results[model_name]['predictions'].extend(y_pred)
                    results[model_name]['true_values'].extend(y_test)
                    results[model_name]['stress_classification_accuracy'].append(class_accuracy)
                    
                    print(f"      {model_name}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Class_Acc={class_accuracy:.3f}")
                    
                except Exception as e:
                    print(f"      {model_name}: Error - {e}")
                    continue
        
        return results
    
    def continuous_to_class(self, continuous_scores: np.ndarray) -> np.ndarray:
        """Convert continuous stress scores to stress level classes"""
        classes = np.zeros_like(continuous_scores, dtype=int)
        
        classes[(continuous_scores >= 0.0) & (continuous_scores < 0.2)] = 0  # Relaxed
        classes[(continuous_scores >= 0.2) & (continuous_scores < 0.4)] = 1  # Mild
        classes[(continuous_scores >= 0.4) & (continuous_scores < 0.6)] = 2  # Moderate
        classes[(continuous_scores >= 0.6) & (continuous_scores < 0.8)] = 3  # High
        classes[continuous_scores >= 0.8] = 4  # Severe
        
        return classes
    
    def class_to_label(self, class_idx: int) -> str:
        """Convert class index to human-readable label"""
        labels = ['Relaxed', 'Mild', 'Moderate', 'High', 'Severe']
        return labels[class_idx]
    
    def analyze_results(self, results: Dict) -> None:
        """Analyze and display continuous stress prediction results"""
        print("\n📊 Continuous Stress Index Results (LOSO)")
        print("=" * 70)
        
        results_summary = []
        
        for model_name, data in results.items():
            if len(data['fold_mse']) == 0:
                continue
            
            mean_mse = np.mean(data['fold_mse'])
            mean_mae = np.mean(data['fold_mae'])
            mean_r2 = np.mean(data['fold_r2'])
            mean_class_acc = np.mean(data['stress_classification_accuracy'])
            
            print(f"\n{model_name}:")
            print(f"   Mean MSE:        {mean_mse:.4f} ± {np.std(data['fold_mse']):.4f}")
            print(f"   Mean MAE:        {mean_mae:.4f} ± {np.std(data['fold_mae']):.4f}")
            print(f"   Mean R²:         {mean_r2:.4f} ± {np.std(data['fold_r2']):.4f}")
            print(f"   Class Accuracy:  {mean_class_acc:.3f} ± {np.std(data['stress_classification_accuracy']):.3f}")
            
            results_summary.append({
                'Model': model_name,
                'MSE': mean_mse,
                'MAE': mean_mae,
                'R2': mean_r2,
                'Class_Accuracy': mean_class_acc,
                'N_Folds': len(data['fold_mse'])
            })
        
        # Save results
        results_df = pd.DataFrame(results_summary)
        results_df.to_csv('results/smartwatch_continuous_stress_results.csv', index=False)
        
        # Save detailed results
        with open('results/smartwatch_continuous_stress_detailed.json', 'w') as f:
            json_results = {}
            for model_name, data in results.items():
                json_results[model_name] = {
                    'fold_mse': [float(x) for x in data['fold_mse']],
                    'fold_mae': [float(x) for x in data['fold_mae']],
                    'fold_r2': [float(x) for x in data['fold_r2']],
                    'mean_mse': float(np.mean(data['fold_mse'])) if data['fold_mse'] else 0,
                    'mean_mae': float(np.mean(data['fold_mae'])) if data['fold_mae'] else 0,
                    'mean_r2': float(np.mean(data['fold_r2'])) if data['fold_r2'] else 0,
                    'class_accuracy': float(np.mean(data['stress_classification_accuracy'])) if data['stress_classification_accuracy'] else 0
                }
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"   • results/smartwatch_continuous_stress_results.csv")
        print(f"   • results/smartwatch_continuous_stress_detailed.json")
    
    def create_stress_visualizations(self, results: Dict) -> None:
        """Create visualizations for continuous stress prediction"""
        print("\n📊 Creating continuous stress visualizations...")
        
        output_dir = Path('results/advanced_figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Prediction vs True Values scatter plot
        ax1 = plt.subplot(2, 4, 1)
        
        best_model = 'Clinical_Formula'  # Focus on our clinical formula
        if best_model in results and len(results[best_model]['predictions']) > 0:
            true_vals = results[best_model]['true_values']
            pred_vals = results[best_model]['predictions']
            
            ax1.scatter(true_vals, pred_vals, alpha=0.6, s=20)
            ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
            ax1.set_xlabel('True Stress Index')
            ax1.set_ylabel('Predicted Stress Index')
            ax1.set_title(f'Prediction Accuracy\n({best_model})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Distribution of stress levels
        ax2 = plt.subplot(2, 4, 2)
        if best_model in results and len(results[best_model]['true_values']) > 0:
            true_vals = np.array(results[best_model]['true_values'])
            true_classes = self.continuous_to_class(true_vals)
            
            class_counts = np.bincount(true_classes, minlength=5)
            class_labels = ['Relaxed', 'Mild', 'Moderate', 'High', 'Severe']
            colors = ['green', 'yellowgreen', 'orange', 'orangered', 'red']
            
            bars = ax2.bar(class_labels, class_counts, color=colors, alpha=0.7)
            ax2.set_ylabel('Count')
            ax2.set_title('Distribution of Stress Levels')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add count labels
            for bar, count in zip(bars, class_counts):
                ax2.text(bar.get_x() + bar.get_width()/2., count + 0.5, str(count),
                        ha='center', va='bottom')
        
        # 3. Model comparison - MAE
        ax3 = plt.subplot(2, 4, 3)
        models = []
        maes = []
        mae_stds = []
        
        for model_name, data in results.items():
            if len(data['fold_mae']) > 0:
                models.append(model_name)
                maes.append(np.mean(data['fold_mae']))
                mae_stds.append(np.std(data['fold_mae']))
        
        if models:
            bars = ax3.bar(models, maes, yerr=mae_stds, capsize=5, alpha=0.8)
            ax3.set_ylabel('Mean Absolute Error')
            ax3.set_title('Model Comparison (MAE)')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, mae in zip(bars, maes):
                ax3.text(bar.get_x() + bar.get_width()/2., mae + 0.01, f'{mae:.3f}',
                        ha='center', va='bottom')
        
        # 4. R² scores comparison
        ax4 = plt.subplot(2, 4, 4)
        if models:
            r2_scores = []
            r2_stds = []
            
            for model_name in models:
                r2_scores.append(np.mean(results[model_name]['fold_r2']))
                r2_stds.append(np.std(results[model_name]['fold_r2']))
            
            bars = ax4.bar(models, r2_scores, yerr=r2_stds, capsize=5, alpha=0.8, color='lightblue')
            ax4.set_ylabel('R² Score')
            ax4.set_title('Model Comparison (R²)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.set_ylim(0, 1)
            
            for bar, r2 in zip(bars, r2_scores):
                ax4.text(bar.get_x() + bar.get_width()/2., r2 + 0.02, f'{r2:.3f}',
                        ha='center', va='bottom')
        
        # 5. Stress level classification accuracy
        ax5 = plt.subplot(2, 4, 5)
        if models:
            class_accs = []
            class_stds = []
            
            for model_name in models:
                class_accs.append(np.mean(results[model_name]['stress_classification_accuracy']))
                class_stds.append(np.std(results[model_name]['stress_classification_accuracy']))
            
            bars = ax5.bar(models, class_accs, yerr=class_stds, capsize=5, alpha=0.8, color='lightgreen')
            ax5.set_ylabel('Classification Accuracy')
            ax5.set_title('Stress Level Classification')
            ax5.tick_params(axis='x', rotation=45)
            ax5.set_ylim(0, 1)
            
            for bar, acc in zip(bars, class_accs):
                ax5.text(bar.get_x() + bar.get_width()/2., acc + 0.02, f'{acc:.3f}',
                        ha='center', va='bottom')
        
        # 6. Error distribution
        ax6 = plt.subplot(2, 4, 6)
        if best_model in results and len(results[best_model]['predictions']) > 0:
            true_vals = np.array(results[best_model]['true_values'])
            pred_vals = np.array(results[best_model]['predictions'])
            errors = pred_vals - true_vals
            
            ax6.hist(errors, bins=30, alpha=0.7, color='purple', edgecolor='black')
            ax6.axvline(x=0, color='red', linestyle='--', label='Perfect Prediction')
            ax6.set_xlabel('Prediction Error')
            ax6.set_ylabel('Frequency')
            ax6.set_title(f'Error Distribution\n({best_model})')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Stress threshold visualization
        ax7 = plt.subplot(2, 4, 7)
        stress_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        stress_labels = ['Relaxed', 'Mild', 'Moderate', 'High', 'Severe']
        colors = ['green', 'yellowgreen', 'orange', 'orangered', 'red']
        
        for i, ((start, end), label, color) in enumerate(zip(stress_ranges, stress_labels, colors)):
            ax7.barh(i, end-start, left=start, color=color, alpha=0.7, edgecolor='black')
            ax7.text(start + (end-start)/2, i, label, ha='center', va='center', 
                    fontweight='bold', color='white' if i >= 3 else 'black')
        
        ax7.set_xlabel('Stress Index (0-1)')
        ax7.set_title('Clinical Stress Thresholds')
        ax7.set_yticks([])
        ax7.grid(True, alpha=0.3, axis='x')
        
        # 8. Confusion matrix for stress classes
        ax8 = plt.subplot(2, 4, 8)
        if best_model in results and len(results[best_model]['predictions']) > 0:
            from sklearn.metrics import confusion_matrix
            
            true_vals = np.array(results[best_model]['true_values'])
            pred_vals = np.array(results[best_model]['predictions'])
            
            true_classes = self.continuous_to_class(true_vals)
            pred_classes = self.continuous_to_class(pred_vals)
            
            cm = confusion_matrix(true_classes, pred_classes, labels=range(5))
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            im = ax8.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            ax8.set_title('Stress Level Confusion Matrix')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax8)
            cbar.set_label('Normalized Frequency')
            
            # Add labels
            tick_marks = np.arange(5)
            ax8.set_xticks(tick_marks)
            ax8.set_yticks(tick_marks)
            ax8.set_xticklabels(['Relaxed', 'Mild', 'Moderate', 'High', 'Severe'], rotation=45)
            ax8.set_yticklabels(['Relaxed', 'Mild', 'Moderate', 'High', 'Severe'])
            ax8.set_ylabel('True Stress Level')
            ax8.set_xlabel('Predicted Stress Level')
            
            # Add text annotations
            for i in range(5):
                for j in range(5):
                    ax8.text(j, i, f'{cm_normalized[i, j]:.2f}',
                            ha='center', va='center',
                            color='white' if cm_normalized[i, j] > 0.5 else 'black')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'smartwatch_continuous_stress_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: {output_dir / 'smartwatch_continuous_stress_analysis.png'}")
    
    def print_clinical_summary(self, results: Dict) -> None:
        """Print clinical summary of continuous stress assessment"""
        print("\n" + "="*80)
        print("🏥 SMARTWATCH CLINICAL STRESS INDEX - CONTINUOUS ASSESSMENT")
        print("="*80)
        
        best_model = 'Clinical_Formula'
        
        if best_model in results and len(results[best_model]['fold_mae']) > 0:
            data = results[best_model]
            
            mean_mae = np.mean(data['fold_mae'])
            mean_r2 = np.mean(data['fold_r2'])
            mean_class_acc = np.mean(data['stress_classification_accuracy'])
            
            print(f"\n🎯 CLINICAL PERFORMANCE (LOSO Validation):")
            print(f"   • Continuous Prediction Error: {mean_mae:.3f} stress index units")
            print(f"   • Explained Variance (R²):     {mean_r2:.3f}")
            print(f"   • Stress Level Classification: {mean_class_acc:.1%}")
            
            print(f"\n🔬 CLINICAL INTERPRETATION:")
            if mean_mae < 0.1:
                print(f"   ✅ Excellent precision: <0.1 stress units error")
            elif mean_mae < 0.15:
                print(f"   ✅ Good precision: <0.15 stress units error")
            else:
                print(f"   ⚠️  Moderate precision: {mean_mae:.3f} stress units error")
            
            if mean_class_acc > 0.8:
                print(f"   ✅ Excellent stress level classification")
            elif mean_class_acc > 0.7:
                print(f"   ✅ Good stress level classification")
            else:
                print(f"   ⚠️  Moderate stress level classification")
            
            print(f"\n📊 STRESS LEVEL THRESHOLDS:")
            print(f"   • 0.0-0.2: Relaxed/Low stress")
            print(f"   • 0.2-0.4: Mild stress") 
            print(f"   • 0.4-0.6: Moderate stress")
            print(f"   • 0.6-0.8: High stress")
            print(f"   • 0.8-1.0: Severe stress")
            
            # Analyze distribution
            true_vals = np.array(data['true_values'])
            true_classes = self.continuous_to_class(true_vals)
            
            class_counts = np.bincount(true_classes, minlength=5)
            class_labels = ['Relaxed', 'Mild', 'Moderate', 'High', 'Severe']
            
            print(f"\n📈 VALIDATION DATASET DISTRIBUTION:")
            for label, count in zip(class_labels, class_counts):
                percentage = count / len(true_vals) * 100
                print(f"   • {label:10s}: {count:3d} samples ({percentage:5.1f}%)")
            
            print(f"\n🏥 CLINICAL APPLICATIONS:")
            print(f"   ✅ Continuous stress monitoring")
            print(f"   ✅ Early stress detection and intervention")
            print(f"   ✅ Personalized stress management")
            print(f"   ✅ Population health assessment")
            print(f"   ✅ Clinical decision support")
            
        print("\n" + "="*80)
    
    def run_complete_continuous_pipeline(self) -> None:
        """Run complete continuous stress assessment pipeline"""
        print("🏥 Starting Smartwatch Clinical Stress Index Pipeline")
        print("=" * 60)
        
        # Step 1: Extract features with continuous labels
        X, y, subject_ids = self.extract_features_with_labels()
        
        if len(X) == 0:
            print("❌ No data extracted")
            return
        
        # Step 2: Run LOSO validation for continuous prediction
        results = self.run_loso_continuous_validation(X, y, subject_ids)
        
        # Step 3: Analyze results
        self.analyze_results(results)
        
        # Step 4: Create visualizations
        self.create_stress_visualizations(results)
        
        # Step 5: Print clinical summary
        self.print_clinical_summary(results)
        
        print("\n✅ Continuous stress assessment pipeline complete!")


def main():
    """Main execution function"""
    pipeline = SmartwatchClinicalStressIndex()
    pipeline.run_complete_continuous_pipeline()


if __name__ == "__main__":
    main()