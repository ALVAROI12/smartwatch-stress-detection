"""
WESAD Data Processor
====================

This module handles the preprocessing and feature extraction from WESAD dataset.

Author: [Your Name]
Date: 2025
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
from scipy import signal
from scipy.stats import skew, kurtosis
import sys
import os

warnings.filterwarnings('ignore')

class WESADProcessor:
    """
    Main class for processing WESAD dataset
    """
    
    def __init__(self, data_path="data/wesad", output_path="data/processed"):
        """
        Initialize the WESAD processor
        
        Parameters:
        -----------
        data_path : str
            Path to WESAD dataset directory
        output_path : str
            Path to save processed data
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.fs = 700  # WESAD sampling frequency
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 WESAD Processor initialized")
        print(f"   Data path: {self.data_path}")
        print(f"   Output path: {self.output_path}")
    
    def process_single_subject(self, subject_path, window_size_sec=180, overlap_sec=90):
        """
        Process single WESAD subject with robust windowing logic
        
        Parameters:
        -----------
        subject_path : Path
            Path to subject .pkl file
        window_size_sec : int
            Window size in seconds (default: 180 = 3 minutes)
        overlap_sec : int
            Overlap in seconds (default: 90 = 50% overlap)
        
        Returns:
        --------
        list : List of extracted windows with features
        """
        
        # Convert to samples
        window_size = window_size_sec * self.fs
        overlap = overlap_sec * self.fs
        step_size = window_size - overlap
        
        subject_id = Path(subject_path).stem
        print(f"\n🔄 Processing {subject_id}")
        print(f"   Window: {window_size_sec}s ({window_size:,} samples)")
        print(f"   Step: {step_size:,} samples")
        
        try:
            # Load data
            with open(subject_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # Extract labels and signals
            labels = data['label']
            if hasattr(labels, 'flatten'):
                labels = labels.flatten()
            
            chest_signals = data['signal']['chest']
            wrist_signals = data['signal']['wrist']
            
            print(f"   📊 Total samples: {len(labels):,}")
            
            # Extract signals
            ecg = chest_signals['ECG'].flatten()
            eda = chest_signals['EDA'].flatten()
            resp = chest_signals['Resp'].flatten()
            acc_chest = chest_signals['ACC']
            
            bvp = wrist_signals['BVP'].flatten()
            eda_wrist = wrist_signals['EDA'].flatten()
            temp = wrist_signals['TEMP'].flatten()
            acc_wrist = wrist_signals['ACC']
            
            # Process conditions
            conditions = {'baseline': 1, 'stress': 2}
            all_windows = []
            
            for condition_name, label_value in conditions.items():
                print(f"\n   🔍 Processing {condition_name} (label {label_value})")
                
                # Find all indices for this condition
                condition_indices = np.where(labels == label_value)[0]
                
                if len(condition_indices) == 0:
                    print(f"      ❌ No {condition_name} data found")
                    continue
                
                print(f"      📊 Found: {len(condition_indices):,} samples ({len(condition_indices)/self.fs/60:.1f} min)")
                
                # Find continuous segments
                segments = self._find_continuous_segments(condition_indices)
                print(f"      📦 Segments: {len(segments)}")
                
                condition_windows = 0
                
                # Process each segment
                for segment_idx, (start_idx, end_idx) in enumerate(segments):
                    segment_length = end_idx - start_idx + 1
                    segment_duration = segment_length / self.fs / 60
                    
                    print(f"         Segment {segment_idx + 1}: {start_idx}-{end_idx} ({segment_duration:.1f} min)")
                    
                    # Check if segment is long enough for windowing
                    if segment_length >= window_size:
                        print(f"         ✅ Long enough for windowing")
                        
                        # Extract windows with sliding window approach
                        window_count = 0
                        
                        for window_start in range(start_idx, end_idx - window_size + 1, step_size):
                            window_end = window_start + window_size
                            
                            if window_end <= end_idx + 1:
                                
                                # Check window purity
                                window_labels = labels[window_start:window_end]
                                condition_purity = np.sum(window_labels == label_value) / len(window_labels)
                                
                                if condition_purity > 0.95:  # 95% purity threshold
                                    
                                    try:
                                        # Extract signals for this window
                                        window_data = {
                                            'ecg': ecg[window_start:window_end],
                                            'eda': eda[window_start:window_end],
                                            'resp': resp[window_start:window_end],
                                            'bvp': bvp[window_start:window_end],
                                            'temp': temp[window_start:window_end],
                                            'acc_chest': acc_chest[window_start:window_end, :],
                                            'acc_wrist': acc_wrist[window_start:window_end, :]
                                        }
                                        
                                        # Extract features
                                        features = self._extract_features(window_data)
                                        
                                        if features is not None:
                                            # Add metadata
                                            features.update({
                                                'subject_id': subject_id,
                                                'condition': condition_name,
                                                'label': 0 if condition_name == 'baseline' else 1,
                                                'window_start': int(window_start),
                                                'window_end': int(window_end),
                                                'segment_idx': segment_idx,
                                                'window_duration_sec': window_size_sec,
                                                'purity': float(condition_purity)
                                            })
                                            
                                            all_windows.append(features)
                                            window_count += 1
                                            condition_windows += 1
                                    
                                    except Exception as e:
                                        print(f"            ❌ Feature extraction failed: {e}")
                                        continue
                            else:
                                break
                        
                        print(f"         ✅ Extracted {window_count} windows from this segment")
                    else:
                        print(f"         ❌ Too short (need {window_size:,}, have {segment_length:,})")
                
                print(f"      🎯 Total {condition_name} windows: {condition_windows}")
            
            # Final summary for this subject
            baseline_count = sum(1 for w in all_windows if w['condition'] == 'baseline')
            stress_count = sum(1 for w in all_windows if w['condition'] == 'stress')
            
            print(f"\n   🎯 Subject {subject_id} Summary:")
            print(f"      Total windows: {len(all_windows)}")
            print(f"      Baseline: {baseline_count}")
            print(f"      Stress: {stress_count}")
            
            if stress_count > 0:
                print(f"      🎉 SUCCESS - Stress windows extracted!")
            else:
                print(f"      ❌ FAILED - No stress windows")
            
            return all_windows
            
        except Exception as e:
            print(f"   ❌ Error processing {subject_id}: {e}")
            return []
    
    def _find_continuous_segments(self, indices):
        """Find continuous segments in label indices"""
        if len(indices) == 0:
            return []
        
        segments = []
        current_start = indices[0]
        current_end = indices[0]
        
        for i in range(1, len(indices)):
            if indices[i] - indices[i-1] == 1:  # Consecutive
                current_end = indices[i]
            else:  # Gap found
                segments.append((current_start, current_end))
                current_start = indices[i]
                current_end = indices[i]
        
        # Add the final segment
        segments.append((current_start, current_end))
        
        return segments
    
    def _extract_features(self, window_data):
        """Extract comprehensive features from physiological signals"""
        
        try:
            features = {}
            
            # 1. Basic statistical features for each signal
            for signal_name, signal_data in [
                ('ecg', window_data['ecg']), 
                ('eda', window_data['eda']), 
                ('bvp', window_data['bvp']), 
                ('temp', window_data['temp']),
                ('resp', window_data['resp'])
            ]:
                try:
                    features.update({
                        f'{signal_name}_mean': float(np.mean(signal_data)),
                        f'{signal_name}_std': float(np.std(signal_data)),
                        f'{signal_name}_max': float(np.max(signal_data)),
                        f'{signal_name}_min': float(np.min(signal_data)),
                        f'{signal_name}_median': float(np.median(signal_data)),
                        f'{signal_name}_range': float(np.max(signal_data) - np.min(signal_data))
                    })
                except Exception as e:
                    # Use default values if extraction fails
                    features.update({
                        f'{signal_name}_mean': 0.0,
                        f'{signal_name}_std': 0.0,
                        f'{signal_name}_max': 0.0,
                        f'{signal_name}_min': 0.0,
                        f'{signal_name}_median': 0.0,
                        f'{signal_name}_range': 0.0
                    })
            
            # 2. Accelerometer features
            try:
                acc_chest_mag = np.sqrt(np.sum(window_data['acc_chest']**2, axis=1))
                acc_wrist_mag = np.sqrt(np.sum(window_data['acc_wrist']**2, axis=1))
                
                features.update({
                    'acc_chest_mean': float(np.mean(acc_chest_mag)),
                    'acc_chest_std': float(np.std(acc_chest_mag)),
                    'acc_wrist_mean': float(np.mean(acc_wrist_mag)),
                    'acc_wrist_std': float(np.std(acc_wrist_mag)),
                    'movement_intensity': float(np.mean(acc_wrist_mag)),
                    'movement_variance': float(np.var(acc_wrist_mag))
                })
            except Exception as e:
                features.update({
                    'acc_chest_mean': 0.0,
                    'acc_chest_std': 0.0,
                    'acc_wrist_mean': 0.0,
                    'acc_wrist_std': 0.0,
                    'movement_intensity': 0.0,
                    'movement_variance': 0.0
                })
            
            # 3. Simple HRV features from BVP
            try:
                hrv_features = self._calculate_hrv_simple(window_data['bvp'])
                features.update(hrv_features)
            except Exception as e:
                # Default HRV values
                features.update({
                    'hr_estimate': 70.0,
                    'hr_variability': 5.0
                })
            
            return features
            
        except Exception as e:
            print(f"         ❌ Feature extraction error: {e}")
            return None
    
    def _calculate_hrv_simple(self, bvp_signal):
        """Calculate simple HR features from BVP signal"""
        
        try:
            # Simple peak detection in BVP
            peaks, _ = signal.find_peaks(bvp_signal, distance=20)  # Min distance between peaks
            
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / 64  # BVP sampled at 64Hz
                hr_estimate = 60 / np.mean(peak_intervals) if len(peak_intervals) > 0 else 70.0
                hr_variability = np.std(60 / peak_intervals) if len(peak_intervals) > 1 else 5.0
                
                return {
                    'hr_estimate': float(hr_estimate),
                    'hr_variability': float(hr_variability)
                }
            else:
                return {
                    'hr_estimate': 70.0,
                    'hr_variability': 5.0
                }
                
        except Exception as e:
            return {
                'hr_estimate': 70.0,
                'hr_variability': 5.0
            }
    
    def process_all_subjects(self, window_size_sec=180, overlap_sec=90):
        """
        Process all WESAD subjects
        
        Parameters:
        -----------
        window_size_sec : int
            Window size in seconds
        overlap_sec : int
            Overlap in seconds
        
        Returns:
        --------
        list : All extracted windows
        """
        
        print("🚀 WESAD PROCESSING - ORGANIZED VERSION")
        print("=" * 50)
        print(f"📂 Data directory: {self.data_path}")
        print(f"⚙️  Window size: {window_size_sec} seconds")
        print(f"⚙️  Overlap: {overlap_sec} seconds")
        
        all_windows = []
        processed_subjects = []
        
        # Find all subject files
        subject_files = list(self.data_path.glob("S*/S*.pkl"))
        
        print(f"📂 Found {len(subject_files)} subject files")
        
        for subject_file in sorted(subject_files):
            try:
                print(f"\n{'='*60}")
                windows = self.process_single_subject(subject_file, window_size_sec, overlap_sec)
                if windows:
                    all_windows.extend(windows)
                    processed_subjects.append(subject_file.stem)
                else:
                    print(f"   ❌ No windows extracted from {subject_file.stem}")
            except Exception as e:
                print(f"❌ Error processing {subject_file.stem}: {e}")
        
        # Save results
        if all_windows:
            output_file = self.output_path / "wesad_features.json"
            
            with open(output_file, 'w') as f:
                json.dump(all_windows, f, indent=2)
            
            # Summary statistics
            baseline_windows = [w for w in all_windows if w['condition'] == 'baseline']
            stress_windows = [w for w in all_windows if w['condition'] == 'stress']
            
            print(f"\n🎯 FINAL RESULTS")
            print("=" * 20)
            print(f"Subjects processed: {len(processed_subjects)}")
            print(f"Total windows: {len(all_windows)}")
            print(f"Baseline windows: {len(baseline_windows)}")
            print(f"Stress windows: {len(stress_windows)}")
            
            if len(all_windows) > 0:
                stress_percentage = (len(stress_windows) / len(all_windows)) * 100
                baseline_percentage = (len(baseline_windows) / len(all_windows)) * 100
                print(f"Class balance: {baseline_percentage:.1f}% baseline, {stress_percentage:.1f}% stress")
            
            print(f"Results saved to: {output_file}")
            
            if len(stress_windows) > 0:
                print(f"\n🎉 SUCCESS! {len(stress_windows)} stress windows extracted!")
                print(f"📊 Ready for machine learning!")
            else:
                print(f"\n❌ Still no stress windows extracted!")
            
            return all_windows
        else:
            print(f"\n❌ No windows extracted at all!")
            return []

# Main execution function
def main():
    """Main function to run WESAD processing"""
    
    # Initialize processor with proper paths
    processor = WESADProcessor(
        data_path="data/wesad",
        output_path="data/processed"
    )
    
    # Process all subjects
    windows = processor.process_all_subjects(
        window_size_sec=180,  # 3 minutes
        overlap_sec=90        # 50% overlap
    )
    
    return windows

if __name__ == "__main__":
    windows = main()