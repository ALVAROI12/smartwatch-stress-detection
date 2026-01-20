"""
PPG Signal Preprocessing Module

This module provides comprehensive preprocessing functions for photoplethysmography (PPG) signals
including filtering, artifact detection, peak detection, and RR interval extraction.

Based on research findings for optimal PPG processing in stress detection applications.

Author: Research Team
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, butter, sosfilt, sosfiltfilt
import logging
from typing import Tuple, Dict, List, Optional, Union
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# PPG processing constants
DEFAULT_SAMPLING_RATE = 64  # Hz (WESAD standard)
PPG_FILTER_PARAMS = {
    'filter_type': 'chebyshev2',
    'order': 4,
    'lowcut': 0.5,   # Hz
    'highcut': 5.0,  # Hz
    'rs': 20         # stopband attenuation in dB
}

RR_INTERVAL_LIMITS = {
    'min_rr': 500,   # ms (120 BPM max)
    'max_rr': 1200,  # ms (50 BPM min)
    'artifact_threshold': 20  # percentage for artifact detection
}


class PPGPreprocessor:
    """
    A comprehensive PPG signal preprocessing class.
    
    This class implements state-of-the-art PPG preprocessing techniques
    for stress detection applications, including filtering, artifact removal,
    and peak detection.
    """
    
    def __init__(self, sampling_rate: float = DEFAULT_SAMPLING_RATE):
        """
        Initialize the PPG preprocessor.
        
        Parameters
        ----------
        sampling_rate : float, default 64
            Sampling rate of the PPG signal in Hz
        """
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
        
        logger.info(f"Initialized PPG preprocessor with sampling rate: {sampling_rate} Hz")
    
    def design_filter(self, 
                     lowcut: float = PPG_FILTER_PARAMS['lowcut'],
                     highcut: float = PPG_FILTER_PARAMS['highcut'],
                     filter_type: str = PPG_FILTER_PARAMS['filter_type'],
                     order: int = PPG_FILTER_PARAMS['order']) -> np.ndarray:
        """
        Design a bandpass filter for PPG signal preprocessing.
        
        Parameters
        ----------
        lowcut : float, default 0.5
            Low cutoff frequency in Hz
        highcut : float, default 5.0
            High cutoff frequency in Hz
        filter_type : str, default 'chebyshev2'
            Filter type ('chebyshev2', 'butter', 'ellip')
        order : int, default 4
            Filter order
            
        Returns
        -------
        np.ndarray
            Second-order sections representation of the filter
        """
        low = lowcut / self.nyquist
        high = highcut / self.nyquist
        
        if filter_type == 'chebyshev2':
            sos = signal.cheby2(order, PPG_FILTER_PARAMS['rs'], [low, high], 
                              btype='band', output='sos')
        elif filter_type == 'butter':
            sos = signal.butter(order, [low, high], btype='band', output='sos')
        elif filter_type == 'ellip':
            sos = signal.ellip(order, 1, PPG_FILTER_PARAMS['rs'], [low, high], 
                             btype='band', output='sos')
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        
        return sos
    
    def filter_signal(self, ppg_signal: np.ndarray, 
                     zero_phase: bool = True) -> np.ndarray:
        """
        Apply bandpass filtering to PPG signal.
        
        Parameters
        ----------
        ppg_signal : np.ndarray
            Raw PPG signal
        zero_phase : bool, default True
            Whether to use zero-phase filtering (filtfilt vs lfilter)
            
        Returns
        -------
        np.ndarray
            Filtered PPG signal
        """
        if len(ppg_signal) == 0:
            logger.warning("Empty PPG signal provided for filtering")
            return ppg_signal
        
        # Remove NaN values
        if np.any(np.isnan(ppg_signal)):
            logger.warning("NaN values detected in PPG signal, interpolating...")
            ppg_signal = self._interpolate_nan(ppg_signal)
        
        # Design filter
        sos = self.design_filter()
        
        # Apply filter
        if zero_phase:
            filtered_signal = sosfiltfilt(sos, ppg_signal)
        else:
            filtered_signal = sosfilt(sos, ppg_signal)
        
        logger.info("PPG signal filtering completed")
        return filtered_signal
    
    def _interpolate_nan(self, signal: np.ndarray) -> np.ndarray:
        """
        Interpolate NaN values in signal using linear interpolation.
        
        Parameters
        ----------
        signal : np.ndarray
            Signal with potential NaN values
            
        Returns
        -------
        np.ndarray
            Signal with interpolated values
        """
        mask = ~np.isnan(signal)
        
        if not np.any(mask):
            logger.error("Signal contains only NaN values")
            return signal
        
        indices = np.arange(len(signal))
        signal[~mask] = np.interp(indices[~mask], indices[mask], signal[mask])
        
        return signal
    
    def detect_peaks(self, ppg_signal: np.ndarray, 
                    min_distance: Optional[float] = None,
                    height_threshold: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        """
        Detect peaks in filtered PPG signal.
        
        Parameters
        ----------
        ppg_signal : np.ndarray
            Filtered PPG signal
        min_distance : float, optional
            Minimum distance between peaks in seconds (default: 0.4s)
        height_threshold : float, optional
            Minimum peak height (default: automatic)
            
        Returns
        -------
        Tuple[np.ndarray, Dict]
            Peak indices and peak properties
        """
        if min_distance is None:
            min_distance = 0.4  # seconds (150 BPM max)
        
        min_distance_samples = int(min_distance * self.sampling_rate)
        
        # Automatic height threshold if not provided
        if height_threshold is None:
            height_threshold = np.percentile(ppg_signal, 60)
        
        # Find peaks
        peaks, properties = find_peaks(
            ppg_signal,
            height=height_threshold,
            distance=min_distance_samples
        )
        
        logger.info(f"Detected {len(peaks)} peaks in PPG signal")
        return peaks, properties
    
    def extract_rr_intervals(self, peak_indices: np.ndarray) -> np.ndarray:
        """
        Extract RR intervals from peak indices.
        
        Parameters
        ----------
        peak_indices : np.ndarray
            Indices of detected peaks
            
        Returns
        -------
        np.ndarray
            RR intervals in milliseconds
        """
        if len(peak_indices) < 2:
            logger.warning("Insufficient peaks for RR interval calculation")
            return np.array([])
        
        # Calculate RR intervals in samples
        rr_samples = np.diff(peak_indices)
        
        # Convert to milliseconds
        rr_intervals = (rr_samples / self.sampling_rate) * 1000
        
        logger.info(f"Extracted {len(rr_intervals)} RR intervals")
        return rr_intervals
    
    def detect_artifacts(self, rr_intervals: np.ndarray,
                        threshold_percent: float = RR_INTERVAL_LIMITS['artifact_threshold']) -> np.ndarray:
        """
        Detect artifacts in RR intervals using percentage threshold method.
        
        Parameters
        ----------
        rr_intervals : np.ndarray
            RR intervals in milliseconds
        threshold_percent : float, default 20
            Percentage threshold for artifact detection
            
        Returns
        -------
        np.ndarray
            Boolean mask indicating artifacts (True = artifact)
        """
        if len(rr_intervals) == 0:
            return np.array([], dtype=bool)
        
        # Calculate local averages (5-point moving average)
        window_size = min(5, len(rr_intervals))
        padded_rr = np.pad(rr_intervals, (window_size//2, window_size//2), mode='edge')
        local_averages = np.convolve(padded_rr, np.ones(window_size)/window_size, mode='valid')
        
        # Calculate percentage differences
        percent_diff = np.abs(rr_intervals - local_averages) / local_averages * 100
        
        # Identify artifacts
        artifacts = percent_diff > threshold_percent
        
        # Also check physiological limits
        physiological_artifacts = (
            (rr_intervals < RR_INTERVAL_LIMITS['min_rr']) |
            (rr_intervals > RR_INTERVAL_LIMITS['max_rr'])
        )
        
        combined_artifacts = artifacts | physiological_artifacts
        
        artifact_count = np.sum(combined_artifacts)
        artifact_percentage = (artifact_count / len(rr_intervals)) * 100
        
        logger.info(f"Detected {artifact_count} artifacts ({artifact_percentage:.2f}%)")
        return combined_artifacts
    
    def remove_artifacts(self, rr_intervals: np.ndarray,
                        artifact_mask: np.ndarray,
                        method: str = 'remove') -> np.ndarray:
        """
        Remove or interpolate artifacts in RR intervals.
        
        Parameters
        ----------
        rr_intervals : np.ndarray
            RR intervals in milliseconds
        artifact_mask : np.ndarray
            Boolean mask indicating artifacts
        method : str, default 'remove'
            Artifact handling method ('remove' or 'interpolate')
            
        Returns
        -------
        np.ndarray
            Cleaned RR intervals
        """
        if len(rr_intervals) == 0:
            return rr_intervals
        
        if method == 'remove':
            cleaned_rr = rr_intervals[~artifact_mask]
        elif method == 'interpolate':
            cleaned_rr = rr_intervals.copy()
            if np.any(artifact_mask):
                # Linear interpolation for artifacts
                valid_indices = np.where(~artifact_mask)[0]
                artifact_indices = np.where(artifact_mask)[0]
                
                if len(valid_indices) >= 2:
                    interp_func = interp1d(valid_indices, rr_intervals[valid_indices],
                                         kind='linear', bounds_error=False,
                                         fill_value='extrapolate')
                    cleaned_rr[artifact_indices] = interp_func(artifact_indices)
        else:
            raise ValueError(f"Unsupported artifact removal method: {method}")
        
        removed_count = len(rr_intervals) - len(cleaned_rr) if method == 'remove' else np.sum(artifact_mask)
        logger.info(f"Artifact handling completed: {removed_count} artifacts {method}d")
        
        return cleaned_rr
    
    def assess_signal_quality(self, ppg_signal: np.ndarray,
                            rr_intervals: np.ndarray,
                            artifact_mask: np.ndarray) -> Dict[str, float]:
        """
        Assess the quality of PPG signal and extracted RR intervals.
        
        Parameters
        ----------
        ppg_signal : np.ndarray
            PPG signal
        rr_intervals : np.ndarray
            Extracted RR intervals
        artifact_mask : np.ndarray
            Artifact detection mask
            
        Returns
        -------
        Dict[str, float]
            Quality metrics
        """
        quality_metrics = {}
        
        # Signal quality metrics
        quality_metrics['signal_length'] = len(ppg_signal)
        quality_metrics['signal_duration'] = len(ppg_signal) / self.sampling_rate
        quality_metrics['signal_snr'] = self._calculate_snr(ppg_signal)
        
        # RR interval quality metrics
        if len(rr_intervals) > 0:
            quality_metrics['total_rr_intervals'] = len(rr_intervals)
            quality_metrics['artifact_count'] = np.sum(artifact_mask)
            quality_metrics['artifact_percentage'] = (np.sum(artifact_mask) / len(rr_intervals)) * 100
            quality_metrics['valid_rr_percentage'] = 100 - quality_metrics['artifact_percentage']
            
            # Physiological plausibility
            valid_rr = rr_intervals[~artifact_mask]
            if len(valid_rr) > 0:
                quality_metrics['mean_hr'] = 60000 / np.mean(valid_rr)  # BPM
                quality_metrics['hr_variability'] = np.std(valid_rr)
        
        return quality_metrics
    
    def _calculate_snr(self, ppg_signal: np.ndarray) -> float:
        """
        Calculate signal-to-noise ratio for PPG signal.
        
        Parameters
        ----------
        ppg_signal : np.ndarray
            PPG signal
            
        Returns
        -------
        float
            Signal-to-noise ratio in dB
        """
        # Simple SNR estimation based on signal power vs noise power
        signal_power = np.var(ppg_signal)
        
        # Estimate noise as high-frequency component
        sos_hp = signal.butter(2, 10/self.nyquist, btype='high', output='sos')
        noise = sosfiltfilt(sos_hp, ppg_signal)
        noise_power = np.var(noise)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = np.inf
        
        return snr_db
    
    def process_ppg_segment(self, ppg_signal: np.ndarray,
                           artifact_method: str = 'remove') -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Complete PPG processing pipeline for a signal segment.
        
        Parameters
        ----------
        ppg_signal : np.ndarray
            Raw PPG signal
        artifact_method : str, default 'remove'
            Method for handling artifacts ('remove' or 'interpolate')
            
        Returns
        -------
        Dict[str, Union[np.ndarray, Dict]]
            Complete processing results
        """
        results = {}
        
        # Step 1: Filter signal
        filtered_ppg = self.filter_signal(ppg_signal)
        results['filtered_ppg'] = filtered_ppg
        
        # Step 2: Detect peaks
        peaks, peak_properties = self.detect_peaks(filtered_ppg)
        results['peaks'] = peaks
        results['peak_properties'] = peak_properties
        
        # Step 3: Extract RR intervals
        rr_intervals = self.extract_rr_intervals(peaks)
        results['raw_rr_intervals'] = rr_intervals
        
        # Step 4: Detect artifacts
        artifact_mask = self.detect_artifacts(rr_intervals)
        results['artifact_mask'] = artifact_mask
        
        # Step 5: Clean RR intervals
        cleaned_rr = self.remove_artifacts(rr_intervals, artifact_mask, artifact_method)
        results['cleaned_rr_intervals'] = cleaned_rr
        
        # Step 6: Quality assessment
        quality_metrics = self.assess_signal_quality(ppg_signal, rr_intervals, artifact_mask)
        results['quality_metrics'] = quality_metrics
        
        logger.info("Complete PPG processing pipeline executed successfully")
        return results


def process_ppg_batch(ppg_signals: List[np.ndarray], 
                     sampling_rate: float = DEFAULT_SAMPLING_RATE,
                     artifact_method: str = 'remove') -> List[Dict]:
    """
    Process multiple PPG signals in batch.
    
    Parameters
    ----------
    ppg_signals : List[np.ndarray]
        List of PPG signals to process
    sampling_rate : float, default 64
        Sampling rate in Hz
    artifact_method : str, default 'remove'
        Artifact handling method
        
    Returns
    -------
    List[Dict]
        List of processing results for each signal
    """
    preprocessor = PPGPreprocessor(sampling_rate)
    results = []
    
    for i, ppg_signal in enumerate(ppg_signals):
        logger.info(f"Processing PPG signal {i+1}/{len(ppg_signals)}")
        try:
            result = preprocessor.process_ppg_segment(ppg_signal, artifact_method)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process signal {i+1}: {str(e)}")
            results.append({})
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    import matplotlib.pyplot as plt
    
    # Generate synthetic PPG signal for testing
    duration = 60  # seconds
    fs = 64  # Hz
    t = np.linspace(0, duration, int(duration * fs))
    
    # Synthetic PPG with heart rate around 70 BPM
    hr = 70  # BPM
    heart_period = 60 / hr
    ppg_synthetic = np.sin(2 * np.pi * t / heart_period) + 0.1 * np.random.randn(len(t))
    
    # Test preprocessing
    preprocessor = PPGPreprocessor(fs)
    results = preprocessor.process_ppg_segment(ppg_synthetic)
    
    print("PPG Preprocessing Test Results:")
    print(f"Signal length: {len(ppg_synthetic)} samples")
    print(f"Peaks detected: {len(results['peaks'])}")
    print(f"RR intervals extracted: {len(results['raw_rr_intervals'])}")
    print(f"Quality metrics: {results['quality_metrics']}")
    
    # Plot results if matplotlib is available
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Original vs filtered signal
        axes[0].plot(t[:1000], ppg_synthetic[:1000], label='Original', alpha=0.7)
        axes[0].plot(t[:1000], results['filtered_ppg'][:1000], label='Filtered')
        axes[0].set_title('PPG Signal: Original vs Filtered')
        axes[0].legend()
        
        # Peaks detection
        peak_times = results['peaks'] / fs
        axes[1].plot(t, results['filtered_ppg'])
        axes[1].plot(peak_times, results['filtered_ppg'][results['peaks']], 'ro', markersize=4)
        axes[1].set_title('Peak Detection Results')
        axes[1].set_xlim(0, 10)  # Show first 10 seconds
        
        # RR intervals
        if len(results['cleaned_rr_intervals']) > 0:
            axes[2].plot(results['cleaned_rr_intervals'])
            axes[2].set_title('Cleaned RR Intervals')
            axes[2].set_ylabel('RR Interval (ms)')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")