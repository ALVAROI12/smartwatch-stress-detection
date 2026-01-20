import numpy as np
import pytest

from src.preprocessing.wesad_processor import WESADProcessor


@pytest.fixture()
def processor(tmp_path):
    # Use temporary folders to avoid touching real data directories during tests
    return WESADProcessor(data_path=tmp_path, output_path=tmp_path)


def test_find_continuous_segments_merges_adjacent(processor):
    indices = np.array([1, 2, 3, 7, 8, 10, 11, 12])
    segments = processor._find_continuous_segments(indices)
    assert segments == [(1, 3), (7, 8), (10, 12)]


def test_extract_features_returns_expected_keys(processor):
    samples = 128
    window_data = {
        'ecg': np.linspace(-1, 1, samples),
        'eda': np.linspace(0.1, 0.5, samples),
        'resp': np.sin(np.linspace(0, 4 * np.pi, samples)),
        'bvp': np.sin(np.linspace(0, 8 * np.pi, samples)) + 0.05,
        'temp': np.linspace(36.0, 36.5, samples),
        'acc_chest': np.ones((samples, 3)),
        'acc_wrist': np.full((samples, 3), 0.5),
    }

    features = processor._extract_features(window_data)

    assert features is not None
    for key in ['ecg_mean', 'eda_std', 'acc_wrist_mean', 'movement_intensity', 'hr_estimate']:
        assert key in features

    # All feature values should be finite numbers
    assert all(np.isfinite(list(features.values())))


def test_calculate_hrv_simple_returns_defaults_when_no_peaks(processor):
    constant_signal = np.ones(64)
    hrv = processor._calculate_hrv_simple(constant_signal)
    assert hrv['hr_estimate'] == pytest.approx(70.0)
    assert hrv['hr_variability'] == pytest.approx(5.0)
