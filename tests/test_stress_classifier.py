import numpy as np
import pandas as pd
import pytest

from src.models.stress_classifier import StressClassifier


@pytest.fixture()
def classifier(tmp_path):
    return StressClassifier(data_path=tmp_path / "dummy.json", results_path=tmp_path)


def test_prepare_features_imputes_missing_values(classifier):
    df = pd.DataFrame({
        'feature_a': [1.0, 2.0, np.nan],
        'feature_b': [0.5, np.nan, 0.2],
        'subject_id': ['S1', 'S2', 'S1'],
        'condition': ['baseline', 'stress', 'baseline'],
        'label': [0, 1, 0],
        'window_start': [0, 0, 10],
        'window_end': [5, 5, 15],
        'segment_idx': [0, 0, 1],
        'window_duration_sec': [180, 180, 180],
        'purity': [1.0, 0.98, 0.99],
    })

    X_scaled, y, subjects, feature_cols, scaler = classifier.prepare_features(df)

    assert not np.isnan(X_scaled).any()
    assert list(feature_cols) == ['feature_a', 'feature_b']
    assert np.array_equal(y, np.array([0, 1, 0]))
    assert np.array_equal(subjects, np.array(['S1', 'S2', 'S1']))


def test_compare_models_marks_expected_winner(classifier):
    rf_scores = {
        'test_accuracy': np.array([0.70, 0.75]),
        'test_precision': np.array([0.68, 0.72]),
        'test_recall': np.array([0.66, 0.70]),
        'test_f1': np.array([0.67, 0.71]),
        'test_roc_auc': np.array([0.74, 0.76]),
    }
    xgb_scores = {
        'accuracy': [0.80, 0.82],
        'precision': [0.78, 0.79],
        'recall': [0.77, 0.78],
        'f1': [0.78, 0.79],
        'roc_auc': [0.83, 0.84],
    }

    comparison = classifier.compare_models(rf_scores, xgb_scores)

    assert set(comparison['Better_Model']) == {'XGBoost'}
    assert np.allclose(comparison['Difference'], comparison['XGBoost'] - comparison['Random Forest'])
