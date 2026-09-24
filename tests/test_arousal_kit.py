import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from arousal_kit import arousal_controlled_auroc, iso_hr_subsample  # noqa: E402


def test_score_equal_to_arousal_carries_nothing_beyond_it():
    rng = np.random.default_rng(1)
    n = 4000
    index = rng.normal(size=n)
    labels = (index + rng.normal(scale=0.5, size=n) > 0).astype(int)  # label depends on arousal only
    subjects = np.repeat(np.arange(40), n // 40)
    pooled_blind = arousal_controlled_auroc(rng.normal(size=n), index, labels, subjects, n_boot=50)
    assert abs(pooled_blind["auc"] - 0.5) < 0.05
    extra = arousal_controlled_auroc(index + 2 * labels, index, labels, subjects, n_boot=50)
    assert extra["auc"] > 0.9 and extra["lo"] > 0.5


def test_iso_hr_balances_classes_within_subject_and_bin():
    rng = np.random.default_rng(0)
    hr = np.array([60, 61, 62, 63, 80, 81, 60, 90], float)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    g = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    k = iso_hr_subsample(hr, y, g, rng)
    assert sorted(y[k]) == [0, 1]  # one pair from subject 0's 60-64 bpm bin; nothing else matches
    assert set(g[k]) == {0}
