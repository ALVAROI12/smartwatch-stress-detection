import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from relabel_windows import STAGE_TO_HARMONIZED, label_windows  # noqa: E402


def test_label_windows_assigns_majority_class_and_purity():
    windows = pd.DataFrame({"timestamp_start": [0.0, 30.0, 60.0, 300.0], "timestamp_end": [60.0, 90.0, 120.0, 360.0]})
    # Baseline 0-60 s, then two back-to-back stressors (both harmonize to Stress) from 60 s to 120 s.
    intervals = [(0.0, 60.0, "Baseline"), (60.0, 90.0, "Stroop"), (90.0, 120.0, "TMCT")]
    out = label_windows(windows, intervals, STAGE_TO_HARMONIZED.get)

    assert out["harmonized_label"].tolist() == ["Baseline", "Baseline", "Stress", "Excluded"]
    assert out["purity"].tolist() == [1.0, 0.5, 1.0, 0.0]  # stressor stages pool into one class
    assert out.loc[3, "original_label"] == "Transition/unlabelled"
