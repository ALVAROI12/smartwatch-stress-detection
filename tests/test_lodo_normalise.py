import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from leave_one_dataset_out import CAUSAL_MIN, normalise  # noqa: E402


def recording(uid: str, values: list[float], labels: list[str]) -> pd.DataFrame:
    start = 30.0 * np.arange(len(values))
    return pd.DataFrame({"subject_uid": uid, "timestamp_start": start, "timestamp_end": start + 60,
                         "harmonized_label": labels, "f": values})


def test_causal_uses_only_earlier_non_overlapping_windows():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 7.0]
    df = recording("a", values, ["Baseline"] * 4 + ["Stress"] * 4)
    out = normalise(df, ["f"], "causal")["f"].to_numpy()
    opening = np.array(values[:CAUSAL_MIN])
    assert np.isclose(out[0], (values[0] - opening.mean()) / opening.std(ddof=1))
    # window 7 (start 210 s) may use windows ending by 210 s: indices 0-5, never the outlier at index 6
    ref = np.array(values[:6])
    assert np.isclose(out[7], (values[7] - ref.mean()) / ref.std(ddof=1))


def test_session_and_baseline_scale_exercise_with_protocol_statistics():
    df = pd.concat([recording("a", [1.0, 3.0, 5.0, 7.0], ["Baseline", "Baseline", "Stress", "Rest"]),
                    recording("a", [50.0], ["Aerobic"])], ignore_index=True)
    session = normalise(df, ["f"], "session")["f"].to_numpy()
    protocol = np.array([1.0, 3.0, 5.0, 7.0])
    assert np.isclose(session[4], (50 - protocol.mean()) / protocol.std(ddof=1))
    baseline = normalise(df, ["f"], "baseline")["f"].to_numpy()
    assert np.isclose(baseline[2], (5 - 2.0) / np.std([1.0, 3.0], ddof=1))
