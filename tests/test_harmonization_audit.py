import unittest

import pandas as pd

from scripts.harmonization_audit import build_overlap_tables, coerce_timestamp_series


class HarmonizationAuditTests(unittest.TestCase):
    def test_coerce_timestamp_series_accepts_numeric_strings(self):
        series = pd.Series(["0", "30", "60"], name="timestamp_start")
        coerced = coerce_timestamp_series(series)
        self.assertEqual(coerced.tolist(), [0, 30, 60])

    def test_coerce_timestamp_series_rejects_malformed_values(self):
        series = pd.Series(["0", "bad", "60"], name="timestamp_start")
        with self.assertRaises(ValueError):
            coerce_timestamp_series(series)

    def test_build_overlap_tables_detects_touching_or_overlapping_windows(self):
        df = pd.DataFrame(
            [
                {"dataset": "WESAD", "subject_id": "S2", "label": "Stress", "timestamp_start": 0, "timestamp_end": 60},
                {"dataset": "WESAD", "subject_id": "S2", "label": "Stress", "timestamp_start": 60, "timestamp_end": 120},
                {"dataset": "WESAD", "subject_id": "S2", "label": "Stress", "timestamp_start": 120, "timestamp_end": 180},
            ]
        )
        details, summary = build_overlap_tables(df)
        self.assertEqual(int(summary.iloc[0]["groups_with_overlap"]), 1)
        self.assertEqual(int(details.iloc[0]["overlapping_windows"]), 2)


if __name__ == "__main__":
    unittest.main()
