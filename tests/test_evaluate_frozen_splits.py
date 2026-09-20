import unittest

import pandas as pd

from scripts.evaluate_frozen_splits import summarize_results


class SummarizeResultsTests(unittest.TestCase):
    def test_skipped_splits_are_counted_but_excluded_from_metrics(self):
        results = pd.DataFrame(
            [
                {
                    "protocol": "loso",
                    "modality": "all_modalities",
                    "status": "ok",
                    "macro_f1": 0.8,
                    "balanced_accuracy": 0.7,
                    "accuracy": 0.75,
                },
                {
                    "protocol": "loso",
                    "modality": "all_modalities",
                    "status": "ok",
                    "macro_f1": 0.6,
                    "balanced_accuracy": 0.5,
                    "accuracy": 0.55,
                },
                {
                    "protocol": "loso",
                    "modality": "all_modalities",
                    "status": "unseen_test_labels",
                    "macro_f1": None,
                    "balanced_accuracy": None,
                    "accuracy": None,
                },
            ]
        )

        summary = summarize_results(results).iloc[0]

        self.assertEqual(summary["n_total_splits"], 3)
        self.assertEqual(summary["n_evaluable_splits"], 2)
        self.assertEqual(summary["n_skipped_splits"], 1)
        self.assertEqual(summary["skip_reasons"], "unseen_test_labels:1")
        self.assertAlmostEqual(summary["macro_f1_mean"], 0.7)
        self.assertAlmostEqual(summary["balanced_accuracy_mean"], 0.6)
        self.assertAlmostEqual(summary["accuracy_mean"], 0.65)


if __name__ == "__main__":
    unittest.main()
