"""
Unit and Integration Tests for SentinelML Data Validation and Preprocessing
Tests schema conformity, error handling on malformed data, stratified splitting,
and zero-leakage preprocessing.
"""

import unittest
import tempfile
import csv
from pathlib import Path
from sentinelml.config import ALL_FEATURES, TARGET_COLUMN, REQUIRED_COLUMNS
from sentinelml.data.validate_data import validate_csv_dataset, DataValidationError
from sentinelml.data.preprocess import stratified_split, FraudPreprocessor


class TestDataValidation(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_csv(self, filename: str, header: list, rows: list) -> Path:
        p = self.temp_path / filename
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        return p

    def test_missing_file_raises_error(self):
        non_existent = self.temp_path / "does_not_exist.csv"
        with self.assertRaises(DataValidationError) as ctx:
            validate_csv_dataset(non_existent)
        self.assertIn("not found", str(ctx.exception).lower())

    def test_missing_column_raises_error(self):
        bad_header = ["Time", "Amount", "Class"]  # Missing V1-V28
        row = [100.0, 50.0, 0]
        csv_path = self._create_csv("bad_header.csv", bad_header, [row])
        with self.assertRaises(DataValidationError) as ctx:
            validate_csv_dataset(csv_path)
        self.assertIn("missing required columns", str(ctx.exception).lower())

    def test_invalid_class_label_raises_error(self):
        header = REQUIRED_COLUMNS
        # Create row with Class = 2 instead of 0 or 1
        row = [0.0] * (len(header) - 1) + [2]
        csv_path = self._create_csv("invalid_class.csv", header, [row])
        with self.assertRaises(DataValidationError) as ctx:
            validate_csv_dataset(csv_path)
        self.assertIn("target class must be binary", str(ctx.exception).lower())

    def test_null_value_raises_error(self):
        header = REQUIRED_COLUMNS
        row = [""] * len(header)
        csv_path = self._create_csv("null_row.csv", header, [row])
        with self.assertRaises(DataValidationError):
            validate_csv_dataset(csv_path)

    def test_valid_dataset_passes(self):
        header = REQUIRED_COLUMNS
        rows = []
        for i in range(100):
            # 98 legit (0), 2 fraud (1)
            label = 1 if i < 2 else 0
            row = [float(i)] * (len(header) - 1) + [label]
            rows.append(row)
        csv_path = self._create_csv("valid.csv", header, rows)
        report = validate_csv_dataset(csv_path)
        self.assertTrue(report["validation_passed"])
        self.assertEqual(report["total_records"], 100)
        self.assertEqual(report["fraud_count"], 2)


class TestPreprocessing(unittest.TestCase):

    def test_stratified_split_preserves_classes(self):
        # 1000 items, 20 positive (2% fraud)
        X = [[float(j) for j in range(30)] for _ in range(1000)]
        y = [1] * 20 + [0] * 980

        X_tr, y_tr, X_val, y_val, X_te, y_te = stratified_split(
            X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
        )

        self.assertEqual(len(y_tr), 700)
        self.assertEqual(len(y_val), 150)
        self.assertEqual(len(y_te), 150)

        # Both positive and negative classes must be present in every split
        self.assertGreater(sum(y_tr), 0)
        self.assertGreater(sum(y_val), 0)
        self.assertGreater(sum(y_te), 0)

    def test_zero_leakage_scaling(self):
        preproc = FraudPreprocessor()
        # Train data has amount median = 50.0
        X_train = [[0.0] * 29 + [50.0], [0.0] * 29 + [150.0], [0.0] * 29 + [250.0]]
        preproc.fit(X_train)

        # Ensure median is strictly 150.0
        self.assertAlmostEqual(preproc.amount_median, 150.0)

        # Transform test data where amount is 5000.0 (an extreme outlier)
        X_test = [[0.0] * 29 + [5000.0]]
        X_test_scaled = preproc.transform(X_test)

        # Verify test amount was scaled using training statistics without changing training median
        self.assertEqual(preproc.amount_median, 150.0)
        self.assertNotEqual(X_test_scaled[0][-1], 5000.0)


if __name__ == "__main__":
    unittest.main()
