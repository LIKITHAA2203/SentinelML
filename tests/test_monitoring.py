"""
Unit Tests for SentinelML Drift Monitoring and SQLite Persistence
Tests Population Stability Index (PSI), Kolmogorov-Smirnov test on stable
vs shifted samples, and SQLite operational storage.
"""

import unittest
import tempfile
import random
from pathlib import Path
from sentinelml.monitoring.drift import calculate_psi, calculate_ks_statistic, analyze_drift_batch
from sentinelml.monitoring.logging_store import PredictionStore


class TestDriftDetection(unittest.TestCase):

    def setUp(self):
        # Use deterministic samples for repeatable unit testing
        rng = random.Random(101)
        self.ref_data = [rng.gauss(0.0, 1.0) for _ in range(600)]
        self.stable_curr = [rng.gauss(0.0, 1.0) for _ in range(600)]
        # Shifted distribution: 600 samples with mean shift N(3.5, 1.5)
        self.shifted_curr = [rng.gauss(3.5, 1.5) for _ in range(600)]

    def test_stable_data_has_low_psi(self):
        psi, status = calculate_psi(self.ref_data, self.stable_curr)
        self.assertLess(psi, 0.10)
        self.assertEqual(status, "STABLE")

    def test_stable_data_ks_test_no_drift(self):
        d_stat, p_val, is_drift = calculate_ks_statistic(self.ref_data, self.stable_curr)
        self.assertFalse(is_drift)
        self.assertGreater(p_val, 0.01)

    def test_shifted_data_triggers_psi_critical(self):
        psi, status = calculate_psi(self.ref_data, self.shifted_curr)
        self.assertGreaterEqual(psi, 0.25)
        self.assertEqual(status, "CRITICAL_DRIFT")

    def test_shifted_data_ks_detects_significant_drift(self):
        d_stat, p_val, is_drift = calculate_ks_statistic(self.ref_data, self.shifted_curr)
        self.assertTrue(is_drift)
        self.assertLess(p_val, 0.001)
        self.assertGreater(d_stat, 0.40)


class TestPredictionLoggingStore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_monitoring.db"
        self.store = PredictionStore(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_log_and_fetch_prediction(self):
        dummy_features = {"Time": 1200.0, "V1": 0.45, "Amount": 89.50}
        self.store.log_prediction(
            request_id="req-test-001",
            model_version="v1.0.0-RandomForest",
            risk_score=0.82,
            decision_threshold=0.42,
            predicted_label=1,
            latency_ms=14.2,
            feature_payload=dummy_features,
            status="SUCCESS"
        )

        rows = self.store.get_recent_predictions(limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request_id"], "req-test-001")
        self.assertEqual(rows[0]["predicted_label"], 1)
        self.assertAlmostEqual(rows[0]["risk_score"], 0.82)

        metrics = self.store.get_operational_metrics()
        self.assertEqual(metrics["total_requests"], 1)
        self.assertEqual(metrics["predicted_frauds"], 1)
        self.assertEqual(metrics["predicted_fraud_rate_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
