"""
Integration and Unit Tests for SentinelML API Endpoints and Schema Validation
"""

import unittest
from sentinelml.models.metadata import ModelMetadata
from sentinelml.config import MODELS_DIR, METADATA_FILENAME, ALL_FEATURES


class TestApiAndSchemas(unittest.TestCase):

    def test_model_metadata_exists_and_loads(self):
        metadata_path = MODELS_DIR / METADATA_FILENAME
        self.assertTrue(metadata_path.exists(), "Trained model metadata must exist")
        meta = ModelMetadata.load(metadata_path)
        self.assertEqual(meta.model_version, "v1.0.0-RandomForest")
        self.assertGreater(len(meta.feature_names), 25)
        self.assertIn("Time", meta.feature_names)
        self.assertIn("Amount", meta.feature_names)
        self.assertIn("precision", meta.metrics_test)
        self.assertIn("recall", meta.metrics_test)
        self.assertIn("pr_auc", meta.metrics_test)

    def test_feature_list_matches_expected_columns(self):
        self.assertEqual(len(ALL_FEATURES), 30)
        self.assertEqual(ALL_FEATURES[0], "Time")
        self.assertEqual(ALL_FEATURES[-1], "Amount")


if __name__ == "__main__":
    unittest.main()
