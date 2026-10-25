"""
Integration and Unit Tests for SentinelML API Endpoints, Model Artifacts, and Schema Validation
"""

import unittest
from sentinelml.models.metadata import ModelMetadata
from sentinelml.models.load_model import load_trained_model
from sentinelml.config import MODELS_DIR, METADATA_FILENAME, MODEL_FILENAME, ALL_FEATURES
from sentinelml.api.main import load_service_state, health_check


class TestApiAndSchemas(unittest.TestCase):

    def test_model_metadata_exists_and_loads(self):
        metadata_path = MODELS_DIR / METADATA_FILENAME
        self.assertTrue(metadata_path.exists(), f"Trained model metadata must exist at {metadata_path}")
        meta = ModelMetadata.load(metadata_path)
        self.assertEqual(meta.model_version, "v1.0.0-RandomForest")
        self.assertGreater(len(meta.feature_names), 25)
        self.assertIn("Time", meta.feature_names)
        self.assertIn("Amount", meta.feature_names)
        self.assertIn("precision", meta.metrics_test)
        self.assertIn("recall", meta.metrics_test)
        self.assertIn("pr_auc", meta.metrics_test)

    def test_model_artifact_exists_and_loads(self):
        model_path = MODELS_DIR / MODEL_FILENAME
        self.assertTrue(model_path.exists(), f"Trained model artifact must exist at {model_path}")
        model, metadata = load_trained_model(MODELS_DIR)
        self.assertIsNotNone(model, "Model estimator must be loaded")
        self.assertIsNotNone(metadata, "Model metadata must be loaded")
        self.assertTrue(hasattr(model, "predict_proba"), "Model estimator must implement predict_proba")

        # Test inference with dummy feature vector
        sample_vector = [0.0] * len(ALL_FEATURES)
        probs = model.predict_proba([sample_vector])[0]
        self.assertEqual(len(probs), 2)
        self.assertGreaterEqual(probs[1], 0.0)
        self.assertLessEqual(probs[1], 1.0)

    def test_api_service_loads_model(self):
        load_service_state()
        health = health_check()
        self.assertTrue(health["model_loaded"], "FastAPI health_check must report model_loaded: True")
        self.assertEqual(health["status"], "healthy")

    def test_feature_list_matches_expected_columns(self):
        self.assertEqual(len(ALL_FEATURES), 30)
        self.assertEqual(ALL_FEATURES[0], "Time")
        self.assertEqual(ALL_FEATURES[-1], "Amount")


if __name__ == "__main__":
    unittest.main()
