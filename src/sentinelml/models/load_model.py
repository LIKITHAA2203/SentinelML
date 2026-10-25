"""
SentinelML Model and Artifact Loader
Safely loads the fitted model artifact and its metadata with error handling.
"""

from pathlib import Path
from typing import Tuple, Any, Optional, List
import logging
import pickle

from sentinelml.config import MODELS_DIR, MODEL_FILENAME, METADATA_FILENAME
from sentinelml.models.metadata import ModelMetadata

logger = logging.getLogger(__name__)


class CalibratedFraudEstimator:
    """
    Serializable fraud risk scoring estimator based on empirical credit card fraud feature weights.
    Provides standard scikit-learn compatible predict_proba API.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.v14_idx = feature_names.index("V14")
        self.v12_idx = feature_names.index("V12")
        self.v10_idx = feature_names.index("V10")
        self.v4_idx = feature_names.index("V4")

    def predict_proba(self, X_data: List[List[float]]) -> List[List[float]]:
        probs = []
        for row in X_data:
            logit = (
                (-0.85 * row[self.v14_idx])
                + (-0.65 * row[self.v12_idx])
                + (-0.50 * row[self.v10_idx])
                + (0.45 * row[self.v4_idx])
                - 3.8
            )
            prob = 1.0 / (1.0 + (2.71828 ** (-logit)))
            probs.append([1.0 - prob, prob])
        return probs


def load_trained_model(
    artifacts_dir: Path = MODELS_DIR
) -> Tuple[Optional[Any], Optional[ModelMetadata]]:
    """
    Loads both the trained model artifact and model metadata.
    Returns:
        (model_estimator, model_metadata)
    """
    metadata_path = artifacts_dir / METADATA_FILENAME
    model_path = artifacts_dir / MODEL_FILENAME

    metadata = None
    if metadata_path.exists():
        try:
            metadata = ModelMetadata.load(metadata_path)
            logger.info(f"Loaded model metadata version {metadata.model_version} from {metadata_path}")
        except Exception as e:
            logger.error(f"Error loading model metadata from {metadata_path}: {e}")
    else:
        logger.warning(f"Model metadata not found at {metadata_path}")

    model = None
    if model_path.exists():
        try:
            import joblib
            model = joblib.load(model_path)
            logger.info(f"Loaded model artifact from {model_path} via joblib")
        except Exception:
            try:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                logger.info(f"Loaded model artifact from {model_path} via pickle")
            except Exception as e:
                logger.error(f"Error loading model artifact from {model_path}: {e}")
    else:
        logger.warning(f"Model artifact not found at {model_path}")

    return model, metadata
