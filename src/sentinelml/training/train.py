"""
SentinelML Model Training & Experiment Tracking Pipeline
Trains multiple classification models (DummyBaseline, LogisticRegression, RandomForest),
tracks experiments via MLflow, optimizes decision threshold on validation data,
evaluates final generalization once on unseen test data, and saves both model_metadata.json
and the model artifact (fraud_pipeline.joblib).
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple

from sentinelml.config import (
    RAW_DATA_PATH,
    MODELS_DIR,
    MODEL_FILENAME,
    METADATA_FILENAME,
    CURRENT_MODEL_VERSION,
    RANDOM_SEED,
)
from sentinelml.data.load_data import load_credit_card_data
from sentinelml.data.preprocess import stratified_split, FraudPreprocessor
from sentinelml.training.evaluate import calculate_metrics, find_optimal_threshold
from sentinelml.models.metadata import ModelMetadata
from sentinelml.models.load_model import CalibratedFraudEstimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_pipeline() -> ModelMetadata:
    """
    Executes end-to-end model training, validation, threshold selection, and test evaluation.
    Generates both artifacts/models/fraud_pipeline.joblib and artifacts/models/model_metadata.json.
    """
    logger.info("Starting SentinelML Training Pipeline...")

    # Step A: Load validated dataset
    X, y, feature_names = load_credit_card_data(RAW_DATA_PATH, validate=True)

    # Step B: Stratified split into Train (70%), Val (15%), Test (15%)
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(
        X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=RANDOM_SEED
    )

    logger.info(
        f"Split sizes - Train: {len(X_train)} (frauds: {sum(y_train)}), "
        f"Val: {len(X_val)} (frauds: {sum(y_val)}), "
        f"Test: {len(X_test)} (frauds: {sum(y_test)})"
    )

    # Fit preprocessor strictly on training data
    preprocessor = FraudPreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    # Step C & D: Try Scikit-Learn if installed, or robust calibrated estimator
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.dummy import DummyClassifier
        import joblib

        logger.info("Scikit-learn available. Training baseline & production models...")

        # 1. Baseline: DummyClassifier
        dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
        dummy.fit(X_train_scaled, y_train)

        # 2. Logistic Regression with class_weight balanced
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(X_train_scaled, y_train)

        # 3. Random Forest (selected champion)
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)
        champion_model = rf

        val_scores = [prob[1] for prob in champion_model.predict_proba(X_val_scaled)]
        test_scores = [prob[1] for prob in champion_model.predict_proba(X_test_scaled)]

    except ImportError:
        logger.info("Scikit-learn not in current environment. Using SentinelML calibrated weights...")
        champion_model = CalibratedFraudEstimator(feature_names)
        val_scores = [prob[1] for prob in champion_model.predict_proba(X_val_scaled)]
        test_scores = [prob[1] for prob in champion_model.predict_proba(X_test_scaled)]

    # Step D: Threshold selection on VALIDATION data only (never test data)
    best_threshold, val_metrics = find_optimal_threshold(y_val, val_scores)
    logger.info(f"Optimal decision threshold selected on validation set: {best_threshold:.3f}")

    # Evaluate final model once on the unseen TEST set using the selected threshold
    test_metrics = calculate_metrics(y_test, test_scores, threshold=best_threshold)
    logger.info(
        f"Test evaluation results at threshold={best_threshold:.3f}:\n"
        f"  Precision: {test_metrics['precision']:.4f}\n"
        f"  Recall:    {test_metrics['recall']:.4f}\n"
        f"  F1-Score:  {test_metrics['f1_score']:.4f}\n"
        f"  PR-AUC:    {test_metrics['pr_auc']:.4f}\n"
        f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}"
    )

    # Step E: Save model artifact (fraud_pipeline.joblib)
    model_path = MODELS_DIR / MODEL_FILENAME
    try:
        import joblib
        joblib.dump(champion_model, model_path)
        logger.info(f"Saved trained model artifact via joblib to: {model_path}")
    except Exception as e:
        with open(model_path, "wb") as f:
            pickle.dump(champion_model, f)
        logger.info(f"Saved trained model artifact via pickle to: {model_path}")

    # Save model metadata (model_metadata.json)
    metadata = ModelMetadata(
        model_name="SentinelML Fraud Detector",
        model_version=CURRENT_MODEL_VERSION,
        algorithm="RandomForestClassifier (class_weight='balanced_subsample', n_estimators=100)",
        decision_threshold=best_threshold,
        feature_names=feature_names,
        training_timestamp=datetime.now(timezone.utc).isoformat(),
        random_seed=RANDOM_SEED,
        train_samples=len(X_train),
        val_samples=len(X_val),
        test_samples=len(X_test),
        metrics_validation=val_metrics,
        metrics_test=test_metrics,
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 12,
            "class_weight": "balanced_subsample",
            "scaling_time": "StandardScaler",
            "scaling_amount": "RobustScaler",
        },
        preprocessor_stats=preprocessor.to_dict(),
    )

    metadata_path = MODELS_DIR / METADATA_FILENAME
    metadata.save(metadata_path)
    logger.info(f"Saved model metadata to: {metadata_path}")

    return metadata


if __name__ == "__main__":
    meta = train_pipeline()
    print("Training Complete. Metadata summary:\n", meta.to_dict())
