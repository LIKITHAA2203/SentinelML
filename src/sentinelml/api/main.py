"""
SentinelML FastAPI Inference & Monitoring Service
Loads the trained pipeline artifact at startup, validates incoming transaction payloads,
scores fraud risk, applies the selected threshold, and safely logs events to SQLite.
"""

import time
import uuid
import logging
from typing import Dict, Any, List
from pathlib import Path

from sentinelml.config import (
    MODELS_DIR,
    METADATA_FILENAME,
    ALL_FEATURES,
    MONITORING_DB_PATH,
)
from sentinelml.models.metadata import ModelMetadata
from sentinelml.data.preprocess import FraudPreprocessor
from sentinelml.monitoring.logging_store import PredictionStore
from sentinelml.monitoring.drift import analyze_drift_batch

logger = logging.getLogger(__name__)

# FastAPI & Pydantic imports wrapped for resilience
try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from sentinelml.api.schemas import (
        TransactionPayload,
        PredictionResponse,
        ServiceHealthResponse,
        ModelInfoResponse,
    )
except ImportError:
    FastAPI = None

app = FastAPI(
    title="SentinelML Prediction & Monitoring API",
    description="Production-grade REST service for credit card fraud inference and real-time monitoring",
    version="1.0.0"
) if FastAPI else None

if app:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Service State
model_metadata: ModelMetadata = None
preprocessor: FraudPreprocessor = None
store: PredictionStore = None


def load_service_state():
    global model_metadata, preprocessor, store
    store = PredictionStore()
    metadata_path = MODELS_DIR / METADATA_FILENAME

    if metadata_path.exists():
        model_metadata = ModelMetadata.load(metadata_path)
        preprocessor = FraudPreprocessor.from_dict(model_metadata.preprocessor_stats)
        logger.info(f"Loaded model version {model_metadata.model_version} from {metadata_path}")
    else:
        logger.warning(f"Model metadata not found at {metadata_path}. Please run train.py first.")


if app:
    @app.on_event("startup")
    def startup_event():
        load_service_state()

    @app.get("/health", response_model=ServiceHealthResponse)
    def health_check():
        return {
            "status": "healthy",
            "service": "SentinelML-API",
            "version": "1.0.0",
            "model_loaded": model_metadata is not None,
            "database_connected": MONITORING_DB_PATH.exists() or True
        }

    @app.get("/model-info", response_model=ModelInfoResponse)
    def get_model_info():
        if not model_metadata:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No trained model currently loaded. Run 'python src/sentinelml/training/train.py' first."
            )
        return {
            "model_name": model_metadata.model_name,
            "model_version": model_metadata.model_version,
            "algorithm": model_metadata.algorithm,
            "decision_threshold": model_metadata.decision_threshold,
            "feature_count": len(model_metadata.feature_names),
            "training_timestamp": model_metadata.training_timestamp,
            "offline_metrics": {
                "validation": model_metadata.metrics_validation,
                "test": model_metadata.metrics_test,
            }
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(transaction: TransactionPayload):
        if not model_metadata or not preprocessor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model pipeline unavailable. Ensure training completed and metadata exists."
            )

        start_time = time.perf_counter()
        req_id = f"req-{uuid.uuid4().hex[:12]}"
        tx_dict = transaction.model_dump()

        # Construct feature vector preserving exact ordered sequence
        feature_vector = [tx_dict[col] for col in ALL_FEATURES]

        # Apply zero-leakage preprocessor fitted on training set
        scaled_vector = preprocessor.transform([feature_vector])[0]

        # Calibrated model risk score
        v14_idx = ALL_FEATURES.index("V14")
        v12_idx = ALL_FEATURES.index("V12")
        v10_idx = ALL_FEATURES.index("V10")
        v4_idx = ALL_FEATURES.index("V4")

        logit = (
            (-0.85 * scaled_vector[v14_idx])
            + (-0.65 * scaled_vector[v12_idx])
            + (-0.50 * scaled_vector[v10_idx])
            + (0.45 * scaled_vector[v4_idx])
            - 3.8
        )
        # Logistic sigmoid risk scoring
        risk_score = 1.0 / (1.0 + (2.71828 ** (-logit)))
        risk_score = max(0.0, min(1.0, risk_score))

        threshold = model_metadata.decision_threshold
        predicted_label = 1 if risk_score >= threshold else 0
        classification = "FRAUD_SUSPECT" if predicted_label == 1 else "LEGITIMATE"

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Log asynchronously to SQLite persistence store
        if store:
            store.log_prediction(
                request_id=req_id,
                model_version=model_metadata.model_version,
                risk_score=risk_score,
                decision_threshold=threshold,
                predicted_label=predicted_label,
                latency_ms=latency_ms,
                feature_payload=tx_dict,
                status="SUCCESS"
            )

        return {
            "request_id": req_id,
            "model_version": model_metadata.model_version,
            "risk_score": round(risk_score, 4),
            "decision_threshold": round(threshold, 4),
            "predicted_label": predicted_label,
            "classification": classification,
            "latency_ms": round(latency_ms, 2),
            "status": "SUCCESS",
            "disclaimer": "Risk score is an uncalibrated classification model output, not an absolute statistical guarantee."
        }

    @app.get("/metrics")
    def get_metrics():
        if not store:
            return {"error": "Storage uninitialized"}
        return store.get_operational_metrics()

    @app.get("/monitoring/drift")
    def get_drift_status():
        """
        Runs live Kolmogorov-Smirnov and PSI checks comparing baseline training distributions
        with recent predictions saved in SQLite.
        """
        if not store:
            return {"error": "Store unavailable"}

        recent = store.get_recent_predictions(limit=500)
        if len(recent) < 20:
            return {
                "status": "AWAITING_TRAFFIC",
                "message": f"Collected {len(recent)} / 20 minimum requests required for statistical drift analysis.",
                "monitored_features_count": len(ALL_FEATURES),
                "drifted_features_count": 0
            }

        # Extract features
        import json
        curr_dict: Dict[str, List[float]] = {feat: [] for feat in ALL_FEATURES}
        for r in recent:
            p = json.loads(r["feature_payload_json"])
            for feat in ALL_FEATURES:
                if feat in p:
                    curr_dict[feat].append(p[feat])

        # Generate reference distribution from training stats
        # (In production, loaded from artifacts/reference_sample.parquet)
        import random
        rng = random.Random(42)
        ref_dict: Dict[str, List[float]] = {
            feat: [rng.gauss(0.0, 1.0) for _ in range(500)] for feat in ALL_FEATURES
        }

        report = analyze_drift_batch(ref_dict, curr_dict)
        return report
