"""
SentinelML Pydantic API Schemas
Strict validation for incoming inference payloads, operational responses,
and drift monitoring summaries.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Any, List, Optional
import math


class TransactionPayload(BaseModel):
    """
    Validated credit card transaction input schema.
    Strictly accepts Time, Amount, and V1-V28 PCA features.
    Rejects extra unexpected fields, missing values, and NaN/Infinity.
    """
    model_config = ConfigDict(extra="forbid")

    Time: float = Field(..., ge=0.0, le=200000.0, description="Elapsed seconds since reference transaction")
    Amount: float = Field(..., ge=0.0, le=50000.0, description="Transaction amount in currency units")

    # V1 to V28 PCA components
    V1: float = Field(..., description="PCA Component V1")
    V2: float = Field(..., description="PCA Component V2")
    V3: float = Field(..., description="PCA Component V3")
    V4: float = Field(..., description="PCA Component V4")
    V5: float = Field(..., description="PCA Component V5")
    V6: float = Field(..., description="PCA Component V6")
    V7: float = Field(..., description="PCA Component V7")
    V8: float = Field(..., description="PCA Component V8")
    V9: float = Field(..., description="PCA Component V9")
    V10: float = Field(..., description="PCA Component V10")
    V11: float = Field(..., description="PCA Component V11")
    V12: float = Field(..., description="PCA Component V12")
    V13: float = Field(..., description="PCA Component V13")
    V14: float = Field(..., description="PCA Component V14")
    V15: float = Field(..., description="PCA Component V15")
    V16: float = Field(..., description="PCA Component V16")
    V17: float = Field(..., description="PCA Component V17")
    V18: float = Field(..., description="PCA Component V18")
    V19: float = Field(..., description="PCA Component V19")
    V20: float = Field(..., description="PCA Component V20")
    V21: float = Field(..., description="PCA Component V21")
    V22: float = Field(..., description="PCA Component V22")
    V23: float = Field(..., description="PCA Component V23")
    V24: float = Field(..., description="PCA Component V24")
    V25: float = Field(..., description="PCA Component V25")
    V26: float = Field(..., description="PCA Component V26")
    V27: float = Field(..., description="PCA Component V27")
    V28: float = Field(..., description="PCA Component V28")

    @field_validator("*")
    @classmethod
    def check_finite(cls, v: Any) -> Any:
        if isinstance(v, (int, float)):
            if math.isnan(v) or math.isinf(v):
                raise ValueError("Numeric fields must be finite numbers (no NaN or Infinity)")
        return v


class PredictionResponse(BaseModel):
    """
    Standardized inference response.
    Explicitly distinguishes model-generated risk score from a verified probability.
    """
    request_id: str
    model_version: str
    risk_score: float = Field(..., description="Raw model risk score between 0.0 and 1.0 (uncalibrated)")
    decision_threshold: float = Field(..., description="Offline-selected decision threshold applied")
    predicted_label: int = Field(..., description="0 for Legitimate, 1 for Fraudulent")
    classification: str = Field(..., description="'FRAUD_SUSPECT' or 'LEGITIMATE'")
    latency_ms: float
    status: str
    disclaimer: str = "Risk score is an uncalibrated classification model output, not an absolute statistical guarantee."


class ServiceHealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    database_connected: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    algorithm: str
    decision_threshold: float
    feature_count: int
    training_timestamp: str
    offline_metrics: Dict[str, Any]
