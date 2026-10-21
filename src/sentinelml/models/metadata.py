"""
SentinelML Model Metadata and Schema Definition
Tracks model lineage, hyperparameters, evaluation metrics, and feature schemas.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import json
from pathlib import Path


@dataclass
class ModelMetadata:
    model_name: str
    model_version: str
    algorithm: str
    decision_threshold: float
    feature_names: List[str]
    training_timestamp: str
    random_seed: int
    train_samples: int
    val_samples: int
    test_samples: int
    metrics_validation: Dict[str, float]
    metrics_test: Dict[str, float]
    hyperparameters: Dict[str, Any]
    preprocessor_stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "ModelMetadata":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
