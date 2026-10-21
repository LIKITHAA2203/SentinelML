"""
SentinelML Configuration Module
Provides centralized, reproducible path resolution and hyperparameter settings.
"""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Core Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MONITORING_DB_PATH = ARTIFACTS_DIR / "monitoring.db"

# Data schema
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
NUMERICAL_FEATURES = ["Time", "Amount"]
ALL_FEATURES = ["Time"] + PCA_FEATURES + ["Amount"]
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = ALL_FEATURES + [TARGET_COLUMN]

# Split settings
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Decision Threshold & Alerting
DEFAULT_DECISION_THRESHOLD = 0.42
DRIFT_P_VALUE_THRESHOLD = 0.05
DRIFT_PSI_WARNING_THRESHOLD = 0.10
DRIFT_PSI_CRITICAL_THRESHOLD = 0.25

# Model versioning
CURRENT_MODEL_VERSION = "v1.0.0-RandomForest"
MODEL_FILENAME = "fraud_pipeline.joblib"
METADATA_FILENAME = "model_metadata.json"

# Ensure runtime directories exist
for directory in [DATA_DIR / "raw", PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
