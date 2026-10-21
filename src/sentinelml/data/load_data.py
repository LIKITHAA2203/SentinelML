"""
SentinelML Data Loader
Loads and parses credit card dataset into structured features and target.
"""

import csv
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any
from sentinelml.config import RAW_DATA_PATH, ALL_FEATURES, TARGET_COLUMN
from sentinelml.data.validate_data import validate_csv_dataset, DataValidationError

logger = logging.getLogger(__name__)


def load_credit_card_data(
    file_path: Path = RAW_DATA_PATH,
    validate: bool = True
) -> Tuple[List[List[float]], List[int], List[str]]:
    """
    Loads credit card dataset from CSV.
    
    Returns:
        X: List of feature rows (floats), preserving exact ALL_FEATURES order.
        y: List of target binary integers (0 or 1).
        feature_names: List of feature column names in order.
    """
    if validate:
        validate_csv_dataset(file_path)

    X: List[List[float]] = []
    y: List[int] = []

    with open(file_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_to_idx = {name: idx for idx, name in enumerate(header)}
        target_idx = col_to_idx[TARGET_COLUMN]
        feat_indices = [col_to_idx[col] for col in ALL_FEATURES]

        for row in reader:
            row_features = [float(row[idx]) for idx in feat_indices]
            row_target = int(row[target_idx])
            X.append(row_features)
            y.append(row_target)

    logger.info(f"Loaded {len(X)} records across {len(ALL_FEATURES)} features.")
    return X, y, ALL_FEATURES


if __name__ == "__main__":
    X, y, feats = load_credit_card_data()
    print(f"Loaded {len(X)} records with {len(feats)} features. Sample target: {y[:10]}")
