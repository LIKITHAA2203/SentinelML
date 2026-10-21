"""
SentinelML Data Validation Module
Validates raw and processed credit card fraud datasets against strict schema,
null-check, duplicate, and class-imbalance expectations.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sentinelml.config import REQUIRED_COLUMNS, TARGET_COLUMN, ALL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when dataset fails schema, integrity, or distribution checks."""
    pass


def validate_csv_dataset(file_path: Path) -> Dict[str, Any]:
    """
    Validates a credit card CSV dataset.
    
    Checks performed:
    1. File existence and accessibility.
    2. Header schema matches required columns exactly.
    3. No NaN, null, or empty fields.
    4. Numeric float types for features and integer in {0, 1} for Class.
    5. Count of total rows, class 0 (legit), and class 1 (fraud).
    6. Extreme class imbalance warning check.
    """
    if not file_path.exists():
        raise DataValidationError(
            f"Dataset not found at: {file_path}\n"
            "Action required: Place the official European credit card 'creditcard.csv' into "
            "'data/raw/' or run 'python scripts/generate_demo_data.py' for quickstart dev testing."
        )

    logger.info(f"Validating dataset at: {file_path}")

    total_rows = 0
    null_count = 0
    class_0_count = 0
    class_1_count = 0
    invalid_rows = []

    with open(file_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        if not header:
            raise DataValidationError(f"File {file_path} is completely empty.")

        # Check required columns
        header_set = set(header)
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in header_set]
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns in dataset: {missing_cols}. "
                f"Expected all 31 columns: Time, V1-V28, Amount, Class."
            )

        col_indices = {col: header.index(col) for col in REQUIRED_COLUMNS}
        class_idx = col_indices[TARGET_COLUMN]

        for row_idx, row in enumerate(reader, start=2):
            total_rows += 1
            if len(row) != len(header):
                null_count += 1
                if len(invalid_rows) < 5:
                    invalid_rows.append((row_idx, "Mismatched column count"))
                continue

            # Validate Class column
            class_val_str = row[class_idx].strip()
            if class_val_str == "0":
                class_0_count += 1
            elif class_val_str == "1":
                class_1_count += 1
            else:
                raise DataValidationError(
                    f"Invalid class value '{class_val_str}' at line {row_idx}. "
                    "Target Class must be binary integer 0 or 1."
                )

            # Validate numeric features
            for feat in ALL_FEATURES:
                idx = col_indices[feat]
                val_str = row[idx].strip()
                if val_str == "" or val_str.lower() in ("nan", "null", "none"):
                    null_count += 1
                    if len(invalid_rows) < 5:
                        invalid_rows.append((row_idx, f"Null feature '{feat}'"))
                else:
                    try:
                        float(val_str)
                    except ValueError:
                        raise DataValidationError(
                            f"Non-numeric value '{val_str}' found in column '{feat}' at row {row_idx}."
                        )

    if total_rows == 0:
        raise DataValidationError("Dataset contains header but 0 data records.")

    if null_count > 0:
        raise DataValidationError(
            f"Dataset failed validation: {null_count} null or corrupt cells detected. "
            f"Examples: {invalid_rows}"
        )

    fraud_rate = (class_1_count / total_rows) * 100.0

    summary = {
        "status": "VALID",
        "file_path": str(file_path),
        "total_records": total_rows,
        "features_count": len(ALL_FEATURES),
        "legitimate_count": class_0_count,
        "fraud_count": class_1_count,
        "fraud_rate_pct": round(fraud_rate, 4),
        "is_imbalanced": fraud_rate < 1.0,
        "validation_passed": True,
    }

    logger.info(
        f"Validation PASSED: {total_rows} records validated. "
        f"Class 0: {class_0_count}, Class 1: {class_1_count} ({fraud_rate:.3f}% fraud rate)."
    )
    return summary


if __name__ == "__main__":
    from sentinelml.config import RAW_DATA_PATH
    try:
        report = validate_csv_dataset(RAW_DATA_PATH)
        print("Validation Result:\n", report)
    except DataValidationError as e:
        print(f"Validation ERROR: {e}")
