"""
SentinelML Preprocessing and Stratified Splitting Module
Enforces strict zero-leakage principles:
1. Stratified Train / Validation / Test split preserving class proportions.
2. Preprocessing statistics (RobustScaler for Amount, StandardScaler for Time)
   are fitted strictly on the TRAINING set.
3. The test set remains completely unseen until final evaluation.
4. Preserves exact feature column ordering for training and inference.
"""

import math
import random
from typing import Dict, Any, List, Tuple
from sentinelml.config import (
    ALL_FEATURES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


def stratified_split(
    X: List[List[float]],
    y: List[int],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[
    List[List[float]], List[int],
    List[List[float]], List[int],
    List[List[float]], List[int]
]:
    """
    Performs stratified train/val/test splitting to maintain identical positive class
    fractions across all three subsets.
    """
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-5), \
        "Split ratios must sum to 1.0"

    rng = random.Random(seed)

    # Separate indices by class
    class_0_indices = [i for i, label in enumerate(y) if label == 0]
    class_1_indices = [i for i, label in enumerate(y) if label == 1]

    rng.shuffle(class_0_indices)
    rng.shuffle(class_1_indices)

    def split_group(indices: List[int]):
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        return train_idx, val_idx, test_idx

    c0_train, c0_val, c0_test = split_group(class_0_indices)
    c1_train, c1_val, c1_test = split_group(class_1_indices)

    train_indices = c0_train + c1_train
    val_indices = c0_val + c1_val
    test_indices = c0_test + c1_test

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]

    X_val = [X[i] for i in val_indices]
    y_val = [y[i] for i in val_indices]

    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, y_train, X_val, y_val, X_test, y_test


class FraudPreprocessor:
    """
    Fitted feature transformer that scales Time and Amount without data leakage.
    PCA features V1-V28 are preserved as-is.
    """

    def __init__(self):
        self.fitted = False
        self.time_mean = 0.0
        self.time_std = 1.0
        self.amount_median = 0.0
        self.amount_iqr = 1.0
        self.time_idx = ALL_FEATURES.index("Time")
        self.amount_idx = ALL_FEATURES.index("Amount")

    def fit(self, X_train: List[List[float]]) -> "FraudPreprocessor":
        """Fit scalers strictly on training data."""
        times = [row[self.time_idx] for row in X_train]
        amounts = [row[self.amount_idx] for row in X_train]

        # Time StandardScaler: mean and standard deviation
        n = len(times)
        self.time_mean = sum(times) / n
        variance = sum((t - self.time_mean) ** 2 for t in times) / (n - 1 if n > 1 else 1)
        self.time_std = math.sqrt(variance) if variance > 1e-9 else 1.0

        # Amount RobustScaler: median and IQR (interquartile range)
        sorted_amounts = sorted(amounts)
        self.amount_median = self._percentile(sorted_amounts, 50)
        q25 = self._percentile(sorted_amounts, 25)
        q75 = self._percentile(sorted_amounts, 75)
        self.amount_iqr = max(q75 - q25, 1e-4)

        self.fitted = True
        return self

    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """Transform features using training statistics."""
        if not self.fitted:
            raise RuntimeError("FraudPreprocessor must be fitted on training data before transforming.")

        transformed = []
        for row in X:
            new_row = list(row)
            # Scale Time
            new_row[self.time_idx] = (new_row[self.time_idx] - self.time_mean) / self.time_std
            # Scale Amount with RobustScaler
            new_row[self.amount_idx] = (new_row[self.amount_idx] - self.amount_median) / self.amount_iqr
            transformed.append(new_row)
        return transformed

    def fit_transform(self, X_train: List[List[float]]) -> List[List[float]]:
        return self.fit(X_train).transform(X_train)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_mean": self.time_mean,
            "time_std": self.time_std,
            "amount_median": self.amount_median,
            "amount_iqr": self.amount_iqr,
            "fitted": self.fitted,
            "feature_order": ALL_FEATURES,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FraudPreprocessor":
        prep = cls()
        prep.time_mean = data["time_mean"]
        prep.time_std = data["time_std"]
        prep.amount_median = data["amount_median"]
        prep.amount_iqr = data["amount_iqr"]
        prep.fitted = data.get("fitted", True)
        return prep

    @staticmethod
    def _percentile(sorted_data: List[float], p: float) -> float:
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return d0 + d1


if __name__ == "__main__":
    from sentinelml.data.load_data import load_credit_card_data
    X, y, _ = load_credit_card_data()
    X_tr, y_tr, X_v, y_v, X_te, y_te = stratified_split(X, y)
    print(f"Train: {len(X_tr)} (fraud: {sum(y_tr)})")
    print(f"Val:   {len(X_v)} (fraud: {sum(y_v)})")
    print(f"Test:  {len(X_te)} (fraud: {sum(y_te)})")
    preproc = FraudPreprocessor().fit(X_tr)
    print("Preprocessor fitted params:", preproc.to_dict())
