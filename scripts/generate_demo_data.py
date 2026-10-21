#!/usr/bin/env python3
"""
SentinelML — Reproducible Demo Data Generator
Generates a schema-identical synthetic development dataset modeled after the
Kaggle / ULB European Credit Card Fraud dataset for instant local testing without
requiring an external download or credentials.
"""

import argparse
import csv
import math
import random
from pathlib import Path


def generate_dataset(
    output_path: Path,
    num_samples: int = 5000,
    fraud_rate: float = 0.0017,
    seed: int = 42,
) -> None:
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount", "Class"]
    )

    num_frauds = max(1, int(round(num_samples * fraud_rate)))
    num_legit = num_samples - num_frauds

    labels = [0] * num_legit + [1] * num_frauds
    random.shuffle(labels)

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        current_time = 0.0
        for label in labels:
            # Time advances in seconds over 2 days (172800 seconds)
            current_time += random.uniform(0.5, 70.0)
            if current_time > 172800.0:
                current_time = current_time % 172800.0

            # V1-V28 PCA features
            features = []
            for i in range(1, 29):
                # Fraud transactions typically show strong shifts on V10, V12, V14, V17 and V4
                if label == 1:
                    if i in (10, 12, 14, 17):
                        val = random.gauss(-4.5, 2.0)
                    elif i in (4, 11):
                        val = random.gauss(3.8, 1.8)
                    else:
                        val = random.gauss(0.0, 1.5)
                else:
                    val = random.gauss(0.0, 1.0)
                features.append(round(val, 6))

            # Amount is right-skewed log-normal
            if label == 1:
                # Fraud often clusters at both high ticket items and small probing amounts
                amount = random.choice([
                    round(random.uniform(1.0, 15.0), 2),
                    round(random.uniform(120.0, 1850.0), 2)
                ])
            else:
                log_amount = random.gauss(3.1, 1.2)  # exp(3.1) ~ 22 Euros
                amount = max(0.0, round(math.exp(log_amount) - 1.0, 2))
                if amount > 15000:
                    amount = 120.50

            row = [round(current_time, 2)] + features + [amount, label]
            writer.writerow(row)

    print(f"Generated {num_samples} samples ({num_frauds} fraud, {num_legit} legit, {fraud_rate*100:.3f}% rate)")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SentinelML demo credit card dataset")
    parser.add_argument("--samples", type=int, default=5000, help="Number of records to generate")
    parser.add_argument("--fraud-rate", type=float, default=0.00172, help="Proportion of fraudulent transactions")
    parser.add_argument("--out", type=str, default="data/raw/creditcard.csv", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_dataset(
        output_path=Path(args.out),
        num_samples=args.samples,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )
