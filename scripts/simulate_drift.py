#!/usr/bin/env python3
"""
SentinelML Drift Simulation Script
Simulates realistic production distribution shifts (e.g. changing spending habits,
novel fraud patterns, or sensor drift) by generating a batch of transactions with
mean shifts and logging them to the prediction store.

EDUCATIONAL NOTE:
This script demonstrates the sensitivity of Kolmogorov-Smirnov and PSI drift metrics.
It NEVER alters the frozen reference dataset or test set.
"""

import argparse
import random
import uuid
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentinelml.config import ALL_FEATURES, CURRENT_MODEL_VERSION
from sentinelml.monitoring.logging_store import PredictionStore
from sentinelml.monitoring.drift import analyze_drift_batch


def run_drift_simulation(
    num_requests: int = 150,
    shift_magnitude: float = 3.2,
    drift_features: list = None,
) -> None:
    if drift_features is None:
        drift_features = ["Amount", "V14", "V10", "V4"]

    print(f"Starting drift simulation: generating {num_requests} requests...")
    print(f"Target drifted features: {drift_features} with shift magnitude +{shift_magnitude} sigma")

    store = PredictionStore()
    simulated_payloads = []

    for i in range(num_requests):
        req_id = f"sim-{uuid.uuid4().hex[:10]}"
        payload = {}

        for feat in ALL_FEATURES:
            if feat in drift_features:
                # Injected mean shift
                val = random.gauss(shift_magnitude, 1.8)
            else:
                val = random.gauss(0.0, 1.0)
            payload[feat] = round(val, 4)

        # Base Amount adjustment if shifted
        if "Amount" in drift_features:
            payload["Amount"] = max(1.0, round(random.gauss(450.0, 120.0), 2))
        else:
            payload["Amount"] = max(1.0, round(random.expovariate(1/45.0), 2))

        payload["Time"] = round(random.uniform(100.0, 150000.0), 2)

        # Risk score calculation
        score = random.uniform(0.65, 0.98) if any(feat in ("V14", "V10") for feat in drift_features) else random.uniform(0.01, 0.25)
        pred_label = 1 if score >= 0.50 else 0

        store.log_prediction(
            request_id=req_id,
            model_version=CURRENT_MODEL_VERSION,
            risk_score=score,
            decision_threshold=0.50,
            predicted_label=pred_label,
            latency_ms=random.uniform(8.0, 22.0),
            feature_payload=payload,
            status="SUCCESS"
        )
        simulated_payloads.append(payload)

    print(f"Logged {num_requests} shifted inference events into {store.db_path}.")
    print("Now evaluating statistical drift across logged samples...")

    # Compare with reference
    rng = random.Random(42)
    ref_dict = {f: [rng.gauss(0.0, 1.0) for _ in range(500)] for f in ALL_FEATURES}
    curr_dict = {f: [p[f] for p in simulated_payloads] for f in ALL_FEATURES}

    report = analyze_drift_batch(ref_dict, curr_dict)
    print("\n--- DRIFT MONITORING REPORT ---")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Monitored Features: {report['monitored_features_count']}")
    print(f"Drifted Features Detected: {report['drifted_features_count']}")
    for f in drift_features:
        if f in report["features"]:
            res = report["features"][f]
            print(f"  Feature {f:6s} -> PSI: {res['psi']:.4f} ({res['psi_status']}), KS p-val: {res['p_value']:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate feature drift for SentinelML")
    parser.add_argument("--samples", type=int, default=100, help="Number of simulated requests")
    parser.add_argument("--shift", type=float, default=2.8, help="Shift magnitude in standard deviations")
    args = parser.parse_args()

    run_drift_simulation(num_requests=args.samples, shift_magnitude=args.shift)
