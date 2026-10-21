"""
SentinelML Statistical Drift Monitoring Engine
Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests
between the baseline training reference dataset and recent live prediction traffic.
"""

import math
from typing import Dict, Any, List, Tuple
from sentinelml.config import (
    DRIFT_P_VALUE_THRESHOLD,
    DRIFT_PSI_WARNING_THRESHOLD,
    DRIFT_PSI_CRITICAL_THRESHOLD,
    ALL_FEATURES
)


def calculate_psi(
    reference: List[float],
    current: List[float],
    num_bins: int = 10
) -> Tuple[float, str]:
    """
    Computes Population Stability Index (PSI) across quantile bins defined on reference data.
    
    Interpretation:
    - PSI < 0.10: Stable (No shift)
    - 0.10 <= PSI < 0.25: Moderate Shift (Warning)
    - PSI >= 0.25: Significant Shift (Critical Alert)
    """
    if len(reference) < 20 or len(current) < 20:
        return 0.0, "INSUFFICIENT_DATA"

    sorted_ref = sorted(reference)
    n_ref = len(sorted_ref)
    n_curr = len(current)

    # Calculate bin edges using deciles of the reference distribution
    bin_edges = []
    for i in range(1, num_bins):
        idx = int(i * (n_ref / num_bins))
        bin_edges.append(sorted_ref[idx])
    bin_edges = sorted(list(set(bin_edges)))  # eliminate duplicate edges

    if len(bin_edges) < 2:
        return 0.0, "STABLE"

    # Count occurrences in bins
    def get_bin_counts(data: List[float], edges: List[float]) -> List[int]:
        counts = [0] * (len(edges) + 1)
        for val in data:
            placed = False
            for b_idx, edge in enumerate(edges):
                if val <= edge:
                    counts[b_idx] += 1
                    placed = True
                    break
            if not placed:
                counts[-1] += 1
        return counts

    ref_counts = get_bin_counts(reference, bin_edges)
    curr_counts = get_bin_counts(current, bin_edges)

    psi_total = 0.0
    epsilon = 1e-4  # Zero-frequency smoothing

    for r_cnt, c_cnt in zip(ref_counts, curr_counts):
        pct_ref = max(r_cnt / n_ref, epsilon)
        pct_curr = max(c_cnt / n_curr, epsilon)
        psi_total += (pct_curr - pct_ref) * math.log(pct_curr / pct_ref)

    if psi_total >= DRIFT_PSI_CRITICAL_THRESHOLD:
        status = "CRITICAL_DRIFT"
    elif psi_total >= DRIFT_PSI_WARNING_THRESHOLD:
        status = "WARNING_DRIFT"
    else:
        status = "STABLE"

    return round(psi_total, 4), status


def calculate_ks_statistic(
    reference: List[float],
    current: List[float]
) -> Tuple[float, float, bool]:
    """
    Computes two-sample Kolmogorov-Smirnov statistic (maximum vertical distance between empirical CDFs)
    and asymptotic p-value approximation.
    """
    n1 = len(reference)
    n2 = len(current)
    if n1 < 10 or n2 < 10:
        return 0.0, 1.0, False

    combined = sorted([(x, 1) for x in reference] + [(x, 2) for x in current], key=lambda t: t[0])
    cdf1 = 0.0
    cdf2 = 0.0
    max_d = 0.0

    for val, sample_id in combined:
        if sample_id == 1:
            cdf1 += 1.0 / n1
        else:
            cdf2 += 1.0 / n2
        d = abs(cdf1 - cdf2)
        if d > max_d:
            max_d = d

    # Asymptotic p-value approximation via Kolmogorov distribution
    en = math.sqrt((n1 * n2) / (n1 + n2))
    lambda_param = (en + 0.12 + 0.11 / en) * max_d

    # Kolmogorov series approximation
    p_value = 0.0
    for j in range(1, 101):
        term = 2.0 * ((-1) ** (j - 1)) * math.exp(-2.0 * (j ** 2) * (lambda_param ** 2))
        p_value += term
        if abs(term) < 1e-6:
            break
    p_value = max(0.0, min(1.0, p_value))
    is_drift = p_value < DRIFT_P_VALUE_THRESHOLD

    return round(max_d, 4), round(p_value, 5), is_drift


def analyze_drift_batch(
    reference_dict: Dict[str, List[float]],
    current_dict: Dict[str, List[float]]
) -> Dict[str, Any]:
    """
    Runs both PSI and KS statistical checks across all monitored transaction features.
    
    IMPORTANT METHODOLOGICAL NOTE:
    Data drift indicates that incoming feature distributions deviate from the training baseline.
    It does NOT prove that model accuracy has degraded, nor does it identify fraud rates without
    verified ground-truth feedback.
    """
    results: Dict[str, Any] = {}
    drifted_count = 0
    total_features = 0

    for feat in ALL_FEATURES:
        if feat in reference_dict and feat in current_dict:
            ref_vals = reference_dict[feat]
            curr_vals = current_dict[feat]
            if len(ref_vals) >= 20 and len(curr_vals) >= 20:
                total_features += 1
                psi_val, psi_status = calculate_psi(ref_vals, curr_vals)
                ks_stat, p_val, is_ks_drift = calculate_ks_statistic(ref_vals, curr_vals)

                is_drifted = (psi_status in ("WARNING_DRIFT", "CRITICAL_DRIFT")) or is_ks_drift
                if is_drifted:
                    drifted_count += 1

                results[feat] = {
                    "psi": psi_val,
                    "psi_status": psi_status,
                    "ks_statistic": ks_stat,
                    "p_value": p_val,
                    "is_drifted": is_drifted
                }

    overall_status = "HEALTHY"
    if drifted_count >= 5:
        overall_status = "CRITICAL_DRIFT"
    elif drifted_count >= 2:
        overall_status = "WARNING_DRIFT"

    return {
        "overall_status": overall_status,
        "monitored_features_count": total_features,
        "drifted_features_count": drifted_count,
        "features": results,
        "notice": "Statistical drift flags distribution shifts against baseline training data. True performance requires verified labels."
    }
