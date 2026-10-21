"""
SentinelML Model Evaluation Module
Calculates fraud-specific metrics: Precision, Recall, F1, PR-AUC, ROC-AUC,
Confusion Matrix, threshold optimization, and illustrative business cost modeling.
"""

from typing import Dict, Any, List, Tuple
import math


def compute_confusion_matrix(
    y_true: List[int],
    y_scores: List[float],
    threshold: float = 0.5
) -> Dict[str, int]:
    """Computes TP, FP, TN, FN for a given decision threshold."""
    tp = fp = tn = fn = 0
    for y, score in zip(y_true, y_scores):
        pred = 1 if score >= threshold else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 0 and pred == 0:
            tn += 1
        elif y == 1 and pred == 0:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def calculate_metrics(
    y_true: List[int],
    y_scores: List[float],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes key classification metrics tailored for imbalanced fraud detection.
    """
    cm = compute_confusion_matrix(y_true, y_scores, threshold)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    roc_auc = calculate_roc_auc(y_true, y_scores)
    pr_auc = calculate_pr_auc(y_true, y_scores)

    return {
        "threshold": round(threshold, 3),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "specificity": round(specificity, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def calculate_roc_auc(y_true: List[int], y_scores: List[float]) -> float:
    """
    Computes ROC-AUC via Mann-Whitney U rank statistic.
    Handles ties by assigning mid-ranks.
    """
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return 0.5

    paired = sorted(zip(y_scores, y_true), key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    n = len(paired)

    while i < n:
        start_i = i
        val = paired[i][0]
        while i < n and paired[i][0] == val:
            i += 1
        end_i = i
        avg_rank = (start_i + 1 + end_i) / 2.0
        for j in range(start_i, end_i):
            if paired[j][1] == 1:
                rank_sum_pos += avg_rank

    u_stat = rank_sum_pos - (positives * (positives + 1)) / 2.0
    return u_stat / (positives * negatives)


def calculate_pr_auc(y_true: List[int], y_scores: List[float]) -> float:
    """
    Computes Precision-Recall Area Under Curve (PR-AUC) using trapezoidal integration.
    """
    thresholds = sorted(set(round(s, 4) for s in y_scores))
    if not thresholds:
        return 0.0

    pr_points = []
    for thresh in thresholds:
        cm = compute_confusion_matrix(y_true, y_scores, thresh)
        prec = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) > 0 else 1.0
        rec = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) > 0 else 0.0
        pr_points.append((rec, prec))

    pr_points.append((0.0, 1.0))
    # Sort by recall ascending
    pr_points = sorted(pr_points, key=lambda x: x[0])

    auc = 0.0
    for i in range(len(pr_points) - 1):
        r1, p1 = pr_points[i]
        r2, p2 = pr_points[i + 1]
        auc += (r2 - r1) * (p1 + p2) / 2.0
    return max(0.0, min(1.0, auc))


def find_optimal_threshold(
    y_val: List[int],
    y_scores: List[float],
    cost_fp: float = 15.0,  # Illustrative friction / manual verification cost (€)
    cost_fn: float = 200.0  # Illustrative average fraud loss (€)
) -> Tuple[float, Dict[str, Any]]:
    """
    Determines optimal decision threshold on VALIDATION data to minimize illustrative
    operational cost while preserving strong F1-score.
    
    NOTE: Cost values are illustrative pedagogical assumptions, NOT a live bank's real ledger.
    """
    best_threshold = 0.5
    best_cost = float("inf")
    best_metrics = {}

    for t_int in range(5, 95, 2):
        t = t_int / 100.0
        cm = compute_confusion_matrix(y_val, y_scores, t)
        total_cost = (cm["fp"] * cost_fp) + (cm["fn"] * cost_fn)
        metrics = calculate_metrics(y_val, y_scores, t)
        metrics["illustrative_cost_eur"] = total_cost

        # Optimize for minimum financial damage + investigation friction
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = t
            best_metrics = metrics

    return best_threshold, best_metrics
