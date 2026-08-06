import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
from typing import List, Dict


def quadratic_weighted_kappa(y_true: List[int], y_pred: List[int]) -> float:
    """Compute Quadratic Weighted Kappa (QWK)."""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def per_class_accuracy(y_true: List[int], y_pred: List[int], num_classes: int = 5) -> Dict[int, float]:
    """Compute per-class accuracy."""
    per_class = {}
    for c in range(num_classes):
        mask = np.array(y_true) == c
        if mask.sum() > 0:
            per_class[c] = accuracy_score(np.array(y_true)[mask], np.array(y_pred)[mask])
        else:
            per_class[c] = 0.0
    return per_class


def overall_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Compute overall accuracy (fraction of correct predictions)."""
    return accuracy_score(y_true, y_pred)


def forgetting_metric(perf_before: float, perf_after: float) -> float:
    """Compute forgetting metric (difference in source performance)."""
    return perf_before - perf_after


def adaptation_trigger_rate(num_adaptations: int, total_batches: int) -> float:
    """Compute adaptation trigger rate."""
    if total_batches == 0:
        return 0.0
    return num_adaptations / total_batches


def recovery_speed(per_batch_acc: List[float], shift_point: int, threshold: float = 0.95) -> int:
    """Compute recovery speed (batches to reach threshold of oracle performance)."""
    if shift_point >= len(per_batch_acc):
        return -1

    oracle_perf = max(per_batch_acc[shift_point:])
    target_perf = threshold * oracle_perf

    for i in range(shift_point, len(per_batch_acc)):
        if per_batch_acc[i] >= target_perf:
            return i - shift_point

    return -1  # Never reached threshold