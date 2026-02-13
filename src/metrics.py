"""Evaluation metrics matching the competition metric."""
import numpy as np
from sklearn.metrics import average_precision_score
from .data import CLASSES


def compute_map(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    """
    Compute macro-averaged mAP over all 9 classes.

    Args:
        y_true: integer labels (0-8), shape (n_samples,)
        y_pred: predicted probabilities, shape (n_samples, 9)

    Returns:
        (overall_mAP, {class_name: AP})
    """
    n_classes = len(CLASSES)
    y_onehot = np.eye(n_classes)[y_true]
    per_class = {}
    for c in range(n_classes):
        if y_onehot[:, c].sum() > 0:
            per_class[CLASSES[c]] = average_precision_score(y_onehot[:, c], y_pred[:, c])
        else:
            per_class[CLASSES[c]] = 0.0
    return np.mean(list(per_class.values())), per_class


def print_results(mAP: float, per_class: dict[str, float], label: str = ""):
    """Pretty-print mAP results with per-class breakdown."""
    if label:
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")
    print(f"\n  Overall mAP: {mAP:.4f}\n")
    for cls, ap in per_class.items():
        marker = " <-- weak" if ap < 0.6 else ""
        print(f"  {cls:15s}: {ap:.4f}{marker}")
