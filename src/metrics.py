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


def bootstrap_map_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap 95% CI for macro mAP and per-class APs.

    Args:
        y_true: integer labels (0-8), shape (n_samples,)
        y_pred: predicted probabilities, shape (n_samples, 9)
        n_bootstrap: number of bootstrap iterations
        ci: confidence interval (e.g. 0.95 for 95%)
        seed: random seed for reproducibility

    Returns:
        dict with keys: mean, std, ci_lo, ci_hi, per_class
        per_class maps class name -> (mean, ci_lo, ci_hi)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    maps = []
    per_class_maps = {cls: [] for cls in CLASSES}
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        m, per = compute_map(y_true[idx], y_pred[idx])
        maps.append(m)
        for cls, ap in per.items():
            per_class_maps[cls].append(ap)
    lo_idx = int((1 - ci) / 2 * n_bootstrap)
    hi_idx = int((1 + ci) / 2 * n_bootstrap)
    maps_sorted = sorted(maps)
    return {
        "mean": np.mean(maps),
        "std": np.std(maps),
        "ci_lo": maps_sorted[lo_idx],
        "ci_hi": maps_sorted[hi_idx],
        "per_class": {
            cls: (np.mean(v), sorted(v)[lo_idx], sorted(v)[hi_idx])
            for cls, v in per_class_maps.items()
        },
    }


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
