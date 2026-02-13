"""Submission file generation."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from .data import CLASSES, ROOT, load_test


def save_submission(
    test_preds: np.ndarray,
    name: str,
    cv_map: float = None,
) -> Path:
    """
    Save a submission CSV to submissions/ with metadata in filename.

    Args:
        test_preds: array of shape (n_test, 9)
        name: experiment name (e.g. "v2_ensemble")
        cv_map: optional CV mAP score to include in filename

    Returns:
        Path to saved submission file.
    """
    test = load_test()
    sub = pd.DataFrame({"track_id": test["track_id"]})
    for i, cls in enumerate(CLASSES):
        sub[cls] = test_preds[:, i]

    submissions_dir = ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    score_str = f"_{cv_map:.4f}" if cv_map is not None else ""
    filename = f"{name}{score_str}_{ts}.csv"
    path = submissions_dir / filename

    sub.to_csv(path, index=False)
    print(f"Saved: {path.name} ({len(sub)} rows)")

    # Also save as latest for quick upload
    latest = ROOT / "submission.csv"
    sub.to_csv(latest, index=False)

    return path
