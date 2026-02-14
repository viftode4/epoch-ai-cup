"""Submission file generation."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from .data import CLASSES, ROOT, load_test, load_sample_submission


def save_submission(
    test_preds: np.ndarray,
    name: str,
    cv_map: float = None,
) -> Path:
    """
    Save a submission CSV to submissions/ with metadata in filename.

    Args:
        test_preds: array of shape (n_test, 9) with columns in CLASSES order (alphabetical)
        name: experiment name (e.g. "v2_ensemble")
        cv_map: optional CV mAP score to include in filename

    Returns:
        Path to saved submission file.
    """
    test = load_test()
    sample_sub = load_sample_submission()
    # Use sample submission column order (NOT alphabetical CLASSES order)
    sub_columns = [c for c in sample_sub.columns if c != "track_id"]

    sub = pd.DataFrame({"track_id": test["track_id"]})
    for col in sub_columns:
        cls_idx = CLASSES.index(col)
        sub[col] = test_preds[:, cls_idx]

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
