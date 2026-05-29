"""Submission file generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASSES, ROOT, load_sample_submission, load_test


def save_submission(
    test_preds: np.ndarray,
    name: str,
    cv_map: float = None,
    *,
    submissions_dir: Path | None = None,
    latest_path: Path | None = None,
    write_latest: bool = True,
    test_df: pd.DataFrame | None = None,
    sample_submission: pd.DataFrame | None = None,
) -> Path:
    """
    Save a submission CSV to submissions/ with metadata in filename.

    Args:
        test_preds: array of shape (n_test, 9) with columns in CLASSES order (alphabetical)
        name: experiment name (e.g. "v2_ensemble")
        cv_map: optional CV mAP score to include in filename
        submissions_dir: optional output directory override
        latest_path: optional latest-file override
        write_latest: when False, skip the repo-root shortcut file
        test_df: optional test dataframe override for callers/tests
        sample_submission: optional sample submission override for callers/tests

    Returns:
        Path to saved submission file.
    """
    test = test_df if test_df is not None else load_test()
    sample_sub = sample_submission if sample_submission is not None else load_sample_submission()
    # Use sample submission column order (NOT alphabetical CLASSES order)
    sub_columns = [c for c in sample_sub.columns if c != "track_id"]

    sub = pd.DataFrame({"track_id": test["track_id"]})
    for col in sub_columns:
        cls_idx = CLASSES.index(col)
        sub[col] = test_preds[:, cls_idx]

    submissions_dir = submissions_dir or (ROOT / "submissions")
    submissions_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    score_str = f"_{cv_map:.4f}" if cv_map is not None else ""
    filename = f"{name}{score_str}_{ts}.csv"
    path = submissions_dir / filename

    sub.to_csv(path, index=False)
    print(f"Saved: {path.name} ({len(sub)} rows)")

    if write_latest:
        latest = latest_path or (ROOT / "submission.csv")
        latest.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(latest, index=False)

    return path
