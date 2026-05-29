from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import validate


def test_load_oof_accepts_explicit_candidates(tmp_path: Path):
    arr = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=float)
    candidate = tmp_path / "custom_oof.npy"
    np.save(candidate, arr)
    loaded, label = validate._load_oof([candidate])
    assert label == candidate.name
    assert loaded.shape == arr.shape


def test_load_test_base_accepts_explicit_candidates(tmp_path: Path, monkeypatch):
    arr = np.array([[0.6, 0.4], [0.2, 0.8]], dtype=float)
    candidate = tmp_path / "custom_test.npy"
    np.save(candidate, arr)
    dummy_test = pd.DataFrame({"timestamp_start_radar_utc": ["2026-02-01", "2026-09-01"]})
    monkeypatch.setattr(validate, "load_test", lambda: dummy_test)
    loaded, test_df, months = validate._load_test_base([candidate])
    assert loaded.shape == arr.shape
    assert list(test_df.columns) == ["timestamp_start_radar_utc"]
    assert months.tolist() == [2, 9]
