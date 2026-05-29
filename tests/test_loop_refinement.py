from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research_os.artifacts import ArtifactValidationError, inspect_prediction_artifact, load_prediction_matrix
from research_os.cli import build_parser
from src.data import CLASSES


def _seed_loop_root(root: Path) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    for relative in [
        "src/data.py",
        "src/submission.py",
        "src/validate.py",
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("seed\n", encoding="utf-8")

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_months = [1, 1, 1, 4, 4, 4, 9, 9, 9, 10, 10, 10]
    train_classes = CLASSES[:9] + CLASSES[:3]
    train_sizes = ["Small bird", "Medium bird", "Large bird", "Flock"] * 3
    train_df = pd.DataFrame(
        {
            "bird_group": train_classes,
            "timestamp_start_radar_utc": [f"2026-{month:02d}-01" for month in train_months],
            "airspeed": np.linspace(10, 25, 12),
            "min_z": np.linspace(10, 60, 12),
            "max_z": np.linspace(20, 80, 12),
            "radar_bird_size": train_sizes,
        }
    )
    train_df.to_csv(data_dir / "train.csv", index=False)

    test_months = [2, 5, 9, 10, 12, 2, 5, 9, 10, 12]
    test_sizes = ["Small bird", "Medium bird", "Large bird", "Flock", "Small bird"] * 2
    test_df = pd.DataFrame(
        {
            "track_id": list(range(1, 11)),
            "timestamp_start_radar_utc": [f"2026-{month:02d}-15" for month in test_months],
            "airspeed": np.linspace(11, 21, 10),
            "min_z": np.linspace(12, 42, 10),
            "max_z": np.linspace(22, 62, 10),
            "radar_bird_size": test_sizes,
        }
    )
    test_df.to_csv(data_dir / "test.csv", index=False)

    sample_submission = pd.DataFrame({"track_id": test_df["track_id"]})
    for cls in CLASSES:
        sample_submission[cls] = 0.0
    sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)

    gbif = pd.DataFrame({"month": list(range(1, 13))})
    for idx, cls in enumerate(CLASSES):
        gbif[cls] = np.linspace(10 + idx, 22 + idx, 12)
    gbif.to_csv(data_dir / "gbif_monthly_counts.csv", index=False)

    n_classes = len(CLASSES)
    train_labels = np.array([CLASSES.index(label) for label in train_classes])
    oof = np.full((len(train_df), n_classes), 0.2 / (n_classes - 1), dtype=float)
    oof[np.arange(len(train_df)), train_labels] = 0.8
    np.save(root / "oof_e175_best.npy", oof)

    test_preds = np.full((len(test_df), n_classes), 1.0 / n_classes, dtype=float)
    test_preds[:, 0] += 0.02
    test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)
    np.save(root / "test_e175_best.npy", test_preds)


def test_detect_lfs_pointer(tmp_path: Path):
    pointer = tmp_path / "pointer.npy"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 123\n",
        encoding="utf-8",
    )
    inspection = inspect_prediction_artifact(pointer)
    assert inspection.artifact_type == "git-lfs-pointer"
    assert inspection.valid is False
    try:
        load_prediction_matrix(pointer)
    except ArtifactValidationError as exc:
        assert "Git LFS pointer" in str(exc)
    else:
        raise AssertionError("Expected ArtifactValidationError")


def test_end_to_end_loop(tmp_path: Path):
    _seed_loop_root(tmp_path)
    parser = build_parser()

    init_args = parser.parse_args(["--root", str(tmp_path), "init", "--force"])
    assert init_args.func(init_args) == 0

    status_args = parser.parse_args(["--root", str(tmp_path), "status"])
    assert status_args.func(status_args) == 0

    doctor_args = parser.parse_args(["--root", str(tmp_path), "doctor"])
    assert doctor_args.func(doctor_args) == 0

    baseline_args = parser.parse_args([
        "--root", str(tmp_path), "run", "baseline-run",
        "--base-test", "test_e175_best.npy",
        "--output-label", "golden-loop",
    ])
    assert baseline_args.func(baseline_args) == 0
    assert not (tmp_path / "submission.csv").exists()

    compare_args = parser.parse_args([
        "--root", str(tmp_path), "validate-compare-runs",
        "--base-oof", "oof_e175_best.npy",
        "--base-test", "test_e175_best.npy",
    ])
    assert compare_args.func(compare_args) == 0

    report_args = parser.parse_args([
        "--root", str(tmp_path), "report-and-memory-update",
        "--experiment-id", "E902",
        "--name", "golden-loop",
        "--cv-map", "0.7000",
        "--note", "golden loop note",
        "--tracker-summary", "golden tracker summary",
        "--research-summary", "golden research summary",
    ])
    assert report_args.func(report_args) == 0

    assert (tmp_path / ".pilot" / "reports").exists()
    assert "E902" in (tmp_path / "EXPERIMENTS.md").read_text(encoding="utf-8")
    assert "golden tracker summary" in (tmp_path / "TRACKER.md").read_text(encoding="utf-8")
    assert "golden research summary" in (tmp_path / "RESEARCH.md").read_text(encoding="utf-8")
