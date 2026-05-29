from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .contracts import ArtifactInspection


_LFS_PREFIX = "version https://git-lfs.github.com/spec/v1"


class ArtifactValidationError(ValueError):
    """Raised when a pilot artifact cannot be used safely."""


def make_run_label(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_")
    return cleaned or "run"


def inspect_prediction_artifact(path: Path) -> ArtifactInspection:
    path = path.resolve()
    if not path.exists():
        return ArtifactInspection(path=str(path), exists=False, kind="missing", valid=False, message="Artifact does not exist")

    try:
        prefix = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
    except Exception:
        prefix = []

    if prefix and prefix[0].strip() == _LFS_PREFIX:
        return ArtifactInspection(
            path=str(path),
            exists=True,
            kind="git-lfs-pointer",
            valid=False,
            message="Artifact is a Git LFS pointer, not a resolved numpy array",
        )

    try:
        arr = np.asarray(np.load(path, allow_pickle=True), dtype=float)
    except Exception as exc:
        return ArtifactInspection(
            path=str(path),
            exists=True,
            kind="invalid",
            valid=False,
            message=str(exc),
        )

    return ArtifactInspection(
        path=str(path),
        exists=True,
        kind="numpy-array",
        valid=True,
        shape=list(arr.shape),
        dtype=str(arr.dtype),
    )


def load_prediction_matrix(path: Path) -> tuple[np.ndarray, ArtifactInspection]:
    inspection = inspect_prediction_artifact(path)
    if not inspection.valid:
        raise ArtifactValidationError(f"Invalid prediction artifact at {inspection.path}: {inspection.message}")
    arr = np.asarray(np.load(path, allow_pickle=True), dtype=float)
    return arr, inspection


def require_prediction_matrix(path: Path) -> tuple[np.ndarray, ArtifactInspection]:
    return load_prediction_matrix(path)
