from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.submission import save_submission


def test_save_submission_can_avoid_repo_root_latest_write(tmp_path: Path):
    preds = np.array(
        [
            [0.7, 0.3],
            [0.2, 0.8],
        ]
    )
    test_df = pd.DataFrame({"track_id": [1, 2]})
    sample_submission = pd.DataFrame(
        {"track_id": [1, 2], "Birds of Prey": [0.0, 0.0], "Clutter": [0.0, 0.0]}
    )

    path = save_submission(
        preds,
        "pilot",
        submissions_dir=tmp_path / "subs",
        write_latest=False,
        test_df=test_df,
        sample_submission=sample_submission,
    )

    assert path.exists()
    assert not (tmp_path / "submission.csv").exists()
