"""E121: Graph-level sharpening (consistency) without averaging away minorities.

Hypothesis
----------
E113/E116 used *averaging* (explicitly or implicitly) inside inferred same-flock
groups, which can wash out minority probabilities and lock us at 0.59.

Instead we apply a multiplicative "consensus prior":
  p'_i ∝ p_i^(1-β) * g_c^β
where g is the component's (weighted) mean probability vector.

This keeps peaks (ranking signal) while nudging within-group consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CLASSES  # noqa: E402
from src.submission import save_submission  # noqa: E402


def renorm_rows(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def load_submission_probs(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    p = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    for j, cls in enumerate(CLASSES):
        p[:, j] = df[cls].to_numpy(dtype=np.float32)
    return renorm_rows(p)


def connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = np.zeros(n, dtype=bool)
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
                    comp.append(v)
        comps.append(comp)
    return comps


def main() -> None:
    print("=" * 72, flush=True)
    print("E121 GRAPH SHARPENING".center(72), flush=True)
    print("=" * 72, flush=True)

    base = ROOT / "submissions" / "e111_mega_ensemble_geo5_20260302_1333.csv"
    if not base.exists():
        raise FileNotFoundError(base)
    p = load_submission_probs(base)

    # Use the already-good E116 edge set implicitly (from its training run):
    # We reconstruct edges by reusing the stored test pairs from E116? Not saved.
    # So we use E113's hard-thresholded component file as a proxy graph definition:
    # (If you want to use E116 edges directly, extend E116 to save ii/jj/w arrays.)
    #
    # Practically: use E113-smoothed submission as a *graph indicator* by reading
    # its track-wise deltas vs base, but we don't have that either.
    #
    # Therefore this script is a template; actual edges should be loaded/saved.
    raise RuntimeError("Template: needs saved edge list from E116/E113 to run.")


if __name__ == "__main__":
    main()

