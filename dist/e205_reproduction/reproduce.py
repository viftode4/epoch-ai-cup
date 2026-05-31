"""Reproduce the E205 winning submission (e205_multi_restart_T09, 0.545 private LB).

Self-contained: needs ONLY the files shipped in this package (model prediction
arrays in models/, the blend weights in weights.json, and track_ids.csv).
No competition data required.

The submission is a weighted probability blend of 11 diverse models, followed by
temperature sharpening (T=0.9) and per-row renormalization.

Usage:
    pip install -r requirements.txt
    python reproduce.py                 # writes e205_multi_restart_T09.csv
    python reproduce.py --variant raw   # the un-sharpened blend
    python reproduce.py --variant T085  # T=0.85 sharpening

Verify it matches the original (if you have the reference CSV):
    python reproduce.py --check e205_multi_restart_T09.reference.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Model output column order (the order of the columns in every models/test_*.npy)
CLASSES = [
    "Birds of Prey", "Clutter", "Cormorants", "Ducks", "Geese",
    "Gulls", "Pigeons", "Songbirds", "Waders",
]
# Column order required by the competition submission CSV
SUB_CLASSES = [
    "Clutter", "Cormorants", "Pigeons", "Ducks", "Geese",
    "Gulls", "Birds of Prey", "Waders", "Songbirds",
]

# Temperature per variant. raw = no sharpening; T<1 sharpens the distribution.
VARIANT_T = {"raw": None, "T085": 0.85, "T09": 0.9}


def build_blend():
    """Weighted arithmetic blend of the 11 model test predictions."""
    spec = json.loads((HERE / "weights.json").read_text())["multi_restart"]
    weights = spec["weights"]

    blend = None
    for name, w in weights.items():
        arr = np.load(HERE / "models" / f"test_{name}.npy", allow_pickle=True).astype(float)
        if arr.shape[1] != len(CLASSES):
            raise ValueError(f"test_{name}.npy has {arr.shape[1]} cols, expected {len(CLASSES)}")
        blend = arr * w if blend is None else blend + arr * w
    return blend, spec["oof_macro_map"]


def sharpen(blend, T):
    if T is None:
        return blend
    s = blend ** (1.0 / T)
    return s / s.sum(axis=1, keepdims=True)


def write_submission(preds, out_path):
    track_ids = pd.read_csv(HERE / "track_ids.csv")["track_id"].values
    if len(track_ids) != len(preds):
        raise ValueError(f"{len(track_ids)} track_ids vs {len(preds)} rows")
    df = pd.DataFrame({"track_id": track_ids})
    for cls in SUB_CLASSES:
        df[cls] = preds[:, CLASSES.index(cls)]
    df.to_csv(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANT_T), default="T09",
                    help="raw | T085 | T09 (default: T09, the 0.545 submission)")
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument("--check", default=None,
                    help="reference CSV to compare against (max abs diff)")
    args = ap.parse_args()

    blend, oof = build_blend()
    print(f"Loaded 11-model blend (OOF macro-mAP = {oof:.4f})")
    preds = sharpen(blend, VARIANT_T[args.variant])

    out = args.out or f"e205_multi_restart_{args.variant}.csv"
    write_submission(preds, out)
    print(f"Wrote {out}  ({preds.shape[0]} rows)")

    if args.check:
        ref = pd.read_csv(args.check)
        got = pd.read_csv(out)
        cols = [c for c in ref.columns if c != "track_id"]
        diff = np.abs(ref[cols].values - got[cols].values).max()
        print(f"Max abs diff vs {args.check}: {diff:.3e}  "
              f"({'MATCH' if diff < 1e-6 else 'MISMATCH'})")


if __name__ == "__main__":
    main()
