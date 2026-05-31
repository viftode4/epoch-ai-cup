"""Assemble + verify + zip the E205 reproduction package."""
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent.parent          # project root
PKG = Path(__file__).resolve().parent / "e205_reproduction"
MODELS_DIR = PKG / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

POOL = ["e79", "e175_best", "e175_lgb", "e179_best", "e185_tabpfn_relabel",
        "e185_tabpfn_all", "e186_ovo", "e180_cnn", "e187_blend", "e173", "e179_cb"]
CLASSES = ["Birds of Prey", "Clutter", "Cormorants", "Ducks", "Geese",
           "Gulls", "Pigeons", "Songbirds", "Waders"]

# 1) weights.json
shutil.copy(ROOT / "submissions" / "e205_weights.json", PKG / "weights.json")

# 2) model arrays (test_ for reproduce, oof_ for optimize)
for name in POOL:
    shutil.copy(ROOT / f"test_{name}.npy", MODELS_DIR / f"test_{name}.npy")
    shutil.copy(ROOT / f"oof_{name}.npy", MODELS_DIR / f"oof_{name}.npy")

# 3) track_ids.csv (from the reference submission so no competition data ships)
ref = pd.read_csv(ROOT / "submissions" / "e205_multi_restart_T09.csv")
ref[["track_id"]].to_csv(PKG / "track_ids.csv", index=False)
# keep a copy of the reference for --check verification (not zipped)
ref.to_csv(PKG / "e205_multi_restart_T09.reference.csv", index=False)

# 4) y.npy training labels as class indices
train = pd.read_csv(ROOT / "data" / "train.csv")
y = np.array([CLASSES.index(c) for c in train["bird_group"].values])
np.save(PKG / "y.npy", y)

print(f"Assembled package at {PKG}")
print(f"  models: {len(list(MODELS_DIR.glob('*.npy')))} arrays")
print(f"  track_ids: {len(ref)}  labels: {len(y)}")

# 5) zip (exclude the reference CSV — it's only for local verification)
zip_path = Path(__file__).resolve().parent / "e205_reproduction.zip"
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(PKG.rglob("*")):
        if f.is_file() and f.name != "e205_multi_restart_T09.reference.csv":
            zf.write(f, f.relative_to(PKG.parent))
size_mb = zip_path.stat().st_size / 1e6
print(f"Wrote {zip_path}  ({size_mb:.1f} MB)")
