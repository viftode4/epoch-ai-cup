# E205 Submission Reproduction — `e205_multi_restart_T09`

Reproduces our E205 submission (**0.545 private LB**, AI Cup 2026 Performance Track,
9-class bird radar classification, macro-averaged mAP).

The submission is a **weighted probability blend of 11 diverse models**, followed by
temperature sharpening (T=0.9) and per-row renormalization. The blend weights were
found by Nelder-Mead optimization (10 seeded restarts) maximizing out-of-fold (OOF)
macro-mAP.

## Contents

```
e205_reproduction/
├── reproduce.py          # Regenerate the submission CSV from shipped model arrays
├── optimize_weights.py   # Re-derive the blend weights from OOF preds + labels
├── weights.json          # The optimized blend weights (all 4 E205 variants)
├── track_ids.csv         # 1872 test track_ids (submission row order)
├── y.npy                 # 2601 training labels (class index 0..8) — for optimize_weights
├── models/               # Per-model predictions
│   ├── test_<model>.npy  #   (1872, 9) test probabilities — for reproduce.py
│   └── oof_<model>.npy   #   (2601, 9) OOF train probabilities — for optimize_weights.py
├── requirements.txt
└── README.md
```

Column order in every `.npy` array (`CLASSES`):
`Birds of Prey, Clutter, Cormorants, Ducks, Geese, Gulls, Pigeons, Songbirds, Waders`

## Run

```bash
pip install -r requirements.txt

# 1) Regenerate the winning submission (no competition data needed)
python reproduce.py                 # -> e205_multi_restart_T09.csv  (0.545)
python reproduce.py --variant raw   # -> the un-sharpened blend
python reproduce.py --variant T085  # -> T=0.85 variant

# 2) (optional) Re-derive the blend weights end-to-end and confirm they match
python optimize_weights.py          # prints weights + diff vs weights.json
```

## The 11 models and their blend weights (`multi_restart`)

| Model | Weight | What it is |
|---|---|---|
| `e186_ovo` | 0.5677 | One-vs-one pairwise-coupled classifier |
| `e185_tabpfn_relabel` | 0.2520 | TabPFN with cleanlab-relabeled training data |
| `e175_lgb` | 0.0945 | LightGBM (75-feature pruned set) |
| `e185_tabpfn_all` | 0.0341 | TabPFN on full feature set |
| `e180_cnn` | 0.0227 | 1D CNN over raw trajectory sequences |
| `e79` | 0.0172 | 36-feature pruned tree ensemble (LGB/XGB/CB) |
| `e179_best` | 0.0052 | Best E179 tree blend |
| `e187_blend` | 0.0048 | E187 feature-improved blend |
| `e185_tabpfn_all`… | | |
| `e179_cb` | 0.0007 | CatBoost variant |
| `e173` | 0.0001 | E173 fixed-leakage tree blend |

OOF macro-mAP of the blend = **0.8724**. Two models (OvO + TabPFN-relabel) carry ~82%
of the weight; the rest add diversity.

> `weights.json` also contains three other E205 variants we generated
> (`power_p2`, `power_p3`, `e188_plus_dart_cnn`); `multi_restart` is the one that
> produced the 0.545 submission.

## Notes

- `reproduce.py` is fully self-contained — it needs only files in this package.
- `optimize_weights.py` additionally uses `y.npy` (training labels) to re-derive the
  weights; the optimization is deterministic (fixed seeds), so it reproduces
  `weights.json` to < 1e-4.
- `.npy` files are plain NumPy float arrays (loaded with `allow_pickle=True` only
  because they were saved from an object-dtype context; no custom classes inside).
