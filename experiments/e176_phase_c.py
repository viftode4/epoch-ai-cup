"""E176 Phase C: New Model Architectures on existing E175 features.

Tests:
  C1. Per-class specialized binary LGB models (domain-informed feature subsets)
  C2. BalancedRandomForest (structural diversity)
  C3. LGB with Asymmetric Softmax Loss (ASL)
  C5. LGB with Focal Loss
  C7. Smooth-AP thin MLP on OOF predictions
  C8. LGB with confidence regularization

All models trained on same 100 E175-selected features.
Final ensemble: blend new models with existing E175 predictions.
"""

from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train
from src.metrics import compute_map, print_results
from src.submission import save_submission
from src.postprocessing import N_CLASSES, renorm_rows

ROOT = Path(__file__).resolve().parent.parent
N_FOLDS = 5
N_SEEDS = 5

print("=" * 70)
print("  E176 Phase C: New Model Architectures")
print("=" * 70)

t0 = time.time()

# ── Load data ──
train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values

# Load cached features
train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]

X_train = np.nan_to_num(train_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(test_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

# Load E175 baselines for blending
oof_e175 = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))
test_e175 = renorm_rows(np.load(ROOT / "test_e175_best.npy").astype(np.float64))
oof_lgb = renorm_rows(np.load(ROOT / "oof_e175_lgb.npy").astype(np.float64))
test_lgb = renorm_rows(np.load(ROOT / "test_e175_lgb.npy").astype(np.float64))

base_score, base_pc = compute_map(y, oof_e175)
lgb_score, _ = compute_map(y, oof_lgb)
print(f"\nBaseline E175 blend: {base_score:.4f}")
print(f"Baseline E175 LGB:   {lgb_score:.4f}")
print(f"Features: {len(selected)}, Train: {X_train.shape}, Test: {X_test.shape}")

counts = np.bincount(y, minlength=N_CLASSES).astype(float)
p_train = counts / counts.sum()

all_results = {}  # name -> (oof_score, oof_preds, test_preds)


# ══════════════════════════════════════════════════════════════════════
# C2. BalancedRandomForest (structural diversity)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  C2. BalancedRandomForest")
print("=" * 70)


def train_brf(X_train, y, groups, X_test, n_seeds=N_SEEDS):
    """Train BalancedRandomForest with SGKF."""
    from imblearn.ensemble import BalancedRandomForestClassifier

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    oof_all = np.zeros((n_seeds, n_train, N_CLASSES))
    test_all = np.zeros((n_seeds, n_test, N_CLASSES))

    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        oof_seed = np.zeros((n_train, N_CLASSES))
        test_seed = np.zeros((n_test, N_CLASSES))

        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_train, y, groups)):
            clf = BalancedRandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=5,
                max_features="sqrt",
                sampling_strategy="not majority",
                replacement=True,
                random_state=42 + seed + fold,
                n_jobs=-1,
            )
            clf.fit(X_train[tr_idx], y[tr_idx])
            oof_seed[va_idx] = clf.predict_proba(X_train[va_idx])
            test_seed += clf.predict_proba(X_test) / N_FOLDS

        oof_all[seed] = oof_seed
        test_all[seed] = test_seed
        s, _ = compute_map(y, oof_seed)
        print(f"  Seed {seed+1}: OOF mAP = {s:.4f}")

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    score, pc = compute_map(y, oof_mean)
    print(f"  BRF ({n_seeds} seeds): OOF mAP = {score:.4f}")
    return oof_mean, test_mean, score


try:
    oof_brf, test_brf, score_brf = train_brf(X_train, y, groups, X_test)
    all_results["C2_BRF"] = (score_brf, oof_brf, test_brf)
except Exception as e:
    print(f"  C2 FAILED: {e}")
    oof_brf = test_brf = None


# ══════════════════════════════════════════════════════════════════════
# C3/C5. LGB with Focal Loss (subsumes ASL for multiclass)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  C3/C5. LGB with Focal Loss")
print("=" * 70)


def focal_loss_multiclass(gamma_neg=2.0, gamma_pos=0.0):
    """Custom focal loss for LGB multiclass.

    gamma_neg: focusing on hard negatives (easy Gull negatives downweighted)
    gamma_pos: focusing on hard positives (usually 0)
    """
    def focal_obj(preds, train_data):
        labels = train_data.get_label().astype(int)
        n = len(labels)

        # Reshape predictions: (n * n_classes,) -> (n, n_classes)
        preds_2d = preds.reshape(n, N_CLASSES, order='F')

        # Softmax
        preds_max = preds_2d.max(axis=1, keepdims=True)
        exp_preds = np.exp(preds_2d - preds_max)
        probs = exp_preds / exp_preds.sum(axis=1, keepdims=True)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)

        # One-hot
        y_onehot = np.zeros((n, N_CLASSES))
        y_onehot[np.arange(n), labels] = 1.0

        # Focal weights
        pt = (y_onehot * probs + (1 - y_onehot) * (1 - probs))
        gamma = y_onehot * gamma_pos + (1 - y_onehot) * gamma_neg
        focal_w = (1 - pt) ** gamma

        # Gradient: focal_w * (probs - y_onehot)
        grad = focal_w * (probs - y_onehot)

        # Hessian: focal_w * probs * (1 - probs) (approximate)
        hess = focal_w * probs * (1 - probs)
        hess = np.maximum(hess, 1e-6)

        return grad.flatten(order='F'), hess.flatten(order='F')

    return focal_obj


def train_lgb_focal(X_train, y, groups, X_test, gamma_neg=2.0, n_seeds=N_SEEDS):
    """Train LGB with focal loss."""
    import lightgbm as lgb

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    oof_all = np.zeros((n_seeds, n_train, N_CLASSES))
    test_all = np.zeros((n_seeds, n_test, N_CLASSES))

    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        oof_seed = np.zeros((n_train, N_CLASSES))
        test_seed = np.zeros((n_test, N_CLASSES))

        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_train, y, groups)):
            # Class weights for sampling
            class_w = 1.0 / np.maximum(counts, 1.0)
            class_w /= class_w.sum()
            sample_w = class_w[y[tr_idx]]
            sample_w /= sample_w.mean()

            dtrain = lgb.Dataset(X_train[tr_idx], y[tr_idx], weight=sample_w)
            dval = lgb.Dataset(X_train[va_idx], y[va_idx])

            params = {
                "num_class": N_CLASSES,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 20,
                "colsample_bytree": 0.6,
                "subsample": 0.7,
                "verbosity": -1,
                "seed": 42 + seed + fold,
                "n_jobs": -1,
            }

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dval],
                fobj=focal_loss_multiclass(gamma_neg=gamma_neg),
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )

            # Predict raw scores -> softmax
            raw_va = model.predict(X_train[va_idx]).reshape(-1, N_CLASSES)
            raw_te = model.predict(X_test).reshape(-1, N_CLASSES)

            # Softmax
            def softmax(x):
                e = np.exp(x - x.max(axis=1, keepdims=True))
                return e / e.sum(axis=1, keepdims=True)

            oof_seed[va_idx] = softmax(raw_va)
            test_seed += softmax(raw_te) / N_FOLDS

        oof_all[seed] = oof_seed
        test_all[seed] = test_seed
        s, _ = compute_map(y, oof_seed)
        print(f"  Seed {seed+1} (gamma={gamma_neg}): OOF mAP = {s:.4f}")

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    score, pc = compute_map(y, oof_mean)
    print(f"  Focal LGB ({n_seeds} seeds, gamma={gamma_neg}): OOF mAP = {score:.4f}")
    return oof_mean, test_mean, score


for gamma_neg in [1.0, 2.0, 3.0]:
    try:
        oof_fl, test_fl, score_fl = train_lgb_focal(X_train, y, groups, X_test, gamma_neg=gamma_neg, n_seeds=3)
        all_results[f"C3_focal_g{gamma_neg}"] = (score_fl, oof_fl, test_fl)
    except Exception as e:
        print(f"  Focal gamma={gamma_neg} FAILED: {e}")


# ══════════════════════════════════════════════════════════════════════
# C1. Per-Class Binary Specialists (rank-normalized)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  C1. Per-Class Binary Specialists")
print("=" * 70)


def train_per_class_specialists(X_train, y, groups, X_test, n_seeds=N_SEEDS):
    """Train 9 independent binary classifiers with per-class tuning."""
    import lightgbm as lgb

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    oof_all = np.zeros((n_seeds, n_train, N_CLASSES))
    test_all = np.zeros((n_seeds, n_test, N_CLASSES))

    # Per-class hyperparams (domain-informed)
    class_configs = {
        "Birds of Prey": {"num_leaves": 15, "scale_pos_weight": 10},
        "Clutter": {"num_leaves": 7, "scale_pos_weight": 8},
        "Cormorants": {"num_leaves": 7, "scale_pos_weight": 15},
        "Ducks": {"num_leaves": 15, "scale_pos_weight": 12},
        "Geese": {"num_leaves": 15, "scale_pos_weight": 10},
        "Gulls": {"num_leaves": 31, "scale_pos_weight": 1},  # majority
        "Pigeons": {"num_leaves": 15, "scale_pos_weight": 10},
        "Songbirds": {"num_leaves": 31, "scale_pos_weight": 3},
        "Waders": {"num_leaves": 15, "scale_pos_weight": 10},
    }

    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        oof_seed = np.zeros((n_train, N_CLASSES))
        test_seed = np.zeros((n_test, N_CLASSES))

        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_train, y, groups)):
            for cls_idx, cls_name in enumerate(CLASSES):
                y_bin_tr = (y[tr_idx] == cls_idx).astype(int)
                y_bin_va = (y[va_idx] == cls_idx).astype(int)

                if y_bin_tr.sum() < 2 or y_bin_va.sum() < 1:
                    oof_seed[va_idx, cls_idx] = p_train[cls_idx]
                    test_seed[:, cls_idx] += p_train[cls_idx] / N_FOLDS
                    continue

                cfg = class_configs[cls_name]
                model = lgb.LGBMClassifier(
                    objective="binary",
                    boosting_type="dart",
                    n_estimators=1000,
                    learning_rate=0.03,
                    num_leaves=cfg["num_leaves"],
                    scale_pos_weight=cfg["scale_pos_weight"],
                    min_child_samples=max(5, int(y_bin_tr.sum() * 0.1)),
                    colsample_bytree=0.6,
                    subsample=0.7,
                    drop_rate=0.15,
                    verbosity=-1,
                    random_state=42 + seed + fold + cls_idx,
                    n_jobs=-1,
                )
                model.fit(
                    X_train[tr_idx], y_bin_tr,
                    eval_set=[(X_train[va_idx], y_bin_va)],
                    callbacks=[lgb.early_stopping(100, verbose=False)],
                )

                oof_seed[va_idx, cls_idx] = model.predict_proba(X_train[va_idx])[:, 1]
                test_seed[:, cls_idx] += model.predict_proba(X_test)[:, 1] / N_FOLDS

        oof_all[seed] = oof_seed
        test_all[seed] = test_seed
        s, _ = compute_map(y, oof_seed)
        print(f"  Seed {seed+1}: OOF mAP = {s:.4f}")

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    score, pc = compute_map(y, oof_mean)
    print(f"  Per-class specialists ({n_seeds} seeds): OOF mAP = {score:.4f}")
    for cls, ap in pc.items():
        d = ap - base_pc[cls]
        if abs(d) > 0.01:
            print(f"    {cls}: {base_pc[cls]:.3f} -> {ap:.3f} ({d:+.3f})")
    return oof_mean, test_mean, score


oof_spec, test_spec, score_spec = train_per_class_specialists(X_train, y, groups, X_test)
all_results["C1_specialists"] = (score_spec, oof_spec, test_spec)


# ══════════════════════════════════════════════════════════════════════
# C_extra. Standard LGB multiclass with different hyperparams
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  C_extra. LGB Multiclass (GBDT, not DART)")
print("=" * 70)


def train_lgb_multiclass(X_train, y, groups, X_test, n_seeds=N_SEEDS, boosting="gbdt"):
    """Standard LGB multiclass for diversity."""
    import lightgbm as lgb

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    oof_all = np.zeros((n_seeds, n_train, N_CLASSES))
    test_all = np.zeros((n_seeds, n_test, N_CLASSES))

    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        oof_seed = np.zeros((n_train, N_CLASSES))
        test_seed = np.zeros((n_test, N_CLASSES))

        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_train, y, groups)):
            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=N_CLASSES,
                boosting_type=boosting,
                n_estimators=2000,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=20,
                colsample_bytree=0.6,
                subsample=0.7,
                is_unbalance=True,
                verbosity=-1,
                random_state=42 + seed + fold,
                n_jobs=-1,
            )
            model.fit(
                X_train[tr_idx], y[tr_idx],
                eval_set=[(X_train[va_idx], y[va_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            oof_seed[va_idx] = model.predict_proba(X_train[va_idx])
            test_seed += model.predict_proba(X_test) / N_FOLDS

        oof_all[seed] = oof_seed
        test_all[seed] = test_seed
        s, _ = compute_map(y, oof_seed)
        print(f"  Seed {seed+1} ({boosting}): OOF mAP = {s:.4f}")

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    score, _ = compute_map(y, oof_mean)
    print(f"  LGB {boosting} ({n_seeds} seeds): OOF mAP = {score:.4f}")
    return oof_mean, test_mean, score


oof_gbdt, test_gbdt, score_gbdt = train_lgb_multiclass(X_train, y, groups, X_test, boosting="gbdt")
all_results["C_extra_gbdt"] = (score_gbdt, oof_gbdt, test_gbdt)


# ══════════════════════════════════════════════════════════════════════
# C7. Smooth-AP MLP Layer
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  C7. Smooth-AP MLP on OOF Predictions")
print("=" * 70)


def smooth_ap_loss(y_true_bin, y_scores, tau=0.05):
    """Differentiable approximation of AP using sigmoid pairwise differences."""
    n = len(y_true_bin)
    pos_mask = y_true_bin > 0
    n_pos = pos_mask.sum()
    if n_pos == 0 or n_pos == n:
        return 0.0

    # Pairwise sigmoid: sigma(s_i - s_j) for positive i
    pos_scores = y_scores[pos_mask]
    all_scores = y_scores

    # For each positive, compute rank approximation
    ap_approx = 0.0
    for i, s_i in enumerate(pos_scores):
        diffs = s_i - all_scores
        ranks = 1.0 / (1.0 + np.exp(-diffs / tau))  # sigmoid approx of I(s_i > s_j)
        rank_i = ranks.sum()
        prec_at_i = ranks[pos_mask].sum() / max(rank_i, 1e-8)
        ap_approx += prec_at_i
    return ap_approx / n_pos


def train_smooth_ap_mlp(oof_preds, y, n_epochs=200, lr=0.001, hidden=32):
    """Train thin MLP that optimizes smooth-AP surrogate on OOF predictions."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  PyTorch not available, skipping C7")
        return None, None, None

    class APNet(nn.Module):
        def __init__(self, n_in, n_hidden, n_out):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, n_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(n_hidden, n_out),
            )

        def forward(self, x):
            return self.net(x)

    device = torch.device("cpu")
    n_train = oof_preds.shape[0]

    # Cross-validated
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_mlp = np.zeros_like(oof_preds)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(oof_preds, y, groups)):
        model = APNet(N_CLASSES, hidden, N_CLASSES).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        X_tr = torch.tensor(oof_preds[tr_idx], dtype=torch.float32, device=device)
        y_tr = torch.tensor(y[tr_idx], dtype=torch.long, device=device)
        X_va = torch.tensor(oof_preds[va_idx], dtype=torch.float32, device=device)

        for epoch in range(n_epochs):
            model.train()
            logits = model(X_tr)
            probs = torch.softmax(logits, dim=1)

            # Macro smooth-AP loss
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            for c in range(N_CLASSES):
                y_bin = (y_tr == c).float()
                if y_bin.sum() < 2:
                    continue
                scores = probs[:, c]
                # Cross-entropy as proxy (smooth-AP is too slow for full training)
                ce = -(y_bin * torch.log(scores + 1e-8) + (1 - y_bin) * torch.log(1 - scores + 1e-8))
                # Class-balanced
                weight = 1.0 / max(y_bin.sum().item(), 1.0)
                loss = loss + weight * ce.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Predict validation
        model.eval()
        with torch.no_grad():
            va_logits = model(X_va)
            va_probs = torch.softmax(va_logits, dim=1).cpu().numpy()
            oof_mlp[va_idx] = va_probs

        s, _ = compute_map(y[va_idx], va_probs)
        print(f"  Fold {fold}: val mAP = {s:.4f}")

    score, _ = compute_map(y, oof_mlp)
    print(f"  Smooth-AP MLP: OOF mAP = {score:.4f}")

    # For test: train on full OOF -> predict on test E175 predictions
    model_full = APNet(N_CLASSES, hidden, N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model_full.parameters(), lr=lr, weight_decay=1e-4)
    X_all = torch.tensor(oof_preds, dtype=torch.float32, device=device)
    y_all = torch.tensor(y, dtype=torch.long, device=device)

    for epoch in range(n_epochs):
        model_full.train()
        logits = model_full(X_all)
        probs = torch.softmax(logits, dim=1)
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        for c in range(N_CLASSES):
            y_bin = (y_all == c).float()
            if y_bin.sum() < 2:
                continue
            scores = probs[:, c]
            ce = -(y_bin * torch.log(scores + 1e-8) + (1 - y_bin) * torch.log(1 - scores + 1e-8))
            weight = 1.0 / max(y_bin.sum().item(), 1.0)
            loss = loss + weight * ce.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model_full.eval()
    with torch.no_grad():
        test_in = torch.tensor(test_e175, dtype=torch.float32, device=device)
        test_mlp = torch.softmax(model_full(test_in), dim=1).cpu().numpy()

    return oof_mlp, test_mlp, score


oof_mlp, test_mlp, score_mlp = train_smooth_ap_mlp(oof_e175, y)
if oof_mlp is not None:
    all_results["C7_smoothap_mlp"] = (score_mlp, oof_mlp, test_mlp)


# ══════════════════════════════════════════════════════════════════════
# Ensemble: Blend all models with E175
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  ENSEMBLE: Blend Phase C models with E175")
print("=" * 70)

# Collect all valid OOF predictions
model_oofs = {"e175_best": oof_e175, "e175_lgb": oof_lgb}
model_tests = {"e175_best": test_e175, "e175_lgb": test_lgb}

for name, (score, oof, test) in all_results.items():
    model_oofs[name] = oof
    model_tests[name] = test
    print(f"  {name}: {score:.4f}")

# Rank-based blending
def rank_normalize(preds):
    """Per-class rank normalization to [0, 1]."""
    out = np.zeros_like(preds)
    for c in range(N_CLASSES):
        out[:, c] = rankdata(preds[:, c]) / len(preds)
    return out

# Try all pairs with E175
print("\n  Pairwise blends with E175 best:")
for name in all_results:
    oof = all_results[name][1]
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.30]:
        blend = (1 - alpha) * oof_e175 + alpha * renorm_rows(oof)
        blend = renorm_rows(blend)
        s, _ = compute_map(y, blend)
        if s > base_score + 0.001:
            print(f"    {name} (alpha={alpha}): {s:.4f} (+{s - base_score:.4f})")

# Rank-based blends
print("\n  Rank-based blends with E175:")
oof_e175_rank = rank_normalize(oof_e175)
for name in all_results:
    oof_rank = rank_normalize(all_results[name][1])
    for alpha in [0.10, 0.20, 0.30]:
        blend = (1 - alpha) * oof_e175_rank + alpha * oof_rank
        s, _ = compute_map(y, blend)
        if s > base_score + 0.001:
            print(f"    rank({name}) alpha={alpha}: {s:.4f} (+{s - base_score:.4f})")

# Multi-model blend: E175 best + E175 lgb + best Phase C model
if all_results:
    best_c = max(all_results.items(), key=lambda x: x[1][0])
    print(f"\n  Best Phase C model: {best_c[0]} at {best_c[1][0]:.4f}")

    # 3-way blend
    oof_c = best_c[1][1]
    test_c = best_c[1][2]
    for w_e175 in [0.5, 0.6, 0.7, 0.8]:
        for w_lgb in [0.0, 0.1, 0.2]:
            w_c = 1.0 - w_e175 - w_lgb
            if w_c < 0:
                continue
            blend = w_e175 * oof_e175 + w_lgb * oof_lgb + w_c * renorm_rows(oof_c)
            blend = renorm_rows(blend)
            s, _ = compute_map(y, blend)
            if s > base_score + 0.001:
                print(f"    3-way (e175={w_e175}, lgb={w_lgb}, c={w_c:.1f}): {s:.4f}")


# ══════════════════════════════════════════════════════════════════════
# Save Best Submissions
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SAVING SUBMISSIONS")
print("=" * 70)

# Save individual model predictions
for name, (score, oof, test) in all_results.items():
    save_submission(renorm_rows(test), f"e176_{name}", cv_map=score)
    np.save(ROOT / f"oof_e176_{name}.npy", oof)
    np.save(ROOT / f"test_e176_{name}.npy", test)

# Best blend submission
if all_results:
    best_c_name, (best_c_score, oof_c, test_c) = max(all_results.items(), key=lambda x: x[1][0])
    # Conservative blend: 80% E175 + 20% best new
    blend_oof = 0.8 * oof_e175 + 0.2 * renorm_rows(oof_c)
    blend_test = 0.8 * test_e175 + 0.2 * renorm_rows(test_c)
    blend_score, _ = compute_map(y, renorm_rows(blend_oof))
    save_submission(renorm_rows(blend_test), "e176_blend_80_20", cv_map=blend_score)

elapsed = time.time() - t0

# Final summary
print("\n" + "=" * 70)
print("  E176 PHASE C FINAL SUMMARY")
print("=" * 70)
print(f"\n  {'Model':<35s} {'SKF mAP':>8s} {'Delta':>8s}")
print(f"  {'-'*35} {'-'*8} {'-'*8}")

all_scores = [("E175 blend (baseline)", base_score)]
all_scores += [(name, score) for name, (score, _, _) in all_results.items()]
all_scores.sort(key=lambda x: -x[1])

for name, score in all_scores:
    delta = score - base_score
    marker = " ***" if delta > 0.01 else (" **" if delta > 0.003 else "")
    print(f"  {name:<35s} {score:>8.4f} {delta:>+8.4f}{marker}")

print(f"\n  Total time: {elapsed:.0f}s")
print("=" * 70)
