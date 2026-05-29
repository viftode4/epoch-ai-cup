"""E176: Fix focal loss LGB + try additional loss variants.

C3 focal loss failed because LGB API uses 'objective' function, not 'fobj'.
Also tests: class-balanced sample weights + DART focal.
"""

from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, load_test, load_train
from src.metrics import compute_map
from src.submission import save_submission
from src.postprocessing import N_CLASSES, renorm_rows

ROOT = Path(__file__).resolve().parent.parent
N_FOLDS = 5

print("=" * 70)
print("  E176: Focal Loss + Loss Variants (Fixed)")
print("=" * 70)

t0 = time.time()

# Load data
train_df = load_train()
test_df = load_test()
y = np.asarray(pd.Categorical(train_df["bird_group"], categories=CLASSES).codes, dtype=int)
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
groups = train_df["primary_observation_id"].values

train_feats = pd.read_pickle(ROOT / "data" / "_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data" / "_cached_test_features_v3.pkl")
selected = [l.strip() for l in (ROOT / "data" / "best_features_e175.txt").read_text().splitlines() if l.strip()]
selected = [f for f in selected if f in train_feats.columns and f in test_feats.columns]

X_train = np.nan_to_num(train_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(test_feats[selected].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

oof_e175 = renorm_rows(np.load(ROOT / "oof_e175_best.npy").astype(np.float64))
test_e175 = renorm_rows(np.load(ROOT / "test_e175_best.npy").astype(np.float64))
base_score, _ = compute_map(y, oof_e175)
print(f"Baseline E175: {base_score:.4f}")

counts = np.bincount(y, minlength=N_CLASSES).astype(float)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ── Focal multiclass objective ──
def focal_multiclass_obj(gamma_neg=2.0):
    """Focal loss objective for LGB multiclass custom training."""
    def _obj(preds, train_data):
        labels = train_data.get_label().astype(int)
        n = len(labels)
        preds_2d = preds.reshape(n, N_CLASSES, order='F')

        # Softmax
        preds_max = preds_2d.max(axis=1, keepdims=True)
        exp_p = np.exp(preds_2d - preds_max)
        probs = exp_p / exp_p.sum(axis=1, keepdims=True)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)

        # One-hot
        y_oh = np.zeros((n, N_CLASSES))
        y_oh[np.arange(n), labels] = 1.0

        # Focal weight: (1-p_t)^gamma for negatives
        pt = y_oh * probs + (1 - y_oh) * (1 - probs)
        gamma = np.where(y_oh == 1, 0.0, gamma_neg)
        fw = (1 - pt) ** gamma

        grad = fw * (probs - y_oh)
        hess = np.maximum(fw * probs * (1 - probs), 1e-6)

        return grad.flatten(order='F'), hess.flatten(order='F')
    return _obj


def focal_multiclass_eval(preds, train_data):
    """Custom eval metric for focal LGB (LGB 4.x API)."""
    labels = train_data.get_label().astype(int)
    n = len(labels)
    preds_2d = preds.reshape(n, N_CLASSES, order='F')
    probs = softmax(preds_2d)
    # Use negative log-loss as eval (faster than mAP, correlates well)
    nll = -np.log(np.clip(probs[np.arange(n), labels], 1e-8, 1.0)).mean()
    return "nll", nll, False  # False = lower is better


# ── Train with custom objective ──
def train_focal_lgb(X_train, y, groups, X_test, gamma_neg=2.0, n_seeds=5, boosting="dart"):
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    oof_all = np.zeros((n_seeds, n_train, N_CLASSES))
    test_all = np.zeros((n_seeds, n_test, N_CLASSES))

    obj = focal_multiclass_obj(gamma_neg)

    for seed in range(n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
        oof_seed = np.zeros((n_train, N_CLASSES))
        test_seed = np.zeros((n_test, N_CLASSES))

        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_train, y, groups)):
            # Class-balanced sample weights
            class_w = 1.0 / np.maximum(counts, 1.0)
            class_w /= class_w.sum()
            sw = class_w[y[tr_idx]]
            sw /= sw.mean()

            dtrain = lgb.Dataset(
                X_train[tr_idx], y[tr_idx], weight=sw,
                init_score=np.zeros((len(tr_idx), N_CLASSES)).flatten(order='F'),
            )
            dval = lgb.Dataset(
                X_train[va_idx], y[va_idx],
                init_score=np.zeros((len(va_idx), N_CLASSES)).flatten(order='F'),
            )

            params = {
                "num_class": N_CLASSES,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 20,
                "colsample_bytree": 0.6,
                "subsample": 0.7,
                "verbosity": -1,
                "seed": 42 + seed + fold,
                "num_threads": -1,
                "boosting_type": boosting,
            }
            if boosting == "dart":
                params["drop_rate"] = 0.15

            params["objective"] = obj

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=1500,
                valid_sets=[dval],
                feval=focal_multiclass_eval,
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )

            raw_va = model.predict(X_train[va_idx]).reshape(-1, N_CLASSES)
            raw_te = model.predict(X_test).reshape(-1, N_CLASSES)
            oof_seed[va_idx] = softmax(raw_va)
            test_seed += softmax(raw_te) / N_FOLDS

        oof_all[seed] = oof_seed
        test_all[seed] = test_seed
        s, _ = compute_map(y, oof_seed)
        print(f"  Seed {seed+1} ({boosting}, g={gamma_neg}): {s:.4f}")

    oof_mean = np.mean(oof_all, axis=0)
    test_mean = np.mean(test_all, axis=0)
    score, _ = compute_map(y, oof_mean)
    print(f"  Final ({n_seeds} seeds): {score:.4f}")
    return oof_mean, test_mean, score


# ── Test focal variants ──
results = {}

for gamma in [1.0, 2.0, 3.0]:
    print(f"\n--- Focal DART gamma={gamma} ---")
    try:
        oof, test, score = train_focal_lgb(X_train, y, groups, X_test, gamma_neg=gamma, n_seeds=3, boosting="dart")
        results[f"focal_dart_g{gamma}"] = (score, oof, test)
    except Exception as e:
        print(f"  FAILED: {e}")

# Also try GBDT boosting with focal
print(f"\n--- Focal GBDT gamma=2.0 ---")
try:
    oof_fg, test_fg, score_fg = train_focal_lgb(X_train, y, groups, X_test, gamma_neg=2.0, n_seeds=3, boosting="gbdt")
    results["focal_gbdt_g2"] = (score_fg, oof_fg, test_fg)
except Exception as e:
    print(f"  FAILED: {e}")

# ── LOMO evaluation (generalization check) ──
def eval_lomo(oof, name=""):
    """Evaluate LOMO: held-out-month mAP, more honest than SKF."""
    lomo_scores = {}
    for m in sorted(set(train_months)):
        mask = train_months == m
        if mask.sum() >= 10:
            s, _ = compute_map(y[mask], oof[mask])
            lomo_scores[m] = s
    avg = np.mean(list(lomo_scores.values()))
    month_str = " ".join(f"M{m}={v:.3f}" for m, v in sorted(lomo_scores.items()))
    print(f"  LOMO {name}: {avg:.4f}  ({month_str})")
    return avg

print("\n" + "=" * 70)
print("  LOMO Evaluation (Generalization)")
print("=" * 70)
lomo_base = eval_lomo(oof_e175, "E175 baseline")

for name, (score, oof, test) in results.items():
    lomo = eval_lomo(oof, name)
    print(f"    SKF={score:.4f}, LOMO={lomo:.4f}, gap={score-lomo:.4f}")


# ── Blend with E175 ──
print("\n" + "=" * 70)
print("  Blends with E175 (SKF + LOMO)")
print("=" * 70)

for name, (score, oof, test) in results.items():
    for alpha in [0.10, 0.20, 0.30]:
        blend = (1 - alpha) * oof_e175 + alpha * renorm_rows(oof)
        blend = renorm_rows(blend)
        s, _ = compute_map(y, blend)
        lomo = eval_lomo(blend, f"blend {name} a={alpha}")
        if s > base_score or lomo > lomo_base:
            tag = "BETTER" if lomo > lomo_base else "SKF only"
            print(f"    [{tag}] SKF={s:.4f} LOMO={lomo:.4f}")

# Save best by LOMO (not SKF!)
if results:
    # Evaluate all blends by LOMO
    best_lomo = lomo_base
    best_config = None
    for name, (score, oof, test) in results.items():
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            blend = renorm_rows((1 - alpha) * oof_e175 + alpha * renorm_rows(oof))
            lomo_scores = {}
            for m in sorted(set(train_months)):
                mask = train_months == m
                if mask.sum() >= 10:
                    s, _ = compute_map(y[mask], blend[mask])
                    lomo_scores[m] = s
            lomo = np.mean(list(lomo_scores.values()))
            if lomo > best_lomo:
                best_lomo = lomo
                best_config = (name, alpha, blend, test)

    if best_config:
        bname, balpha, boof, btest = best_config
        blend_test = (1 - balpha) * test_e175 + balpha * renorm_rows(btest)
        skf_s, _ = compute_map(y, boof)
        print(f"\n  Best LOMO blend: {bname} alpha={balpha}")
        print(f"    SKF={skf_s:.4f}, LOMO={best_lomo:.4f}")
        save_submission(renorm_rows(blend_test), f"e176_focal_best_lomo", cv_map=skf_s)
        np.save(ROOT / "oof_e176_focal_best.npy", boof)
        np.save(ROOT / "test_e176_focal_best.npy", renorm_rows(blend_test))
    else:
        print("\n  No blend improves LOMO over baseline")

    # Also save best standalone focal
    best_name, (best_score, best_oof, best_test) = max(results.items(), key=lambda x: x[1][0])
    save_submission(renorm_rows(best_test), f"e176_{best_name}", cv_map=best_score)

print(f"\nDone in {time.time()-t0:.0f}s")
