"""E18: MOMENT Foundation Model as Feature Extractor (T06)

Uses pretrained MOMENT-1-large (ICML 2024) as a frozen backbone to extract
embeddings from radar trajectories, then trains LightGBM on the embeddings.

Approach: Linear probe (freeze backbone, extract 1024-d embeddings, classify with LGB).
Our data: 8 channels x 128 timesteps -> pad to 512 with input mask.
"""
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeClassifierCV
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train, load_test, CLASSES
from src.sequence import prepare_sequences
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOMENT_SEQ_LEN = 512
OUR_SEQ_LEN = 128
BATCH_SIZE = 32

print(f"Device: {DEVICE}", flush=True)

# ── Data ─────────────────────────────────────────────────────────
print("Loading data...", flush=True)
train_df = load_train()
test_df = load_test()

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

print("Preparing sequences (8ch x 128)...", flush=True)
X_seq_train = prepare_sequences(train_df, seq_len=OUR_SEQ_LEN)  # (2601, 8, 128)
X_seq_test = prepare_sequences(test_df, seq_len=OUR_SEQ_LEN)    # (1872, 8, 128)

# Pad to MOMENT's expected 512 length
print(f"Padding sequences from {OUR_SEQ_LEN} to {MOMENT_SEQ_LEN}...", flush=True)
X_train_padded = np.zeros((len(X_seq_train), 8, MOMENT_SEQ_LEN), dtype=np.float32)
X_test_padded = np.zeros((len(X_seq_test), 8, MOMENT_SEQ_LEN), dtype=np.float32)
X_train_padded[:, :, :OUR_SEQ_LEN] = X_seq_train
X_test_padded[:, :, :OUR_SEQ_LEN] = X_seq_test

# Input masks: 1 for valid positions, 0 for padding
train_mask = np.zeros((len(X_seq_train), MOMENT_SEQ_LEN), dtype=np.float32)
test_mask = np.zeros((len(X_seq_test), MOMENT_SEQ_LEN), dtype=np.float32)
train_mask[:, :OUR_SEQ_LEN] = 1.0
test_mask[:, :OUR_SEQ_LEN] = 1.0

# ── Load MOMENT ──────────────────────────────────────────────────
print("Loading MOMENT-1-large...", flush=True)
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name": "embedding"},
)
model = model.to(DEVICE)
model.eval()
print(f"MOMENT loaded. d_model={model.config.d_model}", flush=True)

# ── Extract embeddings ───────────────────────────────────────────
def extract_embeddings(X_padded, masks, desc=""):
    """Extract MOMENT embeddings in batches."""
    n = len(X_padded)
    embeddings = []
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        x = torch.tensor(X_padded[start:end], dtype=torch.float32).to(DEVICE)
        m = torch.tensor(masks[start:end], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            out = model.embed(x_enc=x, input_mask=m, reduction="mean")
        embeddings.append(out.embeddings.cpu().numpy())
        if (start // BATCH_SIZE) % 10 == 0:
            print(f"  {desc}: {end}/{n}", flush=True)
    print(f"  {desc}: {n}/{n} done", flush=True)
    return np.concatenate(embeddings, axis=0)

print("\nExtracting train embeddings...", flush=True)
train_emb = extract_embeddings(X_train_padded, train_mask, "Train")
print(f"  Train embeddings: {train_emb.shape}", flush=True)

print("Extracting test embeddings...", flush=True)
test_emb = extract_embeddings(X_test_padded, test_mask, "Test")
print(f"  Test embeddings: {test_emb.shape}", flush=True)

# Clean up GPU memory
del model
torch.cuda.empty_cache()

# ── Approach 1: Ridge Classifier on embeddings ───────────────────
print("\n" + "="*60, flush=True)
print("Approach 1: Ridge Classifier on MOMENT embeddings", flush=True)
print("="*60, flush=True)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof_ridge = np.zeros((len(y), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_emb, y)):
    clf = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    clf.fit(train_emb[tr_idx], y[tr_idx])
    # RidgeClassifier decision_function gives scores, convert to pseudo-probs
    scores = clf.decision_function(train_emb[va_idx])
    # Softmax-like normalization
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    oof_ridge[va_idx] = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    fold_map, _ = compute_map(y[va_idx], oof_ridge[va_idx])
    print(f"  Fold {fold} mAP: {fold_map:.4f} (alpha={clf.alpha_:.2f})", flush=True)

ridge_map, ridge_per = compute_map(y, oof_ridge)
print_results(ridge_map, ridge_per, "MOMENT + Ridge (linear probe)")

# ── Approach 2: LightGBM on embeddings ───────────────────────────
print("\n" + "="*60, flush=True)
print("Approach 2: LightGBM on MOMENT embeddings", flush=True)
print("="*60, flush=True)

# Effective Number weights
BETA = 0.999
counts = np.bincount(y, minlength=N_CLASSES)
effective_n = 1.0 - np.power(BETA, counts)
class_w = (1.0 - BETA) / (effective_n + 1e-10)
class_w = class_w / class_w.mean()
sample_weights = np.array([class_w[yi] for yi in y])

lgb_params = {
    "objective": "multiclass", "num_class": N_CLASSES,
    "metric": "multi_logloss", "learning_rate": 0.05,
    "num_leaves": 31, "max_depth": 6, "min_child_samples": 10,
    "subsample": 0.8, "colsample_bytree": 0.5,  # lower due to 1024 features
    "reg_alpha": 0.5, "reg_lambda": 2.0,
    "verbose": -1, "seed": 42, "n_jobs": -1,
    "device": "gpu",
}

oof_lgb = np.zeros((len(y), N_CLASSES))
test_lgb = np.zeros((len(test_emb), N_CLASSES))
emb_feat_names = [f"moment_{i}" for i in range(train_emb.shape[1])]

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_emb, y)):
    dtrain = lgb.Dataset(train_emb[tr_idx], label=y[tr_idx],
                         weight=sample_weights[tr_idx], feature_name=emb_feat_names)
    dval = lgb.Dataset(train_emb[va_idx], label=y[va_idx],
                       feature_name=emb_feat_names, reference=dtrain)
    mdl = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_lgb[va_idx] = mdl.predict(train_emb[va_idx])
    test_lgb += mdl.predict(test_emb) / N_FOLDS
    fold_map, _ = compute_map(y[va_idx], oof_lgb[va_idx])
    print(f"  Fold {fold} mAP: {fold_map:.4f}", flush=True)

lgb_map, lgb_per = compute_map(y, oof_lgb)
print_results(lgb_map, lgb_per, "MOMENT + LGB")

# ── Approach 3: Combined (MOMENT emb + handcrafted features) ────
print("\n" + "="*60, flush=True)
print("Approach 3: MOMENT embeddings + handcrafted features", flush=True)
print("="*60, flush=True)

from src.features import build_features
FEATURE_SETS = ["core", "rcs_fft", "tabular", "targeted", "flight_mode"]
print("Extracting handcrafted features...", flush=True)
train_feats = build_features(train_df, feature_sets=FEATURE_SETS)
test_feats = build_features(test_df, feature_sets=FEATURE_SETS)

X_combined_train = np.hstack([train_emb, train_feats.values.astype(np.float32)])
X_combined_test = np.hstack([test_emb, test_feats.values.astype(np.float32)])
combined_feat_names = emb_feat_names + list(train_feats.columns)

print(f"  Combined features: {X_combined_train.shape[1]} ({train_emb.shape[1]} MOMENT + {train_feats.shape[1]} handcrafted)", flush=True)

lgb_params_combined = lgb_params.copy()
lgb_params_combined["colsample_bytree"] = 0.3  # even lower for 1129 features

oof_combined = np.zeros((len(y), N_CLASSES))
test_combined = np.zeros((len(X_combined_test), N_CLASSES))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_combined_train, y)):
    dtrain = lgb.Dataset(X_combined_train[tr_idx], label=y[tr_idx],
                         weight=sample_weights[tr_idx], feature_name=combined_feat_names)
    dval = lgb.Dataset(X_combined_train[va_idx], label=y[va_idx],
                       feature_name=combined_feat_names, reference=dtrain)
    mdl = lgb.train(lgb_params_combined, dtrain, num_boost_round=2000, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    oof_combined[va_idx] = mdl.predict(X_combined_train[va_idx])
    test_combined += mdl.predict(X_combined_test) / N_FOLDS
    fold_map, _ = compute_map(y[va_idx], oof_combined[va_idx])
    print(f"  Fold {fold} mAP: {fold_map:.4f}", flush=True)

combined_map, combined_per = compute_map(y, oof_combined)
print_results(combined_map, combined_per, "MOMENT + Handcrafted + LGB")

# ── Pick best approach and save ──────────────────────────────────
print(f"\n{'='*60}", flush=True)
print(f"COMPARISON", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Ridge on MOMENT:       {ridge_map:.4f}", flush=True)
print(f"  LGB on MOMENT:         {lgb_map:.4f}", flush=True)
print(f"  LGB on MOMENT+feats:   {combined_map:.4f}", flush=True)
print(f"  E06 CNN (baseline):    0.5238", flush=True)
print(f"  E15 tree ensemble:     0.7451", flush=True)

# Save best MOMENT approach for stacking
best_moment_map = max(ridge_map, lgb_map, combined_map)
if best_moment_map == ridge_map:
    best_oof, best_name = oof_ridge, "ridge"
    # Generate test predictions for ridge
    clf = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    clf.fit(train_emb, y)
    scores = clf.decision_function(test_emb)
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    best_test = exp_scores / exp_scores.sum(axis=1, keepdims=True)
elif best_moment_map == lgb_map:
    best_oof, best_test, best_name = oof_lgb, test_lgb, "lgb"
else:
    best_oof, best_test, best_name = oof_combined, test_combined, "combined"

print(f"\n  Best approach: {best_name} ({best_moment_map:.4f})", flush=True)

np.save(ROOT / "oof_e18.npy", best_oof)
np.save(ROOT / "test_e18.npy", best_test)

# ── Stacking: check if MOMENT adds value ─────────────────────────
print(f"\n{'='*60}", flush=True)
print("Stacking: does MOMENT add value?", flush=True)
print(f"{'='*60}", flush=True)

oof_e15 = np.load(ROOT / "oof_e15.npy")
oof_e06 = np.load(ROOT / "oof_e06.npy")
oof_e08 = np.load(ROOT / "oof_e08.npy")
oof_e09 = np.load(ROOT / "oof_e09.npy")

# Current E15 stack: tree=75% rocket=5% cnn=10% svm=10%
e15_stack_map = 0.7493

# Try replacing CNN with MOMENT
best_replace_map = 0
best_replace_w = None
for w0 in np.arange(0.50, 0.90, 0.05):
    for w1 in np.arange(0.05, 0.25, 0.05):
        for w2 in np.arange(0.05, 0.25, 0.05):
            w3 = 1.0 - w0 - w1 - w2
            if w3 < 0.05:
                continue
            oof_s = w0 * oof_e15 + w1 * oof_e08 + w2 * best_oof + w3 * oof_e09
            m, _ = compute_map(y, oof_s)
            if m > best_replace_map:
                best_replace_map = m
                best_replace_w = (w0, w1, w2, w3)

print(f"  Replace CNN with MOMENT: {best_replace_map:.4f} "
      f"(tree={best_replace_w[0]:.2f} rocket={best_replace_w[1]:.2f} "
      f"moment={best_replace_w[2]:.2f} svm={best_replace_w[3]:.2f})", flush=True)

# Try adding MOMENT as 5th model
best_5model_map = 0
best_5model_w = None
for w0 in np.arange(0.50, 0.85, 0.05):
    for w1 in np.arange(0.05, 0.20, 0.05):
        for w2 in np.arange(0.05, 0.20, 0.05):
            for w3 in np.arange(0.05, 0.20, 0.05):
                w4 = 1.0 - w0 - w1 - w2 - w3
                if w4 < 0.05:
                    continue
                oof_s = (w0 * oof_e15 + w1 * oof_e08 + w2 * oof_e06 +
                         w3 * oof_e09 + w4 * best_oof)
                m, _ = compute_map(y, oof_s)
                if m > best_5model_map:
                    best_5model_map = m
                    best_5model_w = (w0, w1, w2, w3, w4)

print(f"  5-model stack:          {best_5model_map:.4f} "
      f"(tree={best_5model_w[0]:.2f} rocket={best_5model_w[1]:.2f} "
      f"cnn={best_5model_w[2]:.2f} svm={best_5model_w[3]:.2f} moment={best_5model_w[4]:.2f})",
      flush=True)
print(f"  E15 4-model stack:      {e15_stack_map:.4f}", flush=True)
print(f"  Replace CNN delta:      {best_replace_map - e15_stack_map:+.4f}", flush=True)
print(f"  5-model delta:          {best_5model_map - e15_stack_map:+.4f}", flush=True)

save_submission(best_test, "e18_moment", cv_map=best_moment_map)
print("\nDone!", flush=True)
