"""E185: Combined Pipeline — Everything We Validated.

Combines:
1. TabPFN with ALL 327 features (validated: SKF +0.045 over 100-selected)
2. Noise relabeling (validated: +0.143 Cormorant AP on TabPFN SKF)
3. Physics features (validated: +0.023 Wader AP)
4. Likelihood ratio evidence fusion (untested, principled)
5. Blend with E175 components (existing predictions)
6. Proper evaluation: SKF + TRUE LOMO + per-class breakdown

Runtime: ~15 min
"""
import sys, time
sys.path.insert(0, 'G:/Projects/epoch-ai-cup')
import numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score

from src.data import load_train, load_test, CLASSES, parse_ewkb_4d, parse_trajectory_time
from src.features import haversine
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path('G:/Projects/epoch-ai-cup')
N = len(CLASSES)

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
print("=" * 90, flush=True)
print("  E185: COMBINED PIPELINE", flush=True)
print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print("=" * 90, flush=True)

train = load_train()
test = load_test()
y = np.asarray(pd.Categorical(train["bird_group"], categories=CLASSES).codes, dtype=int)
groups = train["primary_observation_id"].values
train_months = pd.to_datetime(train["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test["timestamp_start_radar_utc"]).dt.month.values

# Features
train_feats = pd.read_pickle(ROOT / "data/_cached_train_features_v3.pkl")
test_feats = pd.read_pickle(ROOT / "data/_cached_test_features_v3.pkl")
X_train = np.nan_to_num(train_feats.values.astype(np.float32), nan=0, posinf=0, neginf=0)
X_test = np.nan_to_num(test_feats.values.astype(np.float32), nan=0, posinf=0, neginf=0)
print(f"  Features: {X_train.shape[1]}", flush=True)

# ══════════════════════════════════════════════════════════════
# NOISE RELABELING
# ══════════════════════════════════════════════════════════════
print("\n[1] Noise relabeling...", flush=True)
cache_path = ROOT / "data/_cleanlab_cache.npz"
if cache_path.exists():
    cache = np.load(cache_path, allow_pickle=True)
    agreed_noisy = cache['agreed_noisy'].tolist()
    consensus_labels = cache['consensus_labels']
    quality = cache['quality']
    print(f"  Loaded cached cleanlab: {len(agreed_noisy)} agreed noisy", flush=True)
else:
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    oof_t = np.load(ROOT / "oof_e183_tabpfn.npy")
    oof_e = softmax(np.load(ROOT / "oof_e175_best.npy"), axis=1)
    oof_c = np.load(ROOT / "oof_e175_cb.npy")
    oof_r = np.load(ROOT / "oof_e175_ranker.npy")
    issues_t = set(find_label_issues(labels=y, pred_probs=oof_t, return_indices_ranked_by="self_confidence", n_jobs=1))
    issues_e = set(find_label_issues(labels=y, pred_probs=oof_e, return_indices_ranked_by="self_confidence", n_jobs=1))
    agreed_noisy = sorted(issues_t & issues_e)
    quality = get_label_quality_scores(labels=y, pred_probs=oof_t)
    consensus_labels = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        preds = [oof_t[i].argmax(), oof_e[i].argmax(), oof_c[i].argmax(), oof_r[i].argmax()]
        consensus_labels[i] = Counter(preds).most_common(1)[0][0]
    np.savez(cache_path, agreed_noisy=np.array(agreed_noisy), quality=quality, consensus_labels=consensus_labels)
    print(f"  Computed cleanlab: {len(agreed_noisy)} agreed noisy", flush=True)

y_relabeled = y.copy()
for idx in agreed_noisy:
    y_relabeled[idx] = consensus_labels[idx]
# Remove extreme noise (quality < 0.02)
extreme = set(np.where(quality < 0.02)[0])
keep_mask = np.array([i not in extreme for i in range(len(y))])
n_relabeled = np.sum(y_relabeled != y)
print(f"  Relabeled {n_relabeled}, removing {len(extreme)} extreme", flush=True)

# ══════════════════════════════════════════════════════════════
# TABPFN TRAINING
# ══════════════════════════════════════════════════════════════
from tabpfn import TabPFNClassifier

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def run_tabpfn(X_tr, X_te, y_train, label, mask=None):
    t0 = time.time()
    oof = np.zeros((len(y), N))
    test_preds = np.zeros((len(X_te), N))
    for fold, (tr, va) in enumerate(sgkf.split(X_tr, y, groups)):
        tr_use = tr
        if mask is not None:
            tr_use = np.array([i for i in tr if mask[i]])
        clf = TabPFNClassifier(n_estimators=16, random_state=42)
        clf.fit(X_tr[tr_use], y_train[tr_use])
        oof[va] = clf.predict_proba(X_tr[va])
        test_preds += clf.predict_proba(X_te) / 5
        print(f"    Fold {fold+1}/5...", flush=True)
    # Eval against ORIGINAL y
    skf, pc = compute_map(y, oof)
    elapsed = time.time() - t0
    print(f"  {label} ({elapsed:.0f}s): SKF={skf:.4f}", flush=True)
    for cls in CLASSES:
        mk = " ***" if pc[cls] < 0.4 else ""
        print(f"    {cls:15s}: {pc[cls]:.4f}{mk}", flush=True)
    return oof, test_preds, skf, pc

# A. TabPFN-ALL baseline (original labels)
print("\n[2a] TabPFN-ALL baseline...", flush=True)
oof_a, test_a, skf_a, pc_a = run_tabpfn(X_train, X_test, y, "TabPFN-ALL baseline")

# B. TabPFN-ALL + relabeled
print("\n[2b] TabPFN-ALL + relabeled...", flush=True)
oof_b, test_b, skf_b, pc_b = run_tabpfn(X_train, X_test, y_relabeled, "TabPFN-ALL relabeled", mask=keep_mask)

# Save
np.save(ROOT / "oof_e185_tabpfn_all.npy", oof_a)
np.save(ROOT / "test_e185_tabpfn_all.npy", test_a)
np.save(ROOT / "oof_e185_tabpfn_relabel.npy", oof_b)
np.save(ROOT / "test_e185_tabpfn_relabel.npy", test_b)

# ══════════════════════════════════════════════════════════════
# LIKELIHOOD RATIO FUSION
# ══════════════════════════════════════════════════════════════
print("\n[3] Likelihood ratio evidence fusion...", flush=True)

# Training class proportions (priors)
prior = np.bincount(y, minlength=N).astype(float)
prior /= prior.sum()

# Load all diverse OOF predictions
models = {}
model_files = {
    "tabpfn_all": (oof_a, test_a),
    "tabpfn_relabel": (oof_b, test_b),
}
# Load existing predictions
for name in ["e175_best", "e175_cb", "e175_ranker", "e183_tabpfn"]:
    oof_path = ROOT / f"oof_{name}.npy"
    test_path = ROOT / f"test_{name}.npy"
    if oof_path.exists() and test_path.exists():
        o = np.load(oof_path)
        t = np.load(test_path)
        if o.shape == (len(y), N) and t.shape == (len(X_test), N):
            models[name] = (o, t)

# Also try CNN if shapes match
for name in ["e180_cnn", "e182_cnn_v3"]:
    oof_path = ROOT / f"oof_{name}.npy"
    test_path = ROOT / f"test_{name}.npy"
    if oof_path.exists() and test_path.exists():
        o = np.load(oof_path)
        t = np.load(test_path)
        if o.shape[0] == len(y) and o.shape[1] == N and t.shape[0] == len(X_test) and t.shape[1] == N:
            models[name] = (o, t)

models.update(model_files)
print(f"  {len(models)} models loaded for fusion", flush=True)

def lr_fusion(models_dict, priors, split="oof"):
    """Compute likelihood ratio fusion across models."""
    idx = 0 if split == "oof" else 1
    n_samples = models_dict[list(models_dict.keys())[0]][idx].shape[0]

    # For each model, compute log-likelihood-ratio for each class vs uniform
    log_evidence = np.zeros((n_samples, N))

    for name, (oof, tst) in models_dict.items():
        preds = oof if split == "oof" else tst
        # Ensure valid probabilities
        preds = np.clip(preds, 1e-8, 1.0)
        preds = preds / preds.sum(axis=1, keepdims=True)

        # Log-likelihood ratio: log(P(c|x) / prior(c))
        # This removes the prior, leaving just the evidence
        for c in range(N):
            log_evidence[:, c] += np.log(preds[:, c] / prior[c])

    # Convert back to probabilities via softmax
    # The temperature controls how sharp the fusion is
    for temp in [1.0]:
        fused = softmax(log_evidence / temp, axis=1)
        return fused

oof_lr = lr_fusion(models, prior, "oof")
test_lr = lr_fusion(models, prior, "test")

skf_lr, pc_lr = compute_map(y, oof_lr)
print(f"  LR Fusion: SKF={skf_lr:.4f}", flush=True)
for cls in CLASSES:
    mk = " ***" if pc_lr[cls] < 0.4 else ""
    print(f"    {cls:15s}: {pc_lr[cls]:.4f}{mk}", flush=True)

# ══════════════════════════════════════════════════════════════
# RANK-POWER BLENDING (for comparison)
# ══════════════════════════════════════════════════════════════
print("\n[4] Rank-power blending...", flush=True)

def rpe(preds_list, weights, power=1.5):
    n_s = preds_list[0].shape[0]
    out = np.zeros((n_s, N))
    for c in range(N):
        for p, w in zip(preds_list, weights):
            out[:, c] += w * (rankdata(p[:, c]) / n_s) ** power
    return out

# Best known blend: TabPFN + E175 (from earlier validation)
oof_e175 = np.load(ROOT / "oof_e175_best.npy")
test_e175 = np.load(ROOT / "test_e175_best.npy")

# Search: TabPFN-ALL-relabel + E175
def lomo(oof_pred):
    maps = {}
    for m in sorted(set(train_months)):
        mask = train_months == m
        if mask.sum() >= 10:
            lm, _ = compute_map(y[mask], oof_pred[mask])
            maps[m] = lm
    return np.mean(list(maps.values())), maps

best_lomo = -1
best_cfg = None
for power in [1.0, 1.5, 2.0]:
    for w in np.arange(0, 1.05, 0.1):
        blend = rpe([oof_b, oof_e175], [w, 1-w], power)
        l, _ = lomo(blend)
        skf, pc = compute_map(y, blend)
        if l > best_lomo:
            best_lomo = l
            best_cfg = (w, 1-w, power, skf, pc, l)

w_tab, w_e, pw, skf_blend, pc_blend, lomo_blend = best_cfg
print(f"  Best blend: TabPFN={w_tab:.1f} E175={w_e:.1f} power={pw}", flush=True)
print(f"  SKF={skf_blend:.4f}  LOMO={lomo_blend:.4f}", flush=True)
for cls in CLASSES:
    print(f"    {cls:15s}: {pc_blend[cls]:.4f}", flush=True)

oof_final_blend = rpe([oof_b, oof_e175], [w_tab, w_e], pw)
test_final_blend = rpe([test_b, test_e175], [w_tab, w_e], pw)

# Also: 3-way with LR fusion
best_lomo3 = -1
best_cfg3 = None
for power in [1.0, 1.5, 2.0]:
    for w0 in np.arange(0, 1.05, 0.2):
        for w1 in np.arange(0, 1.05 - w0, 0.2):
            w2 = round(1.0 - w0 - w1, 2)
            if w2 < -0.01: continue
            blend3 = rpe([oof_b, oof_e175, oof_lr], [w0, w1, w2], power)
            l, _ = lomo(blend3)
            skf, pc = compute_map(y, blend3)
            if l > best_lomo3:
                best_lomo3 = l
                best_cfg3 = (w0, w1, w2, power, skf, pc, l)

if best_cfg3:
    w0, w1, w2, pw3, skf3, pc3, lomo3 = best_cfg3
    print(f"\n  3-way (TabPFN+E175+LR): w={w0:.1f}/{w1:.1f}/{w2:.1f} power={pw3}", flush=True)
    print(f"  SKF={skf3:.4f}  LOMO={lomo3:.4f}", flush=True)

# ══════════════════════════════════════════════════════════════
# TRUE LOMO EVALUATION
# ══════════════════════════════════════════════════════════════
print("\n[5] TRUE LOMO evaluation...", flush=True)

import lightgbm as lgb

LGB_P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
             class_weight="balanced", subsample=0.7, colsample_bytree=0.6,
             random_state=42, verbose=-1, n_jobs=1)

def true_lomo(name, X, y_labels, mask=None):
    oof = np.zeros((len(y), N))
    for m in sorted(set(train_months)):
        va = train_months == m
        tr = ~va
        if mask is not None:
            tr = tr & mask
        clf = lgb.LGBMClassifier(**LGB_P)
        clf.fit(X[tr], y_labels[tr])
        oof[va] = clf.predict_proba(X[va])
    overall, pc = compute_map(y, oof)
    ms = {}
    for m in sorted(set(train_months)):
        mask_m = train_months == m
        s, _ = compute_map(y[mask_m], oof[mask_m])
        ms[m] = s
    l = np.mean(list(ms.values()))
    month_str = " ".join(f"m{m}={v:.3f}" for m,v in sorted(ms.items()))
    print(f"  {name}: TRUE_LOMO={l:.4f} ({month_str})", flush=True)
    print(f"    Corm={pc['Cormorants']:.4f} Wader={pc['Waders']:.4f} BoP={pc['Birds of Prey']:.4f}", flush=True)
    return l, pc

# LGB baselines
true_lomo("LGB original labels", X_train, y)
true_lomo("LGB relabeled+clean", X_train, y_relabeled, mask=keep_mask)

# TabPFN TRUE LOMO (already have OOF from SKF, compute post-hoc — honest note: this is FAKE LOMO)
print(f"\n  TabPFN FAKE LOMO (post-hoc on SKF OOF — reference only):", flush=True)
for name, oof in [("TabPFN-ALL baseline", oof_a), ("TabPFN-ALL relabel", oof_b),
                   ("LR fusion", oof_lr), ("Best blend", oof_final_blend)]:
    l, ms = lomo(oof)
    _, pc = compute_map(y, oof)
    print(f"    {name:25s}: FAKE_LOMO={l:.4f} Corm={pc['Cormorants']:.4f} Wader={pc['Waders']:.4f}", flush=True)

# ══════════════════════════════════════════════════════════════
# SAVE SUBMISSIONS
# ══════════════════════════════════════════════════════════════
print("\n[6] Saving submissions...", flush=True)

save_submission(test_a, "e185_tabpfn_all", cv_map=skf_a)
save_submission(test_b, "e185_tabpfn_relabel", cv_map=skf_b)
save_submission(test_lr, "e185_lr_fusion", cv_map=skf_lr)
save_submission(test_final_blend, "e185_blend", cv_map=skf_blend)

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("  E185 RESULTS SUMMARY", flush=True)
print("=" * 90, flush=True)

configs = [
    ("E175 (LB=0.59)", oof_e175),
    ("TabPFN-ALL baseline", oof_a),
    ("TabPFN-ALL relabeled", oof_b),
    ("LR evidence fusion", oof_lr),
    ("Best blend", oof_final_blend),
]

print(f"\n  {'Config':30s} {'SKF':>7s} {'LOMO*':>7s} {'Corm':>7s} {'Wader':>7s} {'BoP':>7s} {'Geese':>7s} {'Ducks':>7s}", flush=True)
print(f"  {'-'*95}", flush=True)

for name, oof in configs:
    skf, pc = compute_map(y, oof)
    l, _ = lomo(oof)
    print(f"  {name:30s} {skf:7.4f} {l:7.4f} {pc['Cormorants']:7.4f} {pc['Waders']:7.4f} "
          f"{pc['Birds of Prey']:7.4f} {pc['Geese']:7.4f} {pc['Ducks']:7.4f}", flush=True)

print(f"\n  * LOMO is FAKE (post-hoc on SKF OOF). TRUE LOMO shown separately above.", flush=True)
print(f"\n  Completed at {time.strftime('%H:%M:%S')}", flush=True)
print("=" * 90, flush=True)
