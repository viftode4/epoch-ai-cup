"""
Validate ALL noise/label approaches for Cormorants using existing OOF predictions.
NO model training — uses only pre-computed predictions.

Checks:
1. Consensus relabeling (4 models)
2. Soft-label analysis (TabPFN)
3. Curriculum learning feasibility (cleanlab + feature stats)
4. Active label cleaning (metadata inspection)
5. Noise gradient analysis
6. Leave-one-out feasibility (fold stability)
"""

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import entropy
from pathlib import Path

# ── Setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CLASSES = [
    "Birds of Prey", "Clutter", "Cormorants", "Ducks", "Geese",
    "Gulls", "Pigeons", "Songbirds", "Waders",
]
N_CLASSES = len(CLASSES)
CORM_IDX = 2  # Cormorants index in CLASSES

# Load data
from src.data import load_train, parse_ewkb_4d
train_df = load_train()
y_labels = train_df["bird_group"].map({c: i for i, c in enumerate(CLASSES)}).values
corm_mask = y_labels == CORM_IDX
corm_positions = np.where(corm_mask)[0]
n_corm = corm_mask.sum()

print(f"Total samples: {len(y_labels)}, Cormorants: {n_corm}")
print("=" * 90)

# Load OOF predictions
oof_tabpfn = np.load(ROOT / "oof_e183_tabpfn.npy")  # already probabilities
oof_best = np.load(ROOT / "oof_e175_best.npy")
oof_cb = np.load(ROOT / "oof_e175_cb.npy")
oof_ranker_raw = np.load(ROOT / "oof_e175_ranker.npy")

# Convert ranker scores to probabilities
oof_ranker = softmax(oof_ranker_raw, axis=1)
# E175 best and cb: check if they sum to ~1 already
best_sums = oof_best.sum(axis=1)
cb_sums = oof_cb.sum(axis=1)
print(f"E175 best row sums: mean={best_sums.mean():.4f}, min={best_sums.min():.4f}, max={best_sums.max():.4f}")
print(f"E175 cb row sums: mean={cb_sums.mean():.4f}, min={cb_sums.min():.4f}, max={cb_sums.max():.4f}")

# Normalize if not already probabilities
if abs(best_sums.mean() - 1.0) > 0.01:
    oof_best_prob = softmax(oof_best, axis=1)
    print("  -> E175 best: applied softmax")
else:
    oof_best_prob = oof_best
    print("  -> E175 best: already probabilities")

if abs(cb_sums.mean() - 1.0) > 0.01:
    oof_cb_prob = softmax(oof_cb, axis=1)
    print("  -> E175 cb: applied softmax")
else:
    oof_cb_prob = oof_cb
    print("  -> E175 cb: already probabilities")

models = {
    "TabPFN": oof_tabpfn,
    "E175_best": oof_best_prob,
    "E175_cb": oof_cb_prob,
    "E175_ranker": oof_ranker,
}

# ═══════════════════════════════════════════════════════════════════════
# 1. CONSENSUS RELABELING
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("1. CONSENSUS RELABELING — What do 4 models predict for each Cormorant?")
print("=" * 90)

corm_df = train_df.iloc[corm_positions].copy()
corm_df = corm_df.reset_index(drop=True)

# Build prediction table
rows = []
for i, global_idx in enumerate(corm_positions):
    row = {
        "idx": i,
        "track_id": corm_df.loc[i, "track_id"],
        "airspeed": corm_df.loc[i, "airspeed"],
        "radar_size": corm_df.loc[i, "radar_bird_size"],
    }
    preds = {}
    agree_corm = 0
    for name, oof in models.items():
        pred_class = np.argmax(oof[global_idx])
        pred_prob = oof[global_idx, pred_class]
        corm_prob = oof[global_idx, CORM_IDX]
        preds[f"{name}_pred"] = CLASSES[pred_class]
        preds[f"{name}_conf"] = pred_prob
        preds[f"{name}_P(corm)"] = corm_prob
        if pred_class == CORM_IDX:
            agree_corm += 1
    row.update(preds)
    row["n_agree_corm"] = agree_corm
    rows.append(row)

consensus_df = pd.DataFrame(rows)

# Summary
print(f"\nAgreement distribution (how many models predict Cormorant):")
for n in range(5):
    count = (consensus_df["n_agree_corm"] == n).sum()
    if count > 0:
        print(f"  {n}/4 models agree = Cormorant: {count} samples")

# Show the would-change samples
would_change = consensus_df[consensus_df["n_agree_corm"] <= 1]
print(f"\nSamples where <=1 model predicts Cormorant ({len(would_change)}):")

# Full table for all 40 Cormorants
print(f"\nFull table (all {n_corm} Cormorants):")
display_cols = ["idx", "track_id", "airspeed", "radar_size", "n_agree_corm"]
for name in models:
    display_cols += [f"{name}_pred", f"{name}_P(corm)"]

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 15)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print(consensus_df[display_cols].to_string(index=False))

# What classes do models predict instead?
print("\nAlternative class predictions (when not Cormorant):")
alt_preds = {}
for name in models:
    col = f"{name}_pred"
    non_corm = consensus_df[consensus_df[col] != "Cormorants"][col]
    for cls in non_corm:
        alt_preds[cls] = alt_preds.get(cls, 0) + 1
for cls, count in sorted(alt_preds.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {count} predictions")


# ═══════════════════════════════════════════════════════════════════════
# 2. SOFT-LABEL ANALYSIS (TabPFN)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("2. SOFT-LABEL ANALYSIS — TabPFN probability distributions for Cormorants")
print("=" * 90)

corm_tabpfn = oof_tabpfn[corm_positions]

print(f"\nMean soft-label distribution across {n_corm} Cormorants (TabPFN):")
mean_probs = corm_tabpfn.mean(axis=0)
for j, cls in enumerate(CLASSES):
    bar = "#" * int(mean_probs[j] * 50)
    print(f"  {cls:18s}: {mean_probs[j]:.4f}  {bar}")

# Entropy comparison: hard vs soft
hard_entropy = 0.0  # one-hot has 0 entropy
soft_entropies = np.array([entropy(row) for row in corm_tabpfn])
print(f"\nSoft-label entropy stats:")
print(f"  Hard label entropy: 0.000 (one-hot)")
print(f"  Soft label entropy: mean={soft_entropies.mean():.4f}, "
      f"std={soft_entropies.std():.4f}, min={soft_entropies.min():.4f}, max={soft_entropies.max():.4f}")
print(f"  Max possible entropy (uniform 9-class): {np.log(9):.4f}")

# Show individual soft labels for the most confused samples
print(f"\nTop 10 highest-entropy Cormorants (most confused by TabPFN):")
entropy_order = np.argsort(-soft_entropies)
for rank, i in enumerate(entropy_order[:10]):
    global_idx = corm_positions[i]
    top3_idx = np.argsort(-corm_tabpfn[i])[:3]
    top3_str = ", ".join(
        f"{CLASSES[j]}={corm_tabpfn[i, j]:.3f}" for j in top3_idx
    )
    print(f"  #{rank+1} track={corm_df.loc[i, 'track_id']}, "
          f"H={soft_entropies[i]:.3f}, "
          f"P(Corm)={corm_tabpfn[i, CORM_IDX]:.3f}, "
          f"Top3: [{top3_str}]")

# Information gain from soft labels
print(f"\nInformation gain from soft labels:")
avg_soft_entropy = soft_entropies.mean()
kl_divs = []
for i in range(n_corm):
    hard = np.zeros(N_CLASSES)
    hard[CORM_IDX] = 1.0
    soft = corm_tabpfn[i].clip(1e-10, 1.0)
    kl = entropy(hard, soft)
    kl_divs.append(kl)
kl_divs = np.array(kl_divs)
print(f"  KL(hard || soft): mean={kl_divs.mean():.4f}, "
      f"std={kl_divs.std():.4f}, max={kl_divs.max():.4f}")
print(f"  Interpretation: Higher KL = soft labels disagree more with hard label")
print(f"  Samples with KL > 1.0 (strong disagreement): {(kl_divs > 1.0).sum()}")
print(f"  Samples with KL > 2.0 (very strong disagreement): {(kl_divs > 2.0).sum()}")


# ═══════════════════════════════════════════════════════════════════════
# 3. CURRICULUM LEARNING FEASIBILITY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("3. CURRICULUM LEARNING FEASIBILITY — Clean vs Noisy Cormorants")
print("=" * 90)

# Use cleanlab on TabPFN predictions
try:
    from cleanlab.rank import get_label_quality_scores
    label_quality = get_label_quality_scores(
        labels=y_labels,
        pred_probs=oof_tabpfn,
        method="self_confidence",
    )
    corm_quality = label_quality[corm_positions]
    print(f"\nCleanlab label quality scores for Cormorants:")
    print(f"  Mean: {corm_quality.mean():.4f}")
    print(f"  Std:  {corm_quality.std():.4f}")
    print(f"  Min:  {corm_quality.min():.4f} (most suspect)")
    print(f"  Max:  {corm_quality.max():.4f} (most confident)")

    # Compare with all classes
    print(f"\nLabel quality by class (mean):")
    for j, cls in enumerate(CLASSES):
        cls_mask = y_labels == j
        cls_q = label_quality[cls_mask]
        print(f"  {cls:18s}: mean={cls_q.mean():.4f}, "
              f"min={cls_q.min():.4f}, n={cls_mask.sum()}")

    # Split clean/noisy
    median_q = np.median(corm_quality)
    clean_mask_local = corm_quality >= median_q  # top 50%
    noisy_mask_local = corm_quality < median_q   # bottom 50%

    print(f"\nSplit: {clean_mask_local.sum()} clean, {noisy_mask_local.sum()} noisy "
          f"(threshold quality={median_q:.4f})")

except ImportError:
    print("cleanlab not installed, using TabPFN P(Cormorant) as quality proxy")
    corm_quality = oof_tabpfn[corm_positions, CORM_IDX]
    median_q = np.median(corm_quality)
    clean_mask_local = corm_quality >= median_q
    noisy_mask_local = corm_quality < median_q
    print(f"\nSplit: {clean_mask_local.sum()} clean, {noisy_mask_local.sum()} noisy "
          f"(threshold P(Corm)={median_q:.4f})")

# Extract physical features for clean vs noisy
print(f"\nPhysical stats: Clean vs Noisy Cormorants:")
clean_corm = corm_df[clean_mask_local]
noisy_corm = corm_df[noisy_mask_local]

# Also get Gulls for comparison
gull_mask = y_labels == CLASSES.index("Gulls")
gull_df = train_df[gull_mask]

stats_features = ["airspeed", "min_z", "max_z"]
print(f"{'Feature':<15} {'Clean Corm':>15} {'Noisy Corm':>15} {'Gulls':>15}")
print("-" * 65)
for feat in stats_features:
    c_mean = clean_corm[feat].mean()
    c_std = clean_corm[feat].std()
    n_mean = noisy_corm[feat].mean()
    n_std = noisy_corm[feat].std()
    g_mean = gull_df[feat].mean()
    g_std = gull_df[feat].std()
    print(f"{feat:<15} {c_mean:7.2f}±{c_std:5.2f}  {n_mean:7.2f}±{n_std:5.2f}  {g_mean:7.2f}±{g_std:5.2f}")

# Parse trajectories for RCS stats
print(f"\nRCS stats (from trajectories):")
for label, subset, indices in [
    ("Clean Corm", clean_corm, corm_positions[clean_mask_local]),
    ("Noisy Corm", noisy_corm, corm_positions[noisy_mask_local]),
]:
    rcs_means = []
    rcs_stds = []
    track_lens = []
    for _, row in train_df.iloc[indices].iterrows():
        pts = parse_ewkb_4d(row["trajectory"])
        rcs_vals = [p[3] for p in pts]
        rcs_means.append(np.mean(rcs_vals))
        rcs_stds.append(np.std(rcs_vals))
        track_lens.append(len(pts))
    print(f"  {label}: RCS_mean={np.mean(rcs_means):.2f}dB, "
          f"RCS_std={np.mean(rcs_stds):.2f}dB, "
          f"track_len={np.mean(track_lens):.1f}")

# Are clean ones separable from Gulls?
print(f"\nSeparability assessment:")
clean_speed = clean_corm["airspeed"].values
noisy_speed = noisy_corm["airspeed"].values
gull_speed = gull_df["airspeed"].values
from scipy.stats import mannwhitneyu
if len(clean_speed) >= 3 and len(gull_speed) >= 3:
    u_stat, p_val = mannwhitneyu(clean_speed, gull_speed, alternative="two-sided")
    print(f"  Clean Corm vs Gulls (airspeed): U={u_stat:.0f}, p={p_val:.4f}")
if len(noisy_speed) >= 3 and len(gull_speed) >= 3:
    u_stat, p_val = mannwhitneyu(noisy_speed, gull_speed, alternative="two-sided")
    print(f"  Noisy Corm vs Gulls (airspeed): U={u_stat:.0f}, p={p_val:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# 4. ACTIVE LABEL CLEANING — Metadata inspection
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("4. ACTIVE LABEL CLEANING — Metadata for all 40 Cormorants")
print("=" * 90)

meta_cols = [
    "track_id", "bird_species", "radar_bird_size",
    "n_birds_observed", "observer_comment",
    "primary_observation_id", "airspeed",
]
corm_meta = train_df.iloc[corm_positions][meta_cols].copy()
corm_meta["quality_score"] = corm_quality
corm_meta = corm_meta.sort_values("quality_score")

print(f"\nAll 40 Cormorants sorted by label quality (worst first):")
pd.set_option("display.max_rows", 50)
print(corm_meta.to_string(index=False))

# Flag suspicious patterns
print(f"\nSuspicious metadata patterns:")
small_corm = corm_meta[corm_meta["radar_bird_size"] == "Small bird"]
print(f"  'Small bird' radar size (Cormorants should be Large): {len(small_corm)}")
if len(small_corm) > 0:
    print(f"    track_ids: {small_corm['track_id'].tolist()}")

flock_but_1 = corm_meta[
    (corm_meta["radar_bird_size"] == "Flock") &
    (corm_meta["n_birds_observed"] == 1)
]
print(f"  Flock radar size but n_birds=1: {len(flock_but_1)}")

# Check observation grouping
obs_groups = corm_meta["primary_observation_id"].value_counts()
shared_obs = obs_groups[obs_groups > 1]
print(f"\n  Shared primary_observation_id groups: {len(shared_obs)}")
for obs_id, count in shared_obs.items():
    print(f"    obs_id={obs_id}: {count} tracks")

# Cormorant-specific red flags
print(f"\n  Speed-based red flags (Cormorant typical: 14-18 m/s):")
slow = corm_meta[corm_meta["airspeed"] < 10]
fast = corm_meta[corm_meta["airspeed"] > 22]
print(f"    Unusually slow (<10 m/s): {len(slow)} tracks")
if len(slow) > 0:
    for _, r in slow.iterrows():
        print(f"      track={r['track_id']}, speed={r['airspeed']:.1f}, "
              f"quality={r['quality_score']:.3f}")
print(f"    Unusually fast (>22 m/s): {len(fast)} tracks")
if len(fast) > 0:
    for _, r in fast.iterrows():
        print(f"      track={r['track_id']}, speed={r['airspeed']:.1f}, "
              f"quality={r['quality_score']:.3f}")


# ═══════════════════════════════════════════════════════════════════════
# 5. NOISE GRADIENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("5. NOISE GRADIENT ANALYSIS — How noisy labels distort learning")
print("=" * 90)

print(f"\nFor each suspect Cormorant: cross-entropy gradient direction")
print(f"  If model predicts Gulls with high P but label=Cormorant,")
print(f"  the gradient pushes TOWARD Cormorant (the label) and AWAY FROM Gulls (the prediction)")
print()

# Use TabPFN as our best model
print(f"{'track_id':>10} {'P(Corm)':>8} {'P(Gull)':>8} {'Top Pred':>15} {'Grad_Corm':>10} "
      f"{'Grad_TopPred':>12} {'CE_loss':>10} {'quality':>8}")
print("-" * 95)

total_ce_loss = 0.0
suspect_ce_loss = 0.0
suspect_count = 0

for i, global_idx in enumerate(corm_positions):
    p = oof_tabpfn[global_idx]
    p_corm = p[CORM_IDX]
    p_gull = p[CLASSES.index("Gulls")]
    top_pred_idx = np.argmax(p)
    top_pred_name = CLASSES[top_pred_idx]
    p_top = p[top_pred_idx]

    # Cross-entropy loss: -log(P(true_class))
    ce_loss = -np.log(np.clip(p_corm, 1e-10, 1.0))
    total_ce_loss += ce_loss

    # Gradient of CE w.r.t. logits (softmax output):
    # grad_j = p_j - 1{j=y}
    # For true class (Cormorant): grad = P(Corm) - 1 (negative, pulls toward Corm)
    # For predicted class: grad = P(pred) (positive, pushes away from pred)
    grad_corm = p_corm - 1.0  # negative: pulls toward Cormorant
    grad_top = p_top if top_pred_idx != CORM_IDX else 0.0  # positive: pushes away

    is_suspect = p_corm < 0.3
    if is_suspect:
        suspect_ce_loss += ce_loss
        suspect_count += 1

    marker = " ***" if is_suspect else ""
    print(f"{corm_df.loc[i, 'track_id']:>10} {p_corm:>8.4f} {p_gull:>8.4f} "
          f"{top_pred_name:>15} {grad_corm:>10.4f} {grad_top:>12.4f} "
          f"{ce_loss:>10.4f} {corm_quality[i]:>8.4f}{marker}")

print(f"\nSummary:")
print(f"  Total CE loss from Cormorant class: {total_ce_loss:.4f}")
print(f"  Suspect samples (P(Corm)<0.3): {suspect_count}")
if suspect_count > 0:
    print(f"  CE loss from suspects: {suspect_ce_loss:.4f} "
          f"({suspect_ce_loss/total_ce_loss*100:.1f}% of total)")
    print(f"  Avg CE loss per suspect: {suspect_ce_loss/suspect_count:.4f}")
    print(f"  Avg CE loss per clean: {(total_ce_loss-suspect_ce_loss)/(n_corm-suspect_count):.4f}")

# Impact analysis: what if we relabel suspects?
print(f"\n  IMPACT: If we relabeled suspects to their model-predicted class:")
relabel_gains = []
for i, global_idx in enumerate(corm_positions):
    p = oof_tabpfn[global_idx]
    p_corm = p[CORM_IDX]
    top_pred_idx = np.argmax(p)
    if p_corm < 0.3 and top_pred_idx != CORM_IDX:
        old_loss = -np.log(np.clip(p_corm, 1e-10, 1.0))
        new_loss = -np.log(np.clip(p[top_pred_idx], 1e-10, 1.0))
        gain = old_loss - new_loss
        relabel_gains.append(gain)
if relabel_gains:
    print(f"  Total CE reduction: {sum(relabel_gains):.4f}")
    print(f"  Average CE reduction per relabel: {np.mean(relabel_gains):.4f}")


# ═══════════════════════════════════════════════════════════════════════
# 6. LEAVE-ONE-OUT FEASIBILITY — Fold stability
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("6. LEAVE-ONE-OUT FEASIBILITY — Cross-fold prediction stability")
print("=" * 90)

print(f"\nWith 40 samples across 5 CV folds, each Cormorant appears in exactly 1 OOF fold.")
print(f"We check stability by comparing predictions across different models (proxy for folds).")
print()

# Compare P(Cormorant) across all 4 models for each sample
print(f"{'track_id':>10} {'TabPFN':>8} {'E175best':>8} {'E175cb':>8} {'Ranker':>8} "
      f"{'Std':>8} {'Range':>8} {'Verdict':>10}")
print("-" * 80)

stabilities = []
for i, global_idx in enumerate(corm_positions):
    p_tabpfn = oof_tabpfn[global_idx, CORM_IDX]
    p_best = oof_best_prob[global_idx, CORM_IDX]
    p_cb = oof_cb_prob[global_idx, CORM_IDX]
    p_ranker = oof_ranker[global_idx, CORM_IDX]

    probs = [p_tabpfn, p_best, p_cb, p_ranker]
    std = np.std(probs)
    rng = max(probs) - min(probs)
    stabilities.append(std)

    verdict = "STABLE" if std < 0.10 else ("MODERATE" if std < 0.20 else "UNSTABLE")
    print(f"{corm_df.loc[i, 'track_id']:>10} {p_tabpfn:>8.4f} {p_best:>8.4f} "
          f"{p_cb:>8.4f} {p_ranker:>8.4f} {std:>8.4f} {rng:>8.4f} {verdict:>10}")

stabilities = np.array(stabilities)
print(f"\nStability summary:")
print(f"  STABLE (std<0.10): {(stabilities < 0.10).sum()} samples")
print(f"  MODERATE (0.10-0.20): {((stabilities >= 0.10) & (stabilities < 0.20)).sum()} samples")
print(f"  UNSTABLE (std>=0.20): {(stabilities >= 0.20).sum()} samples")
print(f"  Mean std: {stabilities.mean():.4f}")

# Cross-reference: unstable AND low quality = strongest candidates for relabeling
print(f"\nCross-reference: unstable + low quality samples:")
for i in range(n_corm):
    if stabilities[i] >= 0.15 and corm_quality[i] < median_q:
        global_idx = corm_positions[i]
        top_pred = CLASSES[np.argmax(oof_tabpfn[global_idx])]
        print(f"  track={corm_df.loc[i, 'track_id']}, "
              f"quality={corm_quality[i]:.3f}, "
              f"stability_std={stabilities[i]:.3f}, "
              f"TabPFN_pred={top_pred}, "
              f"P(Corm)_range=[{min(oof_tabpfn[global_idx, CORM_IDX], oof_best_prob[global_idx, CORM_IDX], oof_cb_prob[global_idx, CORM_IDX], oof_ranker[global_idx, CORM_IDX]):.3f}, "
              f"{max(oof_tabpfn[global_idx, CORM_IDX], oof_best_prob[global_idx, CORM_IDX], oof_cb_prob[global_idx, CORM_IDX], oof_ranker[global_idx, CORM_IDX]):.3f}]")


# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("FINAL SUMMARY — Noise approach feasibility")
print("=" * 90)

# Count truly suspect samples (all evidence combined)
truly_suspect = 0
relabel_candidates = []
for i in range(n_corm):
    global_idx = corm_positions[i]
    score = 0
    if consensus_df.loc[i, "n_agree_corm"] <= 1:
        score += 1
    if corm_quality[i] < median_q:
        score += 1
    if stabilities[i] >= 0.15:
        score += 1
    if oof_tabpfn[global_idx, CORM_IDX] < 0.3:
        score += 1
    if score >= 3:
        truly_suspect += 1
        top_pred = CLASSES[np.argmax(oof_tabpfn[global_idx])]
        relabel_candidates.append({
            "track_id": corm_df.loc[i, "track_id"],
            "evidence_score": score,
            "P_corm_tabpfn": oof_tabpfn[global_idx, CORM_IDX],
            "consensus": consensus_df.loc[i, "n_agree_corm"],
            "quality": corm_quality[i],
            "suggested_class": top_pred,
        })

print(f"\n1. CONSENSUS: {(consensus_df['n_agree_corm'] <= 1).sum()}/40 Cormorants "
      f"have <=1/4 models agreeing they are Cormorants")
print(f"2. SOFT LABELS: Mean entropy = {soft_entropies.mean():.3f} "
      f"(uncertainty is {'HIGH' if soft_entropies.mean() > 0.5 else 'moderate'})")
print(f"3. CURRICULUM: Clean/noisy split shows "
      f"{'clear' if abs(clean_corm['airspeed'].mean() - noisy_corm['airspeed'].mean()) > 2 else 'marginal'} "
      f"physical separation")
print(f"4. METADATA: {len(small_corm)} 'Small bird' radar size (red flag for Cormorants)")
print(f"5. GRADIENTS: {suspect_count} suspects contribute "
      f"{suspect_ce_loss/max(total_ce_loss,1e-10)*100:.0f}% of Cormorant CE loss")
print(f"6. STABILITY: {(stabilities >= 0.20).sum()} samples are prediction-unstable")

print(f"\nTRULY SUSPECT (>=3 evidence flags): {truly_suspect}/{n_corm}")
if relabel_candidates:
    print(f"\nRelabel candidates:")
    for r in sorted(relabel_candidates, key=lambda x: -x["evidence_score"]):
        print(f"  track={r['track_id']}, evidence={r['evidence_score']}/4, "
              f"P(Corm)={r['P_corm_tabpfn']:.3f}, consensus={r['consensus']}/4, "
              f"quality={r['quality']:.3f}, suggest={r['suggested_class']}")

print(f"\nRECOMMENDATION:")
if truly_suspect >= 5:
    print(f"  STRONG: {truly_suspect} samples are likely mislabeled. "
          f"Relabeling or removing them should help.")
elif truly_suspect >= 2:
    print(f"  MODERATE: {truly_suspect} samples may be mislabeled. "
          f"Worth trying relabeling in a controlled experiment.")
else:
    print(f"  WEAK: Only {truly_suspect} truly suspect. "
          f"Noise may not be a major issue for Cormorants.")

print(f"\nDone.")
