"""E52: Month-aware blend (E50 for unseen months, E42+E50 for shared months).

Motivation from Kaggle feedback:
- E50 beats E51 on LB (better unseen-month generalization).
- E42 still carries useful signal on shared test months (Sep/Oct).

Strategy:
1) Grid-search blend weights for month 9 and month 10 on OOF labels.
2) Keep E50 unchanged for all other months (especially unseen 2/5/12).
3) Save tuned and conservative submissions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.metrics import compute_map, print_results
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent

GRID = [i / 20.0 for i in range(0, 21)]  # 0.00 .. 1.00


def renorm_rows(pred):
    pred = np.clip(pred, 1e-9, None)
    return pred / pred.sum(axis=1, keepdims=True)


def apply_month_blend(pred_50, pred_42, months, w9, w10):
    out = pred_50.copy()
    m9 = months == 9
    m10 = months == 10
    out[m9] = (1.0 - w9) * pred_50[m9] + w9 * pred_42[m9]
    out[m10] = (1.0 - w10) * pred_50[m10] + w10 * pred_42[m10]
    return renorm_rows(out)


print("=" * 70, flush=True)
print("E52 MONTH-AWARE BLEND (E42 + E50)".center(70), flush=True)
print("=" * 70, flush=True)

oof42 = np.load(ROOT / "oof_e42.npy")
oof50 = np.load(ROOT / "oof_e50.npy")
test42 = np.load(ROOT / "test_e42.npy")
test50 = np.load(ROOT / "test_e50.npy")

train_df = load_train()
test_df = load_test()
train_months = pd.to_datetime(train_df["timestamp_start_radar_utc"]).dt.month.values
test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train_df["bird_group"])

mask_shared = np.isin(train_months, [9, 10])
y_shared = y[mask_shared]

print("\nBaseline diagnostics:", flush=True)
m50_full, _ = compute_map(y, oof50)
m42_full, _ = compute_map(y, oof42)
m50_sh, _ = compute_map(y_shared, oof50[mask_shared])
m42_sh, _ = compute_map(y_shared, oof42[mask_shared])
print(f"  E50 full OOF mAP:        {m50_full:.4f}", flush=True)
print(f"  E42 full OOF mAP:        {m42_full:.4f}", flush=True)
print(f"  E50 shared-month mAP:    {m50_sh:.4f}", flush=True)
print(f"  E42 shared-month mAP:    {m42_sh:.4f}", flush=True)

print("\nGrid search on shared months (w9, w10):", flush=True)
best = {"w9": 0.0, "w10": 0.0, "shared": -1.0, "full": -1.0}
for w9 in GRID:
    for w10 in GRID:
        oof_blend = apply_month_blend(oof50, oof42, train_months, w9=w9, w10=w10)
        m_shared, _ = compute_map(y_shared, oof_blend[mask_shared])
        if m_shared > best["shared"]:
            m_full, _ = compute_map(y, oof_blend)
            best = {"w9": w9, "w10": w10, "shared": m_shared, "full": m_full}

print(
    f"  Best weights -> w9={best['w9']:.2f}, w10={best['w10']:.2f}, "
    f"shared={best['shared']:.4f}, full={best['full']:.4f}",
    flush=True,
)

oof_tuned = apply_month_blend(oof50, oof42, train_months, w9=best["w9"], w10=best["w10"])
m_tuned, per_tuned = compute_map(y, oof_tuned)
print_results(m_tuned, per_tuned, label="E52 tuned month-aware blend (full OOF)")

test_tuned = apply_month_blend(test50, test42, test_months, w9=best["w9"], w10=best["w10"])

# Conservative variant: shrink E42 weights by 20% to reduce overfit risk.
w9_cons = 0.8 * best["w9"]
w10_cons = 0.8 * best["w10"]
oof_cons = apply_month_blend(oof50, oof42, train_months, w9=w9_cons, w10=w10_cons)
m_cons, _ = compute_map(y, oof_cons)
print(
    f"\nConservative variant weights -> w9={w9_cons:.2f}, w10={w10_cons:.2f}, "
    f"full OOF={m_cons:.4f}",
    flush=True,
)
test_cons = apply_month_blend(test50, test42, test_months, w9=w9_cons, w10=w10_cons)

print("\nTest argmax distributions:", flush=True)
for label, pred in [("E50", test50), ("E52 tuned", test_tuned), ("E52 conservative", test_cons)]:
    dist = np.bincount(pred.argmax(axis=1), minlength=len(CLASSES))
    print(f"  {label:<16s}: " + " ".join(f"{c}:{int(dist[i])}" for i, c in enumerate(CLASSES)), flush=True)

np.save(ROOT / "oof_e52.npy", oof_tuned)
np.save(ROOT / "test_e52.npy", test_tuned)
save_submission(
    test_tuned,
    f"e52_monthaware_w9_{best['w9']:.2f}_w10_{best['w10']:.2f}",
    cv_map=m_tuned,
)
save_submission(
    test_cons,
    f"e52_monthaware_cons_w9_{w9_cons:.2f}_w10_{w10_cons:.2f}",
    cv_map=m_cons,
)

print("\nSaved: oof_e52.npy, test_e52.npy + 2 submissions", flush=True)
print("Done.", flush=True)
