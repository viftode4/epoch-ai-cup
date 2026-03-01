"""E106: Combine E104 base model + E105 specialist PP.

Applies specialist corrections on E104's better base model predictions.
Also compares against E79+specialist and E104+default NB PP.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, load_test, load_train
from src.submission import save_submission
from src.postprocessing import (
    BASE_ALPHA, UNSEEN_MONTHS,
    renorm_rows, top2_margin,
    build_gbif_priors, apply_gated_ratio_priors,
    build_nb_params, compute_log_p_u_given_c, apply_nb_poe,
    apply_specialist_corrections,
)
from src.metrics import compute_map, print_results

ROOT = Path(__file__).resolve().parent.parent
N_CLASSES = len(CLASSES)


def main():
    print("=" * 70, flush=True)
    print("E106 COMBINED: E104 base + E105 specialist PP".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = load_train()
    test_df = load_test()
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values
    y = pd.Categorical(train_df["bird_group"], categories=CLASSES).codes.values.astype(int)

    # Check if E104 predictions exist
    e104_oof_path = ROOT / "oof_e104.npy"
    e104_test_path = ROOT / "test_e104.npy"
    e79_test_path = ROOT / "test_e79.npy"

    if not e104_test_path.exists():
        print("ERROR: E104 predictions not found. Run e104_better_base.py first.", flush=True)
        print("  Falling back to E79 base...", flush=True)
        test_base = renorm_rows(np.load(e79_test_path).astype(float))
        base_label = "E79"
    else:
        test_base = renorm_rows(np.load(e104_test_path).astype(float))
        base_label = "E104"
        print(f"  Using {base_label} base predictions", flush=True)

    # Best configs from E105 evaluation
    configs = [
        ("A_conservative", 0.30, 0.20, 0.15, 0.25),
        ("B_moderate",     0.50, 0.30, 0.20, 0.25),
        ("C_aggressive",   0.70, 0.40, 0.30, 0.25),
    ]

    for name, gc, gb, gr, tau in configs:
        print(f"\n--- {base_label}+specialist {name} ---", flush=True)
        out = apply_specialist_corrections(
            test_base, test_df, test_months, train_df, y,
            gamma_clutter=gc, gamma_bop=gb, gamma_rescue=gr, tau_nb=tau,
        )
        sub_name = f"e106_{base_label.lower()}_{name}"
        save_submission(out, sub_name, cv_map=None)

    # Also save raw E104 (no PP) for comparison
    if base_label == "E104":
        save_submission(test_base, "e106_e104_raw", cv_map=None)

    print("\nDone.", flush=True)


def local_eval():
    """Evaluate via IW-mAP."""
    from src.validate import eval_pp, _cache
    from src.postprocessing import renorm_rows as pp_renorm

    train_df = load_train()
    test_df = load_test()

    e104_oof_path = ROOT / "oof_e104.npy"
    e104_test_path = ROOT / "test_e104.npy"

    if not e104_test_path.exists():
        print("ERROR: E104 predictions not found. Run e104_better_base.py first.", flush=True)
        return

    # Inject E104 predictions into validate cache
    _cache.clear()
    oof_e104 = pp_renorm(np.load(e104_oof_path).astype(float))
    test_e104 = pp_renorm(np.load(e104_test_path).astype(float))
    test_months = pd.to_datetime(test_df["timestamp_start_radar_utc"]).dt.month.values

    _cache["oof"] = (oof_e104, "E104")
    _cache["test"] = (test_e104, test_df, test_months)

    # Evaluate identity PP (raw E104)
    print("\n--- E104 raw (no PP) ---", flush=True)
    def identity_pp(preds, test_df, test_months, train_df, y):
        return preds
    result = eval_pp(identity_pp)
    print(f"  >> Cal.LB={result.get('calibrated_lb', 'N/A')}", flush=True)

    # Evaluate specialist PP
    configs = [
        ("A_conservative", 0.30, 0.20, 0.15, 0.25),
        ("B_moderate",     0.50, 0.30, 0.20, 0.25),
        ("C_aggressive",   0.70, 0.40, 0.30, 0.25),
    ]

    for name, gc, gb, gr, tau in configs:
        print(f"\n--- E104 + specialist {name} ---", flush=True)
        _cache.pop("mlls", None)  # clear MLLS cache between runs
        _cache.pop("mlls_raw", None)
        _cache.pop("p_train", None)
        _cache.pop("T", None)

        def make_pp(gc=gc, gb=gb, gr=gr, tau=tau):
            def pp(preds, test_df, test_months, train_df, y):
                return apply_specialist_corrections(
                    preds, test_df, test_months, train_df, y,
                    gamma_clutter=gc, gamma_bop=gb, gamma_rescue=gr, tau_nb=tau,
                )
            return pp

        result = eval_pp(make_pp())
        print(f"  >> Cal.LB={result.get('calibrated_lb', 'N/A')}", flush=True)


if __name__ == "__main__":
    if "--eval" in sys.argv:
        local_eval()
    else:
        main()
