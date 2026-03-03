"""E113: Graph-Based Flock Smoothing (Exploiting the Test Set Time Leak).

Mathematical Motivation:
1. The 0.61 team likely exploited the fact that tracks from the same flock
   appear almost simultaneously in the test set.
2. Heuristic time clustering (e.g., grouping all tracks within 10s) fails because
   multiple flocks of different species can pass the radar simultaneously.
3. We solve this by training a Pairwise Link Prediction model.
   Given two tracks, we predict P(same_flock) using purely *relative* features:
   Delta time, Delta lon, Delta lat, Delta speed, Delta altitude, and Size match.
4. Relative features are invariant to the seasonal domain shift!
5. We build a graph of test tracks where edges are P(same_flock) > 0.5.
6. We extract connected components and average the predictions of our best
   E111 Mega Ensemble for each flock.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import CLASSES, parse_ewkb_4d
from src.submission import save_submission

ROOT = Path(__file__).resolve().parent.parent

def connected_components(n_nodes, edges):
    parent = list(range(n_nodes))
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
    for i, j in edges:
        union(i, j)
    comps = {}
    for i in range(n_nodes):
        root = find(i)
        if root not in comps: comps[root] = []
        comps[root].append(i)
    return comps.values()

def extract_pairwise_features(df: pd.DataFrame, is_train: bool = False):
    df = df.copy()
    df["dt"] = pd.to_datetime(df["timestamp_start_radar_utc"])
    
    # We must preserve the original row order for predictions!
    df["orig_idx"] = np.arange(len(df))
    df = df.sort_values("dt").reset_index(drop=True)
    
    mean_lon = np.zeros(len(df))
    mean_lat = np.zeros(len(df))
    for i, row in df.iterrows():
        try:
            pts = parse_ewkb_4d(row["trajectory"])
            mean_lon[i] = np.mean([p[0] for p in pts])
            mean_lat[i] = np.mean([p[1] for p in pts])
        except:
            mean_lon[i] = np.nan
            mean_lat[i] = np.nan
            
    df["mean_lon"] = mean_lon
    df["mean_lat"] = mean_lat
    
    time_s = df["dt"].astype(int).values / 10**6
    
    pairs = []
    indices = []
    for i in range(len(df)):
        j = i + 1
        while j < len(df) and time_s[j] - time_s[i] <= 120:
            pair_data = {
                "dt": time_s[j] - time_s[i],
                "dlon": abs(df.loc[i, "mean_lon"] - df.loc[j, "mean_lon"]),
                "dlat": abs(df.loc[i, "mean_lat"] - df.loc[j, "mean_lat"]),
                "dspeed": abs(df.loc[i, "airspeed"] - df.loc[j, "airspeed"]),
                "dalt": abs(df.loc[i, "min_z"] - df.loc[j, "min_z"]),
                "same_size": df.loc[i, "radar_bird_size"] == df.loc[j, "radar_bird_size"]
            }
            if is_train:
                pair_data["same_group"] = (df.loc[i, "primary_observation_id"] == df.loc[j, "primary_observation_id"])
            
            pairs.append(pair_data)
            indices.append((i, j))
            j += 1
            
    df_pairs = pd.DataFrame(pairs)
    X = df_pairs[["dt", "dlon", "dlat", "dspeed", "dalt", "same_size"]].fillna(-1)
    
    if is_train:
        y = df_pairs["same_group"].astype(int)
        return X, y
    else:
        return X, indices, df["orig_idx"].values

def main() -> None:
    print("=" * 70, flush=True)
    print("E113 GRAPH-BASED FLOCK SMOOTHING".center(70), flush=True)
    print("=" * 70, flush=True)

    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    test_df = pd.read_csv(ROOT / "data" / "test.csv")

    print("\nExtracting pairwise features for Train...", flush=True)
    X_tr, y_tr = extract_pairwise_features(train_df, is_train=True)
    
    print(f"Training Pairwise Classifier on {len(X_tr)} pairs...", flush=True)
    clf = HistGradientBoostingClassifier(random_state=42, max_depth=5, learning_rate=0.05)
    clf.fit(X_tr, y_tr)
    
    print("\nExtracting pairwise features for Test...", flush=True)
    X_te, test_indices, test_orig_idx = extract_pairwise_features(test_df, is_train=False)
    
    print(f"Predicting P(same_flock) for {len(X_te)} test pairs...", flush=True)
    p_same = clf.predict_proba(X_te)[:, 1]
    
    # Load E111 Mega Ensemble
    print("\nLoading E111 Mega Ensemble predictions...", flush=True)
    base_df = pd.read_csv(ROOT / "submissions" / "e111_mega_ensemble_geo5_20260302_1333.csv")
    base_preds = base_df[CLASSES].values
    
    # The base_preds are in the original test.csv order.
    # We need to map them to the time-sorted order for smoothing, then map back.
    sorted_preds = base_preds[test_orig_idx]
    
    for thresh in [0.30, 0.50, 0.70]:
        print(f"\nBuilding Graph (Threshold = {thresh:.2f})...", flush=True)
        edges = []
        for k, (i, j) in enumerate(test_indices):
            if p_same[k] > thresh:
                edges.append((i, j))
                
        comps = connected_components(len(test_df), edges)
        n_multi = sum(1 for c in comps if len(c) > 1)
        print(f"Found {len(comps)} connected components ({n_multi} have >1 track).", flush=True)
        
        smoothed_sorted_preds = sorted_preds.copy()
        for comp in comps:
            if len(comp) > 1:
                comp_idx = list(comp)
                smoothed_sorted_preds[comp_idx] = np.mean(sorted_preds[comp_idx], axis=0)
                
        # Map back to original order
        final_preds = np.zeros_like(base_preds)
        final_preds[test_orig_idx] = smoothed_sorted_preds
        
        name = f"e113_graph_smoothed_mega_geo5_thresh{thresh:.2f}"
        save_submission(final_preds, name, cv_map=None)

    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
