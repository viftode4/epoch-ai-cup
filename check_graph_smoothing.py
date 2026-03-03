import pandas as pd
import numpy as np
from src.data import parse_ewkb_4d, CLASSES
from src.metrics import compute_map
from sklearn.ensemble import HistGradientBoostingClassifier

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

train = pd.read_csv("data/train.csv")
train["dt"] = pd.to_datetime(train["timestamp_start_radar_utc"])
train["orig_idx"] = np.arange(len(train))

mean_lon = np.zeros(len(train))
mean_lat = np.zeros(len(train))
for i, row in train.iterrows():
    try:
        pts = parse_ewkb_4d(row["trajectory"])
        mean_lon[i] = np.mean([p[0] for p in pts])
        mean_lat[i] = np.mean([p[1] for p in pts])
    except:
        mean_lon[i] = np.nan
        mean_lat[i] = np.nan

train["mean_lon"] = mean_lon
train["mean_lat"] = mean_lat

train = train.sort_values("dt").reset_index(drop=True)
train["idx"] = np.arange(len(train))

time_s = train["dt"].astype(int).values / 10**6
train_months = train["dt"].dt.month.values

unique_months = sorted(set(train_months))
y = pd.Categorical(train["bird_group"], categories=CLASSES).codes

oof_base_orig = np.load("oof_e50.npy")
oof_base = oof_base_orig[train["orig_idx"].values]

all_p_same = []
all_va_indices = []
all_va_idx_global = []
all_va_df_len = []

for m in unique_months:
    tr_mask = train_months != m
    va_mask = train_months == m
    
    tr_df = train[tr_mask].reset_index(drop=True)
    va_df = train[va_mask].reset_index(drop=True)
    
    tr_time_s = time_s[tr_mask]
    pairs = []
    for i in range(len(tr_df)):
        j = i + 1
        while j < len(tr_df) and tr_time_s[j] - tr_time_s[i] <= 120:
            is_same = (tr_df.loc[i, "primary_observation_id"] == tr_df.loc[j, "primary_observation_id"])
            pairs.append({
                "dt": tr_time_s[j] - tr_time_s[i],
                "dlon": abs(tr_df.loc[i, "mean_lon"] - tr_df.loc[j, "mean_lon"]),
                "dlat": abs(tr_df.loc[i, "mean_lat"] - tr_df.loc[j, "mean_lat"]),
                "dspeed": abs(tr_df.loc[i, "airspeed"] - tr_df.loc[j, "airspeed"]),
                "dalt": abs(tr_df.loc[i, "min_z"] - tr_df.loc[j, "min_z"]),
                "same_size": tr_df.loc[i, "radar_bird_size"] == tr_df.loc[j, "radar_bird_size"],
                "same_group": is_same
            })
            j += 1
            
    df_pairs = pd.DataFrame(pairs)
    X_tr = df_pairs[["dt", "dlon", "dlat", "dspeed", "dalt", "same_size"]].fillna(-1)
    y_tr = df_pairs["same_group"].astype(int)
    
    clf = HistGradientBoostingClassifier(random_state=42, max_depth=5, learning_rate=0.05)
    clf.fit(X_tr, y_tr)
    
    va_time_s = time_s[va_mask]
    va_pairs = []
    va_indices = []
    for i in range(len(va_df)):
        j = i + 1
        while j < len(va_df) and va_time_s[j] - va_time_s[i] <= 120:
            va_pairs.append({
                "dt": va_time_s[j] - va_time_s[i],
                "dlon": abs(va_df.loc[i, "mean_lon"] - va_df.loc[j, "mean_lon"]),
                "dlat": abs(va_df.loc[i, "mean_lat"] - va_df.loc[j, "mean_lat"]),
                "dspeed": abs(va_df.loc[i, "airspeed"] - va_df.loc[j, "airspeed"]),
                "dalt": abs(va_df.loc[i, "min_z"] - va_df.loc[j, "min_z"]),
                "same_size": va_df.loc[i, "radar_bird_size"] == va_df.loc[j, "radar_bird_size"]
            })
            va_indices.append((i, j))
            j += 1
            
    if len(va_pairs) > 0:
        df_va_pairs = pd.DataFrame(va_pairs)
        X_va = df_va_pairs[["dt", "dlon", "dlat", "dspeed", "dalt", "same_size"]].fillna(-1)
        p_same = clf.predict_proba(X_va)[:, 1]
    else:
        p_same = np.array([])
        
    all_p_same.append(p_same)
    all_va_indices.append(va_indices)
    all_va_idx_global.append(train[va_mask]["idx"].values)
    all_va_df_len.append(len(va_df))

for thresh in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]:
    oof_smooth = oof_base.copy()
    for m_idx in range(len(unique_months)):
        p_same = all_p_same[m_idx]
        va_indices = all_va_indices[m_idx]
        va_idx_global = all_va_idx_global[m_idx]
        n_nodes = all_va_df_len[m_idx]
        
        edges = []
        for k, (i, j) in enumerate(va_indices):
            if p_same[k] > thresh:
                edges.append((i, j))
                
        for comp in connected_components(n_nodes, edges):
            if len(comp) > 1:
                comp_idx = list(comp)
                global_idx = va_idx_global[comp_idx]
                oof_smooth[global_idx] = np.mean(oof_base[global_idx], axis=0)
                
    m_smooth, _ = compute_map(y, oof_smooth)
    print(f"Thresh {thresh:.2f} -> mAP: {m_smooth:.4f}")

