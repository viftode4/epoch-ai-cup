import pandas as pd
import numpy as np
import struct
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/train.csv')
CLASSES = ['Clutter', 'Cormorants', 'Pigeons', 'Ducks', 'Geese', 'Gulls', 'Birds of Prey', 'Waders', 'Songbirds']
WEAK = ['Pigeons', 'Clutter', 'Songbirds', 'Ducks']

# ============================================================
# 1. WHAT GETS CONFUSED WITH WHAT? (from OOF predictions)
# ============================================================
# Load the OOF preds from improved baseline (re-run minimal version)
def parse_ewkb_4d(hex_str):
    raw = bytes.fromhex(hex_str)
    offset = 0
    bo = '<' if raw[offset] == 1 else '>'
    offset += 1
    geom_type = struct.unpack_from(f'{bo}I', raw, offset)[0]
    offset += 4
    if geom_type & 0x20000000:
        offset += 4
    n_points = struct.unpack_from(f'{bo}I', raw, offset)[0]
    offset += 4
    points = []
    for _ in range(n_points):
        lon, lat, alt, rcs = struct.unpack_from(f'{bo}4d', raw, offset)
        points.append((lon, lat, alt, rcs))
        offset += 32
    return points

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(train['bird_group'])

print('='*60)
print('1. CLASS OVERLAP ANALYSIS')
print('='*60)

# Key discriminating features per class
print('\n--- Airspeed by class ---')
print(train.groupby('bird_group')['airspeed'].agg(['mean', 'std', 'median']).round(1).to_string())

print('\n--- Radar bird size crosstab ---')
ct = pd.crosstab(train['bird_group'], train['radar_bird_size'])
print(ct.to_string())

print('\n--- Min/Max Z by class ---')
for col in ['min_z', 'max_z']:
    print(f'\n{col}:')
    print(train.groupby('bird_group')[col].agg(['mean', 'std', 'median', 'min', 'max']).round(1).to_string())

# ============================================================
# 2. PIGEONS DEEP DIVE
# ============================================================
print('\n' + '='*60)
print('2. PIGEONS DEEP DIVE')
print('='*60)

pigeons = train[train.bird_group == 'Pigeons']
non_pigeons = train[train.bird_group != 'Pigeons']

print(f'\nPigeons: {len(pigeons)} samples')
print(f'\n--- Pigeon species breakdown ---')
print(pigeons.bird_species.value_counts().to_string())

print(f'\n--- Pigeon radar_bird_size ---')
print(pigeons.radar_bird_size.value_counts().to_string())

# Compare pigeons with classes they likely overlap with
for compare_class in ['Ducks', 'Songbirds', 'Gulls', 'Waders']:
    other = train[train.bird_group == compare_class]
    print(f'\n--- Pigeons vs {compare_class} ---')
    for feat in ['airspeed', 'min_z', 'max_z']:
        p_mean, p_std = pigeons[feat].mean(), pigeons[feat].std()
        o_mean, o_std = other[feat].mean(), other[feat].std()
        overlap = abs(p_mean - o_mean) / max((p_std + o_std) / 2, 0.01)
        print(f'  {feat:12s}: Pigeons={p_mean:.1f}+-{p_std:.1f}  {compare_class}={o_mean:.1f}+-{o_std:.1f}  separation={overlap:.2f}')

# ============================================================
# 3. OBSERVER COMMENTS ANALYSIS (train-only, but diagnostic)
# ============================================================
print('\n' + '='*60)
print('3. OBSERVER COMMENTS (DIAGNOSTIC)')
print('='*60)

for cls in WEAK:
    subset = train[train.bird_group == cls]
    comments = subset.observer_comment.dropna()
    if len(comments) > 0:
        print(f'\n--- {cls} comments ({len(comments)}/{len(subset)} have comments) ---')
        print(comments.value_counts().head(10).to_string())
    else:
        print(f'\n--- {cls}: no comments ---')

# ============================================================
# 4. N_BIRDS_OBSERVED patterns
# ============================================================
print('\n' + '='*60)
print('4. N_BIRDS_OBSERVED BY CLASS')
print('='*60)
print(train.groupby('bird_group')['n_birds_observed'].agg(['mean', 'std', 'median', 'min', 'max']).round(1).to_string())

# ============================================================
# 5. TEMPORAL PATTERNS
# ============================================================
print('\n' + '='*60)
print('5. TEMPORAL PATTERNS')
print('='*60)
train['ts'] = pd.to_datetime(train['timestamp_start_radar_utc'])
train['hour'] = train.ts.dt.hour
train['month'] = train.ts.dt.month

print('\n--- Hour distribution per class ---')
hour_ct = pd.crosstab(train['bird_group'], train['hour'], normalize='index').round(2)
# Show peak hours per class
for cls in CLASSES:
    row = hour_ct.loc[cls]
    top3 = row.nlargest(3)
    print(f'  {cls:15s}: peak hours = {", ".join([f"{h}:00({v:.0%})" for h,v in top3.items()])}')

print('\n--- Month distribution per class ---')
month_ct = pd.crosstab(train['bird_group'], train['month'], normalize='index').round(2)
for cls in CLASSES:
    row = month_ct.loc[cls]
    top3 = row.nlargest(3)
    print(f'  {cls:15s}: peak months = {", ".join([f"M{m}({v:.0%})" for m,v in top3.items()])}')

# ============================================================
# 6. TRAJECTORY CHARACTERISTICS
# ============================================================
print('\n' + '='*60)
print('6. TRAJECTORY STATS BY CLASS')
print('='*60)

traj_stats = []
for _, r in train.iterrows():
    pts = parse_ewkb_4d(r.trajectory)
    times = eval(r.trajectory_time)
    alts = [p[2] for p in pts]
    rcs_vals = [p[3] for p in pts]
    duration = times[-1] - times[0] if len(times) > 1 else 0
    traj_stats.append({
        'bird_group': r.bird_group,
        'n_points': len(pts),
        'duration': duration,
        'alt_mean': np.mean(alts),
        'rcs_mean': np.mean(rcs_vals),
        'rcs_std': np.std(rcs_vals),
    })
traj_df = pd.DataFrame(traj_stats)

for feat in ['n_points', 'duration', 'alt_mean', 'rcs_mean', 'rcs_std']:
    print(f'\n{feat}:')
    print(traj_df.groupby('bird_group')[feat].agg(['mean', 'std', 'median']).round(1).to_string())

# ============================================================
# 7. PRIMARY OBSERVATION - multiple tracks per bird?
# ============================================================
print('\n' + '='*60)
print('7. TRACKS PER OBSERVATION')
print('='*60)
tracks_per_obs = train.groupby('primary_observation_id').agg(
    n_tracks=('track_id', 'count'),
    bird_group=('bird_group', 'first')
).reset_index()
print(tracks_per_obs.groupby('bird_group')['n_tracks'].agg(['mean', 'median', 'max']).round(1).to_string())

# Check if same observation has multiple classes (data quality)
mixed = train.groupby('primary_observation_id')['bird_group'].nunique()
print(f'\nObservations with mixed classes: {(mixed > 1).sum()} / {len(mixed)}')

# ============================================================
# 8. CLUTTER ANALYSIS
# ============================================================
print('\n' + '='*60)
print('8. CLUTTER ANALYSIS')
print('='*60)
clutter = train[train.bird_group == 'Clutter']
print(f'Clutter species: {clutter.bird_species.value_counts().to_string()}')
print(f'\nClutter bird sizes: {clutter.radar_bird_size.value_counts().to_string()}')
print(f'\nClutter n_birds: {clutter.n_birds_observed.value_counts().head().to_string()}')
