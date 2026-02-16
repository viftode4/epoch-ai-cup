"""Fetch real bird observation data from GBIF for Eemshaven area.

Queries GBIF occurrence API for bird observations within ~10km of Eemshaven (53.44N, 6.83E).
Maps species to the 9 competition classes and computes monthly class distributions.
Saves the results as a CSV for use in models.

No authentication needed for GBIF occurrence search API.
"""
import urllib.request
import json
import time
import csv
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

# Eemshaven bounding box (~10km radius)
LAT_MIN, LAT_MAX = 53.38, 53.50
LON_MIN, LON_MAX = 6.75, 6.91

# ======================================================================
# Species-to-class mapping
# ======================================================================
# Map GBIF species/genus/family to competition classes.
# We use taxonomic family as primary key, with species overrides.

# Family-level mapping (covers most cases)
FAMILY_TO_CLASS = {
    # Birds of Prey (Accipitriformes, Falconiformes)
    "Accipitridae": "Birds of Prey",      # hawks, eagles, harriers, buzzards
    "Pandionidae": "Birds of Prey",        # osprey
    "Falconidae": "Birds of Prey",         # falcons, kestrels
    # Cormorants
    "Phalacrocoracidae": "Cormorants",
    # Ducks (Anseriformes - ducks only, not geese)
    # We split Anatidae into ducks vs geese below
    # Geese
    # (handled via genus-level split of Anatidae)
    # Gulls (Laridae + Sternidae)
    "Laridae": "Gulls",                    # gulls
    "Sternidae": "Gulls",                  # terns (grouped with gulls in competition)
    "Stercorariidae": "Gulls",             # skuas (gull-like)
    # Pigeons
    "Columbidae": "Pigeons",
    # Songbirds (Passeriformes - all passerine families)
    "Passeridae": "Songbirds",
    "Fringillidae": "Songbirds",
    "Motacillidae": "Songbirds",
    "Sylviidae": "Songbirds",
    "Turdidae": "Songbirds",
    "Muscicapidae": "Songbirds",
    "Paridae": "Songbirds",
    "Corvidae": "Songbirds",
    "Hirundinidae": "Songbirds",
    "Alaudidae": "Songbirds",
    "Emberizidae": "Songbirds",
    "Sturnidae": "Songbirds",
    "Regulidae": "Songbirds",
    "Troglodytidae": "Songbirds",
    "Certhiidae": "Songbirds",
    "Sittidae": "Songbirds",
    "Laniidae": "Songbirds",
    "Oriolidae": "Songbirds",
    "Prunellidae": "Songbirds",
    "Bombycillidae": "Songbirds",
    "Phylloscopidae": "Songbirds",
    "Acrocephalidae": "Songbirds",
    "Locustellidae": "Songbirds",
    "Cisticolidae": "Songbirds",
    "Aegithalidae": "Songbirds",
    "Calcariidae": "Songbirds",
    "Cardinalidae": "Songbirds",
    "Icteridae": "Songbirds",
    "Parulidae": "Songbirds",
    "Passerellidae": "Songbirds",
    "Vireonidae": "Songbirds",
    # Waders (Charadriiformes - shorebirds)
    "Scolopacidae": "Waders",              # sandpipers, godwits, curlews
    "Charadriidae": "Waders",              # plovers
    "Haematopodidae": "Waders",            # oystercatchers
    "Recurvirostridae": "Waders",          # avocets, stilts
    "Burhinidae": "Waders",                # thick-knees
}

# Genus-level mapping for Anatidae (split ducks vs geese)
GOOSE_GENERA = {
    "Anser", "Branta", "Chen",             # true geese
}
DUCK_GENERA = {
    "Anas", "Aythya", "Bucephala", "Clangula", "Mareca", "Mergus",
    "Mergellus", "Melanitta", "Netta", "Oxyura", "Somateria", "Spatula",
    "Tadorna",                              # shelduck
}
# Swans are closer to geese in radar signature (large, flock)
SWAN_GENERA = {"Cygnus"}

# Order-level fallback for passerines we might miss
PASSERINE_ORDER = "Passeriformes"


def classify_species(record):
    """Map a GBIF occurrence record to one of 9 competition classes."""
    family = record.get("family", "")
    genus = record.get("genus", "")
    order = record.get("order", "")

    # Anatidae: split into ducks vs geese
    if family == "Anatidae":
        if genus in GOOSE_GENERA:
            return "Geese"
        if genus in SWAN_GENERA:
            return "Geese"  # swans ~ geese on radar
        if genus in DUCK_GENERA:
            return "Ducks"
        # Unknown Anatidae -> Ducks (more common than geese)
        return "Ducks"

    # Family-level mapping
    if family in FAMILY_TO_CLASS:
        return FAMILY_TO_CLASS[family]

    # Order-level fallback for passerines
    if order == PASSERINE_ORDER:
        return "Songbirds"

    # Everything else we can't map -> skip (not one of the 9 classes)
    return None


# ======================================================================
# Fetch GBIF data
# ======================================================================
def fetch_gbif_month(month, year_start=2015, year_end=2025, limit=300):
    """Fetch bird occurrences for a specific month from GBIF."""
    base = "https://api.gbif.org/v1/occurrence/search"
    all_records = []
    offset = 0

    while True:
        params = (
            f"taxonKey=212"
            f"&decimalLatitude={LAT_MIN},{LAT_MAX}"
            f"&decimalLongitude={LON_MIN},{LON_MAX}"
            f"&month={month}"
            f"&year={year_start},{year_end}"
            f"&hasCoordinate=true"
            f"&limit={limit}"
            f"&offset={offset}"
        )
        url = f"{base}?{params}"

        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/json",
                "User-Agent": "EpochAICup/1.0 (bird-classification-research)"
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  Error fetching month {month} offset {offset}: {e}", flush=True)
            break

        results = data.get("results", [])
        all_records.extend(results)

        if data.get("endOfRecords", True) or len(results) == 0:
            break

        offset += limit
        # Cap at 5000 records per month to be respectful
        if offset >= 5000:
            break

        time.sleep(0.3)  # rate limiting

    return all_records, data.get("count", 0)


# ======================================================================
# Main
# ======================================================================
print("=" * 60, flush=True)
print("GBIF BIRD DATA FOR EEMSHAVEN", flush=True)
print(f"  Bounding box: ({LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E)", flush=True)
print(f"  Years: 2015-2025", flush=True)
print("=" * 60, flush=True)

CLASSES = [
    "Birds of Prey", "Clutter", "Cormorants", "Ducks", "Geese",
    "Gulls", "Pigeons", "Songbirds", "Waders",
]

# month -> class -> count
monthly_counts = {m: defaultdict(int) for m in range(1, 13)}
monthly_total_gbif = {}
unmapped_families = defaultdict(int)
unmapped_orders = defaultdict(int)

for month in range(1, 13):
    print(f"\nFetching month {month:2d}...", end=" ", flush=True)
    records, total = fetch_gbif_month(month)
    monthly_total_gbif[month] = total
    print(f"  {total} total, fetched {len(records)}", flush=True)

    for r in records:
        cls = classify_species(r)
        if cls is not None:
            monthly_counts[month][cls] += 1
        else:
            fam = r.get("family", "Unknown")
            ord_ = r.get("order", "Unknown")
            unmapped_families[fam] += 1
            unmapped_orders[ord_] += 1

    time.sleep(0.5)  # be nice to GBIF

# ======================================================================
# Print results
# ======================================================================
print("\n" + "=" * 60, flush=True)
print("MONTHLY CLASS DISTRIBUTION (GBIF observations)", flush=True)
print("=" * 60, flush=True)

# Header
header = f"{'Month':>5s} {'Total':>6s}"
for cls in CLASSES:
    header += f" {cls[:5]:>6s}"
print(header, flush=True)

monthly_priors = {}
for month in range(1, 13):
    counts = monthly_counts[month]
    total = sum(counts.values())
    row = f"{month:5d} {total:6d}"
    priors = []
    for cls in CLASSES:
        c = counts.get(cls, 0)
        row += f" {c:6d}"
        priors.append(c)
    print(row, flush=True)

    # Compute normalized prior (with Laplace smoothing)
    priors_arr = [counts.get(cls, 0) for cls in CLASSES]
    total_smooth = sum(priors_arr) + len(CLASSES) * 0.5
    monthly_priors[month] = [(p + 0.5) / total_smooth for p in priors_arr]

# Print normalized priors
print(f"\n{'':>5s} {'':>6s}", end="", flush=True)
for cls in CLASSES:
    print(f" {cls[:5]:>6s}", end="")
print(flush=True)

print("\nNormalized priors (with Laplace smoothing):", flush=True)
for month in range(1, 13):
    row = f"{month:5d}       "
    for p in monthly_priors[month]:
        row += f" {p:6.3f}"
    tag = ""
    if month in [2, 5, 12]:
        tag = " <-- UNSEEN test month"
    elif month in [9, 10]:
        tag = " <-- shared month"
    elif month in [1, 4]:
        tag = " <-- train only"
    print(row + tag, flush=True)

# Show unmapped
print(f"\nUnmapped families (top 10):", flush=True)
for fam, cnt in sorted(unmapped_families.items(), key=lambda x: -x[1])[:10]:
    print(f"  {fam}: {cnt}", flush=True)

# ======================================================================
# Save to CSV
# ======================================================================
out_path = ROOT / "data" / "gbif_monthly_priors.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["month"] + CLASSES)
    for month in range(1, 13):
        writer.writerow([month] + [f"{p:.6f}" for p in monthly_priors[month]])

print(f"\nSaved: {out_path}", flush=True)

# Also save raw counts
out_path2 = ROOT / "data" / "gbif_monthly_counts.csv"
with open(out_path2, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["month", "total_gbif"] + CLASSES)
    for month in range(1, 13):
        counts = monthly_counts[month]
        total = sum(counts.values())
        writer.writerow([month, total] + [counts.get(cls, 0) for cls in CLASSES])

print(f"Saved: {out_path2}", flush=True)

# Compare GBIF priors vs E35 ecological guesses
print("\n" + "=" * 60, flush=True)
print("GBIF vs E35 ECOLOGICAL GUESSES (unseen months)", flush=True)
print("=" * 60, flush=True)

E35_ECO = {
    2:  [0.05, 0.05, 0.05, 0.08, 0.10, 0.50, 0.04, 0.05, 0.08],
    5:  [0.10, 0.05, 0.04, 0.03, 0.01, 0.45, 0.04, 0.18, 0.10],
    12: [0.05, 0.05, 0.04, 0.10, 0.12, 0.45, 0.04, 0.05, 0.10],
}

for month in [2, 5, 12]:
    print(f"\n  Month {month}:", flush=True)
    print(f"  {'Class':<15s} {'E35 guess':>10s} {'GBIF data':>10s} {'Delta':>8s}", flush=True)
    eco = E35_ECO[month]
    # Normalize E35 eco
    eco_sum = sum(eco)
    eco_norm = [e / eco_sum for e in eco]
    for i, cls in enumerate(CLASSES):
        e = eco_norm[i]
        g = monthly_priors[month][i]
        d = g - e
        print(f"  {cls:<15s} {e:10.3f} {g:10.3f} {d:>+8.3f}", flush=True)

print("\nDone!", flush=True)
