import pandas as pd
gbif = pd.read_csv("data/gbif_monthly_counts.csv")
print(gbif[gbif["month"].isin([2, 5, 9, 10, 12])][["month", "Pigeons", "Cormorants", "Waders"]])
