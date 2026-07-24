"""Clean the raw Miami-Dade sales pull and build the modeling table.

- Drops rows missing sqft, beds, or sale date; filters sqft to [200, 10000];
  trims price-per-sqft to the 1st-99th percentile.
- Derives the rent label openly: est_rent = sale_price * zip rent-to-value ratio
  (ZORI/ZHVI for that zip). The county file has no rental amounts, so the label
  is a stated derivation, not observed rent.
- Features: sqft, beds, baths, age at sale, lot size, condo flag, zip,
  rough distance-to-coastline (km) from lat/lon, sale month/year.

Output: data/processed/listings_clean.csv
"""
import numpy as np
import pandas as pd

RAW = "data/raw/mdc_sales_raw.csv"
ZILLOW = "data/processed/zillow_mdc.csv"
OUT = "data/processed/listings_clean.csv"

# Rough Miami-Dade coastline polyline (lat, lon): Atlantic barrier islands from
# Golden Beach to Cape Florida, then the mainland Biscayne Bay shore south to
# Homestead. Coarse by design; documented in data/README.md.
COAST = [
    (25.965, -80.121), (25.940, -80.120), (25.900, -80.120), (25.870, -80.120),
    (25.810, -80.125), (25.770, -80.130), (25.760, -80.140), (25.690, -80.155),
    (25.665, -80.156), (25.850, -80.160), (25.800, -80.180), (25.760, -80.190),
    (25.710, -80.240), (25.660, -80.260), (25.610, -80.300), (25.550, -80.320),
    (25.470, -80.330), (25.400, -80.320),
]

def dist_to_coast_km(lat, lon):
    """Min distance (km) from point to any coast segment, equirectangular approx."""
    kx = 111.32 * np.cos(np.radians(25.7))  # km per degree lon at ~25.7N
    ky = 110.57                              # km per degree lat
    px, py = lon * kx, lat * ky
    best = np.full(len(lat), np.inf)
    pts = [(la * ky, lo * kx) for la, lo in COAST]
    for (ay, ax), (by, bx) in zip(pts[:-1], pts[1:]):
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / seg2, 0, 1)
        d = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
        best = np.minimum(best, d)
    return best

df = pd.read_csv(RAW, dtype={"TRUE_SITE_ZIP_CODE": str, "DOS_1": str})
n0 = len(df)

df["zip"] = df["TRUE_SITE_ZIP_CODE"].str[:5]
df["sale_date"] = pd.to_datetime(df["DOS_1"], format="%Y%m%d", errors="coerce")
df["sqft"] = df["BUILDING_HEATED_AREA"]
df["beds"] = df["BEDROOM_COUNT"]
df["baths"] = df["BATHROOM_COUNT"].fillna(0) + 0.5 * df["HALF_BATHROOM_COUNT"].fillna(0)
df["price"] = df["PRICE_1"]

# required fields
df = df.dropna(subset=["sqft", "beds", "sale_date", "price", "zip", "LON", "LAT"])
df = df[(df["beds"] > 0) & (df["baths"] > 0)]
n_required = len(df)

# outliers
df = df[(df["sqft"] >= 200) & (df["sqft"] <= 10000)]
df["ppsf"] = df["price"] / df["sqft"]
lo, hi = df["ppsf"].quantile([0.01, 0.99])
df = df[(df["ppsf"] >= lo) & (df["ppsf"] <= hi)]
n_outliers = len(df)

# features
df["sale_year"] = df["sale_date"].dt.year
df["sale_month"] = df["sale_date"].dt.month
yb = df["YEAR_BUILT"].where(df["YEAR_BUILT"] >= 1900)
df["age_at_sale"] = (df["sale_year"] - yb).clip(lower=0)
df["lot_size"] = df["LOT_SIZE"].where(df["LOT_SIZE"] > 0)
df["is_condo"] = (df["CONDO_FLAG"] == "Y").astype(int)
df["dist_coast_km"] = dist_to_coast_km(df["LAT"].values, df["LON"].values)

# derived rent label (stated openly): sale price x zip-level ZORI/ZHVI ratio
z = pd.read_csv(ZILLOW, dtype={"zip": str})
df = df.merge(z[["zip", "rent_to_value_ratio", "ratio_imputed", "zori_rent"]], on="zip", how="left")
county_ratio = z["rent_to_value_ratio"].median()
df["ratio_imputed"] = df["ratio_imputed"].fillna(True)
df["rent_to_value_ratio"] = df["rent_to_value_ratio"].fillna(county_ratio)
df["est_rent"] = (df["price"] * df["rent_to_value_ratio"]).round(0)

keep = ["FOLIO", "zip", "TRUE_SITE_CITY", "sale_date", "sale_year", "sale_month",
        "price", "ppsf", "est_rent", "rent_to_value_ratio", "ratio_imputed",
        "sqft", "beds", "baths", "age_at_sale", "lot_size", "is_condo",
        "dist_coast_km", "LAT", "LON", "DOR_DESC"]
df[keep].to_csv(OUT, index=False)

print(f"raw rows:            {n0}")
print(f"after required-field: {n_required}")
print(f"after outlier trim:   {n_outliers}  (ppsf kept in [{lo:.0f}, {hi:.0f}] $/sqft)")
print(f"zips: {df['zip'].nunique()} | condo share: {df['is_condo'].mean():.2f}")
print(f"sale dates: {df['sale_date'].min().date()} .. {df['sale_date'].max().date()}")
print(f"est_rent quartiles: {df['est_rent'].quantile([0.25, 0.5, 0.75]).tolist()}")
print(df[df['zip'] == '33149'][['est_rent', 'sqft', 'beds']].describe().loc[['count', 'mean']].to_string())
