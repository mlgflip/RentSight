"""Filter Zillow ZORI/ZHVI zip-level files to Miami-Dade County and build a
per-zip benchmark table: latest ZORI (rent), latest ZHVI (value), and the
rent-to-value ratio used to derive rent labels from sale prices.

Output: data/processed/zillow_mdc.csv
"""
import pandas as pd

def latest_value(df, id_col="RegionName"):
    month_cols = [c for c in df.columns if c[:2] == "20"]
    month_cols.sort()
    out = df[[id_col, "City", "CountyName"]].copy()
    # use last non-null across the final 6 months to dodge missing latest values
    tail = df[month_cols[-6:]]
    out["value"] = tail.ffill(axis=1).iloc[:, -1]
    out["as_of"] = month_cols[-1]
    return out

zori = pd.read_csv("data/raw/zori_zip.csv", dtype={"RegionName": str})
zhvi = pd.read_csv("data/raw/zhvi_zip.csv", dtype={"RegionName": str})

zori_mdc = zori[zori["CountyName"] == "Miami-Dade County"]
zhvi_mdc = zhvi[zhvi["CountyName"] == "Miami-Dade County"]

r = latest_value(zori_mdc).rename(columns={"value": "zori_rent", "as_of": "zori_as_of"})
v = latest_value(zhvi_mdc).rename(columns={"value": "zhvi_value", "as_of": "zhvi_as_of"})

m = r.merge(v[["RegionName", "zhvi_value", "zhvi_as_of"]], on="RegionName", how="outer")
m["rent_to_value_ratio"] = m["zori_rent"] / m["zhvi_value"]
m = m.rename(columns={"RegionName": "zip"})

# county-median ratio as fallback for zips with ZHVI but no ZORI
county_ratio = m["rent_to_value_ratio"].median()
m["ratio_imputed"] = m["rent_to_value_ratio"].isna()
m["rent_to_value_ratio"] = m["rent_to_value_ratio"].fillna(county_ratio)

m.to_csv("data/processed/zillow_mdc.csv", index=False)
print(f"zips: {len(m)} | with ZORI: {(~m['ratio_imputed']).sum()} | county median ratio: {county_ratio:.5f}")
print(m[m['zip'] == '33149'][['zip', 'City', 'zori_rent', 'zhvi_value', 'rent_to_value_ratio']].to_string(index=False))
