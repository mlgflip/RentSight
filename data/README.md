# Data sources

Pull date for everything below: **2026-07-24**.

## 1. Miami-Dade County Property Appraiser parcels + sales (training rows)

- Source: Miami-Dade County ArcGIS REST service (public, no login), layer
  `MDC.PaGis` — `https://gisweb.miamidade.gov/arcgis/rest/services/MD_ComparableSales/MapServer/5`
  (the same Property Appraiser data exposed on the county Open Data Hub,
  gis-mdc.opendata.arcgis.com).
- Query: single-family (`DOR_CODE_CUR='0101'`) or condo (`CONDO_FLAG='Y'`),
  most-recent sale `PRICE_1 > $25,000`, `BEDROOM_COUNT > 0`,
  `BUILDING_HEATED_AREA > 200`, sale date `DOS_1 >= 2020-01-01`.
  Geometry returned as WGS84 lat/lon. Script: `notebooks/01_pull_mdc_sales.py`.
- Raw file: `data/raw/mdc_sales_raw.csv` — **151,709 rows**.

## 2. Zillow Research (benchmark + rent-ratio source)

- ZORI (Zillow Observed Rent Index, zip level, smoothed, all homes+multifamily):
  `https://files.zillowstatic.com/research/public_csvs/zori/Zip_zori_uc_sfrcondomfr_sm_month.csv`
- ZHVI (Zillow Home Value Index, zip level, mid-tier, smoothed, seasonally adj.):
  `https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv`
- Both linked from zillow.com/research/data. Filtered to Miami-Dade County:
  75 zips with ZORI, 78 with ZHVI. Script: `notebooks/02_zillow_mdc.py` →
  `data/processed/zillow_mdc.csv`.

## Cleaning (`notebooks/03_clean_features.py`)

| step | rows |
|---|---|
| raw pull | 151,709 |
| drop missing sqft/beds/baths/sale date/zip/coords | 151,641 |
| sqft outside [200, 10,000] or price-per-sqft outside 1st–99th pct ($50–$4,064) | **148,487** |

Sale dates span 2020-01-01 to 2026-07-07 across 77 zips (57% condo).

## Rent label — derived, stated openly

The county file records **sales, not rents**. The training label is:

```
est_rent ($/mo) = sale_price × (zip-level ZORI ÷ zip-level ZHVI)
```

i.e. each sale is converted to an estimated monthly rent using the observed
rent-to-value ratio of its zip code (county-median ratio 0.00534 used for the
3 zips lacking ZORI, flagged `ratio_imputed`). The model therefore predicts
*Zillow-ratio-implied rent*, not observed lease amounts — stated in the app
and write-up.

## Features (`data/processed/listings_clean.csv`)

sqft (heated area), beds, baths (full + 0.5×half), age at sale, lot size,
condo flag, zip, sale month/year, and `dist_coast_km` — a rough distance to a
hand-drawn 18-vertex coastline polyline covering the Atlantic barrier islands
(Golden Beach → Cape Florida) and the mainland Biscayne Bay shore, computed
with an equirectangular approximation. Coarse by design.
