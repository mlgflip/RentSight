# PropOS: Rent Prediction for Key Biscayne & Miami-Dade

**Felipe · July 2026 · one-page project summary**

## Problem statement

Given a residential property's square footage, bedrooms, bathrooms, age, lot size, location (zip code and distance to coastline), this model predicts its expected monthly rent in Key Biscayne and greater Miami-Dade County zip codes, in current US dollars.

## Data

Two free, public, citable sources (both pulled 2026-07-24):

1. **Miami-Dade County Property Appraiser** (county ArcGIS open-data service): 151,709 single-family and condo parcels with a sale ≥ $25,000 since Jan 2020 — address zip, heated sqft, beds, baths, year built, lot size, coordinates, sale price and date. **148,487 rows after cleaning** (dropped missing sqft/beds/sale date; kept sqft 200–10,000 and price-per-sqft within the 1st–99th percentile, $50–$4,064).
2. **Zillow Research**: zip-level ZORI (observed rent index) and ZHVI (home value index) for 75–78 Miami-Dade zips — used as an external benchmark and to derive labels.

**Derived label, stated openly:** county records contain sales, not rents. Each sale is converted to an estimated monthly rent: `est_rent = sale_price × (zip ZORI ÷ zip ZHVI)`. The model predicts Zillow-ratio-implied rent, not observed lease amounts.

## Methodology

- **Features:** sqft, beds, baths, age at sale, lot size, condo flag, zip (categorical), sale month/year, and rough distance-to-coastline computed from lat/lon against a hand-drawn 18-vertex Miami-Dade shoreline — a genuine differentiator for a Key Biscayne-aware model.
- **Time-based split (no leakage):** sorted by sale date; trained on the earliest 80% (through **2025-02-05**), tested on the most recent 20% (29,698 sales, Feb 2025 – Jul 2026).
- **Model:** scikit-learn `HistGradientBoostingRegressor` (histogram-based gradient boosting, same family as LightGBM, chosen for zero native dependencies), tuned with `RandomizedSearchCV` — 25 iterations, time-series CV — over learning rate, iterations, depth, leaf size, and L2.

## Results — baseline vs. model, identical test set

| Model | MAE ($/mo) | RMSE ($/mo) | R² |
|---|---|---|---|
| Baseline: zip median $/sqft | 2,231 | 6,604 | 0.331 |
| Baseline: linear (sqft+beds+baths) | 2,708 | 6,352 | 0.381 |
| **Gradient boosting (deployed)** | **1,151** | **3,016** | **0.860** |

The tuned model roughly halves the baseline's error and beats it decisively on all three metrics.

## Limitations (honest list)

- **Labels are derived**, not observed leases; test error measures fit to the derived label, and accuracy against true rents is unmeasured until externally validated.
- Single metro; zip-level ratios ignore within-zip variation (waterfront vs. inland).
- No condition, renovation, amenity, HOA, or view data; coarse coastline approximation.
- Some non-arm's-length transfers may survive the $25k price filter.

## Ask

Sanity-check the app's predictions against a handful of properties you know personally — units you've rented, listed, or managed — and flag where it's off (zip, size, direction, and rough magnitude). That feedback is the next training signal.
