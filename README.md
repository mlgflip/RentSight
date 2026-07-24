# PropOS — Miami-Dade Rent Estimator

> Given a residential property's square footage, bedrooms, bathrooms, age, lot size, location (zip code and distance to coastline), this model predicts its expected monthly rent in Key Biscayne and greater Miami-Dade County zip codes, in current US dollars.

A Streamlit app backed by a gradient-boosting model trained on 148,487 Miami-Dade
County property sales (2020–2026) with rent labels derived from Zillow Research
rent-to-value ratios.

## Results (held-out time-based test set)

Trained on the earliest 80% of sales (through **2025-02-05**), tested on the most
recent 20% (29,698 sales, Feb 2025 – Jul 2026). Same test set for all rows below.

| Model | MAE ($/mo) | RMSE ($/mo) | R² |
|---|---|---|---|
| Baseline: zip median $/sqft | 2,231 | 6,604 | 0.331 |
| Baseline: linear (sqft+beds+baths) | 2,708 | 6,352 | 0.381 |
| **HistGradientBoosting (tuned, deployed)** | **1,151** | **3,016** | **0.860** |

Tuned with `RandomizedSearchCV` (25 iterations, `TimeSeriesSplit` CV) over
`learning_rate`, `max_iter`, `max_depth`, `min_samples_leaf`, `l2_regularization`.
`HistGradientBoostingRegressor` was chosen over LightGBM because it is the same
histogram-based algorithm family with zero extra native dependencies (LightGBM
requires libomp on macOS).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app/streamlit_app.py
```

To rebuild data and model from scratch (in order):

```bash
.venv/bin/python notebooks/01_pull_mdc_sales.py   # county sales pull (~5 min)
.venv/bin/python notebooks/02_zillow_mdc.py       # Zillow benchmark table
.venv/bin/python notebooks/03_clean_features.py   # cleaning + features
.venv/bin/python notebooks/04_train_model.py      # baselines + tuned model
```

## Data

See [data/README.md](data/README.md) for exact source URLs, pull date
(2026-07-24), query filters, and row counts before/after cleaning.

- **Miami-Dade County Property Appraiser** (ArcGIS REST, public): parcel
  attributes + most-recent sale for single-family homes and condos.
- **Zillow Research**: ZORI (rent index) and ZHVI (home value index) at zip
  level — used both as an external benchmark and to derive the rent label.

### The rent label is derived, not observed

Public county records contain sales, not lease amounts. The training label is
`est_rent = sale_price × (zip ZORI ÷ zip ZHVI)` — each sale converted to an
implied monthly rent via its zip's observed rent-to-value ratio. The model
therefore predicts *Zillow-ratio-implied rent*. This is stated in the app UI
and the write-up; treat predictions accordingly.

## Known limitations

- **Derived labels** (above) — the biggest one. Accuracy vs. true lease prices
  is unmeasured; the test MAE measures fit to the derived label.
- Single metro (Miami-Dade only); 76 zips with ≥30 sales.
- Zip-level rent ratio ignores within-zip variation (waterfront vs. inland).
- No condition, renovation, amenity, HOA, or view data.
- Distance-to-coast uses a coarse 18-vertex hand-drawn coastline.
- Sale price ≥ $25k filter keeps some non-arm's-length transfers out but not all.

## Repo layout

```
app/streamlit_app.py      # Streamlit UI (loads serialized model)
notebooks/01..04_*.py     # data pull → benchmark → clean → train
models/                   # rent_model.joblib, metrics.json, zip_stats.csv
data/raw|processed/       # inputs (see data/README.md)
docs/                     # scope.md, one-page write-up
```
