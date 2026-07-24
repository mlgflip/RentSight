"""Train and evaluate the rent model with an honest baseline comparison.

- Time-based split: sort by sale_date, earliest 80% train / latest 20% test.
- Baseline A: per-zip median rent-per-sqft (from train only) x sqft.
- Baseline B: linear regression on sqft + beds + baths.
- Model: sklearn HistGradientBoostingRegressor (histogram-based gradient
  boosting, same family as LightGBM, chosen because LightGBM needs libomp on
  macOS and HistGB is dependency-free), tuned with RandomizedSearchCV (25
  iterations, TimeSeriesSplit CV) over learning_rate, max_depth, max_iter,
  min_samples_leaf, l2_regularization.
- All three evaluated on the same held-out test window: MAE, RMSE, R2.

Outputs: models/rent_model.joblib, models/metrics.json
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

RNG = 42
NUM = ["sqft", "beds", "baths", "age_at_sale", "lot_size", "is_condo",
       "dist_coast_km", "sale_month", "sale_year"]
CAT = ["zip"]

df = pd.read_csv("data/processed/listings_clean.csv",
                 dtype={"zip": str}, parse_dates=["sale_date"])
df = df.sort_values("sale_date").reset_index(drop=True)

cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]
cutoff_date = str(train["sale_date"].max().date())
print(f"train {len(train)} rows (.. {cutoff_date}) | test {len(test)} rows "
      f"({test['sale_date'].min().date()} ..)")

y_tr, y_te = train["est_rent"], test["est_rent"]


def report(name, pred):
    mae = mean_absolute_error(y_te, pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    r2 = r2_score(y_te, pred)
    print(f"{name:28s} MAE ${mae:8.0f}  RMSE ${rmse:8.0f}  R2 {r2:6.3f}")
    return {"MAE": round(mae, 1), "RMSE": round(rmse, 1), "R2": round(r2, 4)}


# Baseline A: zip median rent-per-sqft from train only
zip_rps = (train["est_rent"] / train["sqft"]).groupby(train["zip"]).median()
county_rps = (train["est_rent"] / train["sqft"]).median()
pred_a = test["zip"].map(zip_rps).fillna(county_rps).values * test["sqft"].values

# Baseline B: linear regression on sqft + beds + baths
lin = LinearRegression().fit(train[["sqft", "beds", "baths"]], y_tr)
pred_b = lin.predict(test[["sqft", "beds", "baths"]])

# Model: tuned histogram gradient boosting; zip ordinal-encoded as categorical
pre = ColumnTransformer([
    ("zip", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
    ("num", "passthrough", NUM),
])
gb = HistGradientBoostingRegressor(random_state=RNG, categorical_features=[0])
pipe = Pipeline([("pre", pre), ("gb", gb)])

search = RandomizedSearchCV(
    pipe,
    {
        "gb__learning_rate": loguniform(0.02, 0.3),
        "gb__max_iter": randint(150, 600),
        "gb__max_depth": randint(3, 12),
        "gb__min_samples_leaf": randint(10, 80),
        "gb__l2_regularization": loguniform(1e-3, 10),
    },
    n_iter=25, cv=TimeSeriesSplit(n_splits=3),
    scoring="neg_mean_absolute_error", random_state=RNG, n_jobs=-1, verbose=1,
)
search.fit(train[CAT + NUM], y_tr)
best = search.best_estimator_
print("best params:", {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in search.best_params_.items()})
pred_m = best.predict(test[CAT + NUM])

metrics = {
    "baseline_zip_median_ppsf": report("baseline: zip median $/sqft", pred_a),
    "baseline_linear": report("baseline: linear sqft+bd+ba", pred_b),
    "hist_gradient_boosting": report("HistGradientBoosting tuned", pred_m),
}

meta = {
    "cutoff_date": cutoff_date,
    "n_train": len(train), "n_test": len(test),
    "features": CAT + NUM,
    "best_params": {k: (round(v, 5) if isinstance(v, float) else int(v))
                    for k, v in search.best_params_.items()},
    "metrics_test": metrics,
    "label": "est_rent = sale_price x zip ZORI/ZHVI ratio (monthly) (derived, not observed rent)",
}
joblib.dump({"model": best, "meta": meta}, "models/rent_model.joblib")
with open("models/metrics.json", "w") as fh:
    json.dump(meta, fh, indent=2)
print("saved models/rent_model.joblib + models/metrics.json")
