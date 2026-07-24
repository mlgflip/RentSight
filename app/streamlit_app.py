"""PropOS — Miami-Dade rent estimator (Key Biscayne + greater Miami-Dade zips)."""
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="PropOS Rent Estimator", page_icon="🏝️", layout="centered")


@st.cache_resource
def load_model():
    bundle = joblib.load(ROOT / "models" / "rent_model.joblib")
    return bundle["model"], bundle["meta"]


@st.cache_data
def load_zip_stats():
    return pd.read_csv(ROOT / "models" / "zip_stats.csv", dtype={"zip": str})


model, meta = load_model()
zips = load_zip_stats()
mae = meta["metrics_test"]["hist_gradient_boosting"]["MAE"]
r2 = meta["metrics_test"]["hist_gradient_boosting"]["R2"]

st.title("🏝️ PropOS Rent Estimator")
st.caption("Monthly rent prediction for Key Biscayne and greater Miami-Dade zip codes")

with st.form("inputs"):
    c1, c2 = st.columns(2)
    with c1:
        zip_options = zips.sort_values("zip")
        zip_label = {r.zip: f"{r.zip} — {str(r.city).title()}" for r in zip_options.itertuples()}
        zip_code = st.selectbox(
            "Zip code", zip_options["zip"],
            index=int((zip_options["zip"] == "33149").idxmax()) if "33149" in set(zip_options["zip"]) else 0,
            format_func=lambda z: zip_label.get(z, z),
        )
        sqft = st.number_input("Living area (sqft)", min_value=200, max_value=10_000, value=1_200, step=50)
        beds = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
        baths = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    with c2:
        year_built = st.number_input("Year built", min_value=1900, max_value=date.today().year, value=2000)
        lot_size = st.number_input("Lot size (sqft, 0 for condo)", min_value=0, max_value=200_000, value=0, step=500)
        is_condo = st.toggle("Condo", value=True)
    submitted = st.form_submit_button("Estimate rent", type="primary", use_container_width=True)

if submitted:
    errors = []
    if sqft / max(beds, 1) < 100:
        errors.append("That's less than 100 sqft per bedroom — check sqft and bedrooms.")
    if baths > beds + 3:
        errors.append("Bathrooms far exceed bedrooms — check inputs.")
    if not is_condo and lot_size == 0:
        errors.append("Houses need a lot size (only condos may use 0).")
    if errors:
        for e in errors:
            st.error(e)
    else:
        zrow = zips[zips["zip"] == zip_code].iloc[0]
        today = date.today()
        row = pd.DataFrame([{
            "zip": zip_code,
            "sqft": float(sqft),
            "beds": float(beds),
            "baths": float(baths),
            "age_at_sale": max(0, today.year - year_built),
            "lot_size": float(lot_size) if lot_size > 0 else None,
            "is_condo": int(is_condo),
            "dist_coast_km": float(zrow["med_dist_coast_km"]),
            "sale_month": today.month,
            "sale_year": today.year,
        }])
        pred = float(model.predict(row)[0])

        st.divider()
        st.metric("Predicted monthly rent", f"${pred:,.0f}",
                  help="Point estimate from the gradient-boosting model")
        lo, hi = pred - mae, pred + mae
        st.write(f"**Likely range: ${max(lo, 0):,.0f} – ${hi:,.0f}** "
                 f"(± test-set mean absolute error of ${mae:,.0f})")
        st.caption(
            f"Zip context: {zrow['n_sales']:,} sales in training data, "
            f"median derived rent ${zrow['med_rent']:,.0f}/mo, "
            f"median size {zrow['med_sqft']:,.0f} sqft."
        )

st.divider()
with st.expander("Model details & metrics"):
    m = meta["metrics_test"]
    st.table(pd.DataFrame({
        "Model": ["Baseline: zip median $/sqft", "Baseline: linear (sqft+bd+ba)",
                  "Gradient boosting (deployed)"],
        "MAE ($/mo)": [m["baseline_zip_median_ppsf"]["MAE"], m["baseline_linear"]["MAE"],
                       m["hist_gradient_boosting"]["MAE"]],
        "RMSE ($/mo)": [m["baseline_zip_median_ppsf"]["RMSE"], m["baseline_linear"]["RMSE"],
                        m["hist_gradient_boosting"]["RMSE"]],
        "R²": [m["baseline_zip_median_ppsf"]["R2"], m["baseline_linear"]["R2"],
               m["hist_gradient_boosting"]["R2"]],
    }))
    st.write(
        f"Time-based evaluation: trained on {meta['n_train']:,} Miami-Dade sales through "
        f"**{meta['cutoff_date']}**, tested on the {meta['n_test']:,} most recent sales after it."
    )

st.warning(
    "**Disclaimer — read before using.** Predictions estimate *Zillow-ratio-implied* rent: "
    "the training label is sale price × the zip's ZORI/ZHVI rent-to-value ratio, **not observed "
    "lease amounts** (Miami-Dade public records do not include rents). Data: Miami-Dade County "
    "Property Appraiser sales (2020–2026, single-family & condo) and Zillow Research indices, "
    "pulled 2026-07-24. Single metro, zip-level ratios, no unit-level condition/amenity data. "
    "Not an appraisal; do not use as the sole basis for pricing decisions.",
    icon="⚠️",
)
