"""
House Price Predictor — Streamlit app.

Loads the pipeline saved by the notebook (house_model.joblib) and serves
predictions through a simple web UI. The scaler lives inside the pipeline,
so user inputs are scaled automatically — no manual preprocessing here.

IMPORTANT: the column names in X_new below must match the names created by
the notebook's df.rename(...):  house_age, dist_to_train_station, num_con_store
"""

import streamlit as st
import pandas as pd
from joblib import load

# --- Load the trained model once ---
model = load("house_model.joblib")

# --- Page ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.write(
    "Estimates residential **price per unit area** from three property features. "
    "Demonstration model trained on the public UCI *Real Estate Valuation* dataset "
    "(New Taipei City) — illustrative of an end-to-end ML workflow, not a UK valuation tool."
)

# --- Inputs (units matter: distance is in METRES) ---
house_age = st.number_input(
    "House age (years)", min_value=0.0, max_value=100.0, value=15.0, step=0.5
)
dist_station = st.number_input(
    "Distance to nearest train station (metres)",
    min_value=0.0, max_value=7000.0, value=500.0, step=10.0,
)
num_con_store = st.number_input(
    "Number of nearby convenience stores", min_value=0, max_value=20, value=5, step=1
)

# --- Predict ---
if st.button("Predict price"):
    # Names here MUST match the model's training columns (from df.rename in the notebook):
    # house_age, dist_to_train_station, num_con_store
    X_new = pd.DataFrame([{
        "house_age": house_age,
        "dist_to_train_station": dist_station,
        "num_con_store": num_con_store,
    }])

    prediction = model.predict(X_new)[0]

    st.success(f"Predicted price per unit area: **{prediction:.2f}**")
    st.caption(
        "Units: 10,000 New Taiwan Dollars per Ping (~3.3 m²), per the source dataset."
    )
