import os
import joblib
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go


# This defines the app title, icon, and layout for deployment
st.set_page_config(
    page_title="Airbnb Price Prediction",
    page_icon="🏡",
    layout="wide"
)


# Path to the dataset used for computing global price statistics
# This dataset is not used for prediction, only for contextual visualisation
DATA_PATH = "airbnb.csv"

# Registry of model versions available for deployment
# Additional models can be added here without changing application logic
MODEL_FILES = {
    "Final Iterative (Log-Transformed)": "airbnb_pricing_model.pkl",
}


# Custom CSS is applied to improve visual clarity and user experience
# This ensures a professional, presentation-ready interface
st.markdown(
    """
    <style>
      .stApp {
        background:
          linear-gradient(rgba(15,15,15,0.55), rgba(15,15,15,0.75)),
          url("https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?auto=format&fit=crop&w=1600&q=80");
        background-size: cover;
        background-position: center;
      }

      .block-container {
        background: rgba(255,255,255,0.95);
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        margin-top: 1.2rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.14);
      }

      .card {
        background: rgba(255,255,255,0.92);
        border-radius: 14px;
        padding: 1.15rem 1.25rem;
        border: 1px solid rgba(0,0,0,0.06);
      }
    </style>
    """,
    unsafe_allow_html=True
)


# Main application title and subtitle for clarity
st.markdown("<h1>Airbnb AI Price Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<p>Interactive nightly price estimation using trained machine learning models.</p>",
    unsafe_allow_html=True
)


# Allowed country values are restricted to the countries present in the training data
countries = [
    "United Kingdom", "France", "Italy", "Greece", "Turkey",
    "Morocco", "Japan", "India", "Thailand", "Georgia"
]


# Attempts to automatically identify the price column in the dataset
# This allows flexibility if column names differ slightly
def find_price_column(df: pd.DataFrame) -> str:
    candidates = ["price", "Price", "nightly_price", "nightlyPrice", "price_usd", "Price_USD", "listing_price"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "price" in str(c).lower():
            return c
    return ""


# Loads dataset-level price statistics used for visual context
# Cached to prevent repeated disk reads during interaction
@st.cache_data
def load_dataset_prices(path: str) -> dict:
    df = pd.read_csv(path)
    price_col = find_price_column(df)
    if not price_col:
        raise ValueError("Could not detect a price column in dataset.")

    prices = pd.to_numeric(df[price_col], errors="coerce").dropna()
    prices = prices[prices >= 0]

    if len(prices) == 0:
        raise ValueError("No valid prices found in dataset.")

    return {
        "price_col": price_col,
        "min": float(prices.min()),
        "max": float(prices.max()),
        "median": float(prices.median()),
        "p90": float(prices.quantile(0.90)),
        "p99": float(prices.quantile(0.99)),
        "count": int(len(prices)),
    }


# Loads a serialized model or deployment package only once per session
# This improves responsiveness and avoids repeated file loading
@st.cache_resource
def load_any_model(file_path: str):
    return joblib.load(file_path)


# Filters the model registry to include only files that exist locally
def resolve_available_models() -> dict:
    available = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            available[name] = path
    return available


# Unpacks either a deployment package dictionary or a plain sklearn model
# This supports flexible export formats from the training notebook
def unpack_deployment_package(obj):
    if isinstance(obj, dict) and "model" in obj:
        model_obj = obj["model"]
        engineered_features = obj.get("features", [])
        target_transform = str(obj.get("target_transform", "")).strip().lower()
        return model_obj, engineered_features, target_transform

    return obj, [], ""


# Aligns user input to the exact feature schema expected by the trained model
# Missing features are filled safely to prevent runtime prediction errors
def build_pipeline_input(base_df: pd.DataFrame, model_obj, engineered_features) -> pd.DataFrame:
    if isinstance(engineered_features, (list, tuple)) and engineered_features:
        expected_cols = list(engineered_features)
    elif hasattr(model_obj, "feature_names_in_"):
        expected_cols = list(model_obj.feature_names_in_)
    else:
        return base_df

    df = base_df.copy()

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df.reindex(columns=expected_cols)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("")
        else:
            df[col] = df[col].fillna(0)

    return df


# Converts predictions back to original scale if log transformation was used
def inverse_target_if_needed(pred_value: float, target_transform: str) -> float:
    if target_transform == "log1p":
        return float(np.expm1(pred_value))
    return float(pred_value)


# Applies rounding and clamping to ensure deployment-safe outputs
def clamp_round_safe(value: float, min_v: float, max_v: float, nearest: int) -> float:
    v = float(value)
    if not np.isfinite(v):
        return float("nan")

    v = max(v, float(min_v))
    if max_v > 0:
        v = min(v, float(max_v))

    if nearest and nearest > 0:
        v = round(v / nearest) * nearest

    return float(v)


# Stores prediction history across user interactions
if "history" not in st.session_state:
    st.session_state.history = []


# Validates that at least one model is available before rendering the interface
available_models = resolve_available_models()
if not available_models:
    st.error("No model files found. Place your .pkl file in the same directory as app.py.")
    st.stop()


# Sidebar contains non-critical controls to avoid cluttering the main flow
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_B%C3%A9lo.svg", width=120)
    st.header("App Controls")

    selected_model_name = st.selectbox(
        "Model Version",
        list(available_models.keys()),
        index=0
    )

    show_debug = st.checkbox("Show model input (debug)", value=False)
    show_raw = st.checkbox("Show raw prediction (debug)", value=True)


# Main navigation separates prediction, history, and export for clarity
tab_pred, tab_history, tab_export = st.tabs(["Predict", "History", "Export"])


with tab_pred:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Listing parameters")

        country_selected = st.selectbox("Country", countries)
        guests_selected = st.slider("Guests", 1, 16, 2)
        bedrooms_selected = st.slider("Bedrooms", 0, 10, 1)
        bathrooms_selected = st.slider("Bathrooms (integer)", 0, 10, 1)
        beds_selected = st.slider("Beds", 0, 16, 1)
        studios_selected = st.selectbox("Studios (0 = No, 1 = Yes)", [0, 1])

        toiles_selected = 1 if bathrooms_selected >= 1 else 0

        issues = []
        if bedrooms_selected == 0 and studios_selected == 0:
            issues.append("Bedrooms and Studios are both 0. Select Studios = 1 for studio listings.")
        if beds_selected < max(1, guests_selected // 2):
            issues.append("Beds appear low relative to number of guests.")

        if issues:
            st.warning("Sanity checks:")
            for msg in issues:
                st.write(f"- {msg}")

        st.subheader("Output settings")
        round_to_nearest = st.selectbox("Round to nearest ($)", [1, 5, 10, 50, 100], index=2)
        clamp_min = st.number_input("Minimum clamp ($)", min_value=0, value=0)
        clamp_max = st.number_input("Maximum clamp ($)", min_value=100, value=10000)

        predict_btn = st.button("Predict Price", type="primary")

    with right:
        if predict_btn:
            model_path = available_models[selected_model_name]
            raw_loaded = load_any_model(model_path)
            model_obj, engineered_features, target_transform = unpack_deployment_package(raw_loaded)

            df_input = pd.DataFrame({
                "country": [country_selected],
                "guests": [guests_selected],
                "bedrooms": [bedrooms_selected],
                "bathrooms": [bathrooms_selected],
                "beds": [beds_selected],
                "studios": [studios_selected],
                "toiles": [toiles_selected]
            })

            df_ready = build_pipeline_input(df_input, model_obj, engineered_features)

            raw_pred = float(model_obj.predict(df_ready)[0])
            pred_price = inverse_target_if_needed(raw_pred, target_transform)
            y_pred = clamp_round_safe(pred_price, clamp_min, clamp_max, round_to_nearest)

            st.success(f"Predicted nightly price: ${y_pred:,.0f}")


with tab_history:
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history))
    else:
        st.write("No predictions recorded yet.")


with tab_export:
    if st.session_state.history:
        csv = pd.DataFrame(st.session_state.history).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prediction history",
            data=csv,
            file_name="airbnb_predictions.csv",
            mime="text/csv"
        )
    else:
        st.write("No data available for export.")
