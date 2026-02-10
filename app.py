import os
import joblib
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go


# Page configuration must be the first Streamlit command to avoid runtime errors.
st.set_page_config(
    page_title="Airbnb Price Prediction",
    page_icon="🏡",
    layout="wide"
)

# Dataset path used to compute min, max, and percentile statistics for the price range gauge.
DATA_PATH = "airbnb.csv"

# Model registry for this app. Add more entries here if you export additional model versions.
MODEL_FILES = {
    "Final Iterative (Log-Transformed)": "airbnb_pricing_model.pkl",
}

# App styling to keep the UI clean and consistent across sections.
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

      .title-big {
        font-size: 2.0rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
      }

      .subtitle {
        color: rgba(0,0,0,0.65);
        margin-bottom: 0.9rem;
      }

      .section-title {
        font-size: 1.05rem;
        font-weight: 650;
        margin-bottom: 0.3rem;
      }

      .section-subtitle {
        font-size: 0.9rem;
        color: rgba(0,0,0,0.62);
        margin-bottom: 0.9rem;
      }

      .stButton > button {
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 650;
      }

      div[data-baseweb="select"] > div {
        border-radius: 10px !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# App header for clarity and target-audience friendly wording.
st.markdown("<div class='title-big'>Airbnb AI Price Prediction</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Interactive nightly price estimation using trained machine learning models.</div>",
    unsafe_allow_html=True
)

# Supported country values for the country input field.
countries = [
    "United Kingdom", "France", "Italy", "Greece", "Turkey",
    "Morocco", "Japan", "India", "Thailand", "Georgia"
]


# Finds the most likely price column in the dataset based on common naming patterns.
def find_price_column(df: pd.DataFrame) -> str:
    candidates = ["price", "Price", "nightly_price", "nightlyPrice", "price_usd", "Price_USD", "listing_price"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "price" in str(c).lower():
            return c
    return ""


# Loads and summarizes dataset price statistics used for the gauge visualization.
# Cached to keep the app responsive when re-running.
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


# Loads models only once per session to reduce repeated disk reads.
@st.cache_resource
def load_any_model(file_path: str):
    return joblib.load(file_path)


# Filters the registry to only models that exist in the current folder.
def resolve_available_models() -> dict:
    available = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            available[name] = path
    return available


# Handles two saved formats:
# A deployment package dict with metadata, or a plain scikit-learn model/pipeline object.
def unpack_deployment_package(obj):
    """
    Supports:
    1) deployment_package dict with keys: model, features(optional), target_transform(optional)
    2) plain sklearn model/pipeline object
    """
    if isinstance(obj, dict) and "model" in obj:
        model_obj = obj["model"]
        engineered_features = obj.get("features", [])
        target_transform = str(obj.get("target_transform", "")).strip().lower()
        return model_obj, engineered_features, target_transform

    return obj, [], ""


# Aligns user inputs to the exact feature columns expected by the trained model.
# Missing columns are filled with default values to prevent prediction failures.
def build_pipeline_input(base_df: pd.DataFrame, model_obj, engineered_features) -> pd.DataFrame:
    if isinstance(engineered_features, (list, tuple)) and len(engineered_features) > 0:
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


# Converts prediction back to the original price scale if the model was trained on log1p(price).
def inverse_target_if_needed(pred_value: float, target_transform: str) -> float:
    if target_transform == "log1p":
        return float(np.expm1(pred_value))
    return float(pred_value)


# Applies deployment-friendly constraints so outputs remain stable and readable for users.
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


# Stores prediction history for interactive demo evidence and export.
if "history" not in st.session_state:
    st.session_state.history = []


# Validates that at least one model file is available before rendering the app.
available_models = resolve_available_models()
if not available_models:
    st.error("No model files found. Put your .pkl model file(s) in the same folder as app.py.")
    st.stop()


# Sidebar keeps non-critical controls and debug options without cluttering the main user flow.
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_B%C3%A9lo.svg", width=120)
    st.header("App Controls")

    selected_model_name = st.selectbox(
        "Model Version",
        list(available_models.keys()),
        index=0
    )

    st.divider()
    show_debug = st.checkbox("Show model input (debug)", value=False)
    show_raw = st.checkbox("Show raw prediction (debug)", value=True)


# Main navigation focuses on an interactive prediction workflow plus evidence capture.
tabs = st.tabs(["Predict", "History", "Export"])
tab_pred, tab_history, tab_export = tabs


with tab_pred:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Listing parameters</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Enter listing attributes, then predict.</div>", unsafe_allow_html=True)

        # Listing parameters are placed before output settings to match the user’s mental flow.
        c1, c2 = st.columns(2)
        with c1:
            country_selected = st.selectbox("Country", countries, index=0)
            guests_selected = st.slider("Guests", 1, 16, 2, 1)
            bedrooms_selected = st.slider("Bedrooms", 0, 10, 1, 1)

        with c2:
            bathrooms_selected = st.slider("Bathrooms (integer)", 0, 10, 1, 1)
            beds_selected = st.slider("Beds", 0, 16, 1, 1)
            studios_selected = st.selectbox("Studios (0 = No, 1 = Yes)", [0, 1], index=0)

        # Simple derived field to keep compatibility with the features used in the notebook.
        toiles_selected = 1 if bathrooms_selected >= 1 else 0

        # Lightweight input validation gives immediate feedback while keeping the app responsive.
        issues = []
        if bedrooms_selected == 0 and studios_selected == 0:
            issues.append("Bedrooms and Studios are both 0. If it’s a studio listing, set Studios = 1.")
        if beds_selected > guests_selected + 2:
            issues.append("Beds is unusually high relative to guests. Double-check your input.")
        if guests_selected > 6 and (bedrooms_selected < 2 and studios_selected == 1):
            issues.append("Many guests with a studio is uncommon. Consider more bedrooms or fewer guests.")
        if beds_selected < max(1, guests_selected // 2):
            issues.append("Beds looks low relative to guests. Consider increasing beds.")

        if issues:
            st.warning("Sanity checks:")
            for msg in issues:
                st.write(f"- {msg}")

        st.divider()

        st.markdown("<div class='section-title'>Output settings</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Business safeguards for deployment outputs.</div>", unsafe_allow_html=True)

        # Output controls are placed above the action button so users can adjust them before running prediction.
        round_to_nearest = st.selectbox("Round to nearest ($)", [1, 5, 10, 50, 100], index=2)
        clamp_min = st.number_input("Minimum clamp ($)", min_value=0, value=0, step=10)
        clamp_max = st.number_input("Maximum clamp ($)", min_value=100, value=10000, step=100)

        st.write("")
        predict_btn = st.button("Predict Price", type="primary")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Prediction</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Result is shown with dataset min–max context.</div>", unsafe_allow_html=True)

        if not predict_btn:
            st.info("Fill in the listing parameters and click Predict Price.")

        if predict_btn:
            # Model loading is triggered only when needed to keep the app responsive.
            model_path = available_models[selected_model_name]
            raw_loaded = load_any_model(model_path)
            model_obj, engineered_features, target_transform = unpack_deployment_package(raw_loaded)

            # User inputs are converted into a single-row DataFrame for scikit-learn prediction.
            df_input = pd.DataFrame({
                "country": [country_selected],
                "guests": [int(guests_selected)],
                "bedrooms": [int(bedrooms_selected)],
                "bathrooms": [int(bathrooms_selected)],
                "beds": [int(beds_selected)],
                "studios": [int(studios_selected)],
                "toiles": [int(toiles_selected)]
            })

            # The model may require additional columns from training; these are filled safely.
            df_ready = build_pipeline_input(df_input, model_obj, engineered_features)

            try:
                # Prediction is generated, then converted back if the training used log1p(price).
                raw_pred = float(model_obj.predict(df_ready)[0])
                pred_price = inverse_target_if_needed(raw_pred, target_transform)

                # Output constraints are applied for deployment safety and consistent UX.
                clipped = (np.isfinite(pred_price) and pred_price > float(clamp_max))
                y_pred = clamp_round_safe(pred_price, clamp_min, clamp_max, round_to_nearest)

                if show_raw:
                    if target_transform == "log1p":
                        st.caption(f"Debug: raw model output (log1p scale) = {raw_pred:,.6f}")
                        st.caption(f"Debug: converted back to price (before clamp/round) = {pred_price:,.2f}")
                    else:
                        st.caption(f"Debug: raw model output (before clamp/round) = {raw_pred:,.2f}")

                if not np.isfinite(y_pred):
                    st.error("Prediction returned an invalid value (NaN/Infinity). Try different inputs.")
                else:
                    st.success("Prediction completed")

                    # Gauge shows where the predicted price sits within the dataset range and highlights the long tail.
                    try:
                        stats = load_dataset_prices(DATA_PATH)
                        min_p = stats["min"]
                        max_p = stats["max"]
                        med_p = stats["median"]
                        p90 = stats["p90"]
                        p99 = stats["p99"]

                        pred_val = float(y_pred)
                        if np.isfinite(pred_val):
                            pred_val = max(min_p, min(pred_val, max_p))
                        else:
                            pred_val = med_p

                        fig_range = go.Figure()
                        fig_range.add_trace(go.Indicator(
                            mode="gauge+number",
                            value=pred_val,
                            number={"prefix": "$", "valueformat": ",.0f", "font": {"size": 34}},
                            title={"text": "Predicted Price within Dataset Range"},
                            gauge={
                                "axis": {"range": [min_p, max_p], "tickformat": ",.0f", "tickcolor": "#6b7280"},
                                "bar": {"color": "#FF385C"},
                                "steps": [
                                    {"range": [min_p, p90], "color": "#fde2e7"},
                                    {"range": [p90, p99], "color": "#fbb6c2"},
                                    {"range": [p99, max_p], "color": "#ef4444"},
                                ],
                                "threshold": {
                                    "line": {"color": "#7f1d1d", "width": 4},
                                    "thickness": 0.85,
                                    "value": pred_val
                                }
                            }
                        ))
                        fig_range.update_layout(
                            height=260,
                            margin=dict(t=40, b=10, l=10, r=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            font={"color": "#111827"}
                        )
                        st.plotly_chart(fig_range, use_container_width=True)

                        st.caption(
                            f"Dataset: min ${min_p:,.0f} | median ${med_p:,.0f} | 90th ${p90:,.0f} | "
                            f"99th ${p99:,.0f} | max ${max_p:,.0f}"
                        )
                    except Exception as e:
                        st.metric("Predicted price (per night)", f"${y_pred:,.2f}")
                        st.caption(f"Dataset range graph unavailable (dataset issue): {e}")

                    if clipped:
                        st.warning(f"Note: Raw prediction exceeded the maximum clamp, so output was capped at ${int(clamp_max):,}.")

                    # Each prediction is stored so you can demonstrate interactions and export results during evaluation.
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model_version": selected_model_name,
                        "country": country_selected,
                        "guests": int(guests_selected),
                        "bedrooms": int(bedrooms_selected),
                        "bathrooms": int(bathrooms_selected),
                        "beds": int(beds_selected),
                        "studios": int(studios_selected),
                        "toiles": int(toiles_selected),
                        "target_transform": target_transform if target_transform else "none",
                        "raw_model_output": float(raw_pred),
                        "predicted_price": float(y_pred),
                        "cap_max": float(clamp_max),
                        "capped": "Yes" if clipped else "No",
                    })

                # Debug view helps validate column alignment when deploying the notebook pipeline.
                if show_debug:
                    with st.expander("Debug: final input sent into Pipeline"):
                        st.dataframe(df_ready, use_container_width=True)

            except Exception as e:
                st.error(
                    "Prediction failed. This usually means your model expects different columns.\n"
                    "Fix: ensure your saved model is a Pipeline or the deployment package with correct features."
                )
                st.code(str(e))

        st.markdown("</div>", unsafe_allow_html=True)


with tab_history:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Prediction History</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Your past predictions for demo evidence.</div>", unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.write("No predictions yet. Generate a few predictions first.")

    st.markdown("</div>", unsafe_allow_html=True)


with tab_export:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Export</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Download prediction history (CSV).</div>", unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prediction history (CSV)",
            data=csv,
            file_name="airbnb_predictions.csv",
            mime="text/csv"
        )
    else:
        st.write("Generate predictions first, then you can export them here.")

    st.markdown("</div>", unsafe_allow_html=True)
