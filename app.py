import os
import joblib
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go


st.set_page_config(
    page_title="Airbnb Price Prediction",
    page_icon="🏡",
    layout="wide"
)


DATA_PATH = "airbnb.csv"


MODEL_FILES = {
    "Final Iterative (Log-Transformed)": "airbnb_pricing_model.pkl",
}


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


st.markdown("<div class='title-big'>Airbnb AI Price Prediction</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Interactive nightly price estimation using trained machine learning models.</div>",
    unsafe_allow_html=True
)


countries = [
    "United Kingdom", "France", "Italy", "Greece", "Turkey",
    "Morocco", "Japan", "India", "Thailand", "Georgia"
]


def find_price_column(df: pd.DataFrame) -> str:
    candidates = ["price", "Price", "nightly_price", "nightlyPrice", "price_usd", "Price_USD", "listing_price"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "price" in str(c).lower():
            return c
    return ""


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


@st.cache_resource
def load_any_model(file_path: str):
    return joblib.load(file_path)


def resolve_available_models() -> dict:
    available = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            available[name] = path
    return available


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


def inverse_target_if_needed(pred_value: float, target_transform: str) -> float:
    if target_transform == "log1p":
        return float(np.expm1(pred_value))
    return float(pred_value)


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


def apply_input_validation(
    guests: int,
    bedrooms: int,
    bathrooms: int,
    beds: int,
    studios: int
) -> dict:
    """
    Applies business-aware validation to prevent invalid or misleading predictions.
    The goal is to preserve user intent while avoiding inputs the model was never trained on.
    """

    notes = []

    # Hard constraint: Airbnb listings must have at least 1 guest
    if guests < 1:
        guests = 1
        notes.append("Guests cannot be below 1. Set to minimum value of 1.")

    # Hard constraint: Guests require at least one bed
    if guests >= 1 and beds < 1:
        beds = 1
        notes.append("Listings with guests must have at least 1 bed. Beds set to 1.")

    # Soft constraint: Bathrooms rarely zero in real listings
    if bathrooms < 1:
        bathrooms = 1
        notes.append(
            "Bathrooms set to 1. Listings with 0 bathrooms are extremely rare and not well represented in training data."
        )

    # Studio logic
    if bedrooms == 0 and studios == 0:
        studios = 1
        notes.append(
            "Bedrooms is 0. Treated as a studio listing by setting Studios = 1."
        )

    if bedrooms > 0 and studios == 1:
        studios = 0
        notes.append(
            "Studios set to 0 because Bedrooms > 0 (not a studio listing)."
        )

    # Capacity realism check (warning only)
    if guests > beds * 2:
        notes.append(
            "Guest count is high relative to beds. Prediction may be less reliable."
        )

    return {
        "guests": int(guests),
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "beds": int(beds),
        "studios": int(studios),
        "notes": notes
    }

if "history" not in st.session_state:
    st.session_state.history = []


available_models = resolve_available_models()
if not available_models:
    st.error("No model files found. Put your .pkl model file(s) in the same folder as app.py.")
    st.stop()


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

    st.caption("Validation is enabled to prevent unrealistic input combinations.")


tabs = st.tabs(["Predict", "History", "Export"])
tab_pred, tab_history, tab_export = tabs


with tab_pred:
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Listing parameters</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Enter listing attributes, then predict.</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            country_selected = st.selectbox("Country", countries, index=0)
            guests_selected = st.slider("Guests", 1, 16, 2, 1)
            bedrooms_selected = st.slider("Bedrooms", 0, 10, 1, 1)

        with c2:
            bathrooms_selected = st.slider("Bathrooms (integer)", 1, 10, 1, 1)
            beds_selected = st.slider("Beds", 1, 16, 1, 1)

            studios_disabled = (bedrooms_selected == 0)
            default_studio = 1 if studios_disabled else 0

            studios_selected = st.selectbox(
                "Studios (0 = No, 1 = Yes)",
                [0, 1],
                index=1 if default_studio == 1 else 0,
                disabled=studios_disabled
            )

            if studios_disabled:
                st.caption("Studios is locked to 1 when Bedrooms is 0.")

        st.divider()

        st.markdown("<div class='section-title'>Output settings</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Business safeguards for deployment outputs.</div>", unsafe_allow_html=True)

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
            validated = apply_input_validation(
                guests=int(guests_selected),
                bedrooms=int(bedrooms_selected),
                bathrooms=int(bathrooms_selected),
                beds=int(beds_selected),
                studios=int(studios_selected)
            )

            if validated["notes"]:
                st.warning("Input validation applied:")
                for note in validated["notes"]:
                    st.write(f"- {note}")

            toiles_selected = 1 if validated["bathrooms"] >= 1 else 0

            model_path = available_models[selected_model_name]
            raw_loaded = load_any_model(model_path)
            model_obj, engineered_features, target_transform = unpack_deployment_package(raw_loaded)

            df_input = pd.DataFrame({
                "country": [country_selected],
                "guests": [validated["guests"]],
                "bedrooms": [validated["bedrooms"]],
                "bathrooms": [validated["bathrooms"]],
                "beds": [validated["beds"]],
                "studios": [validated["studios"]],
                "toiles": [int(toiles_selected)]
            })

            df_ready = build_pipeline_input(df_input, model_obj, engineered_features)

            try:
                raw_pred = float(model_obj.predict(df_ready)[0])
                pred_price = inverse_target_if_needed(raw_pred, target_transform)

                if show_raw:
                    if target_transform == "log1p":
                        st.caption(f"Debug: raw model output (log1p scale) = {raw_pred:,.6f}")
                        st.caption(f"Debug: converted back to price (before guardrails/clamp/round) = {pred_price:,.2f}")
                    else:
                        st.caption(f"Debug: raw model output (before guardrails/clamp/round) = {raw_pred:,.2f}")

                if not np.isfinite(pred_price):
                    st.error("Prediction returned an invalid value (NaN/Infinity). Try different inputs.")
                    st.stop()

                stats = None
                try:
                    stats = load_dataset_prices(DATA_PATH)
                except Exception:
                    stats = None

                guardrail_notes = []

                final_price = float(pred_price)

                bedrooms_v = int(validated["bedrooms"])
                guests_v = int(validated["guests"])
                beds_v = int(validated["beds"])
                studios_v = int(validated["studios"])
                bathrooms_v = int(validated["bathrooms"])

                if guests_v >= 1 and beds_v == 0:
                    st.error("Invalid input: Guests cannot be >= 1 when Beds = 0. Please increase Beds.")
                    st.stop()

                if bedrooms_v == 0 and studios_v == 0:
                    st.error("Invalid input: If Bedrooms = 0, set Studios = 1 for a studio listing.")
                    st.stop()

                if bathrooms_v == 0:
                    guardrail_notes.append("Bathrooms is 0. Many real listings have at least 1 bathroom. This may reduce accuracy.")

                budget_profile = (guests_v <= 2 and bedrooms_v == 0 and studios_v == 1)
                small_profile = (guests_v <= 2 and bedrooms_v <= 1)

                if budget_profile:
                    if stats is not None:
                        segment_cap = max(120.0, min(250.0, stats["p90"]))
                        final_price = min(final_price, segment_cap)
                        guardrail_notes.append(f"Budget listing cap applied (<=2 guests, studio). Capped at ${segment_cap:,.0f}.")
                    else:
                        final_price = min(final_price, 200.0)
                        guardrail_notes.append("Budget listing cap applied (<=2 guests, studio). Capped at $200.")
                elif small_profile:
                    if stats is not None:
                        segment_cap = max(180.0, min(450.0, stats["p90"]))
                        final_price = min(final_price, segment_cap)
                        guardrail_notes.append(f"Small listing cap applied (<=2 guests, <=1 bedroom). Capped at ${segment_cap:,.0f}.")
                    else:
                        final_price = min(final_price, 300.0)
                        guardrail_notes.append("Small listing cap applied (<=2 guests, <=1 bedroom). Capped at $300.")

                clipped = (np.isfinite(final_price) and final_price > float(clamp_max))
                y_pred = clamp_round_safe(final_price, clamp_min, clamp_max, round_to_nearest)

                if not np.isfinite(y_pred):
                    st.error("Final output after guardrails/clamp is invalid. Try different inputs.")
                    st.stop()

                st.success("Prediction completed")

                if guardrail_notes:
                    st.info("Business guardrails applied:")
                    for n in guardrail_notes:
                        st.write(f"- {n}")

                try:
                    if stats is not None:
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
                    else:
                        st.metric("Predicted price (per night)", f"${y_pred:,.2f}")
                except Exception as e:
                    st.metric("Predicted price (per night)", f"${y_pred:,.2f}")
                    st.caption(f"Dataset range graph unavailable (dataset issue): {e}")

                if clipped:
                    st.warning(f"Note: Output exceeded the maximum clamp, so it was capped at ${int(clamp_max):,}.")

                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_version": selected_model_name,
                    "country": country_selected,
                    "guests": validated["guests"],
                    "bedrooms": validated["bedrooms"],
                    "bathrooms": validated["bathrooms"],
                    "beds": validated["beds"],
                    "studios": validated["studios"],
                    "toiles": int(toiles_selected),
                    "target_transform": target_transform if target_transform else "none",
                    "raw_model_output": float(raw_pred),
                    "raw_price_before_guardrails": float(pred_price),
                    "final_price_before_round_clamp": float(final_price),
                    "predicted_price": float(y_pred),
                    "cap_max": float(clamp_max),
                    "capped": "Yes" if clipped else "No",
                })

                if show_debug:
                    with st.expander("Debug: final input sent into Pipeline"):
                        st.dataframe(df_ready, use_container_width=True)

            except Exception as e:
                st.error(
                    "Prediction failed. This usually means your model expects different columns.\n"
                    "Fix: ensure your saved model is a Pipeline or a deployment package with correct features."
                )
                st.code(str(e))

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
