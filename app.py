import joblib
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px


MODEL_PATH = "airbnb_pricing_model.pkl"
DATA_PATH = "airbnb.csv"   # <-- change if your dataset file name is different


deployment_package = joblib.load(MODEL_PATH)
model = deployment_package["model"]
engineered_features = deployment_package.get("features", [])
target_transform = str(deployment_package.get("target_transform", "")).strip().lower()


st.set_page_config(
    page_title="Airbnb Price Prediction",
    page_icon="🏡",
    layout="wide"
)

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
    "<div class='subtitle'>Predict an estimated nightly price using listing attributes.</div>",
    unsafe_allow_html=True
)


countries = [
    "United Kingdom", "France", "Italy", "Greece", "Turkey",
    "Morocco", "Japan", "India", "Thailand", "Georgia"
]

country_coords = {
    "United Kingdom": (55.3781, -3.4360),
    "France": (46.2276, 2.2137),
    "Italy": (41.8719, 12.5674),
    "Greece": (39.0742, 21.8243),
    "Turkey": (38.9637, 35.2433),
    "Morocco": (31.7917, -7.0926),
    "Japan": (36.2048, 138.2529),
    "India": (20.5937, 78.9629),
    "Thailand": (15.8700, 100.9925),
    "Georgia": (42.3154, 43.3569),
}


if "history" not in st.session_state:
    st.session_state.history = []


def build_pipeline_input(base_df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(engineered_features, (list, tuple)) and len(engineered_features) > 0:
        expected_cols = list(engineered_features)
    elif hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
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


def inverse_target_if_needed(pred_value: float) -> float:
    if target_transform == "log1p":
        return float(np.expm1(pred_value))
    return float(pred_value)


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def find_price_column(df: pd.DataFrame) -> str:
    candidates = ["price", "Price", "nightly_price", "nightlyPrice", "price_usd", "Price_USD", "listing_price"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "price" in str(c).lower():
            return c
    return ""


with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_B%C3%A9lo.svg", width=120)
    st.header("Listing Parameters")

    country_selected = st.selectbox("Market Location", countries, index=0)

    st.divider()
    guests_selected = st.slider("Guests", 1, 16, 2, 1)
    bedrooms_selected = st.slider("Bedrooms", 0, 10, 1, 1)
    bathrooms_selected = st.slider("Bathrooms", 0, 10, 1, 1)   # draggable + integer only
    beds_selected = st.slider("Beds", 0, 16, 1, 1)
    studios_selected = st.selectbox("Studios (0 = No, 1 = Yes)", [0, 1], index=0)

    st.divider()
    st.subheader("Output settings")
    round_to_nearest = st.selectbox("Round to nearest ($)", [1, 5, 10, 50, 100], index=2)
    clamp_min = st.number_input("Minimum clamp ($)", min_value=0, value=0, step=10)
    clamp_max = st.number_input("Maximum clamp ($)", min_value=100, value=10000, step=100)

    st.divider()
    show_debug = st.checkbox("Show model input (debug)", value=False)
    show_raw = st.checkbox("Show raw prediction (debug)", value=True)


toiles_selected = 1 if bathrooms_selected >= 1 else 0


tab_pred, tab_compare, tab_tail, tab_history, tab_export = st.tabs(
    ["Predict", "Model Comparison", "Long-Tail Analysis", "History", "Export"]
)


with tab_pred:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Listing inputs</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Fill in the listing details and predict price.</div>", unsafe_allow_html=True)

        issues = []
        if bedrooms_selected == 0 and studios_selected == 0:
            issues.append("Bedrooms and Studios are both 0. If it’s a studio listing, set Studios = 1.")
        if beds_selected > guests_selected + 2:
            issues.append("Beds is unusually high relative to guests. Double-check your input.")
        if guests_selected > 6 and (bedrooms_selected < 2 and studios_selected == 1):
            issues.append("Many guests with a studio is uncommon. Consider increasing bedrooms or reducing guests.")
        if beds_selected < max(1, guests_selected // 2):
            issues.append("Beds looks low relative to guests. Consider increasing beds.")

        if issues:
            st.warning("Sanity checks:")
            for msg in issues:
                st.write(f"- {msg}")

        st.write("")
        predict_btn = st.button("Predict Price", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Prediction</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>The model returns an estimated nightly price.</div>", unsafe_allow_html=True)

        if not predict_btn:
            st.info("Adjust inputs in the sidebar and click **Predict Price**.")

        if predict_btn:
            df_input = pd.DataFrame({
                "country": [country_selected],
                "guests": [int(guests_selected)],
                "bedrooms": [int(bedrooms_selected)],
                "bathrooms": [int(bathrooms_selected)],
                "beds": [int(beds_selected)],
                "studios": [int(studios_selected)],
                "toiles": [int(toiles_selected)]
            })

            df_ready = build_pipeline_input(df_input)

            try:
                raw_model_output = float(model.predict(df_ready)[0])
                pred_price = inverse_target_if_needed(raw_model_output)

                clipped = (np.isfinite(pred_price) and pred_price > float(clamp_max))
                y_pred = clamp_round_safe(pred_price, clamp_min, clamp_max, round_to_nearest)

                if show_raw:
                    if target_transform == "log1p":
                        st.caption(f"Debug: raw model output (log1p scale) = {raw_model_output:,.6f}")
                        st.caption(f"Debug: converted back to price (before clamp/round) = {pred_price:,.2f}")
                    else:
                        st.caption(f"Debug: raw model output (before clamp/round) = {raw_model_output:,.2f}")

                if not np.isfinite(y_pred):
                    st.error("Prediction returned an invalid value (NaN/Infinity). Try different inputs or check model training.")
                else:
                    st.success("Prediction completed")
                    st.metric("Predicted price (per night)", f"${y_pred:,.2f}")

                    if clipped:
                        st.warning(
                            f"Note: Raw prediction exceeded the maximum clamp, so output was capped at ${int(clamp_max):,}."
                        )

                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "country": country_selected,
                        "guests": int(guests_selected),
                        "bedrooms": int(bedrooms_selected),
                        "bathrooms": int(bathrooms_selected),
                        "beds": int(beds_selected),
                        "studios": int(studios_selected),
                        "toiles": int(toiles_selected),
                        "raw_model_output": float(raw_model_output),
                        "target_transform": target_transform if target_transform else "none",
                        "predicted_price": float(y_pred),
                        "cap_max": float(clamp_max),
                        "capped": "Yes" if clipped else "No",
                    })

                if show_debug:
                    with st.expander("Debug: final input sent into Pipeline"):
                        st.dataframe(df_ready, use_container_width=True)

            except Exception as e:
                st.error(
                    "Prediction failed. The model expects certain columns from training.\n"
                    "Please confirm your .pkl is the deployment package saved from the notebook."
                )
                st.code(str(e))

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Map visualisation</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Approximate country locations for visualisation. "
        "Shows selected country plus up to 20 recent predictions.</div>",
        unsafe_allow_html=True
    )

    map_rows = []
    lat, lon = country_coords[country_selected]
    map_rows.append({"lat": lat, "lon": lon, "label": f"Selected: {country_selected}"})

    if st.session_state.history:
        for h in st.session_state.history[-20:]:
            c = h["country"]
            lat_h, lon_h = country_coords.get(c, (np.nan, np.nan))
            if np.isfinite(lat_h) and np.isfinite(lon_h):
                map_rows.append({"lat": lat_h, "lon": lon_h, "label": f"{c} | ${h['predicted_price']:.0f}"})

    map_df = pd.DataFrame(map_rows)
    st.map(map_df[["lat", "lon"]], zoom=1)

    with st.expander("See map pin details"):
        st.dataframe(map_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


with tab_compare:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Final Startup Performance Report</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Compare model iterations using MAE (Lower is Better).</div>", unsafe_allow_html=True)

    perf_df = pd.DataFrame([
        {"Model Version": "Baseline (Linear Regression)", "MAE ($)": 8840.91, "Business Readiness": "Unreliable"},
        {"Model Version": "Baseline (Gradient Boosting)", "MAE ($)": 7508.38, "Business Readiness": "MVP Stage"},
        {"Model Version": "Final Iterative (Log-Transformed)", "MAE ($)": 6814.08, "Business Readiness": "Market Ready"},
    ])

    st.dataframe(perf_df, use_container_width=True)

    fig_mae = px.bar(
        perf_df,
        x="Model Version",
        y="MAE ($)",
        text="MAE ($)",
        title="MAE Comparison Across Model Versions"
    )
    fig_mae.update_traces(texttemplate="$%{text:,.2f}", textposition="outside")
    fig_mae.update_layout(yaxis_tickprefix="$", xaxis_title="", yaxis_title="MAE ($)")
    st.plotly_chart(fig_mae, use_container_width=True)

    error_reduction = 8840.91 - 6814.08
    gain_pct = (error_reduction / 8840.91) * 100

    st.success(f"Total Error Reduction (Baseline LR → Final): ${error_reduction:,.2f}  |  Overall Gain: {gain_pct:.2f}%")

    st.markdown(
        "- **Why Iteration 3 improved:** log-transform reduces the impact of extreme luxury listings (long-tail) on training.\n"
        "- **Business readiness:** final model reduces error enough to support more reliable pricing guidance."
    )

    st.markdown("</div>", unsafe_allow_html=True)


with tab_tail:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Long-Tail / Imbalance Analysis (Dataset Evidence)</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Shows why prices are capped high: a small number of luxury listings create extreme values.</div>",
        unsafe_allow_html=True
    )

    df_data = None
    load_error = None

    try:
        df_data = load_dataset(DATA_PATH)
    except Exception as e:
        load_error = str(e)

    if df_data is None:
        st.error(
            f"Could not load dataset at `{DATA_PATH}`.\n\n"
            "Fix: Put the CSV in the same folder as app.py OR update DATA_PATH.\n"
        )
        st.code(load_error)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        price_col = find_price_column(df_data)

        if price_col == "":
            st.error("Could not find a price column in your dataset. Rename your target column to `price` or include 'price' in the column name.")
            st.write("Columns found:", list(df_data.columns))
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            prices = pd.to_numeric(df_data[price_col], errors="coerce").dropna()
            prices = prices[prices >= 0]

            st.write(f"Detected price column: **{price_col}**")
            st.write(f"Rows with valid price: **{len(prices):,}**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Median", f"${prices.median():,.0f}")
            c2.metric("90th percentile", f"${prices.quantile(0.90):,.0f}")
            c3.metric("99th percentile", f"${prices.quantile(0.99):,.0f}")
            c4.metric("Max", f"${prices.max():,.0f}")

            pct_above_cap = (prices > float(clamp_max)).mean() * 100
            st.info(f"With your current cap (${int(clamp_max):,}), **{pct_above_cap:.2f}%** of listings exceed the cap in the dataset.")

            show_log = st.checkbox("Use log scale for x-axis to show long-tail", value=False)
            bins = st.slider("Histogram bins", 10, 120, 50, 5)

            plot_df = pd.DataFrame({"price": prices})

            if show_log:
                plot_df = plot_df[plot_df["price"] > 0]
                plot_df["log_price"] = np.log10(plot_df["price"])
                fig_hist = px.histogram(
                    plot_df,
                    x="log_price",
                    nbins=bins,
                    title="Price Distribution (log10 scale) — Long Tail Visible"
                )
                fig_hist.update_layout(xaxis_title="log10(price)", yaxis_title="Count")
            else:
                fig_hist = px.histogram(
                    plot_df,
                    x="price",
                    nbins=bins,
                    title="Price Distribution (raw scale) — Skewed / Long Tail"
                )
                fig_hist.update_layout(xaxis_title="Price", yaxis_title="Count")

            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("### Why Price is still capped high?")
            st.write(
                "- Airbnb pricing data is **imbalanced / long-tailed**: most listings are low-mid, but a small number are ultra-expensive.\n"
                "- Even after **log1p(price)**, rare combinations can still produce high predicted values.\n"
                "- The cap is a **business safeguard** to prevent unrealistic outputs during deployment.\n"
            )

    st.markdown("</div>", unsafe_allow_html=True)


with tab_history:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Prediction History</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>This table records your demo predictions.</div>", unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.write("No predictions yet. Generate a few predictions first.")

    st.markdown("</div>", unsafe_allow_html=True)


with tab_export:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Export</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Download your prediction history for evidence.</div>", unsafe_allow_html=True)

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
