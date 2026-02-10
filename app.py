

import numpy as np
import pandas as pd
import joblib
import streamlit as st


st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
:root { --border: rgba(0,0,0,.08); --muted: rgba(0,0,0,.55); }
.small-muted { color: var(--muted); font-size: 0.92rem; line-height: 1.35; }
.card { border: 1px solid var(--border); border-radius: 16px; padding: 18px; background: #fff; }
.metric { border: 1px solid var(--border); border-radius: 16px; padding: 18px; background: #fff; }
.metric .label { color: var(--muted); font-size: 0.9rem; margin-bottom: 8px; }
.metric .value { font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; }
.hr { height: 1px; background: rgba(0,0,0,.06); margin: 16px 0; }
.badge {
  display: inline-block; padding: 6px 10px; border-radius: 999px;
  border: 1px solid var(--border); color: var(--muted); font-size: 0.85rem;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_package(model_path: Path) -> Tuple[Any, List[str], str]:
    obj = joblib.load(model_path)

    if not isinstance(obj, dict) or "model" not in obj:
        raise ValueError("Invalid artifact format. Expected a dict with keys: 'model', 'features', 'target_transform'.")

    model = obj["model"]
    features = obj.get("features")
    target_transform = str(obj.get("target_transform", "")).strip().lower()

    if not features or not isinstance(features, (list, tuple)):
        raise ValueError("Artifact missing 'features' list/tuple.")

    return model, list(features), target_transform


def money(x: float, currency: str = "USD") -> str:
    x = float(x)
    if not np.isfinite(x):
        return "—"
    return f"{currency} {max(0.0, x):,.2f}"


def build_input_df(features: List[str], values: Dict[str, Any]) -> pd.DataFrame:
    row = {}
    for f in features:
        row[f] = values.get(f)
    return pd.DataFrame([row], columns=features)


def guess_field_ui(feature_name: str) -> str:
    f = feature_name.lower()
    if f in {"country", "location", "city", "region"}:
        return "text"
    if any(k in f for k in ["guest", "bedroom", "bathroom", "bed", "studio", "toilet", "toile", "room"]):
        return "int"
    return "auto"


MODEL_PATH = Path("airbnb_pricing_model.pkl")

st.title("Airbnb Price Predictor")
st.markdown(
    "<div class='small-muted'>Estimate a nightly listing price using your trained Gradient Boosting pipeline.</div>",
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        "Model artifact not found.\n\n"
        "Expected file: `airbnb_pricing_model.pkl` in the same folder as this app.\n"
        "Fix: upload/copy your pickle into the Streamlit app directory (or repo root)."
    )
    st.stop()

try:
    model, features, target_transform = load_package(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model artifact: {e}")
    st.stop()

with st.sidebar:
    st.markdown("### Inputs")
    st.markdown("<div class='small-muted'>Fields are driven directly by the saved feature list.</div>", unsafe_allow_html=True)

    user_values: Dict[str, Any] = {}

    for feat in features:
        ui_type = guess_field_ui(feat)

        if ui_type == "text":
            default = "Singapore" if feat.lower() == "country" else ""
            user_values[feat] = st.text_input(feat.replace("_", " ").title(), value=default)
        elif ui_type == "int":
            user_values[feat] = st.number_input(
                feat.replace("_", " ").title(),
                min_value=0,
                max_value=50,
                value=1 if feat.lower() in {"guests", "beds", "bedrooms", "bathrooms", "toiles", "toilets"} else 0,
                step=1,
            )
        else:
            user_values[feat] = st.text_input(feat.replace("_", " ").title(), value="")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    currency = st.selectbox("Display currency", options=["USD", "SGD", "EUR", "GBP"], index=0)
    show_input_preview = st.checkbox("Show input preview", value=True)
    show_debug = st.checkbox("Debug: show raw output", value=False)
    predict_btn = st.button("Predict price", type="primary", use_container_width=True)

input_df = build_input_df(features, user_values)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Input Summary")
    st.markdown(
        "<div class='small-muted'>A single-row DataFrame is passed into your saved pipeline (includes preprocessing).</div>",
        unsafe_allow_html=True,
    )

    if show_input_preview:
        st.dataframe(input_df, use_container_width=True, hide_index=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown(
        "<span class='badge'>Deployment note</span> "
        "<span class='small-muted'>Your artifact declares <b>target_transform</b> = "
        f"<b>{target_transform or 'none'}</b>. The app will inverse-transform if needed.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='metric'>", unsafe_allow_html=True)
    st.markdown("<div class='label'>Predicted nightly price</div>", unsafe_allow_html=True)

    if "pred_price" not in st.session_state:
        st.markdown("<div class='value'>—</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='small-muted'>Enter inputs on the left and click <b>Predict price</b>.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='value'>{money(st.session_state['pred_price'], currency=currency)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='small-muted'>This is an estimate based on patterns learned from your training data.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if show_debug and "raw_pred" in st.session_state:
        st.caption(f"Raw model output: {st.session_state['raw_pred']}")


if predict_btn:
    try:
        raw_pred = float(model.predict(input_df)[0])

        if target_transform == "log1p":
            pred_price = float(np.expm1(raw_pred))
        else:
            pred_price = raw_pred

        if not np.isfinite(pred_price):
            raise ValueError("Prediction is not a finite number.")

        pred_price = max(0.0, pred_price)

        st.session_state["raw_pred"] = raw_pred
        st.session_state["pred_price"] = pred_price

        st.toast("Prediction generated.", icon="✅")
        st.rerun()

    except Exception as e:
        st.error(
            "Prediction failed.\n\n"
            "Common causes:\n"
            "- Feature mismatch between app inputs and training\n"
            "- Different scikit-learn version from when the model was saved\n"
            "- Unexpected values in categorical fields (e.g., country naming)\n\n"
            f"Error: {e}"
        )
