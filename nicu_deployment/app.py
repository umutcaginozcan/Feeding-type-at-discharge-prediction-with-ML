import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import shap

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="NICU Feeding Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .main-header {
        background: linear-gradient(135deg, #0A2540 0%, #1E3A5F 50%, #006B7D 100%);
        padding: 2rem 2.5rem; border-radius: 16px; color: white;
        margin-bottom: 1.5rem; font-family: 'Inter', sans-serif;
    }
    .main-header h1 { margin: 0; font-weight: 700; font-size: 1.6rem; }
    .main-header p  { margin: 0.5rem 0 0 0; opacity: 0.85; font-size: 1rem; }
    .metric-item .label {
        font-size: 0.65rem; opacity: 0.7; text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-item .value {
        font-size: 1.1rem; font-weight: 600;
        font-family: 'Courier New', monospace;
    }
    .prediction-box {
        border-radius: 12px; padding: 2rem; text-align: center; margin: 1rem 0;
    }
    .prediction-box.ebf { background: #ECFDF5; border: 2px solid #059669; }
    .prediction-box.formula { background: #FEF2F2; border: 2px solid #DC2626; }
    .prediction-box.mixed { background: #EFF6FF; border: 2px solid #2563EB; }
    .stButton>button {
        background: linear-gradient(135deg, #0A2540 0%, #006B7D 100%);
        color: white; font-weight: 600; border: none;
        padding: 0.75rem 2rem; border-radius: 8px; font-size: 1rem;
    }
    .stButton>button:hover { opacity: 0.9; }
    .field-warning {
        background: #FEF2F2;
        border-left: 4px solid #DC2626;
        padding: 0.5rem 0.8rem;
        border-radius: 0 6px 6px 0;
        margin: 0.3rem 0 0.8rem 0;
        font-size: 0.82rem;
        color: #991B1B;
        font-family: 'Inter', sans-serif;
    }
    .impute-box {
        background: #FFFBEB;
        border-left: 4px solid #D97706;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #92400E;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ====================

FORMULA_CLASS_IDX = 1
CLASS_LABELS = ["Exclusive Breastfeeding", "Formula Feeding", "Mixed Feeding"]
CLASS_COLORS = ["#059669", "#DC2626", "#2563EB"]
APP_DIR = Path(__file__).parent

MODEL_FILES = {
    "Baseline (Admission)": "baseline_model.pkl",
    "Day 1 (0–24h)": "day1_model.pkl",
    "Day 1+2 (0–48h)": "day1_2_model.pkl",
    "Full (0–72h)": "full_model.pkl",
}

WINDOW_META = {
    "Baseline (Admission)": {"icon": "🏥", "short": "Baseline",
        "desc": "Admission data only",
        "detail": "Birth weight, gestational age, maternal factors"},
    "Day 1 (0–24h)": {"icon": "🍼", "short": "Day 1",
        "desc": "First 24 hours",
        "detail": "+ Day 1 feeding volumes & breastfeeding status"},
    "Day 1+2 (0–48h)": {"icon": "📊", "short": "Day 1+2",
        "desc": "First 48 hours",
        "detail": "+ Day 2 volumes & intake trajectory",
        "recommended": True},
    "Full (0–72h)": {"icon": "🔬", "short": "Full",
        "desc": "First 72 hours",
        "detail": "Most complete — highest accuracy"},
}

EXAMPLE_PATIENT = {
    "birth_weight": 2350,
    "ga_weeks": 35.0,
    "ga_days": 4,
    "weight_followup": 2280,
    "mat_age": 29,
    "bf_education": "Yes",
    "bfhi_status": "Yes",
    "d1_formula": 12.0,
    "d1_bm": 8.0,
    "d1_bm_flag": "Yes",
    "d1_bf_flag": "Yes",
    "weight_d1": 2310,
    "d2_bm": 18.0,
    "d2_formula": 22.0,
    "weight_d2": 2290,
    "d3_bm": 28.0,
    "d3_formula": 15.0,
    "weight_d3": 2300,
    "d3_route": "BF+PO",
}

# Human-readable names for pipeline features
FEATURE_DISPLAY_NAMES = {
    "anneyasi": "Maternal Age",
    "dogumagirligi(gram)": "Birth Weight (g)",
    "gebelikhaftası": "Gestational Age (weeks)",
    "gebelikhaftagunu": "Gestational Days",
    "takipilkgün_kilo_gram": "Follow-up Weight (g)",
    "eng_weight_per_week": "Weight per GA Week ★",
    "annesutuemzirmeeğitimidurumu": "Breastfeeding Education",
    "bebek_dostu_20temmuz2018": "BFHI Status",
    "aldığıannesütü_ilkgün": "Day 1 Breast Milk (cc)",
    "aldığımamamiktari1.gün": "Day 1 Formula (cc)",
    "kilo1.gun": "Day 1 Weight (g)",
    "ilk_gün_anne_sütü_1111": "Day 1 BM Given (flag)",
    "ilk_gün_emzirme_111": "Day 1 Breastfeeding (flag)",
    "eng_bm_ratio_d1": "Day 1 BM Ratio ★",
    "beslenmetotali2.gün": "Day 2 Total Intake (cc)",
    "beslenme2.gunannesutucc": "Day 2 Breast Milk (cc)",
    "beslenmemamamiktarı2.guncc": "Day 2 Formula (cc)",
    "kilo2.gun": "Day 2 Weight (g)",
    "eng_bm_ratio_d2": "Day 2 BM Ratio ★",
    "eng_delta_vol_d1_d2": "Volume Change D1→D2 ★",
    "beslenmetotali3.gun": "Day 3 Total Intake (cc)",
    "aldıgıannesütü3.gun": "Day 3 Breast Milk (cc)",
    "aldıgımamamiktari3.gun": "Day 3 Formula (cc)",
    "kilo3.gun": "Day 3 Weight (g)",
    "verilisyolu3gun": "Day 3 Feeding Route",
    "eng_bm_ratio_d3": "Day 3 BM Ratio ★",
    "eng_delta_vol_d2_d3": "Volume Change D2→D3 ★",
    "eng_lactation_momentum": "Lactation Momentum ★",
    "eng_resilience_index": "Resilience Index ★",
}

# ==================== MODEL LOADING ====================

@st.cache_resource
def load_all_models():
    bundles = {}
    for name, filename in MODEL_FILES.items():
        path = APP_DIR / filename
        if path.exists():
            with open(path, "rb") as f:
                bundles[name] = pickle.load(f)
    return bundles

ALL_BUNDLES = load_all_models()

# ==================== HELPERS ====================

def safe_val(v, default=np.nan):
    return default if v is None else float(v)


def compute_engineered(raw):
    for k in list(raw.keys()):
        raw[k] = safe_val(raw[k])
    eps = 1e-6
    bw = raw.get("dogumagirligi(gram)", np.nan)
    gw = raw.get("gebelikhaftası", np.nan)
    raw["eng_weight_per_week"] = (
        bw / (gw + eps) if not np.isnan(bw) and not np.isnan(gw) else np.nan)
    d1_bm = raw.get("aldığıannesütü_ilkgün", np.nan)
    d1_fm = raw.get("aldığımamamiktari1.gün", np.nan)
    raw["eng_bm_ratio_d1"] = (
        d1_bm / (d1_bm + d1_fm + eps)
        if not np.isnan(d1_bm) and not np.isnan(d1_fm) else np.nan)
    d2_bm = raw.get("beslenme2.gunannesutucc", np.nan)
    d2_total = raw.get("beslenmetotali2.gün", np.nan)
    raw["eng_bm_ratio_d2"] = (
        d2_bm / (d2_total + eps)
        if not np.isnan(d2_bm) and not np.isnan(d2_total) else np.nan)
    d1_total = ((d1_bm if not np.isnan(d1_bm) else 0)
                + (d1_fm if not np.isnan(d1_fm) else 0))
    raw["eng_delta_vol_d1_d2"] = (
        d2_total - d1_total if not np.isnan(d2_total) else np.nan)
    d3_bm = raw.get("aldıgıannesütü3.gun", np.nan)
    d3_fm = raw.get("aldıgımamamiktari3.gun", np.nan)
    d3_total_val = raw.get("beslenmetotali3.gun", np.nan)
    raw["eng_bm_ratio_d3"] = (
        d3_bm / (d3_bm + d3_fm + eps)
        if not np.isnan(d3_bm) and not np.isnan(d3_fm) else np.nan)
    raw["eng_delta_vol_d2_d3"] = (
        d3_total_val - d2_total
        if not np.isnan(d3_total_val) and not np.isnan(d2_total) else np.nan)
    raw["eng_lactation_momentum"] = (
        raw.get("eng_bm_ratio_d3", np.nan) - raw.get("eng_bm_ratio_d1", np.nan)
        if not np.isnan(raw.get("eng_bm_ratio_d3", np.nan))
        and not np.isnan(raw.get("eng_bm_ratio_d1", np.nan)) else np.nan)
    raw["eng_resilience_index"] = (
        d3_total_val / (bw + eps)
        if not np.isnan(d3_total_val) and not np.isnan(bw) else np.nan)
    return raw


def apply_threshold(proba, threshold):
    if proba[FORMULA_CLASS_IDX] >= threshold:
        return FORMULA_CLASS_IDX
    p = proba.copy()
    p[FORMULA_CLASS_IDX] = -1
    return int(np.argmax(p))


def compute_tree_ci(pipeline, input_df, confidence=0.95):
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    X_processed = prep.transform(input_df)
    n_trees = len(clf.estimators_)
    tree_preds = np.zeros((n_trees, len(CLASS_LABELS)))
    for i, tree in enumerate(clf.estimators_):
        tree_preds[i] = tree.predict_proba(X_processed)[0]
    mean_proba = tree_preds.mean(axis=0)
    std_proba = tree_preds.std(axis=0)
    from scipy import stats
    alpha = 1 - confidence
    t_val = stats.t.ppf(1 - alpha / 2, df=n_trees - 1)
    se = std_proba / np.sqrt(n_trees)
    lower = np.clip(mean_proba - t_val * se, 0, 1)
    upper = np.clip(mean_proba + t_val * se, 0, 1)
    return mean_proba, lower, upper, std_proba


# ==================== HEADER ====================

st.markdown("""
<div class="main-header">
    <h1>🏥 NICU Feeding Prediction Calculator</h1>
    <p>Clinical Decision Support · Multi-Window Model · Tree-Variance CIs</p>
</div>
""", unsafe_allow_html=True)


# ==================== WINDOW SELECTOR (native Streamlit) ====================

st.markdown("#### 🕐 Select Your Data Window")
st.caption("Choose the model that matches the clinical data you have available.")

# Use native Streamlit columns for the cards — works on mobile
c1, c2, c3, c4 = st.columns(4)
window_keys = list(MODEL_FILES.keys())

for i, (col, wname) in enumerate(zip([c1, c2, c3, c4], window_keys)):
    meta = WINDOW_META[wname]
    with col:
        b = ALL_BUNDLES.get(wname, {})
        n_feat = len(b.get("features", []))
        auc = b.get("test_metrics", {}).get("AUC_ROC", "—")
        rec_label = " ⭐" if meta.get("recommended") else ""
        st.metric(
            label=f"{meta['icon']} {meta['short']}{rec_label}",
            value=f"AUC {auc}",
            delta=f"{n_feat} features",
            delta_color="off",
        )
        st.caption(meta["detail"])

# The actual functional selector
selected_window = st.segmented_control(
    "Select model window:",
    window_keys,
    default="Day 1+2 (0–48h)",
    label_visibility="collapsed",
)
if not selected_window:
    selected_window = "Day 1+2 (0–48h)"

# Load selected model
if selected_window in ALL_BUNDLES:
    bundle = ALL_BUNDLES[selected_window]
    model_pipeline = bundle["pipeline"]
    OPTIMAL_THRESHOLD = bundle["threshold"]
    MODEL_FEATURES = bundle["features"]
    TEST_METRICS = bundle.get("test_metrics", {})
    CV_METRICS = bundle.get("cv_metrics", {})
else:
    st.error(f"Model '{selected_window}' not found.")
    bundle = model_pipeline = None
    OPTIMAL_THRESHOLD = 0.3
    MODEL_FEATURES = []
    TEST_METRICS = {}
    CV_METRICS = {}



st.markdown("---")


# ==================== TABS ====================

tab1, tab2, tab3 = st.tabs([
    "📝 Patient Data Entry",
    "📈 Results & Visualization",
    "ℹ️ About & Explainability"
])

# ==================== EXAMPLE HELPERS ====================

def _load_example():
    """Populate session_state widget keys with example values."""
    ex = EXAMPLE_PATIENT
    st.session_state["inp_bw"] = ex["birth_weight"]
    st.session_state["inp_ga"] = ex["ga_weeks"]
    st.session_state["inp_ga_days"] = ex["ga_days"]
    st.session_state["inp_wt_fu"] = ex["weight_followup"]
    st.session_state["inp_mat_age"] = ex["mat_age"]
    st.session_state["inp_bf_edu"] = ex["bf_education"]
    st.session_state["inp_bfhi"] = ex["bfhi_status"]
    st.session_state["inp_d1_fm"] = ex["d1_formula"]
    st.session_state["inp_d1_bm"] = ex["d1_bm"]
    st.session_state["inp_d1_bm_flag"] = ex["d1_bm_flag"]
    st.session_state["inp_d1_bf_flag"] = ex["d1_bf_flag"]
    st.session_state["inp_wt_d1"] = ex["weight_d1"]
    st.session_state["inp_d2_bm"] = ex["d2_bm"]
    st.session_state["inp_d2_fm"] = ex["d2_formula"]
    st.session_state["inp_wt_d2"] = ex["weight_d2"]
    st.session_state["inp_d3_bm"] = ex["d3_bm"]
    st.session_state["inp_d3_fm"] = ex["d3_formula"]
    st.session_state["inp_wt_d3"] = ex["weight_d3"]
    st.session_state["inp_d3_route"] = ex["d3_route"]

needs_day1 = selected_window in [
    "Day 1 (0–24h)", "Day 1+2 (0–48h)", "Full (0–72h)"]
needs_day2 = selected_window in ["Day 1+2 (0–48h)", "Full (0–72h)"]
needs_day3 = selected_window == "Full (0–72h)"


# ==================== TAB 1: DATA ENTRY ====================

with tab1:
    st.markdown(f"### Patient Information — {selected_window}")

    st.button("📋 Load Example Patient", on_click=_load_example,
              help="Fill all fields with a realistic 35-week preterm case")

    col1, col2 = st.columns(2)

    bf_opts = ["", "No", "Yes"]

    with col2:
        st.markdown("#### 👶 Infant Characteristics")
        birth_weight = st.number_input(
            "Birth Weight (g) *", 300, 7000,
            value=None, key="inp_bw", step=10)
        ga_weeks = st.number_input(
            "Gestational Age (weeks) *", 22.0, 44.0,
            value=None, key="inp_ga", step=0.1)
        ga_days = st.number_input(
            "Gestational Days", 0, 6,
            value=0, key="inp_ga_days", step=1)
        weight_followup = st.number_input(
            "Follow-up Weight (g)", 0, 7000,
            value=0, key="inp_wt_fu", step=10)

        st.markdown("---")
        st.markdown("#### 👩 Maternal & Institutional")
        mat_age = st.number_input(
            "Maternal Age (years) *", 12, 55,
            value=None, key="inp_mat_age", step=1)
        bf_education = st.selectbox(
            "Breastfeeding Education", bf_opts, key="inp_bf_edu")
        bfhi_status = st.selectbox(
            "Baby-Friendly Hospital Initiative (BFHI)", bf_opts,
            key="inp_bfhi")

    with col1:
        if needs_day1:
            st.markdown("#### 🍼 Day 1 (0–24h)")
            d1_formula = st.number_input(
                "Day 1 Formula (cc)", 0.0,
                value=0.0, key="inp_d1_fm", step=0.1)
            d1_bm = st.number_input(
                "Day 1 Breast Milk (cc)", 0.0,
                value=0.0, key="inp_d1_bm", step=0.1)
            d1_bm_flag = st.selectbox(
                "Day 1 Breast Milk Given?", bf_opts, key="inp_d1_bm_flag")
            d1_bf_flag = st.selectbox(
                "Day 1 Breastfeeding?", bf_opts, key="inp_d1_bf_flag")
            weight_d1 = st.number_input(
                "Day 1 Weight (g)", 0, 7000,
                value=0, key="inp_wt_d1", step=10)
        else:
            d1_formula = d1_bm = 0.0
            d1_bm_flag = d1_bf_flag = ""
            weight_d1 = 0

        if needs_day2:
            st.markdown("---")
            st.markdown("#### 🍼 Day 2 (24–48h)")
            d2_bm = st.number_input(
                "Day 2 Breast Milk (cc)", 0.0,
                value=0.0, key="inp_d2_bm", step=0.1)
            d2_formula = st.number_input(
                "Day 2 Formula (cc)", 0.0,
                value=0.0, key="inp_d2_fm", step=0.1)
            d2_total = d2_bm + d2_formula
            st.info(f"Day 2 Total: {d2_total:.1f} cc")
            weight_d2 = st.number_input(
                "Day 2 Weight (g)", 0, 7000,
                value=0, key="inp_wt_d2", step=10)
        else:
            d2_bm = d2_formula = d2_total = 0.0
            weight_d2 = 0

        if needs_day3:
            st.markdown("---")
            st.markdown("#### 🍼 Day 3 (48–72h)")
            d3_bm = st.number_input(
                "Day 3 Breast Milk (cc)", 0.0,
                value=0.0, key="inp_d3_bm", step=0.1)
            d3_formula = st.number_input(
                "Day 3 Formula (cc)", 0.0,
                value=0.0, key="inp_d3_fm", step=0.1)
            d3_total = d3_bm + d3_formula
            st.info(f"Day 3 Total: {d3_total:.1f} cc")
            weight_d3 = st.number_input(
                "Day 3 Weight (g)", 0, 7000,
                value=0, key="inp_wt_d3", step=10)
            route_opts = ["", "PO", "OG", "PO+OG", "BF", "BF+PO", "BF+OG"]
            d3_route = st.selectbox(
                "Day 3 Feeding Route", route_opts, key="inp_d3_route")
        else:
            d3_bm = d3_formula = d3_total = 0.0
            weight_d3 = 0
            d3_route = ""

    st.markdown("---")

    # ---- Validation ----
    missing_fields = []
    if birth_weight is None:
        missing_fields.append("Birth Weight")
    if ga_weeks is None:
        missing_fields.append("Gestational Age")
    if mat_age is None:
        missing_fields.append("Maternal Age")

    predict_button = st.button("🔬 Generate Prediction", type="primary",
                                use_container_width=True)
    if predict_button and missing_fields:
        field_list = ", ".join(f"**{f}**" for f in missing_fields)
        st.markdown(
            f'<div class="field-warning">'
            f'⚠️ Cannot predict — required fields missing: {field_list}.</div>',
            unsafe_allow_html=True)
        predict_button = False
    elif predict_button:
        st.success(
            "✅ Prediction generated! "
            "**Click the 'Results & Visualization' tab.**")


# ==================== TAB 2: RESULTS ====================

with tab2:
    if predict_button and model_pipeline:
        try:
            data = {
                "anneyasi": mat_age,
                "dogumagirligi(gram)": birth_weight,
                "gebelikhaftası": ga_weeks,
                "gebelikhaftagunu": ga_days,
                "takipilkgün_kilo_gram": (
                    weight_followup if weight_followup > 0 else np.nan),
                "annesutuemzirmeeğitimidurumu": (
                    bf_opts.index(bf_education) - 1
                    if bf_education else np.nan),
                "bebek_dostu_20temmuz2018": (
                    bf_opts.index(bfhi_status) - 1
                    if bfhi_status else np.nan),
                "aldığıannesütü_ilkgün": d1_bm if needs_day1 else np.nan,
                "aldığımamamiktari1.gün": d1_formula if needs_day1 else np.nan,
                "kilo1.gun": (weight_d1 if weight_d1 > 0 else np.nan),
                "ilk_gün_anne_sütü_1111": (
                    bf_opts.index(d1_bm_flag) - 1
                    if d1_bm_flag else np.nan),
                "ilk_gün_emzirme_111": (
                    bf_opts.index(d1_bf_flag) - 1
                    if d1_bf_flag else np.nan),
                "beslenme2.gunannesutucc": d2_bm if needs_day2 else np.nan,
                "beslenmemamamiktarı2.guncc": (
                    d2_formula if needs_day2 else np.nan),
                "beslenmetotali2.gün": d2_total if needs_day2 else np.nan,
                "kilo2.gun": (weight_d2 if weight_d2 > 0 else np.nan),
                "beslenmetotali3.gun": d3_total if needs_day3 else np.nan,
                "aldıgıannesütü3.gun": d3_bm if needs_day3 else np.nan,
                "aldıgımamamiktari3.gun": d3_formula if needs_day3 else np.nan,
                "kilo3.gun": (weight_d3 if weight_d3 > 0 else np.nan),
                "verilisyolu3gun": (d3_route if d3_route else np.nan),
            }

            data = compute_engineered(data)
            for feat in MODEL_FEATURES:
                if feat not in data:
                    data[feat] = np.nan

            input_df = pd.DataFrame([data])[MODEL_FEATURES]
            probabilities = model_pipeline.predict_proba(input_df)[0]
            prediction = apply_threshold(probabilities, OPTIMAL_THRESHOLD)
            predicted_class = CLASS_LABELS[prediction]
            mean_proba, ci_lower, ci_upper, std_proba = compute_tree_ci(
                model_pipeline, input_df)

            box_class = ["ebf", "formula", "mixed"][prediction]

            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h2 style="color:#0A2540; margin-bottom:0.5rem;">
                    Predicted Feeding Type at Discharge</h2>
                <h1 style="color:{CLASS_COLORS[prediction]};
                           font-size:2.2rem; margin:0.5rem 0;">
                    {predicted_class}</h1>
                <p style="font-size:1.1rem; color:#475569;">
                    P(Formula) = <strong>{probabilities[FORMULA_CLASS_IDX]*100:.1f}%</strong>
                    &nbsp;|&nbsp; Threshold = {OPTIMAL_THRESHOLD}
                    &nbsp;|&nbsp; Model: {selected_window}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                "### 📊 Class Probabilities with 95% Confidence Intervals")
            st.caption(
                "Intervals from individual tree predictions (tree-variance)")

            gc1, gc2, gc3 = st.columns(3)
            for i, (label, prob, color) in enumerate(
                    zip(CLASS_LABELS, probabilities, CLASS_COLORS)):
                with [gc1, gc2, gc3][i]:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob * 100,
                        title={"text": label, "font": {"size": 13}},
                        number={"suffix": "%", "font": {"size": 28}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": color},
                            "bgcolor": "white",
                            "borderwidth": 1, "bordercolor": "#E2E8F0",
                            "steps": [
                                {"range": [0, 33], "color": "#F8FAFC"},
                                {"range": [33, 67], "color": "#F1F5F9"},
                                {"range": [67, 100], "color": "#E2E8F0"},
                            ],
                        },
                    ))
                    fig.update_layout(
                        height=220, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                    uncertainty = std_proba[i] * 100
                    emoji = ("🟢" if uncertainty < 3 else
                             "🟡" if uncertainty < 8 else "🔴")
                    conf = ("High" if uncertainty < 3 else
                            "Moderate" if uncertainty < 8 else "Low")
                    st.caption(
                        f"{emoji} {conf} confidence · "
                        f"95% CI: [{ci_lower[i]*100:.1f}%"
                        f" – {ci_upper[i]*100:.1f}%]")

            st.markdown("### 📈 Confidence Interval Comparison")
            fig_ci = go.Figure()
            for i, (label, color) in enumerate(
                    zip(CLASS_LABELS, CLASS_COLORS)):
                fig_ci.add_trace(go.Bar(
                    y=[label], x=[probabilities[i]], orientation="h",
                    marker=dict(color=color, opacity=0.85),
                    text=f"{probabilities[i]*100:.1f}%",
                    textposition="auto",
                    textfont=dict(color="white", size=14),
                    showlegend=False,
                    error_x=dict(
                        type="data", symmetric=False,
                        array=[ci_upper[i] - probabilities[i]],
                        arrayminus=[probabilities[i] - ci_lower[i]],
                        color="#1E293B", thickness=2, width=8),
                ))
            fig_ci.add_vline(
                x=OPTIMAL_THRESHOLD, line_dash="dash",
                line_color="#DC2626", line_width=1.5,
                annotation_text=f"Threshold ({OPTIMAL_THRESHOLD})",
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color="#DC2626")
            fig_ci.update_layout(
                xaxis=dict(title="Probability", range=[0, 1],
                           tickformat=".0%"),
                yaxis=dict(title=""),
                height=250,
                margin=dict(l=10, r=10, t=40, b=40),
                plot_bgcolor="white", font=dict(size=12))
            fig_ci.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
            st.plotly_chart(fig_ci, use_container_width=True)

            # ---- Imputation Transparency ----
            imputed_rows = []
            prep = model_pipeline.named_steps.get("prep")
            if prep and hasattr(prep, "transformers_"):
                for tr_name, transformer, cols in prep.transformers_:
                    if hasattr(transformer, "statistics_"):
                        for col, median_val in zip(cols, transformer.statistics_):
                            if col in MODEL_FEATURES:
                                user_val = input_df[col].iloc[0]
                                if pd.isna(user_val):
                                    display = FEATURE_DISPLAY_NAMES.get(
                                        col, col)
                                    imputed_rows.append({
                                        "Field": display,
                                        "Imputed Value": round(
                                            median_val, 2),
                                        "Method":
                                            "Training-set median",
                                    })

            if imputed_rows:
                st.markdown("### ℹ️ Data Completeness")
                n_total = len(MODEL_FEATURES)
                n_provided = n_total - len(imputed_rows)
                pct = n_provided / n_total * 100
                st.markdown(
                    f'<div class="impute-box">'
                    f'You provided <strong>{n_provided}/{n_total}'
                    f'</strong> features ({pct:.0f}%). '
                    f'The {len(imputed_rows)} field(s) below were '
                    f'automatically filled using <strong>training-set '
                    f'median</strong> values (the typical value '
                    f'observed in the development cohort).</div>',
                    unsafe_allow_html=True)
                st.dataframe(
                    pd.DataFrame(imputed_rows),
                    use_container_width=True, hide_index=True)
                st.caption(
                    "★ = Engineered feature computed from other "
                    "inputs. If its source fields are missing, "
                    "it is also imputed.")
            else:
                st.markdown("### ✅ Data Completeness")
                st.success(
                    f"All {len(MODEL_FEATURES)} features were "
                    f"provided — no imputation was needed.")

            # ---- Per-Patient Explainability (SHAP) ----
            st.markdown("### 🔍 Why This Prediction?")
            st.caption(
                "Feature contributions (SHAP values) for the "
                "predicted class. Positive = pushes toward this "
                "prediction; negative = pushes away.")

            try:
                prep = model_pipeline.named_steps["prep"]
                clf = model_pipeline.named_steps["clf"]
                X_transformed = prep.transform(input_df)

                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_transformed)

                # Handle both old (list) and new (3D array) SHAP API
                sv_arr = np.array(shap_values)
                if sv_arr.ndim == 3:
                    # v0.48+: shape (n_samples, n_features, n_classes)
                    sv = sv_arr[0, :, prediction]
                else:
                    # older: list of (n_samples, n_features) per class
                    sv = shap_values[prediction][0]
                feat_vals = X_transformed[0]  # preprocessed values

                # Build sorted dataframe
                explain_df = pd.DataFrame({
                    "feature_raw": MODEL_FEATURES,
                    "shap": sv,
                    "value": feat_vals,
                })
                explain_df["abs_shap"] = explain_df["shap"].abs()
                explain_df = explain_df.sort_values(
                    "abs_shap", ascending=True)  # bottom=smallest
                top_n = min(12, len(explain_df))
                explain_df = explain_df.tail(top_n)

                # Human-readable names + values
                labels = [
                    f"{FEATURE_DISPLAY_NAMES.get(f, f)}\n"
                    f"= {v:.2f}"
                    for f, v in zip(
                        explain_df["feature_raw"],
                        explain_df["value"])
                ]
                colors = [
                    "#DC2626" if s > 0 else "#059669"
                    for s in explain_df["shap"]
                ]

                fig_shap = go.Figure(go.Bar(
                    y=labels,
                    x=explain_df["shap"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{s:+.3f}" for s in explain_df["shap"]],
                    textposition="outside",
                    textfont=dict(size=11),
                ))
                pred_label = CLASS_LABELS[prediction]
                fig_shap.update_layout(
                    title=dict(
                        text=f"Feature Contributions → "
                             f"{pred_label}",
                        font=dict(size=14)),
                    xaxis_title="SHAP Value (impact on prediction)",
                    yaxis_title="",
                    height=max(350, top_n * 42),
                    margin=dict(l=10, r=60, t=40, b=40),
                    plot_bgcolor="white",
                    font=dict(size=11),
                )
                fig_shap.update_xaxes(
                    showgrid=True, gridcolor="#E2E8F0",
                    zeroline=True, zerolinecolor="#1E293B",
                    zerolinewidth=1.5)
                st.plotly_chart(fig_shap, use_container_width=True)

                st.markdown(
                    '<div class="impute-box" style="background:#EFF6FF;'
                    'border-left-color:#2563EB;color:#1E40AF;">'
                    '<strong>How to read:</strong> '
                    'Each bar shows how much a feature pushed the '
                    'model toward (<span style="color:#DC2626">'
                    'red</span>) or away from '
                    '(<span style="color:#059669">green</span>) '
                    'the predicted class. The number after "=" is '
                    'the actual patient value used by the model.'
                    '</div>',
                    unsafe_allow_html=True)
            except Exception as shap_err:
                st.warning(
                    f"Could not generate SHAP explanation: "
                    f"{shap_err}")


        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
    else:
        st.info("👈 Enter patient data and click 'Generate Prediction'.")


# ==================== TAB 3: ABOUT ====================

with tab3:
    st.markdown("### About This Tool")
    st.markdown("""
    #### Multi-Window Prediction
    | Window | Features | Threshold | AUC | Use Case |
    |:---|:---:|:---:|:---:|:---|
    | Baseline | 8 | 0.32 | 0.738 | Admission — earliest |
    | Day 1 | 14 | 0.355 | 0.825 | First 24 hours |
    | **Day 1+2** | **19** | **0.23** | **0.836** | **48 hours — recommended** |
    | Full | 29 | 0.215 | 0.869 | 72 hours — highest accuracy |

    #### Design
    - **F₂-optimized Random Forest** + constrained threshold (precision ≥ 0.40)
    - **COVID/Epoch excluded** — post-pandemic generalization
    - **BFHI auto-included** where it improved cross-validated F₂
    - **Tree-variance CIs** — patient-specific uncertainty

    ⚠️ This tool **supports** clinical decision-making.
    It should **not replace** professional medical judgment.

    Patient data is processed locally, **not stored or transmitted**.
    """)

    with st.expander("📊 Cross-Validation Details"):
        if CV_METRICS:
            cv_table = "| Metric | Mean ± SD |\n|:---|:---:|\n"
            for k, v in CV_METRICS.items():
                cv_table += f"| {k} | {v['mean']:.3f} ± {v['std']:.3f} |\n"
            st.markdown(cv_table)

    with st.expander("❓ FAQ"):
        st.markdown("""
        **Q: Can I trust these predictions?**
        Validated with 5-fold CV and held-out test set. Use as screening, not diagnosis.

        **Q: What if I don't fill all fields?**
        Empty fields → median imputation from training data.

        **Q: Why no COVID variables?**
        Ablation showed they're redundant post-pandemic.
        """)
