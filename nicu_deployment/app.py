import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="NICU Feeding Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .main-header {
        background: linear-gradient(135deg, #0A2540 0%, #1E3A5F 50%, #006B7D 100%);
        padding: 2rem 2.5rem; border-radius: 12px; color: white;
        margin-bottom: 1.5rem; font-family: 'Inter', sans-serif;
    }
    .main-header h1 { margin: 0; font-weight: 700; font-size: 1.6rem; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.85; font-size: 1rem; }
    .metric-row {
        display: flex; gap: 2rem; margin-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.2); padding-top: 1rem;
        flex-wrap: wrap;
    }
    .metric-item .label {
        font-size: 0.7rem; opacity: 0.7; text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-item .value {
        font-size: 1.15rem; font-weight: 600;
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
    .window-card {
        border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.8rem;
        margin-bottom: 0.5rem; background: #F8FAFC;
    }
    .window-card.active {
        border-color: #006B7D; background: #F0FDFA; box-shadow: 0 0 0 1px #006B7D;
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

WINDOW_DESCRIPTIONS = {
    "Baseline (Admission)": "Admission data only — birth weight, gestational age, maternal age",
    "Day 1 (0–24h)": "Adds Day 1 feeding volumes and breastfeeding status",
    "Day 1+2 (0–48h)": "Adds Day 2 feeding volumes and intake trajectory — recommended default",
    "Full (0–72h)": "Includes Day 3 data — most complete, highest accuracy",
}


# ==================== MODEL LOADING ====================

@st.cache_resource
def load_all_models():
    """Load all 4 model bundles."""
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
    """Compute auto-derived features from raw inputs."""
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
    d1_total = (d1_bm if not np.isnan(d1_bm) else 0) + (d1_fm if not np.isnan(d1_fm) else 0)
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
    else:
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


# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown("### 🕐 Select Data Window")
    st.caption("Choose the model based on how much data you have:")

    selected_window = st.radio(
        "Data available:",
        list(MODEL_FILES.keys()),
        index=2,  # Default to Day 1+2
        help="Each model is tuned specifically for its temporal window",
    )

    st.caption(WINDOW_DESCRIPTIONS[selected_window])

    if selected_window in ALL_BUNDLES:
        bundle = ALL_BUNDLES[selected_window]
        model_pipeline = bundle["pipeline"]
        OPTIMAL_THRESHOLD = bundle["threshold"]
        MODEL_FEATURES = bundle["features"]
        TEST_METRICS = bundle.get("test_metrics", {})
        CV_METRICS = bundle.get("cv_metrics", {})
    else:
        st.error(f"Model for '{selected_window}' not found.")
        bundle = None
        model_pipeline = None

    st.markdown("---")
    st.markdown("### 📊 Model Info")

    with st.expander("Performance Metrics", expanded=False):
        if bundle:
            m = TEST_METRICS
            st.markdown(f"""
| Metric | Value |
|:---|:---:|
| Formula Recall | {m.get('Formula_Recall', '—')} |
| Formula Precision | {m.get('Formula_Precision', '—')} |
| Formula F2 | {m.get('Formula_F2', '—')} |
| F1-Macro | {m.get('F1_Macro', '—')} |
| MCC | {m.get('MCC', '—')} |
| Brier Score | {m.get('Brier_Formula', '—')} |
| AUC-ROC | {m.get('AUC_ROC', '—')} |
| Threshold | {OPTIMAL_THRESHOLD} |
| Features | {len(MODEL_FEATURES)} |
""")

    with st.expander("Clinical Context", expanded=False):
        st.info("""
        **Threshold Strategy:**
        The threshold maximizes Formula F₂-score subject to
        Formula Precision ≥ 0.40, ensuring clinically acceptable
        positive predictive value.

        **Missing Values:** Imputed with training-set medians.

        **Note:** This tool supports, not replaces, clinical judgment.
        """)

    st.markdown("---")
    if st.button("📋 Load Example Patient"):
        st.session_state.example_loaded = True
        st.rerun()


# ==================== HEADER ====================

if bundle:
    m = TEST_METRICS
    st.markdown(f"""
    <div class="main-header">
        <h1>🏥 NICU Feeding Prediction Calculator</h1>
        <p>Clinical Decision Support — {selected_window} Model</p>
        <div class="metric-row">
            <div class="metric-item">
                <div class="label">ROC-AUC</div>
                <div class="value">{m.get('AUC_ROC', '—')}</div>
            </div>
            <div class="metric-item">
                <div class="label">Formula Recall</div>
                <div class="value">{m.get('Formula_Recall', '—')}</div>
            </div>
            <div class="metric-item">
                <div class="label">Formula Precision</div>
                <div class="value">{m.get('Formula_Precision', '—')}</div>
            </div>
            <div class="metric-item">
                <div class="label">MCC</div>
                <div class="value">{m.get('MCC', '—')}</div>
            </div>
            <div class="metric-item">
                <div class="label">Features</div>
                <div class="value">{len(MODEL_FEATURES)}</div>
            </div>
            <div class="metric-item">
                <div class="label">Threshold</div>
                <div class="value">{OPTIMAL_THRESHOLD}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==================== TABS ====================

tab1, tab2, tab3 = st.tabs([
    "📝 Patient Data Entry",
    "📈 Results & Visualization",
    "ℹ️ About & Explainability"
])

# ==================== DEFAULTS ====================

example = "example_loaded" in st.session_state and st.session_state.example_loaded
if example:
    st.session_state.example_loaded = False

needs_day1 = selected_window in ["Day 1 (0–24h)", "Day 1+2 (0–48h)", "Full (0–72h)"]
needs_day2 = selected_window in ["Day 1+2 (0–48h)", "Full (0–72h)"]
needs_day3 = selected_window == "Full (0–72h)"


# ==================== TAB 1: DATA ENTRY ====================

with tab1:
    st.markdown(f"### Patient Information — {selected_window}")

    col1, col2 = st.columns(2)

    with col2:
        st.markdown("#### 👶 Infant Characteristics")
        birth_weight = st.number_input(
            "Birth Weight (g) *", 300, 7000,
            value=2500 if example else None, step=10)
        ga_weeks = st.number_input(
            "Gestational Age (weeks) *", 22.0, 44.0,
            value=37.0 if example else None, step=0.1)
        ga_days = st.number_input(
            "Gestational Days", 0, 6, value=0, step=1)
        weight_followup = st.number_input(
            "Follow-up Weight (g)", 0, 7000, value=0, step=10)

        st.markdown("---")
        st.markdown("#### 👩 Maternal & Institutional")
        mat_age = st.number_input(
            "Maternal Age (years) *", 12, 55,
            value=28 if example else None, step=1)
        bf_education = st.selectbox(
            "Breastfeeding Education", ["", "No", "Yes"])
        bfhi_status = st.selectbox(
            "Baby-Friendly Hospital Initiative (BFHI)", ["", "No", "Yes"])

    with col1:
        if needs_day1:
            st.markdown("#### 🍼 Day 1 (0–24h)")
            d1_formula = st.number_input(
                "Day 1 Formula (cc)", 0.0,
                value=10.0 if example else 0.0, step=0.1)
            d1_bm = st.number_input(
                "Day 1 Breast Milk (cc)", 0.0,
                value=5.0 if example else 0.0, step=0.1)
            d1_bm_flag = st.selectbox(
                "Day 1 Breast Milk Given?", ["", "No", "Yes"])
            d1_bf_flag = st.selectbox(
                "Day 1 Breastfeeding?", ["", "No", "Yes"])
            weight_d1 = st.number_input(
                "Day 1 Weight (g)", 0, 7000, value=0, step=10)
        else:
            d1_formula = d1_bm = 0.0
            d1_bm_flag = d1_bf_flag = ""
            weight_d1 = 0

        if needs_day2:
            st.markdown("---")
            st.markdown("#### 🍼 Day 2 (24–48h)")
            d2_bm = st.number_input(
                "Day 2 Breast Milk (cc)", 0.0,
                value=15.0 if example else 0.0, step=0.1)
            d2_formula = st.number_input(
                "Day 2 Formula (cc)", 0.0,
                value=20.0 if example else 0.0, step=0.1)
            d2_total = d2_bm + d2_formula
            st.info(f"Day 2 Total: {d2_total:.1f} cc")
            weight_d2 = st.number_input(
                "Day 2 Weight (g)", 0, 7000, value=0, step=10)
        else:
            d2_bm = d2_formula = d2_total = 0.0
            weight_d2 = 0

        if needs_day3:
            st.markdown("---")
            st.markdown("#### 🍼 Day 3 (48–72h)")
            d3_bm = st.number_input(
                "Day 3 Breast Milk (cc)", 0.0,
                value=25.0 if example else 0.0, step=0.1)
            d3_formula = st.number_input(
                "Day 3 Formula (cc)", 0.0,
                value=15.0 if example else 0.0, step=0.1)
            d3_total = d3_bm + d3_formula
            st.info(f"Day 3 Total: {d3_total:.1f} cc")
            weight_d3 = st.number_input(
                "Day 3 Weight (g)", 0, 7000, value=0, step=10)
            d3_route = st.selectbox(
                "Day 3 Feeding Route", ["", "PO", "OG", "PO+OG",
                                        "BF", "BF+PO", "BF+OG"])
        else:
            d3_bm = d3_formula = d3_total = 0.0
            weight_d3 = 0
            d3_route = ""

    st.markdown("---")
    predict_button = st.button("🔬 Generate Prediction", type="primary",
                                use_container_width=True)
    if predict_button:
        st.success("✅ Prediction generated! **Click the 'Results & Visualization' tab.**")


# ==================== TAB 2: RESULTS ====================

with tab2:
    if predict_button and model_pipeline:
        try:
            # Build raw data dict
            data = {
                "anneyasi": mat_age,
                "dogumagirligi(gram)": birth_weight,
                "gebelikhaftası": ga_weeks,
                "gebelikhaftagunu": ga_days,
                "takipilkgün_kilo_gram": (
                    weight_followup if weight_followup > 0 else np.nan),
                "annesutuemzirmeeğitimidurumu": (
                    ["", "No", "Yes"].index(bf_education) - 1
                    if bf_education else np.nan),
                "bebek_dostu_20temmuz2018": (
                    ["", "No", "Yes"].index(bfhi_status) - 1
                    if bfhi_status else np.nan),
                "aldığıannesütü_ilkgün": d1_bm if needs_day1 else np.nan,
                "aldığımamamiktari1.gün": d1_formula if needs_day1 else np.nan,
                "kilo1.gun": (weight_d1 if weight_d1 > 0 else np.nan),
                "ilk_gün_anne_sütü_1111": (
                    ["", "No", "Yes"].index(d1_bm_flag) - 1
                    if d1_bm_flag else np.nan),
                "ilk_gün_emzirme_111": (
                    ["", "No", "Yes"].index(d1_bf_flag) - 1
                    if d1_bf_flag else np.nan),
                "beslenme2.gunannesutucc": d2_bm if needs_day2 else np.nan,
                "beslenmemamamiktarı2.guncc": d2_formula if needs_day2 else np.nan,
                "beslenmetotali2.gün": d2_total if needs_day2 else np.nan,
                "kilo2.gun": (weight_d2 if weight_d2 > 0 else np.nan),
                "beslenmetotali3.gun": d3_total if needs_day3 else np.nan,
                "aldıgıannesütü3.gun": d3_bm if needs_day3 else np.nan,
                "aldıgımamamiktari3.gun": d3_formula if needs_day3 else np.nan,
                "kilo3.gun": (weight_d3 if weight_d3 > 0 else np.nan),
                "verilisyolu3gun": (
                    d3_route if d3_route else np.nan),
            }

            # Compute engineered features
            data = compute_engineered(data)

            # Ensure all model features present
            for feat in MODEL_FEATURES:
                if feat not in data:
                    data[feat] = np.nan

            input_df = pd.DataFrame([data])[MODEL_FEATURES]

            # Predict
            probabilities = model_pipeline.predict_proba(input_df)[0]
            prediction = apply_threshold(probabilities, OPTIMAL_THRESHOLD)
            predicted_class = CLASS_LABELS[prediction]

            # Tree-variance CI
            mean_proba, ci_lower, ci_upper, std_proba = compute_tree_ci(
                model_pipeline, input_df)

            box_class = ["ebf", "formula", "mixed"][prediction]

            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h2 style="color:#0A2540; margin-bottom:0.5rem;">
                    Predicted Feeding Type at Discharge
                </h2>
                <h1 style="color:{CLASS_COLORS[prediction]}; font-size:2.2rem; margin:0.5rem 0;">
                    {predicted_class}
                </h1>
                <p style="font-size:1.1rem; color:#475569;">
                    P(Formula) = <strong>{probabilities[FORMULA_CLASS_IDX]*100:.1f}%</strong>
                    &nbsp;|&nbsp; Threshold = {OPTIMAL_THRESHOLD}
                    &nbsp;|&nbsp; Model: {selected_window}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Gauges with CIs
            st.markdown("### 📊 Class Probabilities with 95% Confidence Intervals")
            st.caption("Intervals from individual tree predictions (tree-variance method)")

            col1, col2, col3 = st.columns(3)
            for i, (label, prob, color) in enumerate(
                    zip(CLASS_LABELS, probabilities, CLASS_COLORS)):
                with [col1, col2, col3][i]:
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
                    fig.update_layout(height=220,
                                      margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                    uncertainty = std_proba[i] * 100
                    emoji = "🟢" if uncertainty < 3 else "🟡" if uncertainty < 8 else "🔴"
                    conf = ("High" if uncertainty < 3 else
                            "Moderate" if uncertainty < 8 else "Low")
                    st.caption(
                        f"{emoji} {conf} confidence · "
                        f"95% CI: [{ci_lower[i]*100:.1f}% – {ci_upper[i]*100:.1f}%]")

            # Horizontal bar with CI error bars
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
                        color="#1E293B", thickness=2, width=8,
                    ),
                ))
            fig_ci.add_vline(
                x=OPTIMAL_THRESHOLD, line_dash="dash",
                line_color="#DC2626", line_width=1.5,
                annotation_text=f"Threshold ({OPTIMAL_THRESHOLD})",
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color="#DC2626",
            )
            fig_ci.update_layout(
                xaxis=dict(title="Probability", range=[0, 1],
                           tickformat=".0%"),
                yaxis=dict(title=""),
                height=250,
                margin=dict(l=10, r=10, t=40, b=40),
                plot_bgcolor="white", font=dict(size=12),
            )
            fig_ci.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
            st.plotly_chart(fig_ci, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
    else:
        st.info("👈 Enter patient data and click 'Generate Prediction'.")


# ==================== TAB 3: ABOUT ====================

with tab3:
    st.markdown("### About This Tool")

    st.markdown(f"""
    #### Multi-Window Prediction
    The app provides **4 models** matched to the data you have available:

    | Window | Features | Use Case |
    |:---|:---:|:---|
    | Baseline | 8 | Admission — earliest possible prediction |
    | Day 1 | 14 | End of first 24 hours |
    | **Day 1+2** | **19** | **End of 48 hours — recommended default** |
    | Full D1-3 | 29 | End of 72 hours — highest accuracy |

    #### Key Design Decisions
    - **F₂-optimized RF** with constrained threshold (precision ≥ 0.40)
    - **COVID/Epoch excluded** — better generalization post-pandemic
    - **BFHI auto-included** where it improved cross-validated F₂
    - **Tree-variance CIs** — patient-specific uncertainty from {len(ALL_BUNDLES.get(selected_window, {}).get('params', {}))} trees

    #### Clinical Disclaimer
    ⚠️ This tool **supports** clinical decision-making and should
    **not replace** professional medical judgment.

    #### Data Privacy
    Patient data is processed locally and is **not stored or transmitted**.
    """)
