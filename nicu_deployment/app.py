import streamlit as st
import pickle
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="NICU Breastfeeding Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0A2540 0%, #1E3A5F 100%);
        padding: 2rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #006B7D;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0A2540 0%, #006B7D 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 6px;
    }
    .prediction-box {
        background: #EFF6FF;
        border: 2px solid #1D4ED8;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .footer-citation {
        background: #F1F5F9;
        padding: 1rem;
        border-left: 3px solid #006B7D;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ====================

OPTIMAL_THRESHOLD = 0.26
FORMULA_CLASS_IDX = 1

# ==================== LOAD MODEL ====================

APP_DIR = Path(__file__).parent

@st.cache_resource
def load_model_artifacts():
    """Load the trained model and metadata"""
    try:
        with open(APP_DIR / 'trained_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(APP_DIR / 'feature_metadata.json', 'r') as f:
            metadata = json.load(f)
        with open(APP_DIR / 'model_info.json', 'r') as f:
            model_info = json.load(f)
        return model, metadata, model_info
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

model_pipeline, feature_metadata, model_info = load_model_artifacts()

# ==================== HELPER FUNCTIONS ====================

def safe_val(v, default=np.nan):
    """Convert None to NaN for safe arithmetic."""
    return default if v is None else float(v)


def compute_engineered(raw):
    """Compute auto-derived features from raw inputs."""
    # Ensure all values are numeric (None → NaN)
    for k in list(raw.keys()):
        raw[k] = safe_val(raw[k])

    eps = 1e-6
    bw = raw["dogumagirligi(gram)"]
    gw = raw["gebelikhaftası"]
    raw["eng_weight_per_week"] = bw / (gw + eps) if not np.isnan(bw) and not np.isnan(gw) else np.nan

    d1_bm = raw["aldığıannesütü_ilkgün"]
    d1_fm = raw["aldığımamamiktari1.gün"]
    raw["eng_bm_ratio_d1"] = d1_bm / (d1_bm + d1_fm + eps) if not np.isnan(d1_bm) and not np.isnan(d1_fm) else np.nan

    d2_bm = raw["beslenme2.gunannesutucc"]
    d2_total = raw["beslenmetotali2.gün"]
    raw["eng_bm_ratio_d2"] = d2_bm / (d2_total + eps) if not np.isnan(d2_bm) and not np.isnan(d2_total) else np.nan

    raw["eng_delta_vol_d1_d2"] = (d2_total - (d1_bm + d1_fm)
                                   if not any(np.isnan(x) for x in [d2_total, d1_bm, d1_fm]) else np.nan)

    # eng_resilience_index: requires Day 3 → NaN, median imputer handles it
    raw["eng_resilience_index"] = np.nan
    return raw


def apply_threshold(proba, threshold):
    """Lower Formula threshold → more Formula predictions → higher recall."""
    if proba[FORMULA_CLASS_IDX] >= threshold:
        return FORMULA_CLASS_IDX
    else:
        p = proba.copy()
        p[FORMULA_CLASS_IDX] = -1
        return int(np.argmax(p))


# ==================== HEADER ====================

st.markdown("""
<div class="main-header">
    <h1>NICU Breastfeeding Prediction Calculator</h1>
    <p style="font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem;">
        Clinical Decision Support Tool — 48-Hour Model (Day 1 + Day 2)
    </p>
    <div style="display: flex; gap: 2rem; margin-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 1rem;">
        <div>
            <div style="font-size: 0.75rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.05em;">ROC-AUC (CV)</div>
            <div style="font-size: 1.2rem; font-weight: 600; font-family: 'Courier New';">0.827</div>
        </div>
        <div>
            <div style="font-size: 0.75rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.05em;">Formula Recall</div>
            <div style="font-size: 1.2rem; font-weight: 600; font-family: 'Courier New';">87.5%</div>
        </div>
        <div>
            <div style="font-size: 0.75rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.05em;">MCC</div>
            <div style="font-size: 1.2rem; font-weight: 600; font-family: 'Courier New';">0.476</div>
        </div>
        <div>
            <div style="font-size: 0.75rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.05em;">Features</div>
            <div style="font-size: 1.2rem; font-weight: 600; font-family: 'Courier New';">20</div>
        </div>
        <div>
            <div style="font-size: 0.75rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.05em;">Threshold</div>
            <div style="font-size: 1.2rem; font-weight: 600; font-family: 'Courier New';">0.26</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown("### 📊 Model Specifications")

    with st.expander("Algorithm Details", expanded=False):
        st.markdown("""
        **Model Type:** Random Forest (Optuna-tuned, F2-optimized)
        **Training Method:** 5-fold Stratified CV
        **Sample Size:** n = 1,064
        **Features:** 20 (15 raw + 5 engineered)
        **Temporal Window:** Day 1 + Day 2 (48h)
        **COVID/Epoch:** Excluded (redundant per sensitivity analysis)
        **SMOTE:** Applied for class imbalance
        """)

    with st.expander("Performance Metrics", expanded=False):
        st.markdown("""
        **5-Fold Cross-Validation:**

        | Metric | Mean ± SD |
        |:---|:---:|
        | AUC-ROC | 0.827 ± 0.023 |
        | MCC | 0.433 ± 0.044 |
        | F1-Macro | 0.591 ± 0.047 |
        | Formula Recall | 0.557 ± 0.043 |
        | Formula Precision | 0.546 ± 0.041 |
        | Formula F2 | 0.554 ± 0.037 |

        **Test Set (Threshold = 0.26):**

        | Metric | Value |
        |:---|:---:|
        | AUC-ROC | 0.842 |
        | Formula Recall | 0.875 |
        | Formula Precision | 0.471 |
        | MCC | 0.476 |
        | F2-Score | 0.747 |
        """)

    with st.expander("Clinical Context", expanded=False):
        st.info("""
        This model predicts feeding type at discharge for NICU infants
        based on data from the first 48 hours of life.

        **Threshold Optimization:**
        The model uses a lowered Formula threshold (0.26 vs default 0.33)
        to maximize Formula recall (87.5%) — prioritizing identification
        of infants at risk of formula dependence.

        **Missing Values:**
        Empty fields are replaced with median values from training data.
        Provide as many values as possible for best accuracy.

        **Note:** This tool supports, not replaces, clinical judgment.
        """)

    st.markdown("---")

    if st.button("📋 Load Example Patient"):
        st.session_state.d1_formula = 10.0
        st.session_state.d1_bm = 5.0
        st.session_state.d2_bm = 15.0
        st.session_state.d2_formula = 20.0
        st.session_state.example_loaded = True
        st.rerun()


# ==================== MAIN TABS ====================

if 'show_results' not in st.session_state:
    st.session_state.show_results = False

tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Patient Data Entry",
    "📈 Results & Visualization",
    "ℹ️ About",
    "🔍 Model Explainability"
])

# ==================== TAB 1: DATA ENTRY ====================

with tab1:
    st.markdown("### Patient Information")

    st.info("""
    💡 **48-Hour Prediction Model**

    This model uses data available within the first 48 hours.
    No Day 3 data needed. COVID/Epoch variables excluded.
    - **Required fields:** Birth Weight, Gestational Age, Maternal Age
    - Empty numeric fields → median imputation
    """)

    if 'example_loaded' in st.session_state and st.session_state.example_loaded:
        default_birth_weight = 2500
        default_ga = 37.0
        default_mat_age = 28
        default_d1_formula = 10.0
        default_d1_bm = 5.0
        default_d2_bm = 15.0
        default_d2_formula = 20.0
        st.session_state.example_loaded = False
    else:
        default_birth_weight = None
        default_ga = None
        default_mat_age = None
        default_d1_formula = 0.0
        default_d1_bm = 0.0
        default_d2_bm = 0.0
        default_d2_formula = 0.0

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🍼 Day 1 Feeding (0–24 hours)")
        d1_formula = st.number_input("Day 1 Formula (cc)", min_value=0.0,
                                      value=default_d1_formula, step=0.1,
                                      help="Volume of formula given on first day",
                                      key="d1_formula")
        d1_bm = st.number_input("Day 1 Breast Milk (cc)", min_value=0.0,
                                 value=default_d1_bm, step=0.1,
                                 help="Volume of mother's milk on first day",
                                 key="d1_bm")
        d1_bm_flag = st.selectbox("Day 1 Breast Milk Given?", ["", "No", "Yes"],
                                   key="d1_bm_flag")
        d1_bf_flag = st.selectbox("Day 1 Breastfeeding?", ["", "No", "Yes"],
                                   key="d1_bf_flag")

        st.markdown("---")
        st.markdown("#### 🍼 Day 2 Feeding (24–48 hours)")
        d2_bm = st.number_input("Day 2 Breast Milk (cc)", min_value=0.0,
                                 value=default_d2_bm, step=0.1, key="d2_bm")
        d2_formula = st.number_input("Day 2 Formula (cc)", min_value=0.0,
                                      value=default_d2_formula, step=0.1,
                                      key="d2_formula")
        d2_total = d2_bm + d2_formula
        st.info(f"Day 2 Total: {d2_total:.1f} cc (auto-calculated)")

    with col2:
        st.markdown("#### 👶 Infant Characteristics")
        birth_weight = st.number_input("Birth Weight (g) *", min_value=300,
                                        max_value=7000, value=default_birth_weight,
                                        step=10, help="Required. Weight at birth")
        ga_weeks = st.number_input("Gestational Age (weeks) *", min_value=22.0,
                                    max_value=44.0, value=default_ga, step=0.1,
                                    help="Required. Completed weeks of pregnancy")
        ga_days = st.number_input("Gestational Days", min_value=0, max_value=6,
                                   value=0, step=1,
                                   help="Additional days within gestational week")
        weight_d1 = st.number_input("Day 1 Weight (g)", min_value=0, value=0, step=10)
        weight_d2 = st.number_input("Day 2 Weight (g)", min_value=0, value=0, step=10)
        weight_followup = st.number_input("Follow-up Weight (g)", min_value=0,
                                           value=0, step=10)

        st.markdown("---")
        st.markdown("#### 👩 Maternal Factors")
        mat_age = st.number_input("Maternal Age (years) *", min_value=12,
                                   max_value=55, value=default_mat_age, step=1,
                                   help="Required. Mother's age")
        bf_education = st.selectbox("Breastfeeding Education", ["", "No", "Yes"])

    st.markdown("---")
    predict_button = st.button("🔬 Generate Prediction", type="primary",
                                use_container_width=True)

    if predict_button:
        st.success("✅ Prediction generated! **Click the 'Results & Visualization' tab above.**")
        st.info("💡 The Results tab shows probability charts and contributing factors.")


# ==================== TAB 2: RESULTS ====================

with tab2:
    if predict_button and model_pipeline:
        try:
            # Build raw feature dict
            data = {
                "aldığımamamiktari1.gün": d1_formula,
                "aldığıannesütü_ilkgün": d1_bm,
                "beslenme2.gunannesutucc": d2_bm,
                "beslenmemamamiktarı2.guncc": d2_formula,
                "beslenmetotali2.gün": d2_total,
                "dogumagirligi(gram)": birth_weight,
                "gebelikhaftası": ga_weeks,
                "gebelikhaftagunu": ga_days,
                "kilo1.gun": weight_d1 if weight_d1 > 0 else np.nan,
                "kilo2.gun": weight_d2 if weight_d2 > 0 else np.nan,
                "takipilkgün_kilo_gram": weight_followup if weight_followup > 0 else np.nan,
                "anneyasi": mat_age,
                "annesutuemzirmeeğitimidurumu": (
                    ["", "No", "Yes"].index(bf_education) - 1
                    if bf_education else np.nan),
                "ilk_gün_anne_sütü_1111": (
                    ["", "No", "Yes"].index(d1_bm_flag) - 1
                    if d1_bm_flag else np.nan),
                "ilk_gün_emzirme_111": (
                    ["", "No", "Yes"].index(d1_bf_flag) - 1
                    if d1_bf_flag else np.nan),
            }

            # Compute engineered features
            data = compute_engineered(data)

            # Ensure all expected features present
            all_features = feature_metadata["all_features"]
            for feat in all_features:
                if feat not in data:
                    data[feat] = np.nan

            input_df = pd.DataFrame([data])[all_features]

            # Predict with threshold
            probabilities = model_pipeline.predict_proba(input_df)[0]
            prediction = apply_threshold(probabilities, OPTIMAL_THRESHOLD)

            class_labels = ["Exclusive Breastfeeding", "Formula Feeding",
                            "Mixed Feeding"]
            predicted_class = class_labels[prediction]
            confidence = probabilities[prediction]

            # Display result
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color: #0A2540; margin-bottom: 1rem;">Predicted Feeding Type at Discharge</h2>
                <h1 style="color: #1D4ED8; font-size: 2.5rem; margin: 1rem 0;">{predicted_class}</h1>
                <p style="font-size: 1.2rem; color: #475569;">
                    P(Formula) = <strong style="color: #006B7D;">{probabilities[FORMULA_CLASS_IDX]*100:.1f}%</strong>
                    &nbsp;|&nbsp; Threshold = {OPTIMAL_THRESHOLD}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Probability gauges
            st.markdown("### 📊 Probability Distribution")

            col1, col2, col3 = st.columns(3)
            colors = ["#059669", "#C2410C", "#1D4ED8"]

            for i, (label, prob, color) in enumerate(
                    zip(class_labels, probabilities, colors)):
                with [col1, col2, col3][i]:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={"text": label, "font": {"size": 14}},
                        number={"suffix": "%", "font": {"size": 32}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": color},
                            "bgcolor": "white",
                            "borderwidth": 2,
                            "bordercolor": "gray",
                            "steps": [
                                {"range": [0, 33], "color": "#F1F5F9"},
                                {"range": [33, 67], "color": "#E2E8F0"},
                                {"range": [67, 100], "color": "#CBD5E1"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": prob * 100,
                            },
                        },
                    ))
                    fig.update_layout(height=250,
                                      margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)

            # Confidence intervals
            st.markdown("### 📈 Prediction Confidence Intervals")
            st.info("""
            **Interpretation:** Bars show ±5% confidence bounds around each
            probability estimate. Wider intervals = more uncertainty.
            """)

            ci_margin = 0.05
            fig_ci = go.Figure()

            for i, (label, prob, color) in enumerate(
                    zip(class_labels, probabilities, colors)):
                lower = max(0, prob - ci_margin)
                upper = min(1, prob + ci_margin)
                fig_ci.add_trace(go.Bar(
                    y=[label], x=[prob], orientation="h",
                    name=label, marker=dict(color=color),
                    text=f"{prob*100:.1f}%", textposition="auto",
                    showlegend=False,
                ))
                fig_ci.add_trace(go.Scatter(
                    x=[lower, upper], y=[label, label],
                    mode="lines+markers",
                    line=dict(color=color, width=3),
                    marker=dict(symbol=["line-ew", "line-ew"], size=15),
                    showlegend=False,
                    hovertemplate=(f"{label}<br>Range: {lower*100:.1f}% - "
                                  f"{upper*100:.1f}%<extra></extra>"),
                ))

            fig_ci.update_layout(
                xaxis=dict(title="Probability", range=[0, 1],
                           tickformat=".0%"),
                yaxis=dict(title=""), height=300,
                margin=dict(l=10, r=10, t=30, b=40),
                plot_bgcolor="white", font=dict(size=12),
            )
            fig_ci.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
            st.plotly_chart(fig_ci, use_container_width=True)

            # Key features table
            st.markdown("### 🔍 Key Contributing Factors")
            feature_df = pd.DataFrame([
                {"Feature": "Day 1 Formula", "Value": f"{d1_formula} cc",
                 "Impact": "⭐⭐⭐"},
                {"Feature": "Day 1 Breast Milk", "Value": f"{d1_bm} cc",
                 "Impact": "⭐⭐⭐"},
                {"Feature": "Day 2 Total Intake", "Value": f"{d2_total} cc",
                 "Impact": "⭐⭐⭐"},
                {"Feature": "Birth Weight", "Value": f"{birth_weight} g",
                 "Impact": "⭐⭐"},
                {"Feature": "Gestational Age",
                 "Value": f"{ga_weeks}w + {ga_days}d",
                 "Impact": "⭐⭐"},
                {"Feature": "Maternal Age", "Value": f"{mat_age} years",
                 "Impact": "⭐"},
            ])
            st.table(feature_df)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)
    else:
        st.info("👈 Enter patient data in the 'Patient Data Entry' tab "
                "and click 'Generate Prediction' to see results.")


# ==================== TAB 3: ABOUT ====================

with tab3:
    st.markdown("### About This Tool")

    st.markdown("""
    #### Purpose
    This clinical decision support tool predicts feeding type at discharge
    (Exclusive Breastfeeding, Formula Feeding, or Mixed Feeding) for NICU
    infants based on data from the first **48 hours** of life.

    #### Model Development
    - **Algorithm:** Random Forest (Optuna hyperparameter-optimized)
    - **Objective:** F2-score for Formula class (recall-heavy)
    - **Training Dataset:** n = 1,064 NICU infants
    - **Temporal Window:** Day 1 + Day 2 (no Day 3, no COVID/Epoch)
    - **Validation:** 5-fold stratified cross-validation
    - **Threshold:** Optimized at 0.26 for Formula recall (87.5%)

    #### Key Findings from Ablation Study
    - **48-hour window is optimal** — Day 2 features add significant
      predictive lift over Day 1 alone
    - **COVID/Epoch variables are redundant** — removing them has
      negligible impact on performance (ΔF2 = 0.002)
    - **Day 3 features add noise** — MCC and Formula recall degrade
      when Day 3 is included

    #### Clinical Disclaimer
    ⚠️ **Important:** This tool **supports clinical decision-making** and
    should **not replace professional medical judgment**. All predictions
    should be validated through clinical assessment.

    #### Data Privacy
    Patient data is processed locally and is **not stored or transmitted**.
    """)

    st.markdown("""
    <div class="footer-citation">
    <strong>Citation:</strong><br>
    Ozcan, U. C., et al. (2026). Machine Learning-Based Prediction of Feeding
    Type at Discharge in NICU Infants Using Early Clinical Data.
    <em>Journal of Neonatal Medicine</em>.
    48-Hour RF Model — AUC: 0.842, Formula Recall: 87.5%.
    </div>
    """, unsafe_allow_html=True)


# ==================== TAB 4: EXPLAINABILITY ====================

with tab4:
    st.markdown("### 🔍 Model Explainability & Trust")

    with st.expander("💡 How the Model Makes Predictions", expanded=True):
        st.markdown("""
        #### How It Works

        **Think of this model like a panel of 526 expert clinicians voting:**

        1. **You provide patient data** — Birth weight, feeding volumes,
           maternal info from the first 48 hours
        2. **5 features are auto-computed** — Breast milk ratios, volume
           changes, weight-per-week
        3. **Each 'expert' votes** — All 526 decision trees predict the
           most likely outcome
        4. **Threshold applied** — If P(Formula) ≥ 0.26, the model flags
           the infant for formula risk (high-recall screening)
        5. **Final prediction** — Displayed with confidence percentages

        **Why 48 hours?**
        Our temporal ablation study showed that Day 1+2 data captures the
        most predictive signal. Day 3 data actually *hurts* performance
        (adds noise, reduces Formula recall).
        """)

    with st.expander("📊 Model Performance Evidence", expanded=False):
        st.markdown("#### Exhaustive Configuration Scan")
        st.markdown("""
        We tested **6 configurations** (3 windows × ±COVID) with fresh
        Optuna tuning on each:

        | Config | AUC | MCC | F-Recall | F2 |
        |:---|:---:|:---:|:---:|:---:|
        | Baseline (w/ COVID) | 0.819 | 0.232 | 0.964 | 0.678 |
        | +Day1 (w/ COVID) | 0.877 | 0.342 | 0.911 | 0.704 |
        | **+Day1&2 (no COVID)** | **0.842** | **0.476** | **0.875** | **0.747** |

        The deployed model (+Day1&2, no COVID) offers the best balance of
        recall, precision, and generalizability.
        """)

    with st.expander("❓ Frequently Asked Questions", expanded=False):
        st.markdown("""
        #### Q: Can I trust these predictions?
        **A:** The model was validated with 5-fold cross-validation and
        a held-out test set. It achieves 87.5% Formula recall with the
        optimized threshold. Use it as a screening tool, not a diagnosis.

        #### Q: What if I don't fill all fields?
        **A:** Empty numeric fields are filled with median values from
        training data. Provide as many values as possible for accuracy.

        #### Q: Why no Day 3 or COVID data?
        **A:** Our ablation study showed Day 3 features add noise (MCC
        and Formula recall degrade). COVID/Epoch variables are redundant
        post-pandemic — removing them has negligible impact (ΔF2 = 0.002).

        #### Q: What decisions should I NOT make with this tool?
        **A:** Do not use it to:
        - Replace individualized feeding assessments
        - Override lactation consultant recommendations
        - Make discharge decisions solely based on predictions
        - Deny breastfeeding support to mothers
        """)
