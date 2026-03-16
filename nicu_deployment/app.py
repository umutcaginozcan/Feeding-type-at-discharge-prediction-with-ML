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
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        font-family: 'Inter', sans-serif;
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
        font-size: 1.15rem; font-weight: 600; font-family: 'Courier New', monospace;
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
    .footer-citation {
        background: #F1F5F9; padding: 1rem; border-left: 3px solid #006B7D;
        font-family: 'Courier New', monospace; font-size: 0.85rem; margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== CONSTANTS ====================

FORMULA_CLASS_IDX = 1
CLASS_LABELS = ["Exclusive Breastfeeding", "Formula Feeding", "Mixed Feeding"]
CLASS_COLORS = ["#059669", "#DC2626", "#2563EB"]

# ==================== LOAD MODEL ====================

APP_DIR = Path(__file__).parent
MODEL_FILE = "final_model.pkl"


@st.cache_resource
def load_model():
    """Load the model bundle."""
    try:
        with open(APP_DIR / MODEL_FILE, "rb") as f:
            bundle = pickle.load(f)
        return bundle
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


bundle = load_model()
if bundle:
    model_pipeline = bundle["pipeline"]
    OPTIMAL_THRESHOLD = bundle["threshold"]
    MODEL_FEATURES = bundle["features"]
    TEST_METRICS = bundle.get("test_metrics", {})
    CV_METRICS = bundle.get("cv_metrics", {})
else:
    model_pipeline = None
    OPTIMAL_THRESHOLD = 0.15
    MODEL_FEATURES = []
    TEST_METRICS = {}
    CV_METRICS = {}


# ==================== HELPER FUNCTIONS ====================

def safe_val(v, default=np.nan):
    return default if v is None else float(v)


def compute_engineered(raw):
    """Compute auto-derived features from raw inputs."""
    for k in list(raw.keys()):
        raw[k] = safe_val(raw[k])

    eps = 1e-6
    bw = raw.get("dogumagirligi(gram)", np.nan)
    gw = raw.get("gebelikhaftası", np.nan)
    raw["eng_weight_per_week"] = (bw / (gw + eps)
                                   if not np.isnan(bw) and not np.isnan(gw)
                                   else np.nan)

    d1_bm = raw.get("aldığıannesütü_ilkgün", np.nan)
    d1_fm = raw.get("aldığımamamiktari1.gün", np.nan)
    raw["eng_bm_ratio_d1"] = (d1_bm / (d1_bm + d1_fm + eps)
                               if not np.isnan(d1_bm) and not np.isnan(d1_fm)
                               else np.nan)

    d2_bm = raw.get("beslenme2.gunannesutucc", np.nan)
    d2_total = raw.get("beslenmetotali2.gün", np.nan)
    raw["eng_bm_ratio_d2"] = (d2_bm / (d2_total + eps)
                               if not np.isnan(d2_bm) and not np.isnan(d2_total)
                               else np.nan)

    raw["eng_delta_vol_d1_d2"] = (
        d2_total - (d1_bm + d1_fm)
        if not any(np.isnan(x) for x in [d2_total, d1_bm, d1_fm])
        else np.nan
    )
    return raw


def apply_threshold(proba, threshold):
    """Apply Formula-specific threshold."""
    if proba[FORMULA_CLASS_IDX] >= threshold:
        return FORMULA_CLASS_IDX
    else:
        p = proba.copy()
        p[FORMULA_CLASS_IDX] = -1
        return int(np.argmax(p))


def compute_tree_ci(pipeline, input_df, confidence=0.95):
    """
    Compute confidence intervals from individual RF tree predictions.
    Returns (mean_proba, lower, upper) for each class.
    """
    # Get the preprocessor and RF from the pipeline
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    # Transform input through preprocessor
    X_processed = prep.transform(input_df)

    # Get predictions from each individual tree
    n_trees = len(clf.estimators_)
    tree_preds = np.zeros((n_trees, len(CLASS_LABELS)))

    for i, tree in enumerate(clf.estimators_):
        tree_preds[i] = tree.predict_proba(X_processed)[0]

    mean_proba = tree_preds.mean(axis=0)
    std_proba = tree_preds.std(axis=0)

    # Compute CI using t-distribution approximation
    from scipy import stats
    alpha = 1 - confidence
    t_val = stats.t.ppf(1 - alpha / 2, df=n_trees - 1)
    se = std_proba / np.sqrt(n_trees)

    lower = np.clip(mean_proba - t_val * se, 0, 1)
    upper = np.clip(mean_proba + t_val * se, 0, 1)

    return mean_proba, lower, upper, std_proba


# ==================== HEADER ====================

auc_val = TEST_METRICS.get("AUC_ROC", "—")
f_rec_val = TEST_METRICS.get("Formula_Recall", "—")
mcc_val = TEST_METRICS.get("MCC", "—")

st.markdown(f"""
<div class="main-header">
    <h1>🏥 NICU Feeding Prediction Calculator</h1>
    <p>Clinical Decision Support — 48-Hour Model (Day 1 + Day 2)</p>
    <div class="metric-row">
        <div class="metric-item">
            <div class="label">ROC-AUC</div>
            <div class="value">{auc_val}</div>
        </div>
        <div class="metric-item">
            <div class="label">Formula Recall</div>
            <div class="value">{f'{f_rec_val*100:.1f}%' if isinstance(f_rec_val, float) else f_rec_val}</div>
        </div>
        <div class="metric-item">
            <div class="label">MCC</div>
            <div class="value">{mcc_val}</div>
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


# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown("### 📊 Model Specifications")

    with st.expander("Algorithm Details", expanded=False):
        st.markdown(f"""
        **Model Type:** Random Forest (Optuna-tuned, F2-optimized)
        **Training:** 5-fold Stratified CV
        **Sample Size:** n = 1,064
        **Features:** {len(MODEL_FEATURES)} (Day 1+2, no COVID)
        **SMOTE:** Applied for class imbalance
        **Confidence Intervals:** Tree-variance (per-tree predictions)
        """)

    with st.expander("Performance Metrics", expanded=False):
        if CV_METRICS:
            st.markdown("**5-Fold Cross-Validation:**\n")
            cv_table = "| Metric | Mean ± SD |\n|:---|:---:|\n"
            for k, v in CV_METRICS.items():
                cv_table += f"| {k} | {v['mean']:.3f} ± {v['std']:.3f} |\n"
            st.markdown(cv_table)

        if TEST_METRICS:
            st.markdown(f"\n**Test Set (Threshold = {OPTIMAL_THRESHOLD}):**\n")
            test_table = "| Metric | Value |\n|:---|:---:|\n"
            for k, v in TEST_METRICS.items():
                test_table += f"| {k} | {v} |\n"
            st.markdown(test_table)

    with st.expander("Clinical Context", expanded=False):
        st.info("""
        This model predicts feeding type at discharge for NICU infants
        using data from the **first 48 hours** of life.

        **Threshold Optimization:**
        The model uses a lowered Formula threshold to maximize
        Formula recall — prioritizing identification of infants
        at risk of formula dependence.

        **Missing Values:**
        Empty fields are replaced with median values from training data.

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


# ==================== TABS ====================

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

    if "example_loaded" in st.session_state and st.session_state.example_loaded:
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
            for feat in MODEL_FEATURES:
                if feat not in data:
                    data[feat] = np.nan

            input_df = pd.DataFrame([data])[MODEL_FEATURES]

            # Predict
            probabilities = model_pipeline.predict_proba(input_df)[0]
            prediction = apply_threshold(probabilities, OPTIMAL_THRESHOLD)
            predicted_class = CLASS_LABELS[prediction]

            # Compute tree-variance confidence intervals
            mean_proba, ci_lower, ci_upper, std_proba = compute_tree_ci(
                model_pipeline, input_df)

            # Determine CSS class for prediction box
            box_class = ["ebf", "formula", "mixed"][prediction]

            # Display result
            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h2 style="color: #0A2540; margin-bottom: 0.5rem;">
                    Predicted Feeding Type at Discharge
                </h2>
                <h1 style="color: {CLASS_COLORS[prediction]}; font-size: 2.2rem; margin: 0.5rem 0;">
                    {predicted_class}
                </h1>
                <p style="font-size: 1.1rem; color: #475569;">
                    P(Formula) = <strong>{probabilities[FORMULA_CLASS_IDX]*100:.1f}%</strong>
                    &nbsp;|&nbsp; Threshold = {OPTIMAL_THRESHOLD}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Probability gauges
            st.markdown("### 📊 Class Probabilities with Confidence Intervals")
            st.caption("Intervals computed from individual tree predictions (95% CI)")

            col1, col2, col3 = st.columns(3)

            for i, (label, prob, color) in enumerate(
                    zip(CLASS_LABELS, probabilities, CLASS_COLORS)):
                with [col1, col2, col3][i]:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={"text": label, "font": {"size": 13}},
                        number={"suffix": "%", "font": {"size": 28}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": color},
                            "bgcolor": "white",
                            "borderwidth": 1,
                            "bordercolor": "#E2E8F0",
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

                    # Show CI text
                    ci_text = f"95% CI: [{ci_lower[i]*100:.1f}% – {ci_upper[i]*100:.1f}%]"
                    uncertainty = std_proba[i] * 100
                    if uncertainty < 3:
                        emoji = "🟢"
                        conf_label = "High confidence"
                    elif uncertainty < 8:
                        emoji = "🟡"
                        conf_label = "Moderate confidence"
                    else:
                        emoji = "🔴"
                        conf_label = "Low confidence"
                    st.caption(f"{emoji} {conf_label} · {ci_text}")

            # CI visualization
            st.markdown("### 📈 Confidence Interval Comparison")
            st.caption("""
            Error bars represent 95% confidence intervals derived from the variance
            across individual decision trees. Wider bars = more uncertainty for that class.
            """)

            fig_ci = go.Figure()

            for i, (label, color) in enumerate(zip(CLASS_LABELS, CLASS_COLORS)):
                fig_ci.add_trace(go.Bar(
                    y=[label], x=[probabilities[i]], orientation="h",
                    marker=dict(color=color, opacity=0.85),
                    text=f"{probabilities[i]*100:.1f}%",
                    textposition="auto",
                    textfont=dict(color="white", size=14),
                    showlegend=False,
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[ci_upper[i] - probabilities[i]],
                        arrayminus=[probabilities[i] - ci_lower[i]],
                        color="#1E293B",
                        thickness=2,
                        width=8,
                    ),
                ))

            # Threshold line
            fig_ci.add_vline(
                x=OPTIMAL_THRESHOLD, line_dash="dash",
                line_color="#DC2626", line_width=1.5,
                annotation_text=f"Formula threshold ({OPTIMAL_THRESHOLD})",
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
                plot_bgcolor="white",
                font=dict(size=12),
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
    infants using data from the first **48 hours** of life.

    #### Model Development
    - **Algorithm:** Random Forest (Optuna hyperparameter-optimized)
    - **Objective:** F2-score for Formula class (recall-heavy)
    - **Training Dataset:** n = 1,064 NICU infants
    - **Temporal Window:** Day 1 + Day 2 (no Day 3, no COVID/Epoch)
    - **Validation:** 5-fold stratified cross-validation
    - **Confidence Intervals:** Tree-variance method (per-tree prediction variance)

    #### Key Findings from Ablation Study
    - **48-hour window is optimal** — Day 2 features add significant
      predictive lift over Day 1 alone
    - **COVID/Epoch variables are redundant** — removing them has
      negligible impact on performance
    - **Simpler model, same power** — 19 features capture the essential
      clinical signal for feeding outcome prediction

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
    48-Hour RF Model with tree-variance confidence intervals.
    </div>
    """, unsafe_allow_html=True)


# ==================== TAB 4: EXPLAINABILITY ====================

with tab4:
    st.markdown("### 🔍 Model Explainability & Trust")

    with st.expander("💡 How the Model Makes Predictions", expanded=True):
        n_trees = 0
        if model_pipeline:
            try:
                n_trees = len(model_pipeline.named_steps["clf"].estimators_)
            except:
                n_trees = 130

        st.markdown(f"""
        #### How It Works

        **Think of this model like a panel of {n_trees} expert clinicians voting:**

        1. **You provide patient data** — Birth weight, feeding volumes,
           maternal info from the first 48 hours
        2. **4 features are auto-computed** — Breast milk ratios, volume
           change, weight-per-week
        3. **Each 'expert' votes** — All {n_trees} decision trees predict the
           most likely outcome
        4. **Threshold applied** — If P(Formula) ≥ {OPTIMAL_THRESHOLD}, the model flags
           the infant for formula risk (high-recall screening)
        5. **Confidence interval** — The spread of votes across trees gives
           a patient-specific uncertainty estimate

        **Why 48 hours?**
        Our temporal ablation study showed that Day 1+2 data captures the
        most predictive signal. COVID/Epoch variables are redundant
        post-pandemic.
        """)

    with st.expander("📊 How Confidence Intervals Work", expanded=False):
        st.markdown(f"""
        #### Tree-Variance Confidence Intervals

        Unlike a single point estimate, our model provides **patient-specific
        uncertainty bounds** based on the variance across {n_trees} individual
        decision trees in the Random Forest.

        **How it works:**
        - Each tree was trained on a different bootstrap sample of the data
        - For each patient, every tree gives an independent probability estimate
        - The **mean** across trees = our point prediction
        - The **standard deviation** across trees = our uncertainty measure
        - We compute a 95% CI using: mean ± t × SE

        **Interpretation:**
        - 🟢 **Narrow CI** (std < 3%): Trees agree strongly — high confidence
        - 🟡 **Moderate CI** (std 3–8%): Some disagreement — moderate confidence
        - 🔴 **Wide CI** (std > 8%): Trees disagree — the patient may be near
          a decision boundary. Exercise extra clinical judgment.
        """)

    with st.expander("❓ Frequently Asked Questions", expanded=False):
        st.markdown(f"""
        #### Q: Can I trust these predictions?
        **A:** The model was validated with 5-fold cross-validation and
        a held-out test set. Use it as a screening tool, not a diagnosis.

        #### Q: What do the confidence intervals mean?
        **A:** They show how much the {n_trees} individual trees in the
        model agree. Narrow intervals = the model is confident. Wide
        intervals = the patient is ambiguous, and clinical judgment
        should weigh more heavily.

        #### Q: What if I don't fill all fields?
        **A:** Empty numeric fields are filled with median values from
        training data. Provide as many values as possible for best accuracy.

        #### Q: Why no Day 3 or COVID data?
        **A:** Our ablation study showed Day 3 features add noise to the
        minority classes. COVID/Epoch variables are redundant post-pandemic.

        #### Q: What decisions should I NOT make with this tool?
        **A:** Do not use it to:
        - Replace individualized feeding assessments
        - Override lactation consultant recommendations
        - Make discharge decisions solely based on predictions
        - Deny breastfeeding support to mothers
        """)
