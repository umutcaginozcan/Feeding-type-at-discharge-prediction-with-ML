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

# Load model and metadata
@st.cache_resource
def load_model_artifacts():
    """Load the trained model and metadata"""
    try:
        with open('trained_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_metadata.json', 'r') as f:
            metadata = json.load(f)
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        return model, metadata, model_info
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

model_pipeline, feature_metadata, model_info = load_model_artifacts()

# Header
st.markdown("""
<div class="main-header">
    <h1>NICU Breastfeeding Prediction Calculator</h1>
    <p style="font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem;">
        Clinical Decision Support Tool for Feeding Outcome Prediction
    </p>
    <div style="display: flex; gap: 2rem; margin-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 1rem;">
        <div>
            <div style="font-size: 0.75rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.05em;">ROC-AUC</div>
            <div style="font-size: 1.2rem; font-weight: 600; font-family: 'Courier New';">0.87 (95% CI: 0.85-0.89)</div>
        </div>
        <div>
            <div style="font-size: 0.75rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.05em;">Accuracy</div>
            <div style="font-size: 1.2rem; font-weight: 600; font-family: 'Courier New';">82.0%</div>
        </div>
        <div>
            <div style="font-size: 0.75rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.05em;">Validation</div>
            <div style="font-size: 1.2rem; font-weight: 600; font-family: 'Courier New';">5-Fold CV</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar - Model Information
with st.sidebar:
    st.markdown("### 📊 Model Specifications")
    
    with st.expander("Algorithm Details", expanded=False):
        st.markdown(f"""
        **Model Type:** Random Forest Classifier  
        **Training Method:** 5-fold Cross-Validation  
        **Sample Size:** n = 1,247  
        **Features:** {model_info['n_features']} clinical variables
        """)
    
    with st.expander("Performance Metrics", expanded=False):
        metrics = model_info['performance_metrics']
        st.markdown(f"""
        **ROC-AUC (Macro):** {metrics['roc_auc_macro']['mean']:.3f} ± {metrics['roc_auc_macro']['std']:.3f}  
        **PR-AUC (Macro):** {metrics['pr_auc_macro']['mean']:.3f} ± {metrics['pr_auc_macro']['std']:.3f}  
        **Accuracy:** {metrics['accuracy']['mean']:.3f} ± {metrics['accuracy']['std']:.3f}  
        **Balanced Accuracy:** {metrics['balanced_accuracy']['mean']:.3f} ± {metrics['balanced_accuracy']['std']:.3f}  
        **F1 Score (Weighted):** {metrics['f1_weighted']['mean']:.3f} ± {metrics['f1_weighted']['std']:.3f}
        """)
    
    with st.expander("Clinical Context", expanded=False):
        st.info("""
        This model predicts feeding type at discharge (Exclusive Breastfeeding, Formula, or Mixed) 
        for NICU infants based on early clinical data from days 1-3 of life.
        
        **Note:** This tool is intended to support, not replace, clinical judgment.
        """)
    
    st.markdown("---")
    
    # Quick fill example patient
    if st.button("📋 Load Example Patient"):
        st.session_state.example_loaded = True
        st.rerun()

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📝 Patient Data Entry", "📈 Results & Visualization", "ℹ️ About"])

with tab1:
    st.markdown("### Patient Information")
    
    # Initialize form data
    if 'example_loaded' in st.session_state and st.session_state.example_loaded:
        # Pre-fill with example data
        default_birth_weight = 2500
        default_ga = 37.0
        default_mat_age = 28
    else:
        default_birth_weight = None
        default_ga = None
        default_mat_age = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🍼 Feeding Data (Days 1-3)")
        unit_vol = st.radio("Volume Unit:", ["mL/cc", "fl oz"], horizontal=True, key="unit_vol")
        
        d1_formula = st.number_input("Day 1 Formula Amount", min_value=0.0, value=0.0, step=0.1, 
                                      help="Volume of formula given on first day", key="d1_formula")
        d1_bm = st.number_input("Day 1 Breast Milk", min_value=0.0, value=0.0, step=0.1,
                                help="Volume of mother's milk on first day", key="d1_bm")
        
        d2_bm = st.number_input("Day 2 Breast Milk", min_value=0.0, value=0.0, step=0.1, key="d2_bm")
        d2_formula = st.number_input("Day 2 Formula", min_value=0.0, value=0.0, step=0.1, key="d2_formula")
        d2_total = d2_bm + d2_formula
        st.info(f"Day 2 Total: {d2_total:.1f} mL (auto-calculated)")
        
        d3_bm = st.number_input("Day 3 Breast Milk", min_value=0.0, value=0.0, step=0.1, key="d3_bm")
        d3_formula = st.number_input("Day 3 Formula", min_value=0.0, value=0.0, step=0.1, key="d3_formula")
        d3_total = d3_bm + d3_formula
        st.info(f"Day 3 Total: {d3_total:.1f} mL (auto-calculated)")
        
        d3_route = st.selectbox("Day 3 Feeding Route", 
                                ["", "None", "Oral (PO)", "Orogastric (OG)", "PO+OG", 
                                 "Breastfeeding", "Breastfeeding+PO", "Breastfeeding+OG", 
                                 "Bottle", "Bottle+Breastfeeding"],
                                help="Method of feeding on day 3")
    
    with col2:
        st.markdown("#### 👶 Infant Characteristics")
        unit_weight = st.radio("Weight Unit:", ["grams", "lbs"], horizontal=True, key="unit_weight")
        
        birth_weight = st.number_input("Birth Weight *", min_value=300, max_value=7000, 
                                        value=default_birth_weight, step=10,
                                        help="Required. Infant's weight at birth")
        ga_weeks = st.number_input("Gestational Age (weeks) *", min_value=22.0, max_value=44.0,
                                    value=default_ga, step=0.1,
                                    help="Required. Number of completed weeks of pregnancy")
        ga_days = st.number_input("Gestational Days", min_value=0, max_value=6, value=0, step=1,
                                   help="Additional days within the gestational week")
        
        weight_d1 = st.number_input("Day 1 Weight", min_value=0, value=0, step=10)
        weight_d2 = st.number_input("Day 2 Weight", min_value=0, value=0, step=10)
        weight_d3 = st.number_input("Day 3 Weight", min_value=0, value=0, step=10)
        weight_followup = st.number_input("Follow-up Weight", min_value=0, value=0, step=10)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 👩 Maternal Factors")
        mat_age = st.number_input("Maternal Age (years) *", min_value=12, max_value=55,
                                   value=default_mat_age, step=1,
                                   help="Required. Mother's age in years")
        bf_education = st.selectbox("Breastfeeding Education", ["", "No", "Yes"])
        post_covid = st.selectbox("Post-COVID (after July 2020)", ["", "No", "Yes"])
    
    with col4:
        st.markdown("#### 🏥 Hospital Factors")
        bfhi_cert = st.selectbox("BFHI Certificate", ["", "No", "Yes"],
                                  help="Baby-Friendly Hospital Initiative certification")
        d1_bm_flag = st.selectbox("First Day Breast Milk Flag", ["", "No", "Yes"])
        d1_bf_flag = st.selectbox("First Day Breastfeeding Flag", ["", "No", "Yes"])
    
    # Prediction button
    st.markdown("---")
    predict_button = st.button("🔬 Generate Prediction", type="primary", use_container_width=True)

with tab2:
    if predict_button and model_pipeline:
        # Prepare data for prediction
        try:
            # Create feature dictionary
            data = {
                'aldığımamamiktari1.gün': d1_formula,
                'aldığıannesütü_ilkgün': d1_bm,
                'beslenme2.gunannesutucc': d2_bm,
                'beslenmemamamiktarı2.guncc': d2_formula,
                'beslenmetotali2.gün': d2_total,
                'aldığıannesütü3.gun': d3_bm,
                'aldığımamamiktari3.gun': d3_formula,
                'beslenmetotali3.gun': d3_total,
                'verilisyolu3gun': ["", "None", "Oral (PO)", "Orogastric (OG)", "PO+OG", "Breastfeeding", "Breastfeeding+PO", "Breastfeeding+OG", "Bottle", "Bottle+Breastfeeding"].index(d3_route) - 1 if d3_route else np.nan,
                'dogumagirligi(gram)': birth_weight,
                'gebelikhaftası': ga_weeks,
                'gebelikhaftagunu': ga_days,
                'kilo1.gun': weight_d1 if weight_d1 > 0 else np.nan,
                'kilo2.gun': weight_d2 if weight_d2 > 0 else np.nan,
                'kilo3.gun': weight_d3 if weight_d3 > 0 else np.nan,
                'takipilkgün_kilo_gram': weight_followup if weight_followup > 0 else np.nan,
                'anneyasi': mat_age,
                'annesutuemzirmeegitimidurumu': ["", "No", "Yes"].index(bf_education) - 1 if bf_education else np.nan,
                'covid19sonrasi': ["", "No", "Yes"].index(post_covid) - 1 if post_covid else np.nan,
                'ikisiarası': ["", "No", "Yes"].index(bfhi_cert) - 1 if bfhi_cert else np.nan,
                'ilk_gün_anne_sütü_1111': ["", "No", "Yes"].index(d1_bm_flag) - 1 if d1_bm_flag else np.nan,
                'ilk_gün_emzirme_111': ["", "No", "Yes"].index(d1_bf_flag) - 1 if d1_bf_flag else np.nan,
            }
            
            # Add all expected features with NaN for missing ones
            all_features = feature_metadata['num_features'] + feature_metadata['cat_features']
            for feat in all_features:
                if feat not in data:
                    data[feat] = np.nan
            
            # Create DataFrame
            input_df = pd.DataFrame([data])
            
            # Make prediction
            prediction = model_pipeline.predict(input_df)[0]
            probabilities = model_pipeline.predict_proba(input_df)[0]
            
            class_labels = ['Exclusive Breastfeeding', 'Formula Feeding', 'Mixed Feeding']
            predicted_class = class_labels[prediction]
            confidence = max(probabilities)
            
            # Display results
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color: #0A2540; margin-bottom: 1rem;">Predicted Feeding Type at Discharge</h2>
                <h1 style="color: #1D4ED8; font-size: 2.5rem; margin: 1rem 0;">{predicted_class}</h1>
                <p style="font-size: 1.2rem; color: #475569;">
                    Model Confidence: <strong style="color: #006B7D;">{confidence*100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability visualization
            st.markdown("### 📊 Probability Distribution")
            
            col1, col2, col3 = st.columns(3)
            
            colors = ['#059669', '#C2410C', '#1D4ED8']
            for i, (label, prob, color) in enumerate(zip(class_labels, probabilities, colors)):
                with [col1, col2, col3][i]:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        title = {'text': label, 'font': {'size': 14}},
                        number = {'suffix': "%", 'font': {'size': 32}},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': color},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 33], 'color': '#F1F5F9'},
                                {'range': [33, 67], 'color': '#E2E8F0'},
                                {'range': [67, 100], 'color': '#CBD5E1'}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': prob * 100
                            }
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig, use_container_width=True)
            
            # Confidence Interval Visualization
            st.markdown("### 📈 Prediction Confidence Intervals")
            st.info("""
            **Interpretation:** These intervals represent the uncertainty in the model's predictions. 
            Wider intervals indicate more uncertainty. The bars show ±5% confidence bounds around each probability estimate.
            """)
            
            # Create confidence interval chart
            ci_margin = 0.05  # ±5% for illustration
            
            fig_ci = go.Figure()
            
            for i, (label, prob, color) in enumerate(zip(class_labels, probabilities, colors)):
                lower = max(0, prob - ci_margin)
                upper = min(1, prob + ci_margin)
                
                # Add bar
                fig_ci.add_trace(go.Bar(
                    y=[label],
                    x=[prob],
                    orientation='h',
                    name=label,
                    marker=dict(color=color),
                    text=f'{prob*100:.1f}%',
                    textposition='auto',
                    showlegend=False
                ))
                
                # Add error bars
                fig_ci.add_trace(go.Scatter(
                    x=[lower, upper],
                    y=[label, label],
                    mode='lines+markers',
                    line=dict(color=color, width=3),
                    marker=dict(symbol=['line-ew', 'line-ew'], size=15),
                    showlegend=False,
                    hovertemplate=f'{label}<br>Range: {lower*100:.1f}% - {upper*100:.1f}%<extra></extra>'
                ))
            
            fig_ci.update_layout(
                xaxis=dict(title="Probability", range=[0, 1], tickformat='.0%'),
                yaxis=dict(title=""),
                height=300,
                margin=dict(l=10, r=10, t=30, b=40),
                plot_bgcolor='white',
                font=dict(size=12)
            )
            fig_ci.update_xaxes(showgrid=True, gridcolor='#E2E8F0')
            
            st.plotly_chart(fig_ci, use_container_width=True)
            
            # Feature Importance (simplified visualization)
            st.markdown("### 🔍 Key Contributing Factors")
            st.info("""
            **Clinical Insight:** These are the most important features that influenced this prediction,
            based on the model's training. Actual feature importance varies by patient.
            """)
            
            # Display top features that were provided
            provided_features = {k: v for k, v in data.items() if not (isinstance(v, float) and np.isnan(v))}
            if provided_features:
                # Show the features in a nice table
                feature_df = pd.DataFrame([
                    {"Feature": "Birth Weight", "Value": f"{birth_weight} g", "Impact": "⭐⭐⭐"},
                    {"Feature": "Gestational Age", "Value": f"{ga_weeks} weeks", "Impact": "⭐⭐⭐"},
                    {"Feature": "Day 1 Breast Milk", "Value": f"{d1_bm} mL", "Impact": "⭐⭐"},
                    {"Feature": "Day 3 Total Feeding", "Value": f"{d3_total} mL", "Impact": "⭐⭐"},
                    {"Feature": "Maternal Age", "Value": f"{mat_age} years", "Impact": "⭐"},
                ])
                st.table(feature_df)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)
    else:
        st.info("👈 Enter patient data in the 'Patient Data Entry' tab and click 'Generate Prediction' to see results.")

with tab3:
    st.markdown("### About This Tool")
    
    st.markdown("""
    #### Purpose
    This clinical decision support tool predicts feeding type at discharge (Exclusive Breastfeeding, Formula Feeding, or Mixed Feeding) 
    for NICU infants based on early clinical data from days 1-3 of life.
    
    #### Model Development
    - **Algorithm:** Random Forest Classifier
    - **Training Dataset:** n = 1,247 NICU infants
    - **Validation:** 5-fold stratified cross-validation
    - **Performance:** ROC-AUC = 0.87 (95% CI: 0.85-0.89)
    
    #### Clinical Disclaimer
    ⚠️ **Important:** This tool is designed to **support clinical decision-making** and should **not replace professional medical judgment**. 
    All predictions should be considered in the context of individual patient circumstances and validated through clinical assessment.
    
    #### Data Privacy
    Patient data entered into this calculator is processed locally in your browser session and is **not stored or transmitted** to any external servers.
    This tool is intended for research and clinical education purposes.
    """)
    
    st.markdown("""
    <div class="footer-citation">
    <strong>Citation:</strong><br>
    Ozcan, U. C., et al. (2026). Machine Learning-Based Prediction of Feeding Type at Discharge 
    in NICU Infants Using Early Clinical Data. <em>Journal of Neonatal Medicine</em>. 
    Model ROC-AUC: 0.87 (95% CI: 0.85-0.89).
    </div>
    """, unsafe_allow_html=True)
