# 🏥 NICU Feeding Prediction at Discharge

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nicu-prediction.streamlit.app)

**Live app:** [nicu-prediction.streamlit.app](https://nicu-prediction.streamlit.app)

## Overview

Machine learning clinical decision support tool that predicts feeding type at discharge (**Exclusive Breastfeeding**, **Formula Feeding**, or **Mixed Feeding**) for NICU infants using early clinical data.

### Key Features

- **4 data-window models** — Baseline (admission), Day 1, Day 1+2 (recommended), Full (0–72h)
- **Per-patient explainability** — SHAP waterfall chart showing which features drove each prediction
- **Imputation transparency** — clearly reports which missing fields were filled and how
- **Confidence intervals** — tree-variance-based uncertainty from 422 Random Forest estimators

### Model Performance (Day 1+2 — Recommended)

| Metric | Value |
|---|---|
| **AUC-ROC** | 0.842 |
| **Formula Recall** | 0.875 |
| **Formula Precision** | 0.471 |
| **Optimization** | F2-score, threshold = 0.26 |

## Project Structure

```
├── nicu_deployment/            🎯 Production app (deployed to Streamlit Cloud)
│   ├── app.py                  Streamlit web application
│   ├── baseline_model.pkl      Baseline model (admission only)
│   ├── day1_model.pkl          Day 1 model
│   ├── day1_2_model.pkl        Day 1+2 model (recommended)
│   ├── full_model.pkl          Full model (0–72h)
│   └── deployment_model_configs.json
│
├── pipeline/                   💾 ML training pipeline (11 stages)
│   ├── stage2_f2_optimize.py       F2-score optimization & tuning
│   ├── stage3_evaluation.py        Model evaluation & comparison
│   ├── stage4_shap.py              SHAP explainability plots
│   ├── stage5a_ablation.py         Feature ablation study
│   ├── stage5b_stat_validation.py  Statistical validation (DeLong)
│   ├── stage5c_covid_sensitivity.py COVID sensitivity analysis
│   ├── stage5d_transition_analysis.py Feeding transition analysis
│   ├── stage6a_deploy.py           Deployment model training
│   ├── stage6b_export.py           Model export
│   ├── stage6c_final_tuning.py     Final threshold tuning
│   └── stage7_final_model.py       Final 4-model production export
│
├── scripts/                    🔧 Utility scripts
│   ├── figures/                Figure generation (trajectories, SHAP, etc.)
│   ├── tables/                 Table 1 & Table 2 generation
│   ├── analysis/               EBF epoch analyses
│   ├── fix_mislabeled_data.py
│   ├── organize_paper_structure.py
│   └── test_predictions.py
│
├── src/                        📦 Reusable Python modules
│   ├── data/loader.py          NICU data loading & encodings
│   ├── statistics/categorical.py Chi-square, Cramér's V
│   └── visualization/associations.py Heatmaps, bar charts
│
├── data/                       💾 Datasets (gitignored, sensitive)
├── .streamlit/config.toml      Streamlit theme
├── requirements.txt            Python dependencies
└── README.md
```

## Local Development

```bash
# Clone & install
git clone https://github.com/umutcaginozcan/Feeding-type-at-discharge-prediction-with-ML.git
cd Feeding-type-at-discharge-prediction-with-ML
pip install -r requirements.txt

# Run locally
streamlit run nicu_deployment/app.py
```

## Usage

1. **Select a data window** — choose how much clinical data is available (Baseline → Day 1 → Day 1+2 → Full)
2. **Enter patient data** — or click **Load Example Patient** to try a sample
3. **Generate Prediction** — view class probabilities, confidence intervals, and the SHAP explainability chart

### Required fields
- Birth Weight (g)
- Gestational Age (weeks)
- Maternal Age

All other fields are imputed with training-set medians if left empty — the app transparently discloses which fields were imputed.

## Clinical Disclaimer

⚠️ This tool **supports** clinical decision-making and should **not replace** professional medical judgment. All predictions should be considered in the context of individual patient circumstances.

## Data Privacy

Patient data is processed in-browser and is **not stored or transmitted**.

## Citation

```
...
```

## License

[Add your license information]
