# Project Organization Guide

## 📁 New Folder Structure

```
NICU Breastfeeding Paper/
│
├── 🎯 CORE APPLICATION
│   ├── streamlit_app.py          # Main Streamlit web app (ACTIVE)
│   ├── trained_model.pkl          # Trained Random Forest model
│   ├── feature_metadata.json      # Feature definitions
│   ├── model_info.json            # Model performance metrics
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # Main documentation
│
├── 📊 ASSETS (Deployable Images)
│   ├── model_plots/               # Performance visualizations
│   │   ├── roc_curve.png
│   │   ├── confusion_matrix.png
│   │   └── calibration.png
│   └── shap_plots/                # Feature importance plots
│       ├── overall_importance.png
│       ├── ebf_importance.png
│       ├── formula_importance.png
│       └── mixed_importance.png
│
├── 💾 CODE (Training Pipeline)
│   ├── Stage-0.py                 # Data cleaning
│   ├── Stage-1.py                 # Feature engineering
│   ├── Stage-2.py                 # Model training
│   ├── Stage-3.py                 # Model evaluation
│   ├── Stage-4.py                 # Hyperparameter tuning
│   ├── Stage-5.py                 # Final model selection
│   └── Stage-6-export-model.py    # Export for production
│
├── 📚 DOCS (Documentation)
│   └── FEATURE_REMOVAL_GUIDE.md   # How to remove features
│
├── 📦 SCRIPTS (Utility Scripts)
│   ├── test_predictions.py        # Testing script
│   ├── maintenance/               # One-time maintenance scripts
│   │   ├── create_guide.py
│   │   ├── fix_streamlit.py
│   │   └── update_streamlit.py
│   └── legacy/                    # Old versions (Flask, HTML)
│       ├── app.py                 # Flask web app
│       ├── calculator.html        # Standalone HTML calculator
│       └── calculator_screenshot.png
│
├── 💼 BACKUPS
│   ├── streamlit_app.py.backup
│   └── streamlit_app.py.backup2
│
├── 📊 OUTPUTS (All Experimental Results)
│   ├── model plots/               # All model visualizations
│   ├── shap plots/                # All SHAP analyses
│   ├── model performances/        # Performance comparisons
│   ├── paper plots/               # Publication-ready plots
│   └── statistics/                # Statistical analyses
│
├── 📁 EXCELS-NICU-BREATSFEEDING-DATA (Training Data)
│   └── nicu_stage0_5_cleaned.xlsx
│
└── ⚙️ CONFIG
    ├── .streamlit/                # Streamlit configuration
    ├── .gitignore                 # Git ignore rules
    ├── .devcontainer/             # Dev container setup
    └── .vscode/                   # VS Code settings
```

## 📖 Folder Purposes

### Core Application
**Active production files** - Don't modify unless you know what you're doing!

### Assets
**Images that deploy with the app** - These load in the Streamlit app's Explainability tab.

### Code
**Model training pipeline** - Run these scripts to retrain the model with new data or features.

### Docs
**User guides and documentation** - Reference materials for using/modifying the model.

### Scripts
- **scripts/** - Active utility scripts
- **scripts/maintenance/** - One-time scripts used during development
- **scripts/legacy/** - Old versions of the web app (Flask, HTML)

### Backups
**Safety copies** - Automatic backups created during major changes.

### Outputs
**Complete experimental results** - All plots and analyses from model development.

### Excels-NICU-Breatsfeeding-Data
**Training dataset** - The cleaned NICU patient data used to train the model.

### Config
**System configuration files** - IDE and deployment settings.

---

## 🎯 Quick Reference

**To run the app locally:**
```bash
streamlit run streamlit_app.py
```

**To retrain the model:**
```bash
cd Code
python Stage-6-export-model.py
```

**To test predictions:**
```bash
python scripts/test_predictions.py
```

**To modify features:**
See `docs/FEATURE_REMOVAL_GUIDE.md`

---

## 🧹 Maintenance

**Backups:** Created automatically during major updates. Safe to delete old ones after verifying new version works.

**Scripts/maintenance:** One-time scripts. Safe to archive or delete after project is stable.

**Scripts/legacy:** Old web app versions. Keep for reference or delete if you only use Streamlit version.

---

## 🚀 Deployment

**Streamlit Cloud automatically uses:**
- Root directory files (streamlit_app.py, model files, requirements.txt)
- assets/ folder (for images)
- .streamlit/ folder (for configuration)

**Not deployed:**
- Code/ (training scripts)
- scripts/ (utility scripts)
- backups/ (backup files)
- outputs/ (experimental results)
- excels-NICU-breatsfeeding-data/ (raw data)

---

## 📊 Storage Info

**Essential files (deployed):** ~12 MB
- Model: 11.6 MB
- Images: 2.8 MB
- Code: <1 MB

**Development files (local only):** ~450 MB
- Excel data: ~450 KB
- Outputs: ~67 files
- Backups: ~70 KB

---

**Last organized:** 2026-02-08
**Structure version:** 1.0
