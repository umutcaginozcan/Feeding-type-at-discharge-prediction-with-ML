# Feature Removal Guide

## Overview
Removing features from the NICU Breastfeeding Prediction model is straightforward but requires retraining.

## When to Remove Features

Common scenarios:
- **Temporal relevance**: Features like `covid19sonrasi` (post-COVID flag) may become outdated
- **Data availability**: Some features may not be routinely collected
- **Model simplification**: Reducing complexity while maintaining performance

## Step-by-Step Process

### Step 1: Identify Features to Remove

Check current features in `feature_metadata.json`:
```bash
cat feature_metadata.json
```

Example features to consider removing:
- `covid19sonrasi` - COVID-19 temporal flag (may be irrelevant in future)
- Engineered features if you want a simpler model

### Step 2: Update Feature Selection

Edit the feature selection file:
```bash
# Location depends on your training pipeline
# Usually in: excels-NICU-breatsfeeding-data/nicu_selected_features.csv
```

Remove rows corresponding to features you want to exclude.

### Step 3: Retrain the Model

```bash
cd Code
python Stage-6-export-model.py
```

This regenerates:
- `trained_model.pkl` - Updated model without removed features
- `feature_metadata.json` - Updated feature list
- `model_info.json` - Updated performance metrics

### Step 4: Update Streamlit App

If you removed user-input features, update `streamlit_app.py`:

1. Remove corresponding input fields (e.g., lines ~240-300)
2. Remove from data dictionary (lines ~308-330)
3. Test the app locally

### Step 5: Test Locally

```bash
streamlit run streamlit_app.py
```

Verify:
- ✅ App loads without errors
- ✅ Predictions still work
- ✅ No references to removed features

### Step 6: Compare Performance

Check `model_info.json` for new metrics:
```python
import json
with open('model_info.json') as f:
    info = json.load(f)
print(f"ROC-AUC: {info['performance_metrics']['roc_auc_macro']['mean']}")
print(f"Accuracy: {info['performance_metrics']['accuracy']['mean']}")
```

### Step 7: Deploy

```bash
git add trained_model.pkl feature_metadata.json model_info.json streamlit_app.py
git commit -m "Remove outdated features and retrain model"
git push
```

Streamlit Cloud will automatically redeploy.

## Impact Assessment

### Minimal Impact Scenarios
- Features with low importance (check SHAP plots)
- Redundant engineered features
- Temporal flags (like COVID)

### Potential Impact Scenarios
- High-importance features (top 10 in SHAP)
- Core clinical variables (birth weight, GA, feeding volumes)

## Best Practices

1. **Check feature importance first**
   - View SHAP plots in the Explainability tab
   - Remove only low-importance features

2. **Document changes**
   - Note why features were removed
   - Record performance before/after

3. **Validate thoroughly**
   - Test on held-out data
   - Compare metrics to previous version
   - Ensure clinical validity

4. **Communicate changes**
   - Update README.md
   - Update app documentation
   - Notify users of changes

## Pipeline Flexibility

The preprocessing pipeline is designed to be flexible:

```
Input Data → ColumnTransformer → Imputer → SMOTE → Random Forest
```

- **ColumnTransformer**: Automatically adapts to feature list
- **Imputer**: Uses median of available features
- **SMOTE**: Rebalances based on actual data
- **Random Forest**: Adjusts to feature count

**Result**: No breaking changes when features are removed, only retraining needed.

## Example: Removing COVID Feature

```bash
# 1. Check current performance
cat model_info.json | grep roc_auc

# 2. Edit feature list (remove covid19sonrasi)
# ... edit file ...

# 3. Retrain
cd Code
python Stage-6-export-model.py

# 4. Check new performance
cat model_info.json | grep roc_auc

# 5. If acceptable, deploy
git add -A
git commit -m "Remove COVID-19 temporal feature"
git push
```

## Troubleshooting

**Error: Feature not found in dataset**
- Solution: Ensure training data still contains all features in selection list

**Error: Model performance degraded significantly**
- Solution: Feature was important, consider keeping it or finding alternative

**Error: Streamlit app crashes**
- Solution: Update app to remove input fields for removed features

## Questions?

For issues or questions about feature removal, refer to:
- Model training scripts in `/Code`
- Feature metadata in `feature_metadata.json`
- SHAP plots for feature importance
