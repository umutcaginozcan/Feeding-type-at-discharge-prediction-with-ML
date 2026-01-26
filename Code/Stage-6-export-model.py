#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Stage 6: Model Export for Web Deployment
----------------------------------------------
Purpose:
Train the winning Random Forest model on the full dataset and export
it as a pickle file for use in the web calculator.

Outputs:
- trained_model.pkl: The trained Random Forest pipeline
- feature_metadata.json: Feature names and metadata
- model_info.json: Model performance and training metadata
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import pickle
import json
from datetime import datetime

# Scikit-learn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier

# Imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# -------------------- CONFIGURATION --------------------

# Base directory: project root
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_stage0_5_cleaned.xlsx"
FEAT_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_selected_features.csv"

OUTPUT_MODEL_PATH = BASE_DIR / "trained_model.pkl"
OUTPUT_METADATA_PATH = BASE_DIR / "feature_metadata.json"
OUTPUT_INFO_PATH = BASE_DIR / "model_info.json"

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
N_SPLITS = 5

# Optimal Random Forest Parameters (from Stage-3)
RF_PARAMS = {
    'n_estimators': 449,
    'max_depth': 25,
    'min_samples_split': 10,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': True,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': RANDOM_STATE
}

# -------------------- LOAD DATA --------------------

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    
    print(f"Loading features from {FEAT_PATH}...")
    try:
        selected_df = pd.read_csv(FEAT_PATH)
        selected_feat_names = set(selected_df['Selected_Features'].tolist())
    except:
        print("Error reading feature file.")
        sys.exit(1)

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    y = df[TARGET_COL]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    class_names = list(le.classes_)
    
    # Map Selected Features to Raw Columns
    all_raw_cols = df.columns.tolist()
    if TARGET_COL in all_raw_cols: all_raw_cols.remove(TARGET_COL)
    
    cols_to_keep = set()
    for raw_col in all_raw_cols:
        if raw_col in selected_feat_names:
            cols_to_keep.add(raw_col)
            continue
        for sel in selected_feat_names:
            if sel.startswith(str(raw_col)):
                cols_to_keep.add(raw_col)
                break
    
    final_cols = list(cols_to_keep)
    print(f"Using {len(final_cols)} raw columns based on selection.")
    X = df[final_cols]
    
    # Split Num/Cat
    num_cols = []
    cat_cols = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            X[c] = X[c].astype(str)
            cat_cols.append(c)
            
    return X, y_enc, num_cols, cat_cols, n_classes, class_names, le

# -------------------- EVALUATE MODEL --------------------

def evaluate_model(pipeline, X, y, n_classes):
    """Perform cross-validation to get performance metrics"""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    roc_scores = []
    pr_scores = []
    acc_scores = []
    bal_acc_scores = []
    f1_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        clf_fold = clone(pipeline)
        clf_fold.fit(X_train, y_train)
        
        y_pred = clf_fold.predict(X_val)
        y_proba = clf_fold.predict_proba(X_val)
        
        # Calculate metrics
        Y_bin = label_binarize(y_val, classes=np.arange(n_classes))
        if n_classes == 2 and Y_bin.shape[1] == 1:
            Y_bin = np.hstack([1 - Y_bin, Y_bin])
        
        try:
            roc = roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro")
            roc_scores.append(roc)
        except:
            pass
        
        pr = average_precision_score(Y_bin, y_proba, average="macro")
        pr_scores.append(pr)
        
        acc_scores.append(accuracy_score(y_val, y_pred))
        bal_acc_scores.append(balanced_accuracy_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred, average="weighted"))
    
    return {
        'roc_auc_macro': {
            'mean': float(np.mean(roc_scores)),
            'std': float(np.std(roc_scores)),
            'scores': [float(s) for s in roc_scores]
        },
        'pr_auc_macro': {
            'mean': float(np.mean(pr_scores)),
            'std': float(np.std(pr_scores)),
            'scores': [float(s) for s in pr_scores]
        },
        'accuracy': {
            'mean': float(np.mean(acc_scores)),
            'std': float(np.std(acc_scores))
        },
        'balanced_accuracy': {
            'mean': float(np.mean(bal_acc_scores)),
            'std': float(np.std(bal_acc_scores))
        },
        'f1_weighted': {
            'mean': float(np.mean(f1_scores)),
            'std': float(np.std(f1_scores))
        }
    }

# -------------------- MAIN --------------------

def main():
    print("=" * 60)
    print("NICU Breastfeeding Prediction - Model Export")
    print("=" * 60)
    
    # Load data
    X, y, num_cols, cat_cols, n_classes, class_names, label_encoder = load_data()
    print(f"\nDataset: {len(X)} samples, {len(X.columns)} features")
    print(f"Classes: {class_names}")
    print(f"Numeric features: {len(num_cols)}")
    print(f"Categorical features: {len(cat_cols)}")
    
    # Build pipeline
    print("\n" + "=" * 60)
    print("Building Pipeline...")
    print("=" * 60)
    
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    
    model = RandomForestClassifier(**RF_PARAMS)
    
    pipeline = ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", model)
    ])
    
    # Evaluate with CV
    print("\n" + "=" * 60)
    print("Evaluating Model Performance (5-Fold CV)...")
    print("=" * 60)
    
    metrics = evaluate_model(pipeline, X, y, n_classes)
    
    print("\nPerformance Metrics:")
    print(f"  ROC-AUC (Macro):       {metrics['roc_auc_macro']['mean']:.3f} ± {metrics['roc_auc_macro']['std']:.3f}")
    print(f"  PR-AUC (Macro):        {metrics['pr_auc_macro']['mean']:.3f} ± {metrics['pr_auc_macro']['std']:.3f}")
    print(f"  Accuracy:              {metrics['accuracy']['mean']:.3f} ± {metrics['accuracy']['std']:.3f}")
    print(f"  Balanced Accuracy:     {metrics['balanced_accuracy']['mean']:.3f} ± {metrics['balanced_accuracy']['std']:.3f}")
    print(f"  F1-Score (Weighted):   {metrics['f1_weighted']['mean']:.3f} ± {metrics['f1_weighted']['std']:.3f}")
    
    # Train on full dataset
    print("\n" + "=" * 60)
    print("Training Final Model on Full Dataset...")
    print("=" * 60)
    
    pipeline.fit(X, y)
    print("✓ Model trained successfully")
    
    # Save model
    print("\n" + "=" * 60)
    print("Exporting Model...")
    print("=" * 60)
    
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"✓ Model saved to: {OUTPUT_MODEL_PATH}")
    
    # Save feature metadata
    feature_metadata = {
        'num_features': num_cols,
        'cat_features': cat_cols,
        'total_features': len(num_cols) + len(cat_cols),
        'class_names': [int(c) for c in class_names],  # Convert numpy int64 to Python int
        'class_labels': {
            '1': 'Exclusive Breastfeeding',
            '2': 'Formula Feeding',
            '3': 'Mixed Feeding'
        }
    }
    
    with open(OUTPUT_METADATA_PATH, 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    print(f"✓ Feature metadata saved to: {OUTPUT_METADATA_PATH}")
    
    # Save model info
    model_info = {
        'model_type': 'Random Forest',
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(X),
        'n_features': len(X.columns),
        'performance_metrics': metrics,
        'hyperparameters': RF_PARAMS,
        'random_state': RANDOM_STATE
    }
    
    with open(OUTPUT_INFO_PATH, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"✓ Model info saved to: {OUTPUT_INFO_PATH}")
    
    print("\n" + "=" * 60)
    print("✅ MODEL EXPORT COMPLETE!")
    print("=" * 60)
    print("\nYou can now use these files in the web application:")
    print(f"  - {OUTPUT_MODEL_PATH.name}")
    print(f"  - {OUTPUT_METADATA_PATH.name}")
    print(f"  - {OUTPUT_INFO_PATH.name}")
    print("\n")

if __name__ == "__main__":
    main()
