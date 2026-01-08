#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Stage 2: Bayesian Hyperparameter Tuning (Optuna)
-----------------------------------------------------
Selected Winners:
1. Random Forest (Bagging)
2. CatBoost (Boosting)
3. RBF SVM (Kernel Method)

Additions:
- optimizing for ROC-AUC (Weighted)
- After finding best params, re-evaluates to calculate MCC, F1, Brier, etc.
- Output format matches Stage 1.5 (Mean [Fold Scores])
"""

import pandas as pd
import numpy as np
from pathlib import Path
import optuna
import sys
import warnings

# Scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef, average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier

# Imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -------------------- CONFIGURATION --------------------

# Base directory: project root
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "outputs" / "nicu_stage0_5_cleaned.xlsx" 
FEAT_PATH = BASE_DIR / "outputs" / "nicu_selected_features.csv"
OUTPUT_PARAMS_PATH = BASE_DIR / "outputs" / "nicu_optuna_best_params.xlsx"

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
N_TRIALS = 50 
N_FOLDS = 5

# -------------------- UTILS --------------------

def calculate_multiclass_brier(y_true, y_prob):
    """Calculates Brier Score for Multi-class problems."""
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
    return np.mean(np.sum((y_prob - y_true_bin)**2, axis=1))

brier_scorer = make_scorer(calculate_multiclass_brier, needs_proba=True, greater_is_better=False)

def calculate_multiclass_pr_auc_macro(y_true, y_prob):
    """Macro PR-AUC (Average Precision) for multiclass (OvR)."""
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
    return average_precision_score(y_true_bin, y_prob, average="macro")

pr_auc_macro_scorer = make_scorer(
    calculate_multiclass_pr_auc_macro,
    needs_proba=True,
    greater_is_better=True
)

def calculate_multiclass_roc_auc_macro(y_true, y_prob):
    """Macro ROC-AUC for multiclass (OvR). Works even if sklearn lacks 'roc_auc_ovr_macro' alias."""
    classes = np.unique(y_true)
    if len(classes) == 2:
        # use positive class probability
        return roc_auc_score(y_true, y_prob[:, 1])
    return roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

roc_auc_macro_scorer = make_scorer(
    calculate_multiclass_roc_auc_macro,
    needs_proba=True,
    greater_is_better=True
)

def format_cell(scores, is_loss=False):
    """Returns string format: '0.850 [0.840, ...]'"""
    if is_loss: scores = np.abs(scores)
    mean_val = np.mean(scores)
    list_str = ", ".join([f"{s:.3f}" for s in scores])
    return f"{mean_val:.3f} [{list_str}]"

# -------------------- DATA LOADING --------------------

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    
    print(f"Loading features from {FEAT_PATH}...")
    try:
        selected_df = pd.read_csv(FEAT_PATH)
        selected_feat_names = set(selected_df['Selected_Features'].tolist())
    except Exception as e:
        print(f"Error reading feature file: {e}")
        sys.exit(1)

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    y = df[TARGET_COL]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
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
    
    X = df[list(cols_to_keep)]
    
    num_cols = []
    cat_cols = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            X[c] = X[c].astype(str)
            cat_cols.append(c)
            
    print(f"Data Loaded. Rows: {len(df)}, Features: {len(X.columns)}")
    return X, y_enc, num_cols, cat_cols

# -------------------- PIPELINE --------------------

def get_pipeline(model, num_cols, cat_cols, scale=False):
    from sklearn.pipeline import Pipeline as SkPipeline
    
    steps_num = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps_num.append(("scaler", StandardScaler()))
        
    numeric_transformer = SkPipeline(steps_num)
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])
    
    return ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", model)
    ])

# -------------------- RE-EVALUATION LOGIC --------------------

def evaluate_best_model(model, X, y, num_cols, cat_cols, scale=False):
    """
    Runs full CV on the optimized model to get all metrics.
    """
    pipeline = get_pipeline(model, num_cols, cat_cols, scale=scale)
    
    scoring = {
        'ROC_AUC_macro': roc_auc_macro_scorer,
        'PR__AUC_macro': pr_auc_macro_scorer,
        'Accuracy': 'accuracy',
        'Balanced_Accuracy': 'balanced_accuracy',
        'F1_weighted': 'f1_weighted',
        'F1_macro': 'f1_macro',
        'MCC': make_scorer(matthews_corrcoef),
        'Brier_score': brier_scorer
    }
    
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    metrics_row = {}
    for metric, scores in results.items():
        if metric.startswith('test_'):
            metric_name = metric.replace('test_', '')
            is_loss = (metric_name == 'Brier_score')
            metrics_row[metric_name] = format_cell(scores, is_loss=is_loss)
            
    return metrics_row

# -------------------- OBJECTIVES --------------------

def objective_rf(trial, X, y, num_cols, cat_cols):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    }
    model = RandomForestClassifier(**param)
    pipeline = get_pipeline(model, num_cols, cat_cols, scale=False)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc_ovr_weighted', n_jobs=-1).mean()

def objective_cat(trial, X, y, num_cols, cat_cols):
    param = {
        'iterations': trial.suggest_int('iterations', 500, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
        'border_count': trial.suggest_categorical('border_count', [32, 64, 128]),
        'verbose': False, 
        'allow_writing_files': False,
        'random_state': RANDOM_STATE,
        'thread_count': -1
    }
    model = CatBoostClassifier(**param)
    pipeline = get_pipeline(model, num_cols, cat_cols, scale=False)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc_ovr_weighted', n_jobs=-1).mean()

def objective_svm(trial, X, y, num_cols, cat_cols):
    param = {
        'C': trial.suggest_float('C', 0.1, 100, log=True),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'kernel': 'rbf',
        'class_weight': 'balanced',
        'probability': True,
        'random_state': RANDOM_STATE
    }
    model = SVC(**param)
    pipeline = get_pipeline(model, num_cols, cat_cols, scale=True)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc_ovr_weighted', n_jobs=-1).mean()

# -------------------- MAIN --------------------

def main():
    print("--- Starting Stage 2: Optuna Tuning with Extended Metrics ---")
    X, y, num_cols, cat_cols = load_data()
    
    results_list = []
    
    # 1. Random Forest
    print(f"\n>>> Optimizing Random Forest ({N_TRIALS} trials)...")
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(lambda t: objective_rf(t, X, y, num_cols, cat_cols), n_trials=N_TRIALS)
    print("   Evaluating best RF model on all metrics...")
    
    best_rf_params = study_rf.best_params
    best_rf_params.update({'class_weight': 'balanced', 'n_jobs': -1, 'random_state': RANDOM_STATE})
    rf_model = RandomForestClassifier(**best_rf_params)
    
    rf_metrics = evaluate_best_model(rf_model, X, y, num_cols, cat_cols, scale=False)
    
    row_rf = {"Model": "Random Forest"}
    row_rf.update(rf_metrics) # Add MCC, F1, etc.
    row_rf.update(best_rf_params) # Add Best Params
    results_list.append(row_rf)
    
    # 2. CatBoost
    print(f"\n>>> Optimizing CatBoost ({N_TRIALS} trials)...")
    study_cat = optuna.create_study(direction='maximize')
    study_cat.optimize(lambda t: objective_cat(t, X, y, num_cols, cat_cols), n_trials=N_TRIALS)
    print("   Evaluating best CatBoost model on all metrics...")
    
    best_cat_params = study_cat.best_params
    best_cat_params.update({'verbose': False, 'allow_writing_files': False, 'random_state': RANDOM_STATE, 'thread_count': -1})
    cat_model = CatBoostClassifier(**best_cat_params)
    
    cat_metrics = evaluate_best_model(cat_model, X, y, num_cols, cat_cols, scale=False)
    
    row_cat = {"Model": "CatBoost"}
    row_cat.update(cat_metrics)
    row_cat.update(best_cat_params)
    results_list.append(row_cat)

    # 3. RBF SVM
    print(f"\n>>> Optimizing RBF SVM ({N_TRIALS} trials)...")
    study_svm = optuna.create_study(direction='maximize')
    study_svm.optimize(lambda t: objective_svm(t, X, y, num_cols, cat_cols), n_trials=N_TRIALS)
    print("   Evaluating best SVM model on all metrics...")
    
    best_svm_params = study_svm.best_params
    best_svm_params.update({'kernel': 'rbf', 'class_weight': 'balanced', 'probability': True, 'random_state': RANDOM_STATE})
    svm_model = SVC(**best_svm_params)
    
    svm_metrics = evaluate_best_model(svm_model, X, y, num_cols, cat_cols, scale=True)
    
    row_svm = {"Model": "RBF SVM"}
    row_svm.update(svm_metrics)
    row_svm.update(best_svm_params)
    results_list.append(row_svm)
    
    # Save
    print(f"\nSaving detailed results to {OUTPUT_PARAMS_PATH}...")
    df_res = pd.DataFrame(results_list)
    
    # Reorder columns: Model -> Metrics -> Params
    metric_cols = ['Model', 'ROC_AUC_macro', 'PR__AUC_macro', 'Accuracy', 'Balanced_Accuracy',
               'F1_weighted', 'F1_macro', 'MCC', 'Brier_score']
    param_cols = [c for c in df_res.columns if c not in metric_cols]
    df_res = df_res[metric_cols + param_cols]
    
    df_res.to_excel(OUTPUT_PARAMS_PATH, index=False)
    print("DONE. Ready for Stage 3.")

if __name__ == "__main__":
    main()