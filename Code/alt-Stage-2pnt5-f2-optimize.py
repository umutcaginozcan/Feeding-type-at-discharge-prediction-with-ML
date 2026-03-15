#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Alt-Stage 2.5 v2: State-of-the-Art Formula-Aware Model
-------------------------------------------------------------
Three complementary techniques:
  1. Optuna tuning with Formula F2-score (recall-heavy but precision-aware)
  2. Probability threshold optimization per class
  3. Stacking ensemble (meta-learner on base model probabilities)

Models: Random Forest, CatBoost, KNN, + Stacking Ensemble
Goal:  Maximize Formula recall while keeping precision reasonable.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import optuna
import sys
import warnings
import time
import copy

# Scikit-learn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, train_test_split
)
from sklearn.metrics import (
    make_scorer, matthews_corrcoef,
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, fbeta_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_stage0_5_cleaned.xlsx"
FEAT_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_selected_features.csv"
OUTPUT_FILE = Path.home() / "Desktop" / "nicu_alt_stage2pnt5_v2_results.xlsx"

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
N_TRIALS = 50
N_FOLDS = 5

# After LabelEncoder (alphabetical): Exclusive BF=0, Formula=1, Mixed=2
FORMULA_CLASS_IDX = 1

# ==================== UTILS ====================

def format_cell(scores):
    """Returns 'mean [s1, s2, ...]' or 'val [val]' for single values."""
    if isinstance(scores, (float, np.floating, int)):
        return f"{float(scores):.3f}"
    mean_val = np.mean(scores)
    list_str = ", ".join([f"{s:.3f}" for s in scores])
    return f"{mean_val:.3f} [{list_str}]"


def formula_f2_scorer_fn(y_true, y_pred):
    """F2-score for the Formula class (β=2: recall weighted 2× vs precision)."""
    return fbeta_score(y_true, y_pred, beta=2,
                       labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0)

formula_f2_scorer = make_scorer(formula_f2_scorer_fn)


def compute_all_metrics(y_true, y_pred, le, prefix=""):
    """Compute comprehensive metrics including Formula-specific."""
    n_classes = len(le.classes_)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))

    tp = cm[FORMULA_CLASS_IDX, FORMULA_CLASS_IDX]
    fn = cm[FORMULA_CLASS_IDX, :].sum() - tp
    fp = cm[:, FORMULA_CLASS_IDX].sum() - tp
    tn = cm.sum() - tp - fn - fp

    m = {}
    m[f"{prefix}Accuracy"] = accuracy_score(y_true, y_pred)
    m[f"{prefix}Balanced_Acc"] = balanced_accuracy_score(y_true, y_pred)
    m[f"{prefix}F1_Weighted"] = f1_score(y_true, y_pred, average='weighted')
    m[f"{prefix}F1_Macro"] = f1_score(y_true, y_pred, average='macro')
    m[f"{prefix}MCC"] = matthews_corrcoef(y_true, y_pred)
    m[f"{prefix}Precision_Weighted"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    m[f"{prefix}Recall_Weighted"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    m[f"{prefix}Formula_Recall"] = recall_score(y_true, y_pred, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0)
    m[f"{prefix}Formula_Precision"] = precision_score(y_true, y_pred, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0)
    m[f"{prefix}Formula_F2"] = formula_f2_scorer_fn(y_true, y_pred)
    m[f"{prefix}Formula_Accuracy"] = (tp + tn) / cm.sum() if cm.sum() > 0 else 0.0
    return m, cm


def apply_threshold(y_proba, threshold_formula, n_classes):
    """
    Custom argmax with a lowered threshold for Formula.
    If P(Formula) > threshold_formula, predict Formula.
    Otherwise, standard argmax among remaining classes.
    """
    y_pred = np.zeros(len(y_proba), dtype=int)
    for i in range(len(y_proba)):
        if y_proba[i, FORMULA_CLASS_IDX] >= threshold_formula:
            y_pred[i] = FORMULA_CLASS_IDX
        else:
            # Among non-Formula classes, pick argmax
            probs_adj = y_proba[i].copy()
            probs_adj[FORMULA_CLASS_IDX] = -1  # exclude Formula
            y_pred[i] = np.argmax(probs_adj)
    return y_pred


# ==================== DATA LOADING ====================

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
    if TARGET_COL in all_raw_cols:
        all_raw_cols.remove(TARGET_COL)

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

    print(f"Data: {len(df)} rows, {len(X.columns)} features")
    print(f"Classes: {dict(zip(le.classes_, np.bincount(y_enc)))}")
    return X, y_enc, num_cols, cat_cols, le


# ==================== PIPELINE ====================

def get_preprocessor(num_cols, cat_cols, scale=False):
    steps_num = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps_num.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("num", SkPipeline(steps_num), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])


def get_pipeline(model, num_cols, cat_cols, scale=False):
    return ImbPipeline([
        ("prep", get_preprocessor(num_cols, cat_cols, scale=scale)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", model)
    ])


# ==================== OPTUNA OBJECTIVES (F2-optimized) ====================

def objective_rf(trial, X, y, num_cols, cat_cols):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 5, 40),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    }
    model = RandomForestClassifier(**param)
    pipeline = get_pipeline(model, num_cols, cat_cols, scale=False)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        y_tr = y[train_idx]
        X_va = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_va = y[val_idx]
        pipe = clone(pipeline)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_va)
        scores.append(formula_f2_scorer_fn(y_va, y_pred))
    return np.mean(scores)


def objective_cat(trial, X, y, num_cols, cat_cols):
    # Moderate class weight boost for Formula (capped at 4×)
    w_formula = trial.suggest_float('w_formula', 1.0, 4.0)
    w_mixed = trial.suggest_float('w_mixed', 1.0, 3.0)

    param = {
        'iterations': trial.suggest_int('iterations', 500, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
        'border_count': trial.suggest_categorical('border_count', [32, 64, 128]),
        'class_weights': {0: 1.0, 1: w_formula, 2: w_mixed},
        'verbose': False,
        'allow_writing_files': False,
        'random_state': RANDOM_STATE,
        'thread_count': -1
    }
    model = CatBoostClassifier(**param)
    pipeline = get_pipeline(model, num_cols, cat_cols, scale=False)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        y_tr = y[train_idx]
        X_va = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_va = y[val_idx]
        pipe = clone(pipeline)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_va)
        scores.append(formula_f2_scorer_fn(y_va, y_pred))
    return np.mean(scores)


def objective_knn(trial, X, y, num_cols, cat_cols):
    param = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 25),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
        'p': trial.suggest_int('p', 1, 5),
        'n_jobs': -1
    }
    model = KNeighborsClassifier(**param)
    pipeline = get_pipeline(model, num_cols, cat_cols, scale=True)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        y_tr = y[train_idx]
        X_va = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_va = y[val_idx]
        pipe = clone(pipeline)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_va)
        scores.append(formula_f2_scorer_fn(y_va, y_pred))
    return np.mean(scores)


# ==================== THRESHOLD OPTIMIZATION ====================

def optimize_threshold(pipeline, X_train, y_train, le):
    """
    Use CV to find the best Formula probability threshold that maximizes F2.
    Returns: (best_threshold, sweep_results_list)
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_classes = len(le.classes_)

    # Collect out-of-fold probabilities
    oof_proba = np.zeros((len(y_train), n_classes))
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
        y_tr = y_train[train_idx]
        X_va = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]

        pipe = clone(pipeline)
        pipe.fit(X_tr, y_tr)
        oof_proba[val_idx] = pipe.predict_proba(X_va)

    # Sweep thresholds
    thresholds = np.arange(0.10, 0.55, 0.01)
    sweep = []
    best_f2 = -1
    best_thr = 0.33  # default (≈ 1/3 for 3 classes)

    for thr in thresholds:
        y_pred_thr = apply_threshold(oof_proba, thr, n_classes)
        f2 = formula_f2_scorer_fn(y_train, y_pred_thr)
        rec = recall_score(y_train, y_pred_thr, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0)
        prec = precision_score(y_train, y_pred_thr, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0)
        acc = accuracy_score(y_train, y_pred_thr)

        sweep.append({
            'Threshold': round(thr, 2),
            'Formula_F2': round(f2, 4),
            'Formula_Recall': round(rec, 4),
            'Formula_Precision': round(prec, 4),
            'Overall_Accuracy': round(acc, 4)
        })

        if f2 > best_f2:
            best_f2 = f2
            best_thr = round(thr, 2)

    return best_thr, sweep


# ==================== FULL EVALUATION ====================

def evaluate_model(model_name, pipeline, X_train, y_train, X_test, y_test,
                   le, threshold=None):
    """
    Full evaluation: CV metrics + holdout test, with optional threshold.
    Returns: (row_dict, cm_test)
    """
    n_classes = len(le.classes_)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # --- CV metrics ---
    cv_metrics_val = []
    cv_metrics_train = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
        y_tr = y_train[train_idx]
        X_va = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
        y_va = y_train[val_idx]

        pipe = clone(pipeline)
        pipe.fit(X_tr, y_tr)

        # Validation predictions
        if threshold is not None:
            proba_va = pipe.predict_proba(X_va)
            y_pred_va = apply_threshold(proba_va, threshold, n_classes)
            proba_tr = pipe.predict_proba(X_tr)
            y_pred_tr = apply_threshold(proba_tr, threshold, n_classes)
        else:
            y_pred_va = pipe.predict(X_va)
            y_pred_tr = pipe.predict(X_tr)

        m_val, _ = compute_all_metrics(y_va, y_pred_va, le)
        m_train, _ = compute_all_metrics(y_tr, y_pred_tr, le)
        cv_metrics_val.append(m_val)
        cv_metrics_train.append(m_train)

    # Aggregate CV
    row = {"Model": model_name}
    metric_keys = list(cv_metrics_val[0].keys())

    for k in metric_keys:
        val_scores = np.array([m[k] for m in cv_metrics_val])
        train_scores = np.array([m[k] for m in cv_metrics_train])
        row[f"Train-CV {k}"] = format_cell(train_scores)
        row[f"Val-CV {k}"] = format_cell(val_scores)

    # --- Holdout test ---
    pipeline_full = clone(pipeline)
    pipeline_full.fit(X_train, y_train)

    if threshold is not None:
        proba_test = pipeline_full.predict_proba(X_test)
        y_pred_test = apply_threshold(proba_test, threshold, n_classes)
    else:
        y_pred_test = pipeline_full.predict(X_test)

    m_test, cm_test = compute_all_metrics(y_test, y_pred_test, le)
    for k, v in m_test.items():
        row[f"Test {k}"] = format_cell(v)

    if threshold is not None:
        row["Threshold"] = threshold

    return row, cm_test


# ==================== MAIN ====================

def main():
    print("=" * 70)
    print("  ALT-STAGE 2.5 v2: STATE-OF-THE-ART FORMULA-AWARE MODEL")
    print("=" * 70)

    X, y, num_cols, cat_cols, le = load_data()
    n_classes = len(le.classes_)

    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Split: Train={len(X_train)}, Test={len(X_test)}\n")

    results_list = []
    confusion_matrices = {}
    threshold_sweeps = {}

    # ============================================================
    # PHASE 1: OPTUNA TUNING (F2-optimized)
    # ============================================================
    print("=" * 50)
    print("  PHASE 1: OPTUNA HYPERPARAMETER TUNING (F2)")
    print("=" * 50)

    tuned_models = {}

    # --- Random Forest ---
    print(f"\n>>> Random Forest ({N_TRIALS} trials)...")
    t0 = time.time()
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective_rf(t, X_train, y_train, num_cols, cat_cols),
                   n_trials=N_TRIALS)
    print(f"    Best F2: {study.best_value:.4f} ({time.time()-t0:.1f}s)")

    bp = study.best_params.copy()
    bp.update({'class_weight': 'balanced', 'n_jobs': -1, 'random_state': RANDOM_STATE})
    tuned_models['RF (Tuned)'] = {
        'model': RandomForestClassifier(**bp),
        'scale': False,
        'params': bp
    }

    # --- CatBoost ---
    print(f"\n>>> CatBoost ({N_TRIALS} trials)...")
    t0 = time.time()
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective_cat(t, X_train, y_train, num_cols, cat_cols),
                   n_trials=N_TRIALS)
    print(f"    Best F2: {study.best_value:.4f} ({time.time()-t0:.1f}s)")

    bp = study.best_params.copy()
    w_f = bp.pop('w_formula')
    w_m = bp.pop('w_mixed')
    bp['class_weights'] = {0: 1.0, 1: w_f, 2: w_m}
    bp.update({'verbose': False, 'allow_writing_files': False,
               'random_state': RANDOM_STATE, 'thread_count': -1})
    tuned_models['CatBoost (Tuned)'] = {
        'model': CatBoostClassifier(**bp),
        'scale': False,
        'params': {**{k: v for k, v in bp.items() if k != 'class_weights'},
                   'w_formula': w_f, 'w_mixed': w_m}
    }

    # --- KNN ---
    print(f"\n>>> KNN ({N_TRIALS} trials)...")
    t0 = time.time()
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective_knn(t, X_train, y_train, num_cols, cat_cols),
                   n_trials=N_TRIALS)
    print(f"    Best F2: {study.best_value:.4f} ({time.time()-t0:.1f}s)")

    bp = study.best_params.copy()
    bp.update({'n_jobs': -1})
    tuned_models['KNN (Tuned)'] = {
        'model': KNeighborsClassifier(**bp),
        'scale': True,
        'params': bp
    }

    # ============================================================
    # PHASE 2: EVALUATE BASE MODELS + THRESHOLD OPTIMIZATION
    # ============================================================
    print("\n" + "=" * 50)
    print("  PHASE 2: THRESHOLD OPTIMIZATION")
    print("=" * 50)

    for name, info in tuned_models.items():
        pipeline = get_pipeline(info['model'], num_cols, cat_cols, scale=info['scale'])

        # 2a. Evaluate with default threshold (argmax)
        print(f"\n>>> {name} — Default (argmax)...")
        row_default, cm_default = evaluate_model(
            f"{name}", pipeline, X_train, y_train, X_test, y_test, le
        )
        row_default.update({f"param_{k}": v for k, v in info['params'].items()})
        results_list.append(row_default)
        confusion_matrices[name] = cm_default

        # 2b. Threshold optimization
        print(f"    Optimizing threshold...")
        best_thr, sweep = optimize_threshold(pipeline, X_train, y_train, le)
        threshold_sweeps[name] = sweep
        print(f"    Best threshold: {best_thr}")

        # 2c. Evaluate with optimized threshold
        row_thr, cm_thr = evaluate_model(
            f"{name} + Thr={best_thr}", pipeline,
            X_train, y_train, X_test, y_test, le, threshold=best_thr
        )
        row_thr.update({f"param_{k}": v for k, v in info['params'].items()})
        results_list.append(row_thr)
        confusion_matrices[f"{name} + Thr"] = cm_thr

    # ============================================================
    # PHASE 3: STACKING ENSEMBLE
    # ============================================================
    print("\n" + "=" * 50)
    print("  PHASE 3: STACKING ENSEMBLE")
    print("=" * 50)

    # Build stacking with all tuned base models
    # Each base model needs its own preprocessing pipeline
    rf_pipe = get_pipeline(
        clone(tuned_models['RF (Tuned)']['model']),
        num_cols, cat_cols, scale=False
    )
    cat_pipe = get_pipeline(
        clone(tuned_models['CatBoost (Tuned)']['model']),
        num_cols, cat_cols, scale=False
    )
    knn_pipe = get_pipeline(
        clone(tuned_models['KNN (Tuned)']['model']),
        num_cols, cat_cols, scale=True
    )

    stacking = StackingClassifier(
        estimators=[
            ('rf', rf_pipe),
            ('catboost', cat_pipe),
            ('knn', knn_pipe)
        ],
        final_estimator=LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        stack_method='predict_proba',
        cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        passthrough=False
    )

    # Wrap stacker — no extra SMOTE/preprocessing since base estimators handle it
    # StackingClassifier manages its own CV internally for generating meta-features

    print("\n>>> Stacking Ensemble (RF + CatBoost + KNN)...")
    # We need to evaluate the stacker manually since it's not an ImbPipeline
    # Evaluate with default predictions
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_val_metrics = []
    cv_train_metrics = []
    oof_proba_stack = np.zeros((len(y_train), n_classes))

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
        y_tr = y_train[train_idx]
        X_va = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
        y_va = y_train[val_idx]

        stk = clone(stacking)
        stk.fit(X_tr, y_tr)

        y_pred_va = stk.predict(X_va)
        y_pred_tr = stk.predict(X_tr)
        oof_proba_stack[val_idx] = stk.predict_proba(X_va)

        m_val, _ = compute_all_metrics(y_va, y_pred_va, le)
        m_train, _ = compute_all_metrics(y_tr, y_pred_tr, le)
        cv_val_metrics.append(m_val)
        cv_train_metrics.append(m_train)

    # Build result row for stacker (default)
    row_stack = {"Model": "Stacking Ensemble"}
    metric_keys = list(cv_val_metrics[0].keys())
    for k in metric_keys:
        val_scores = np.array([m[k] for m in cv_val_metrics])
        train_scores = np.array([m[k] for m in cv_train_metrics])
        row_stack[f"Train-CV {k}"] = format_cell(train_scores)
        row_stack[f"Val-CV {k}"] = format_cell(val_scores)

    # Test set
    stacking_full = clone(stacking)
    stacking_full.fit(X_train, y_train)
    y_pred_test_stack = stacking_full.predict(X_test)
    m_test_stack, cm_stack = compute_all_metrics(y_test, y_pred_test_stack, le)
    for k, v in m_test_stack.items():
        row_stack[f"Test {k}"] = format_cell(v)
    results_list.append(row_stack)
    confusion_matrices["Stacking Ensemble"] = cm_stack

    # Threshold optimization for stacker
    print("    Optimizing stacking threshold...")
    thresholds = np.arange(0.10, 0.55, 0.01)
    best_f2_stack = -1
    best_thr_stack = 0.33
    sweep_stack = []

    for thr in thresholds:
        y_pred_thr = apply_threshold(oof_proba_stack, thr, n_classes)
        f2 = formula_f2_scorer_fn(y_train, y_pred_thr)
        rec = recall_score(y_train, y_pred_thr, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0)
        prec = precision_score(y_train, y_pred_thr, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0)
        acc = accuracy_score(y_train, y_pred_thr)
        sweep_stack.append({
            'Threshold': round(thr, 2), 'Formula_F2': round(f2, 4),
            'Formula_Recall': round(rec, 4), 'Formula_Precision': round(prec, 4),
            'Overall_Accuracy': round(acc, 4)
        })
        if f2 > best_f2_stack:
            best_f2_stack = f2
            best_thr_stack = round(thr, 2)

    threshold_sweeps["Stacking Ensemble"] = sweep_stack
    print(f"    Best stacking threshold: {best_thr_stack}")

    # Evaluate stacker with threshold on test
    proba_test_stack = stacking_full.predict_proba(X_test)
    y_pred_test_stack_thr = apply_threshold(proba_test_stack, best_thr_stack, n_classes)
    m_test_stack_thr, cm_stack_thr = compute_all_metrics(y_test, y_pred_test_stack_thr, le)

    row_stack_thr = {"Model": f"Stacking + Thr={best_thr_stack}"}
    # Reuse CV metrics with threshold
    for k in metric_keys:
        vals = []
        trains = []
        # Recompute from oof for threshold
        for train_idx, val_idx in cv.split(X_train, y_train):
            y_va = y_train[val_idx]
            y_tr = y_train[train_idx]
            yp_va = apply_threshold(oof_proba_stack[val_idx], best_thr_stack, n_classes)
            m_v, _ = compute_all_metrics(y_va, yp_va, le)
            vals.append(m_v[k])
        row_stack_thr[f"Val-CV {k}"] = format_cell(np.array(vals))
    for k, v in m_test_stack_thr.items():
        row_stack_thr[f"Test {k}"] = format_cell(v)
    row_stack_thr["Threshold"] = best_thr_stack
    results_list.append(row_stack_thr)
    confusion_matrices["Stacking + Thr"] = cm_stack_thr

    # ============================================================
    # SAVE TO EXCEL
    # ============================================================
    print(f"\nSaving to {OUTPUT_FILE}...")

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        # --- Results ---
        df_res = pd.DataFrame(results_list)
        priority = [
            "Model", "Threshold",
            "Val-CV Formula_Recall", "Test Formula_Recall",
            "Val-CV Formula_Precision", "Test Formula_Precision",
            "Val-CV Formula_F2", "Test Formula_F2",
            "Val-CV Formula_Accuracy", "Test Formula_Accuracy",
        ]
        other = [c for c in df_res.columns if c not in priority]
        col_order = [c for c in priority if c in df_res.columns] + other
        df_res = df_res[col_order]
        df_res.to_excel(writer, sheet_name="Results", index=False)

        # --- Threshold Sweep sheets ---
        for model_name, sweep_data in threshold_sweeps.items():
            df_sweep = pd.DataFrame(sweep_data)
            sheet = f"Thr_{model_name}"[:31]
            df_sweep.to_excel(writer, sheet_name=sheet, index=False)

        # --- Confusion Matrices ---
        class_names = list(le.classes_)
        for model_name, cm_arr in confusion_matrices.items():
            df_cm = pd.DataFrame(
                cm_arr,
                index=[f"Actual: {c}" for c in class_names],
                columns=[f"Predicted: {c}" for c in class_names]
            )
            sheet = f"CM_{model_name}"[:31]
            df_cm.to_excel(writer, sheet_name=sheet)

    print("\n" + "=" * 70)
    print("  DONE! Check Desktop for results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
