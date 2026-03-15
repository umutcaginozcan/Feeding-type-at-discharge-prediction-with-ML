#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Stage 1.5: The Grand Model Race (Comprehensive Benchmark)
--------------------------------------------------------------
Purpose: 
To benchmarking a wide variety of algorithms (Linear, Non-Linear, Trees, Probabilistic)
using the selected features from Stage 1.

Methodology:
- 80% Train / 20% Test Split
- 5-Fold Cross Validation on Training Set
- Reporting: Train-CV, Val-CV (Validation), and Test (Holdout) scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import time

# Sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    make_scorer, f1_score, roc_auc_score, accuracy_score, 
    balanced_accuracy_score, precision_score, recall_score, matthews_corrcoef,
    confusion_matrix
)

# Models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# -------------------- CONFIGURATION --------------------

# Base directory: project root
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_stage0_5_cleaned.xlsx"
FEAT_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_selected_features.csv"
OUTPUT_FILE = Path.home() / "Desktop" / "nicu_stage1_5_grand_race_results.xlsx"

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
N_FOLDS = 5

# -------------------- 1. MODEL ZOO --------------------

def get_model_zoo():
    models = {}
    
    # --- Baseline ---
    models["Baseline (Most Freq)"] = DummyClassifier(strategy="most_frequent")
    
    # --- Probabilistic / Linear ---
    models["Naive Bayes"] = GaussianNB()
    models["Logistic Regression"] = LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    
    # --- Neural Networks ---
    models["Neural Net (MLP)"] = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=RANDOM_STATE
    )
    
    # --- SVM (Scaled data required) ---
    # Linear SVC via SVC kernel='linear' to allow probability=True for AUC
    models["Linear SVM"] = SVC(
        kernel='linear',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    models["RBF SVM"] = SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    
    # --- KNN Series ---
    for k in [1, 3, 5, 7, 9, 11]:
        models[f"KNN (k={k})"] = KNeighborsClassifier(n_neighbors=k)
        
    # --- Trees & Ensembles ---
    models["Decision Tree"] = DecisionTreeClassifier(
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    models["AdaBoost"] = AdaBoostClassifier(
        n_estimators=50,
        random_state=RANDOM_STATE
    )
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    
    # --- Gradient Boosting Giants ---
    models["XGBoost"] = XGBClassifier(
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    models["CatBoost"] = CatBoostClassifier(
        verbose=0,
        allow_writing_files=False,
        random_state=RANDOM_STATE,
        thread_count=1
    )
    
    return models

# -------------------- 2. UTILS & SCORING --------------------

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    
    # Load Features
    try:
        selected_df = pd.read_csv(FEAT_PATH)
        selected_feat_names = set(selected_df['Selected_Features'].tolist())
    except:
        print("Feature file not found. Using all features.")
        sys.exit(1)

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    y = df[TARGET_COL]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Map Features
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
    
    # Define Types for Preprocessing
    num_cols = []
    cat_cols = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            X[c] = X[c].astype(str)
            cat_cols.append(c)
            
    return X, y_enc, num_cols, cat_cols, le

def format_cell(scores):
    """
    Returns string format: "0.850 [0.840, 0.860, 0.850, ...]"
    """
    if isinstance(scores, (float, np.floating)):
        return f"{scores:.3f} [{scores:.3f}]"  # Single score (Test set)
    
    mean_val = np.mean(scores)
    list_str = ", ".join([f"{s:.3f}" for s in scores])
    return f"{mean_val:.3f} [{list_str}]"

def brier_score_multiclass(y_true, y_proba):
    """
    Multi-class Brier score.

    y_true : shape (n_samples,), integer labels (0..K-1)
    y_proba: shape (n_samples, n_classes), predicted probabilities

    Returns:
        scalar Brier score (lower is better)
        1/N * sum_i sum_k (p_ik - o_ik)^2
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Safety: ensure probabilities are valid
    y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    # One-hot encode y_true
    n_samples, n_classes = y_proba.shape
    Y_true = np.zeros_like(y_proba)
    Y_true[np.arange(n_samples), y_true] = 1.0

    # Brier score per sample, then mean
    sq_diff = (y_proba - Y_true) ** 2
    return np.mean(np.sum(sq_diff, axis=1))

# -------------------- 3. RACE ENGINE --------------------

def main():
    X, y, num_cols, cat_cols, le = load_data()
    models = get_model_zoo()
    
    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    
    print(f"Data Split: Train={len(X_train)}, Test={len(X_test)}")
    
    # Preprocessor (With SCALER for SVM/KNN/MLP)
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())  # Crucial for SVM/KNN
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    
    # Metrics
    scoring = {
        'ROC_AUC_Macro': 'roc_auc_ovr',
        'PR_AUC_Macro': 'average_precision',  # Approximation for multi-class
        'Accuracy': 'accuracy',
        'Balanced_Acc': 'balanced_accuracy',
        'F1_Weighted': 'f1_weighted',
        'F1_Macro': 'f1_macro',
        'MCC': make_scorer(matthews_corrcoef),

        # NEW: Multi-class Brier score (lower is better, so greater_is_better=False)
        'Brier_Score': make_scorer(
            brier_score_multiclass,
            needs_proba=True,
            greater_is_better=False
        ),

        # NEW: Overall Precision & Recall (weighted)
        'Precision_Weighted': make_scorer(precision_score, average='weighted', zero_division=0),
        'Recall_Weighted': make_scorer(recall_score, average='weighted', zero_division=0),
    }

    # ---------- Formula-class-specific metrics (not in cross_validate scoring) ----------
    # These are computed manually on train/val folds and holdout test set.
    # Formula = class index 1 after LabelEncoder (Exclusive BF=0, Formula=1, Mixed=2)
    FORMULA_CLASS_IDX = 1
    formula_metrics = ['Formula_Accuracy', 'Formula_Precision', 'Formula_Recall']
    
    # Results Container
    # Structure: { Metric: [ {Algorithm, Type, Score}, ... ] }
    final_output = {m: [] for m in list(scoring.keys()) + formula_metrics}
    
    # Confusion matrices for selected models
    CM_MODELS = {"Random Forest", "RBF SVM", "CatBoost"}
    confusion_matrices = {}  # name -> cm array

    print(f"\n--- 🏁 STARTING THE GRAND RACE ({len(models)} Models) ---")
    
    for name, model in models.items():
        print(f"Running: {name}...")
        start_time = time.time()
        
        # 1. Pipeline Construction
        pipeline = ImbPipeline([
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", model)
        ])
        
        # 2. Cross Validation (Train & Val Scores)
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        try:
            cv_results = cross_validate(
                pipeline, X_train, y_train, cv=cv, scoring=scoring, 
                return_train_score=True, n_jobs=-1
            )
            
            # 3. Final Test Prediction (Holdout)
            # Re-fit on FULL train set
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
            
            # Calculate Test Metrics Manually
            test_scores = {}
            test_scores['Accuracy'] = accuracy_score(y_test, y_pred)
            test_scores['Balanced_Acc'] = balanced_accuracy_score(y_test, y_pred)
            test_scores['F1_Weighted'] = f1_score(y_test, y_pred, average='weighted')
            test_scores['F1_Macro'] = f1_score(y_test, y_pred, average='macro')
            test_scores['MCC'] = matthews_corrcoef(y_test, y_pred)
            test_scores['Precision_Weighted'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            test_scores['Recall_Weighted'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)

            # --- Formula-class-specific metrics on test set ---
            cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
            # Store confusion matrix for selected models
            if name in CM_MODELS:
                confusion_matrices[name] = cm
            # Formula Accuracy: (TP_formula + TN_formula) / total
            tp_f = cm[FORMULA_CLASS_IDX, FORMULA_CLASS_IDX]
            fn_f = cm[FORMULA_CLASS_IDX, :].sum() - tp_f
            fp_f = cm[:, FORMULA_CLASS_IDX].sum() - tp_f
            tn_f = cm.sum() - tp_f - fn_f - fp_f
            test_scores['Formula_Accuracy'] = (tp_f + tn_f) / cm.sum() if cm.sum() > 0 else 0.0
            test_scores['Formula_Precision'] = precision_score(y_test, y_pred, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0)
            test_scores['Formula_Recall'] = recall_score(y_test, y_pred, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0)
            
            # Handle Probabilistic Metrics
            if y_proba is not None:
                try:
                    test_scores['ROC_AUC_Macro'] = roc_auc_score(
                        y_test, y_proba, multi_class='ovr', average='macro'
                    )
                except:
                    test_scores['ROC_AUC_Macro'] = 0.5
                
                # PR AUC requires binarization for macro average
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import average_precision_score

                Y_bin = label_binarize(y_test, classes=np.unique(y))
                if Y_bin.shape[1] == 1:
                    Y_bin = np.hstack([1 - Y_bin, Y_bin])  # Binary case fix
                try:
                    test_scores['PR_AUC_Macro'] = average_precision_score(
                        Y_bin, y_proba, average='macro'
                    )
                except:
                    test_scores['PR_AUC_Macro'] = 0.0

                # NEW: multi-class Brier score on the test set (positive, lower = better)
                test_scores['Brier_Score'] = brier_score_multiclass(y_test, y_proba)

            else:
                test_scores['ROC_AUC_Macro'] = 0.0
                test_scores['PR_AUC_Macro'] = 0.0
                # If no probabilities, Brier score undefined
                test_scores['Brier_Score'] = np.nan
                test_scores['Precision_Weighted'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                test_scores['Recall_Weighted'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)

                # Formula-specific metrics still work without probabilities
                cm = confusion_matrix(y_test, y_pred)
                tp_f = cm[FORMULA_CLASS_IDX, FORMULA_CLASS_IDX]
                fn_f = cm[FORMULA_CLASS_IDX, :].sum() - tp_f
                fp_f = cm[:, FORMULA_CLASS_IDX].sum() - tp_f
                tn_f = cm.sum() - tp_f - fn_f - fp_f
                test_scores['Formula_Accuracy'] = (tp_f + tn_f) / cm.sum() if cm.sum() > 0 else 0.0
                test_scores['Formula_Precision'] = precision_score(y_test, y_pred, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0)
                test_scores['Formula_Recall'] = recall_score(y_test, y_pred, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0)

            # 4. Format & Store — standard CV-based metrics
            for m in scoring.keys():
                train_key = f"train_{m}"
                test_key = f"test_{m}"

                # --- Train-CV ---
                if train_key in cv_results:
                    train_scores = cv_results[train_key]

                    # For Brier_Score, cv_results contains NEGATIVE values
                    # (because greater_is_better=False). Flip sign for display.
                    if m == 'Brier_Score':
                        train_scores = -train_scores  # -> positive Brier

                    final_output[m].append({
                        "Algorithm": name,
                        "Type": "Train-CV",
                        "Score": format_cell(train_scores),
                        "_sort": np.mean(cv_results.get(test_key, train_scores))
                    })

                # --- Val-CV ---
                if test_key in cv_results:
                    val_scores = cv_results[test_key]

                    if m == 'Brier_Score':
                        val_scores = -val_scores  # -> positive Brier

                    final_output[m].append({
                        "Algorithm": name,
                        "Type": "Val-CV",
                        "Score": format_cell(val_scores),
                        # For sorting, we still use the raw cv_results[test_key]:
                        # for Brier this is negative, so higher (less negative) = better.
                        "_sort": np.mean(cv_results[test_key])
                    })

                # --- Test (Holdout) ---
                if m in test_scores:
                    # test_scores['Brier_Score'] is already positive (lower is better)
                    final_output[m].append({
                        "Algorithm": name,
                        "Type": "Test",
                        "Score": format_cell(test_scores[m]),
                        "_sort": np.mean(cv_results[test_key]) if test_key in cv_results else 0.0
                    })

            # 4b. Format & Store — Formula-class-specific metrics
            #     These are NOT in cross_validate scoring, so we compute
            #     per-fold metrics manually from the CV indices.
            formula_fold_train_scores = {fm: [] for fm in formula_metrics}
            formula_fold_val_scores = {fm: [] for fm in formula_metrics}

            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
                y_tr_fold = y_train[train_idx]
                X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
                y_val_fold = y_train[val_idx]

                fold_pipe = ImbPipeline([
                    ("prep", preprocessor),
                    ("smote", SMOTE(random_state=RANDOM_STATE)),
                    ("clf", model)
                ])
                try:
                    fold_pipe.fit(X_tr_fold, y_tr_fold)
                except Exception:
                    for fm in formula_metrics:
                        formula_fold_train_scores[fm].append(0.0)
                        formula_fold_val_scores[fm].append(0.0)
                    continue

                for fold_X, fold_y, fold_dict in [
                    (X_tr_fold, y_tr_fold, formula_fold_train_scores),
                    (X_val_fold, y_val_fold, formula_fold_val_scores)
                ]:
                    y_fp = fold_pipe.predict(fold_X)
                    cm_f = confusion_matrix(fold_y, y_fp, labels=np.arange(len(le.classes_)))
                    tp = cm_f[FORMULA_CLASS_IDX, FORMULA_CLASS_IDX]
                    fn = cm_f[FORMULA_CLASS_IDX, :].sum() - tp
                    fp = cm_f[:, FORMULA_CLASS_IDX].sum() - tp
                    tn = cm_f.sum() - tp - fn - fp
                    fold_dict['Formula_Accuracy'].append((tp + tn) / cm_f.sum() if cm_f.sum() > 0 else 0.0)
                    fold_dict['Formula_Precision'].append(precision_score(fold_y, y_fp, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0))
                    fold_dict['Formula_Recall'].append(recall_score(fold_y, y_fp, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0))

            for fm in formula_metrics:
                tr_arr = np.array(formula_fold_train_scores[fm])
                va_arr = np.array(formula_fold_val_scores[fm])
                final_output[fm].append({
                    "Algorithm": name,
                    "Type": "Train-CV",
                    "Score": format_cell(tr_arr),
                    "_sort": np.mean(va_arr)
                })
                final_output[fm].append({
                    "Algorithm": name,
                    "Type": "Val-CV",
                    "Score": format_cell(va_arr),
                    "_sort": np.mean(va_arr)
                })
                if fm in test_scores:
                    final_output[fm].append({
                        "Algorithm": name,
                        "Type": "Test",
                        "Score": format_cell(test_scores[fm]),
                        "_sort": np.mean(va_arr)
                    })

        except Exception as e:
            print(f"Error running {name}: {e}")
            
        print(f"   -> Finished in {time.time() - start_time:.2f}s")

    # -------------------- 4. SAVE TO EXCEL --------------------
    print(f"\nSaving detailed report to {OUTPUT_FILE}...")
    
    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        for metric_name, rows in final_output.items():
            if not rows:
                continue
            
            df_m = pd.DataFrame(rows)
            
            # Custom Sorting:
            # 1. Sort by Algorithm (to keep Train/Val/Test grouped)
            # 2. But we want the Algorithm chunks to be ordered by their Best Score
            # Let's create a map of Algo -> Max Val Score
            algo_score_map = df_m.groupby('Algorithm')['_sort'].max().to_dict()
            df_m['_algo_rank'] = df_m['Algorithm'].map(algo_score_map)
            
            # Sort: Rank (Desc), Algorithm Name, Type Order (Train, Val, Test)
            type_order = {"Train-CV": 1, "Val-CV": 2, "Test": 3}
            df_m['_type_rank'] = df_m['Type'].map(type_order)
            
            df_m = df_m.sort_values(
                by=['_algo_rank', 'Algorithm', '_type_rank'],
                ascending=[False, True, True]
            )
            
            # Clean up helper cols
            df_m = df_m.drop(columns=['_sort', '_algo_rank', '_type_rank'])
            
            df_m.to_excel(writer, sheet_name=metric_name, index=False)

        # --- Confusion Matrices for selected models ---
        class_names = list(le.classes_)
        for model_name, cm_arr in confusion_matrices.items():
            # Build a labelled DataFrame: rows = Actual, cols = Predicted
            df_cm = pd.DataFrame(
                cm_arr,
                index=[f"Actual: {c}" for c in class_names],
                columns=[f"Predicted: {c}" for c in class_names]
            )
            sheet = f"CM_{model_name}"[:31]  # Excel sheet name limit
            df_cm.to_excel(writer, sheet_name=sheet)
            
    print("DONE. Check the Excel file for the leaderboard.")

if __name__ == "__main__":
    main()
