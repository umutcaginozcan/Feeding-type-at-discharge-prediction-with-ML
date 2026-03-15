#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Stage 3: Final Production Leaderboard (The "Winner" Script)
----------------------------------------------------------------
Purpose:
Generate the final results for the paper using:
1. The Cleaned Data (Stage 0.5)
2. The Selected Features (Stage 1)
3. The Optimized Hyperparameters (Stage 2)

Includes:
- Individual Tuned Models (XGB, CatBoost, RF)
- Ensemble Model (Soft Voting)
- Academic Reporting Format
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

# Scikit-learn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, balanced_accuracy_score, brier_score_loss
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Models
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
OUTPUT_FILE = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_final_results_table.xlsx"

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
N_SPLITS = 5

# -------------------- 1. LOAD & FILTER DATA --------------------

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
    
    # Map Selected Features to Raw Columns (The Logic Fix)
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
            
    return X, y_enc, num_cols, cat_cols, n_classes

# -------------------- 2. DEFINE TUNED MODELS --------------------

def get_models():
    zoo = {}
    
    # XGBoost (Tuned Params)
    zoo["XGBoost (Tuned)"] = XGBClassifier(
        n_estimators=827,
        learning_rate=0.028840882720509959,
        max_depth=6,
        min_child_weight=4,
        subsample=0.76582409867318413,
        colsample_bytree=0.62252251153402693,
        gamma=0.40726140833576441,
        reg_alpha=1.529969916955737,
        reg_lambda=0.92068103133158941,
        n_jobs=-1,
        tree_method='hist',
        random_state=RANDOM_STATE,
        eval_metric='mlogloss'
    )
    
    # CatBoost (Tuned Params)
    zoo["CatBoost (Tuned)"] = CatBoostClassifier(
        iterations=911,
        learning_rate=0.01546307199186798,
        depth=4,
        l2_leaf_reg=6,
        random_strength=5.235468329194088,
        bagging_temperature=0.1117606075955955,
        border_count=32,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
        random_state=RANDOM_STATE
    )
    
    # Random Forest (Tuned Params)
    zoo["Random Forest (Tuned)"] = RandomForestClassifier(
        n_estimators=449,
        max_depth=25,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    
    # Ensemble (Voting)
    # We clone the instances to avoid fitting conflicts
    zoo["Ensemble (XGB+Cat+RF)"] = VotingClassifier(
        estimators=[
            ('xgb', clone(zoo["XGBoost (Tuned)"])),
            ('cat', clone(zoo["CatBoost (Tuned)"])),
            ('rf', clone(zoo["Random Forest (Tuned)"]))
        ],
        voting='soft',
        n_jobs=-1
    )
    
    return zoo

# -------------------- 3. UTILITIES --------------------

def format_cell(scores):
    if not scores: return "N/A"
    mean_val = np.mean(scores)
    list_str = ", ".join([f"{s:.3f}" for s in scores])
    return f"{mean_val:.3f} [{list_str}]"

def calculate_metrics(y_true, y_pred, y_proba, n_classes):
    # Binarize for AUC
    Y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    if n_classes == 2 and Y_bin.shape[1] == 1:
        Y_bin = np.hstack([1 - Y_bin, Y_bin])
        
    brier = np.mean(np.sum((y_proba - Y_bin) ** 2, axis=1))
    
    try: roc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except: roc = 0.5 
    
    pr = average_precision_score(Y_bin, y_proba, average="macro")

    return {
        "ROC_AUC_macro": roc,
        "PR_AUC_macro": pr,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Brier_Score": brier,
        "F1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "F1_macro": f1_score(y_true, y_pred, average="macro")
    }

# -------------------- 4. MAIN LOOP --------------------

def main():
    X, y, num_cols, cat_cols, n_classes = load_data()
    
    # 80/20 Train/Test Split (Holdout for Final Validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    # Pipeline Construction
    # No Scaler for trees (Stage 0.5 cleanup handled data types)
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    
    models = get_models()
    metrics_list = ["ROC_AUC_macro", "PR_AUC_macro", "Accuracy", "Balanced_Accuracy", "Brier_Score", "F1_weighted", "F1_macro"]
    final_results = {m: [] for m in metrics_list}
    
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    print(f"\nStarting Final Benchmark on {len(models)} models...")
    
    for name, model in models.items():
        print(f"Processing: {name}")
        
        pipeline = ImbPipeline([
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)), # Integrity: SMOTE only on train
            ("clf", model)
        ])
        
        fold_scores_train = {m: [] for m in metrics_list}
        fold_scores_val = {m: [] for m in metrics_list}
        
        # --- CROSS VALIDATION ---
        for train_idx, val_idx in cv.split(X_train, y_train):
            Xt, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            yt, yv = y_train[train_idx], y_train[val_idx]
            
            clf_fold = clone(pipeline)
            clf_fold.fit(Xt, yt)
            
            # Validation
            pv = clf_fold.predict(Xv)
            prob_v = clf_fold.predict_proba(Xv)
            res_val = calculate_metrics(yv, pv, prob_v, n_classes)
            
            # Train (Overfit Check)
            pt = clf_fold.predict(Xt)
            prob_t = clf_fold.predict_proba(Xt)
            res_train = calculate_metrics(yt, pt, prob_t, n_classes)
            
            for m in metrics_list:
                fold_scores_val[m].append(res_val[m])
                fold_scores_train[m].append(res_train[m])
        
        # --- FINAL TEST (HOLDOUT) ---
        final_clf = clone(pipeline)
        final_clf.fit(X_train, y_train)
        p_test = final_clf.predict(X_test)
        prob_test = final_clf.predict_proba(X_test)
        res_test = calculate_metrics(y_test, p_test, prob_test, n_classes)
        
        # --- STORE RESULTS ---
        for m in metrics_list:
            final_results[m].append({
                "Algorithm": name,
                "Type": "Train-CV",
                "Score": format_cell(fold_scores_train[m])
            })
            final_results[m].append({
                "Algorithm": name,
                "Type": "Val-CV",
                "Score": format_cell(fold_scores_val[m])
            })
            final_results[m].append({
                "Algorithm": name,
                "Type": "Test",
                "Score": f"{res_test[m]:.3f} [{res_test[m]:.3f}]"
            })

    # --- SORT & SAVE ---
    print(f"\nSaving results to {OUTPUT_FILE}...")
    
    def extract_score(s):
        try: return float(s.split()[0])
        except: return -1.0

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        for metric_name, rows in final_results.items():
            df_m = pd.DataFrame(rows)
            
            # Sort Logic: Get Test Score for each Algo -> Sort Algo by that -> Sort Rows
            test_scores = df_m[df_m['Type'] == 'Test'].copy()
            test_scores['Sort_Key'] = test_scores['Score'].apply(extract_score)
            score_map = dict(zip(test_scores['Algorithm'], test_scores['Sort_Key']))
            
            df_m['Sort_Key'] = df_m['Algorithm'].map(score_map)
            df_m = df_m.sort_values(by=['Sort_Key', 'Algorithm'], ascending=[False, True])
            df_m = df_m.drop(columns=['Sort_Key'])
            
            df_m.to_excel(writer, sheet_name=metric_name, index=False)
            
    print("Done. Check the 'Ensemble' row in the output. It should be your winner.")

if __name__ == "__main__":
    main()