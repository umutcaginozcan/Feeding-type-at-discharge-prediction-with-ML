#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Alt-Stage 5: Temporal Ablation Study
-------------------------------------------
Purpose:
  Determine if reliable discharge-feeding predictions can be made using
  only Day 1 data, supporting the clinical argument for early intervention.

Design:
  Four temporal windows, each adding one more day of feeding data:
    1. Baseline — demographics/diagnosis only (no feeding data)
    2. Day 1    — + Day 1 intake, weight, route, breastmilk ratio
    3. Day 1+2  — + Day 2 intake, weight, trajectory
    4. Full     — + Day 3 (current production model)

Models (UNTUNED — default hyperparameters to avoid bias):
  - Random Forest  (default + class_weight='balanced')
  - CatBoost       (default + auto_class_weights='Balanced')

Metrics:
  Formula Precision, Formula Recall, F1-Macro, MCC, AUC-ROC Macro

Output:
  ~/Desktop/nicu_temporal_ablation.xlsx
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, LabelEncoder, label_binarize
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_stage0_5_cleaned.xlsx"
FEAT_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_selected_features.csv"
OUTPUT_FILE = Path.home() / "Desktop" / "nicu_temporal_ablation.xlsx"

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
N_FOLDS = 5
FORMULA_CLASS_IDX = 1

# ==================== UNTUNED MODELS (default params — fair across all windows) ==

# ==================== TEMPORAL FEATURE GROUPS ====================
#
# These are the RAW column names (before OHE).
# Selected features after OHE (e.g. "covid19sonrasi_0") map back to
# the raw column that produced them (e.g. "covid19sonrasi").
# We define inclusion at the RAW column level.

# --- BASELINE: demographics, diagnosis, admission (no feeding data) ---
BASELINE_COLS = [
    # Maternal
    "anneyasi",
    # Infant
    "dogumagirligi(gram)", "gebelikhaftası", "gebelikhaftagunu",
    "takipilkgün_kilo_gram",
    # Engineered (day-independent)
    "eng_weight_per_week",
    # Categoricals (day-independent)
    "annesutuemzirmeeğitimidurumu",
    "covid19sonrasi",
    "ikisiarası",
]

# --- DAY 1: first-day feeding, intake, weight ---
DAY1_COLS = [
    # Numeric
    "aldığıannesütü_ilkgün", "aldığımamamiktari1.gün", "kilo1.gun",
    # Engineered
    "eng_bm_ratio_d1", "eng_resilience_index",
    # Categoricals
    "ilk_gün_anne_sütü_1111", "ilk_gün_emzirme_111",
]

# --- DAY 2: second-day feeding, intake, weight, trajectory ---
DAY2_COLS = [
    # Numeric
    "beslenmetotali2.gün", "beslenme2.gunannesutucc",
    "beslenmemamamiktarı2.guncc", "kilo2.gun",
    # Engineered
    "eng_bm_ratio_d2", "eng_delta_vol_d1_d2",
]

# --- DAY 3: third-day feeding, intake, weight, trajectory ---
DAY3_COLS = [
    # Numeric
    "beslenmetotali3.gun", "aldıgıannesütü3.gun",
    "aldıgımamamiktari3.gun", "kilo3.gun",
    # Engineered
    "eng_bm_ratio_d3", "eng_delta_vol_d2_d3",
    "eng_lactation_momentum",  # requires D2+D3 trajectory
    # Categoricals
    "verilisyolu3gun",
]

# The four temporal windows (cumulative)
TEMPORAL_WINDOWS = {
    "Baseline":   BASELINE_COLS,
    "Day 1":      BASELINE_COLS + DAY1_COLS,
    "Day 1+2":    BASELINE_COLS + DAY1_COLS + DAY2_COLS,
    "Full (D1-3)": BASELINE_COLS + DAY1_COLS + DAY2_COLS + DAY3_COLS,
}

# ==================== DATA LOADING ====================

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    selected_df = pd.read_csv(FEAT_PATH)
    selected_feat_names = set(selected_df["Selected_Features"].tolist())

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    y = df[TARGET_COL]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Identify raw columns that produced selected features
    all_raw_cols = df.columns.tolist()
    if TARGET_COL in all_raw_cols:
        all_raw_cols.remove(TARGET_COL)

    selected_raw_cols = set()
    for raw_col in all_raw_cols:
        if raw_col in selected_feat_names:
            selected_raw_cols.add(raw_col)
            continue
        for sel in selected_feat_names:
            if sel.startswith(str(raw_col)):
                selected_raw_cols.add(raw_col)
                break

    print(f"  {len(df)} samples, {len(selected_raw_cols)} selected raw columns")
    print(f"  Classes: {dict(zip(le.classes_, np.bincount(y_enc)))}")

    return df, y_enc, selected_raw_cols, le


def build_feature_set(df, allowed_raw_cols, selected_raw_cols):
    """
    Intersect the temporal window's allowed columns with the RFECV-selected
    columns. Returns X, num_cols, cat_cols.
    """
    # Only keep columns that are BOTH in the temporal window AND selected by RFECV
    cols_to_use = [c for c in allowed_raw_cols if c in selected_raw_cols and c in df.columns]

    X = df[cols_to_use].copy()
    num_cols, cat_cols = [], []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            X[c] = X[c].astype(str)
            cat_cols.append(c)

    return X, num_cols, cat_cols


# ==================== PIPELINE ====================

def get_pipeline(model, num_cols, cat_cols):
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    return ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", model)
    ])


# ==================== EVALUATION ====================

def evaluate(pipeline, X_train, y_train, X_test, y_test, le):
    """Full evaluation: 5-fold CV + holdout test."""
    n_classes = len(le.classes_)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # --- CV ---
    cv_metrics = {"F1_Macro": [], "MCC": [], "Formula_Prec": [],
                  "Formula_Rec": [], "AUC_ROC": []}

    for train_idx, val_idx in cv.split(X_train, y_train):
        Xt = X_train.iloc[train_idx]
        yt = y_train[train_idx]
        Xv = X_train.iloc[val_idx]
        yv = y_train[val_idx]

        pipe = clone(pipeline)
        pipe.fit(Xt, yt)
        yp = pipe.predict(Xv)
        yproba = pipe.predict_proba(Xv)

        cv_metrics["F1_Macro"].append(f1_score(yv, yp, average="macro"))
        cv_metrics["MCC"].append(matthews_corrcoef(yv, yp))
        cv_metrics["Formula_Prec"].append(
            precision_score(yv, yp, labels=[FORMULA_CLASS_IDX], average="micro", zero_division=0))
        cv_metrics["Formula_Rec"].append(
            recall_score(yv, yp, labels=[FORMULA_CLASS_IDX], average="micro", zero_division=0))
        try:
            cv_metrics["AUC_ROC"].append(
                roc_auc_score(yv, yproba, multi_class="ovr", average="macro"))
        except Exception:
            cv_metrics["AUC_ROC"].append(np.nan)

    # --- Holdout test ---
    pipe_full = clone(pipeline)
    pipe_full.fit(X_train, y_train)
    y_pred = pipe_full.predict(X_test)
    y_proba = pipe_full.predict_proba(X_test)

    test_metrics = {
        "F1_Macro": f1_score(y_test, y_pred, average="macro"),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Formula_Prec": precision_score(
            y_test, y_pred, labels=[FORMULA_CLASS_IDX], average="micro", zero_division=0),
        "Formula_Rec": recall_score(
            y_test, y_pred, labels=[FORMULA_CLASS_IDX], average="micro", zero_division=0),
    }
    try:
        test_metrics["AUC_ROC"] = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        test_metrics["AUC_ROC"] = np.nan

    return cv_metrics, test_metrics


# ==================== MAIN ====================

def main():
    print("=" * 65)
    print("  ALT-STAGE 5: TEMPORAL ABLATION STUDY")
    print("=" * 65)

    df, y, selected_raw_cols, le = load_data()

    # 80/20 split (same seed as all other stages)
    all_idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  Split: Train={len(train_idx)}, Test={len(test_idx)}\n")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "CatBoost": CatBoostClassifier(
            auto_class_weights="Balanced",
            verbose=False, allow_writing_files=False,
            random_state=RANDOM_STATE, thread_count=-1
        ),
    }

    results = []

    for window_name, allowed_cols in TEMPORAL_WINDOWS.items():
        print(f"\n{'='*50}")
        print(f"  Window: {window_name}")
        print(f"{'='*50}")

        X, num_cols, cat_cols = build_feature_set(df, allowed_cols, selected_raw_cols)
        n_features = len(X.columns)

        X_train = X.iloc[train_idx]
        X_test_w = X.iloc[test_idx]

        print(f"  Features: {n_features} raw columns ({len(num_cols)} num, {len(cat_cols)} cat)")
        print(f"  Columns: {list(X.columns)}")

        for model_name, model in models.items():
            print(f"\n  >>> {model_name}...")
            t0 = time.time()

            pipeline = get_pipeline(clone(model), num_cols, cat_cols)
            cv_m, test_m = evaluate(pipeline, X_train, y_train, X_test_w, y_test, le)

            elapsed = time.time() - t0

            row = {
                "Window": window_name,
                "Model": model_name,
                "N_Features": n_features,
                # CV (mean ± std)
                "CV F1-Macro": f"{np.mean(cv_m['F1_Macro']):.3f} ± {np.std(cv_m['F1_Macro']):.3f}",
                "CV AUC-ROC": f"{np.nanmean(cv_m['AUC_ROC']):.3f} ± {np.nanstd(cv_m['AUC_ROC']):.3f}",
                "CV Formula Rec": f"{np.mean(cv_m['Formula_Rec']):.3f} ± {np.std(cv_m['Formula_Rec']):.3f}",
                # Test
                "Test Formula Prec": round(test_m["Formula_Prec"], 3),
                "Test Formula Rec": round(test_m["Formula_Rec"], 3),
                "Test F1-Macro": round(test_m["F1_Macro"], 3),
                "Test MCC": round(test_m["MCC"], 3),
                "Test AUC-ROC": round(test_m["AUC_ROC"], 3),
                "Time (s)": round(elapsed, 1),
            }
            results.append(row)

            print(f"      Test AUC-ROC: {test_m['AUC_ROC']:.3f}  |  "
                  f"F1: {test_m['F1_Macro']:.3f}  |  "
                  f"Formula Rec: {test_m['Formula_Rec']:.3f}  |  "
                  f"MCC: {test_m['MCC']:.3f}  ({elapsed:.1f}s)")

    # ==================== RESULTS TABLE ====================
    df_res = pd.DataFrame(results)

    print("\n\n" + "=" * 100)
    print("  TEMPORAL ABLATION — FULL RESULTS")
    print("=" * 100)
    print(df_res.to_string(index=False))

    # Save
    df_res.to_excel(OUTPUT_FILE, index=False, sheet_name="Ablation")
    print(f"\nSaved: {OUTPUT_FILE}")

    # --- Summary: Delta from Full model ---
    print("\n\n--- PERFORMANCE DELTA (vs Full model) ---")
    for model_name in models.keys():
        full_row = df_res[(df_res["Model"] == model_name) & (df_res["Window"] == "Full (D1-3)")]
        if full_row.empty:
            continue
        full_auc = full_row["Test AUC-ROC"].values[0]
        full_f1 = full_row["Test F1-Macro"].values[0]
        full_rec = full_row["Test Formula Rec"].values[0]

        print(f"\n  {model_name}:")
        for _, row in df_res[df_res["Model"] == model_name].iterrows():
            d_auc = row["Test AUC-ROC"] - full_auc
            d_f1 = row["Test F1-Macro"] - full_f1
            d_rec = row["Test Formula Rec"] - full_rec
            print(f"    {row['Window']:12s}  ΔAUC={d_auc:+.3f}  ΔF1={d_f1:+.3f}  ΔRec={d_rec:+.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
