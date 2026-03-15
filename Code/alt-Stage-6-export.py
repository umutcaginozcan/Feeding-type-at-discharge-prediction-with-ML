#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Alt-Stage 6: Export Deployment Model
--------------------------------------------
Imitates Stage-6-export-model.py but for the Day 1+2 (no COVID) model.

1. Loads RFECV features, filters to Day1+2 window (no COVID)
2. Trains tuned RF on FULL dataset
3. Exports: pickle, feature_metadata.json, model_info.json
4. These files are consumed by the Streamlit app

Output: ~/Desktop/nicu_deployment/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import pickle
import json
from datetime import datetime

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, balanced_accuracy_score, recall_score, precision_score,
    fbeta_score, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_stage0_5_cleaned.xlsx"
FEAT_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_selected_features.csv"
OUTPUT_DIR = Path.home() / "Desktop" / "nicu_deployment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
N_FOLDS = 5
FORMULA_CLASS_IDX = 1

# Tuned RF params (from alt-Stage-6-final-tuning.py, +Day1&2 no COVID)
RF_PARAMS = {
    "n_estimators": 526,
    "max_depth": 22,
    "min_samples_split": 8,
    "min_samples_leaf": 7,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

OPTIMAL_THRESHOLD = 0.26

# Day 1+2 window (no COVID)
BASELINE_COLS = [
    "anneyasi",
    "dogumagirligi(gram)", "gebelikhaftası", "gebelikhaftagunu",
    "takipilkgün_kilo_gram",
    "eng_weight_per_week",
    "annesutuemzirmeeğitimidurumu",
]

DAY1_COLS = [
    "aldığıannesütü_ilkgün", "aldığımamamiktari1.gün", "kilo1.gun",
    "eng_bm_ratio_d1", "eng_resilience_index",
    "ilk_gün_anne_sütü_1111", "ilk_gün_emzirme_111",
]

DAY2_COLS = [
    "beslenmetotali2.gün", "beslenme2.gunannesutucc",
    "beslenmemamamiktarı2.guncc", "kilo2.gun",
    "eng_bm_ratio_d2", "eng_delta_vol_d1_d2",
]

ALLOWED_COLS = BASELINE_COLS + DAY1_COLS + DAY2_COLS

# ==================== DATA LOADING ====================

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    selected_df = pd.read_csv(FEAT_PATH)
    selected_feat_names = set(selected_df["Selected_Features"].tolist())

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    le = LabelEncoder()
    y_enc = le.fit_transform(df[TARGET_COL])

    # Map OHE feature names to raw columns (same logic as all stages)
    all_raw_cols = [c for c in df.columns if c != TARGET_COL]
    selected_raw_cols = set()
    for raw_col in all_raw_cols:
        if raw_col in selected_feat_names:
            selected_raw_cols.add(raw_col)
            continue
        for sel in selected_feat_names:
            if sel.startswith(str(raw_col)):
                selected_raw_cols.add(raw_col)
                break

    # Filter to Day 1+2 no-COVID window
    cols = [c for c in ALLOWED_COLS
            if c in selected_raw_cols and c in df.columns]

    X = df[cols].copy()
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in cat_cols:
        X[c] = X[c].astype(str)

    print(f"  {len(df)} samples, {len(X.columns)} features "
          f"({len(num_cols)} num, {len(cat_cols)} cat)")
    print(f"  Columns: {sorted(X.columns.tolist())}")
    print(f"  Classes: {dict(zip(le.classes_, np.bincount(y_enc)))}")

    return X, y_enc, num_cols, cat_cols, le


# ==================== PIPELINE ====================

def build_pipeline(num_cols, cat_cols):
    transformers = [("num", SimpleImputer(strategy="median"), num_cols)]
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False), cat_cols))
    preprocessor = ColumnTransformer(transformers)
    model = RandomForestClassifier(**RF_PARAMS)
    return ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", model),
    ])


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("  ALT-STAGE 6: EXPORT DEPLOYMENT MODEL")
    print("  Model: Day 1+2 (no COVID) — Tuned RF")
    print("=" * 60)

    X, y, num_cols, cat_cols, le = load_data()
    n_classes = len(le.classes_)
    class_names = list(le.classes_)

    pipeline = build_pipeline(num_cols, cat_cols)

    # ---- Evaluate with CV ----
    print("\n" + "=" * 60)
    print("  5-Fold CV Evaluation")
    print("=" * 60)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    cv_metrics = {"AUC": [], "MCC": [], "F1_Macro": [],
                  "Formula_Rec": [], "Formula_Prec": [], "Formula_F2": []}

    for ti, vi in cv.split(X, y):
        p = clone(pipeline)
        p.fit(X.iloc[ti], y[ti])
        yp = p.predict(X.iloc[vi])
        ypr = p.predict_proba(X.iloc[vi])
        cv_metrics["MCC"].append(matthews_corrcoef(y[vi], yp))
        cv_metrics["F1_Macro"].append(f1_score(y[vi], yp, average="macro"))
        cv_metrics["Formula_Rec"].append(recall_score(
            y[vi], yp, labels=[FORMULA_CLASS_IDX], average="micro", zero_division=0))
        cv_metrics["Formula_Prec"].append(precision_score(
            y[vi], yp, labels=[FORMULA_CLASS_IDX], average="micro", zero_division=0))
        cv_metrics["Formula_F2"].append(fbeta_score(
            y[vi], yp, beta=2, labels=[FORMULA_CLASS_IDX], average="micro", zero_division=0))
        try:
            cv_metrics["AUC"].append(roc_auc_score(
                y[vi], ypr, multi_class="ovr", average="macro"))
        except:
            cv_metrics["AUC"].append(np.nan)

    for k, v in cv_metrics.items():
        print(f"  {k:15s}: {np.mean(v):.3f} ± {np.std(v):.3f}")

    # ---- Train on FULL dataset ----
    print("\n" + "=" * 60)
    print("  Training on FULL dataset...")
    print("=" * 60)
    pipeline.fit(X, y)
    print("  ✓ Model trained")

    # ---- Export ----
    print("\n" + "=" * 60)
    print("  Exporting...")
    print("=" * 60)

    # 1. Pickle
    model_path = OUTPUT_DIR / "trained_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"  ✓ {model_path}")

    # 2. Feature metadata
    feature_metadata = {
        "num_features": num_cols,
        "cat_features": cat_cols,
        "all_features": sorted(X.columns.tolist()),
        "total_features": len(X.columns),
        "class_names": [int(c) for c in class_names],
        "class_labels": {
            "0": "Exclusive Breastfeeding",
            "1": "Formula Feeding",
            "2": "Mixed Feeding",
        },
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "auto_computed": {
            "eng_weight_per_week": {
                "formula": "dogumagirligi(gram) / (gebelikhaftası + 1e-6)",
                "inputs": ["dogumagirligi(gram)", "gebelikhaftası"],
            },
            "eng_bm_ratio_d1": {
                "formula": "d1_bm / (d1_bm + d1_formula + 1e-6)",
                "inputs": ["aldığıannesütü_ilkgün", "aldığımamamiktari1.gün"],
            },
            "eng_resilience_index": {
                "formula": "NaN (requires Day 3 data, median-imputed)",
                "inputs": [],
                "note": "Always NaN at inference — median imputer handles it",
            },
            "eng_bm_ratio_d2": {
                "formula": "d2_bm / (d2_total + 1e-6)",
                "inputs": ["beslenme2.gunannesutucc", "beslenmetotali2.gün"],
            },
            "eng_delta_vol_d1_d2": {
                "formula": "d2_total - (d1_bm + d1_formula)",
                "inputs": ["beslenmetotali2.gün", "aldığıannesütü_ilkgün",
                           "aldığımamamiktari1.gün"],
            },
        },
        "raw_user_inputs": [
            {"name": "anneyasi", "label": "Mother's Age", "type": "number", "unit": "years"},
            {"name": "dogumagirligi(gram)", "label": "Birth Weight", "type": "number", "unit": "grams"},
            {"name": "gebelikhaftası", "label": "Gestational Age (weeks)", "type": "number", "unit": "weeks"},
            {"name": "gebelikhaftagunu", "label": "Gestational Age (days)", "type": "number", "unit": "days"},
            {"name": "takipilkgün_kilo_gram", "label": "Day 1 Weight", "type": "number", "unit": "grams"},
            {"name": "annesutuemzirmeeğitimidurumu", "label": "BF Education Status", "type": "number"},
            {"name": "aldığıannesütü_ilkgün", "label": "Day 1 Breast Milk (cc)", "type": "number", "unit": "cc"},
            {"name": "aldığımamamiktari1.gün", "label": "Day 1 Formula (cc)", "type": "number", "unit": "cc"},
            {"name": "kilo1.gun", "label": "Day 1 Weight (follow-up)", "type": "number", "unit": "grams"},
            {"name": "ilk_gün_anne_sütü_1111", "label": "Day 1 BM Given (0/1)", "type": "number"},
            {"name": "ilk_gün_emzirme_111", "label": "Day 1 Breastfeeding (0/1)", "type": "number"},
            {"name": "beslenmetotali2.gün", "label": "Day 2 Total Intake (cc)", "type": "number", "unit": "cc"},
            {"name": "beslenme2.gunannesutucc", "label": "Day 2 Breast Milk (cc)", "type": "number", "unit": "cc"},
            {"name": "beslenmemamamiktarı2.guncc", "label": "Day 2 Formula (cc)", "type": "number", "unit": "cc"},
            {"name": "kilo2.gun", "label": "Day 2 Weight", "type": "number", "unit": "grams"},
        ],
    }
    meta_path = OUTPUT_DIR / "feature_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(feature_metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {meta_path}")

    # 3. Model info
    model_info = {
        "model_type": "Random Forest (Optuna-tuned, F2-optimized)",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": len(X),
        "n_features": len(X.columns),
        "temporal_window": "Day 1 + Day 2 (no COVID)",
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "hyperparameters": RF_PARAMS,
        "cv_performance": {k: {"mean": round(float(np.mean(v)), 3),
                               "std": round(float(np.std(v)), 3)}
                           for k, v in cv_metrics.items()},
    }
    info_path = OUTPUT_DIR / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2, default=str)
    print(f"  ✓ {info_path}")

    print("\n" + "=" * 60)
    print("  ✅ EXPORT COMPLETE!")
    print("=" * 60)
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.suffix in (".pkl", ".json"):
            print(f"  {f.name:40s}  {f.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
