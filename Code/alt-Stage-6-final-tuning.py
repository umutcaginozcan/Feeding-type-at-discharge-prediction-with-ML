#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Alt-Stage 6: Final Deployment Tuning
--------------------------------------------
Exhaustive scan: 3 temporal windows × 2 COVID variants = 6 configurations.
Each configuration gets fresh Optuna RF tuning (F2-score) + threshold
optimization. Best model is pickled for deployment.

Architecture: alt-Stage-5 data loading + alt-Stage-2pnt5 tuning engine.

Output: ~/Desktop/nicu_deployment/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import optuna
import json
import pickle
import warnings
import time

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    fbeta_score, matthews_corrcoef,
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_stage0_5_cleaned.xlsx"
FEAT_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_selected_features.csv"
OUTPUT_DIR = Path.home() / "Desktop" / "nicu_deployment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
N_TRIALS = 50
N_FOLDS = 5
FORMULA_CLASS_IDX = 1

# ==================== TEMPORAL FEATURE GROUPS (from alt-Stage-5) =============

BASELINE_COLS = [
    "anneyasi",
    "dogumagirligi(gram)", "gebelikhaftası", "gebelikhaftagunu",
    "takipilkgün_kilo_gram",
    "eng_weight_per_week",
    "annesutuemzirmeeğitimidurumu",
    "covid19sonrasi", "ikisiarası",
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

COVID_COLS = {"covid19sonrasi", "ikisiarası"}

# ==================== BUILD ALL 6 CONFIGURATIONS =============================

def build_configs():
    """3 windows × 2 COVID variants = 6 configs."""
    windows = {
        "Baseline":  BASELINE_COLS,
        "+Day1":     BASELINE_COLS + DAY1_COLS,
        "+Day1&2":   BASELINE_COLS + DAY1_COLS + DAY2_COLS,
    }
    configs = {}
    for win_name, cols in windows.items():
        configs[f"{win_name} (with COVID)"] = cols
        configs[f"{win_name} (no COVID)"] = [c for c in cols
                                              if c not in COVID_COLS]
    return configs

# ==================== UTILS ==================================================

def formula_f2(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2,
                       labels=[FORMULA_CLASS_IDX], average="micro",
                       zero_division=0)


def apply_threshold(y_proba, thr, n_classes):
    y_pred = np.zeros(len(y_proba), dtype=int)
    for i in range(len(y_proba)):
        if y_proba[i, FORMULA_CLASS_IDX] >= thr:
            y_pred[i] = FORMULA_CLASS_IDX
        else:
            p = y_proba[i].copy()
            p[FORMULA_CLASS_IDX] = -1
            y_pred[i] = np.argmax(p)
    return y_pred


# ==================== DATA LOADING (from alt-Stage-5) ========================

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    selected_df = pd.read_csv(FEAT_PATH)
    selected_feat_names = set(selected_df["Selected_Features"].tolist())

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    le = LabelEncoder()
    y_enc = le.fit_transform(df[TARGET_COL])

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

    print(f"  {len(df)} samples, {len(selected_raw_cols)} RFECV-selected")
    print(f"  Classes: {dict(zip(le.classes_, np.bincount(y_enc)))}")
    return df, y_enc, selected_raw_cols, le


def build_feature_set(df, allowed_raw_cols, selected_raw_cols):
    """Intersect temporal window with RFECV-selected."""
    cols = [c for c in allowed_raw_cols
            if c in selected_raw_cols and c in df.columns]
    X = df[cols].copy()
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in cat_cols:
        X[c] = X[c].astype(str)
    return X, num_cols, cat_cols


# ==================== PIPELINE ===============================================

def get_pipeline(model, num_cols, cat_cols):
    transformers = [("num", SimpleImputer(strategy="median"), num_cols)]
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False), cat_cols))
    return ImbPipeline([
        ("prep", ColumnTransformer(transformers)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", model),
    ])


# ==================== OPTUNA (from alt-Stage-2pnt5) ==========================

def objective_rf(trial, X, y, num_cols, cat_cols):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features",
                                                   ["sqrt", "log2"]),
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    model = RandomForestClassifier(**param)
    pipe = get_pipeline(model, num_cols, cat_cols)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    scores = []
    for ti, vi in cv.split(X, y):
        p = clone(pipe)
        p.fit(X.iloc[ti], y[ti])
        yp = p.predict(X.iloc[vi])
        scores.append(formula_f2(y[vi], yp))
    return np.mean(scores)


# ==================== MAIN ===================================================

def main():
    print("=" * 70)
    print("  ALT-STAGE 6: FINAL DEPLOYMENT TUNING")
    print("  3 windows × 2 COVID variants = 6 configurations")
    print("=" * 70)

    df, y, selected_raw_cols, le = load_data()
    n_classes = len(le.classes_)

    all_idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  Split: Train={len(train_idx)}, Test={len(test_idx)}\n")

    configs = build_configs()
    all_results = []
    best_overall = {"F2": -1, "config": None}

    for config_idx, (config_name, allowed_cols) in enumerate(configs.items(), 1):
        print(f"\n{'='*65}")
        print(f"  [{config_idx}/6] {config_name}")
        print(f"{'='*65}")

        X, num_cols, cat_cols = build_feature_set(
            df, allowed_cols, selected_raw_cols)
        n_feats = len(X.columns)
        X_train, X_test_df = X.iloc[train_idx], X.iloc[test_idx]

        print(f"  Features: {n_feats} ({len(num_cols)} num, {len(cat_cols)} cat)")
        print(f"  Columns: {sorted(X.columns.tolist())}")

        # ---- OPTUNA TUNING ----
        print(f"\n  ⏳ Optuna ({N_TRIALS} trials)...")
        t0 = time.time()
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(
            lambda trial: objective_rf(trial, X_train, y_train,
                                       num_cols, cat_cols),
            n_trials=N_TRIALS)

        best_params = {**study.best_params,
                       "class_weight": "balanced",
                       "n_jobs": -1, "random_state": RANDOM_STATE}
        elapsed_tune = time.time() - t0
        print(f"  ✓ Best CV F2 = {study.best_value:.4f} ({elapsed_tune:.0f}s)")
        print(f"  Params: {study.best_params}")

        # ---- EVALUATE DEFAULT THRESHOLD ----
        best_model = RandomForestClassifier(**best_params)
        pipeline = get_pipeline(best_model, num_cols, cat_cols)

        # CV evaluation
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)
        cv_m = {"AUC": [], "MCC": [], "F_Rec": [], "F_Prec": []}
        oof_proba = np.zeros((len(y_train), n_classes))

        for ti, vi in cv.split(X_train, y_train):
            p = clone(pipeline)
            p.fit(X_train.iloc[ti], y_train[ti])
            yp = p.predict(X_train.iloc[vi])
            ypr = np.asarray(p.predict_proba(X_train.iloc[vi]))
            oof_proba[vi] = ypr
            cv_m["MCC"].append(matthews_corrcoef(y_train[vi], yp))
            cv_m["F_Rec"].append(recall_score(y_train[vi], yp,
                                              labels=[FORMULA_CLASS_IDX],
                                              average="micro", zero_division=0))
            cv_m["F_Prec"].append(precision_score(y_train[vi], yp,
                                                  labels=[FORMULA_CLASS_IDX],
                                                  average="micro", zero_division=0))
            try:
                cv_m["AUC"].append(roc_auc_score(y_train[vi], ypr,
                                                 multi_class="ovr", average="macro"))
            except:
                cv_m["AUC"].append(np.nan)

        # Holdout
        pipe_final = clone(pipeline)
        pipe_final.fit(X_train, y_train)
        yp_test = np.ravel(pipe_final.predict(X_test_df))
        ypr_test = np.asarray(pipe_final.predict_proba(X_test_df))

        test_auc = roc_auc_score(y_test, ypr_test, multi_class="ovr", average="macro")
        test_mcc = matthews_corrcoef(y_test, yp_test)
        test_f_rec = recall_score(y_test, yp_test, labels=[FORMULA_CLASS_IDX],
                                  average="micro", zero_division=0)
        test_f_prec = precision_score(y_test, yp_test, labels=[FORMULA_CLASS_IDX],
                                      average="micro", zero_division=0)
        test_f2 = formula_f2(y_test, yp_test)

        print(f"\n  Default threshold:")
        print(f"    CV  AUC={np.nanmean(cv_m['AUC']):.3f}±{np.nanstd(cv_m['AUC']):.3f}  "
              f"MCC={np.mean(cv_m['MCC']):.3f}  "
              f"F-Rec={np.mean(cv_m['F_Rec']):.3f}  "
              f"F-Prec={np.mean(cv_m['F_Prec']):.3f}")
        print(f"    Test AUC={test_auc:.3f}  MCC={test_mcc:.3f}  "
              f"F-Rec={test_f_rec:.3f}  F-Prec={test_f_prec:.3f}  F2={test_f2:.3f}")

        all_results.append({
            "Config": config_name, "Variant": "Default",
            "N_Features": n_feats, "Threshold": 0.33,
            "CV_AUC": round(np.nanmean(cv_m["AUC"]), 3),
            "CV_MCC": round(np.mean(cv_m["MCC"]), 3),
            "CV_Formula_Rec": round(np.mean(cv_m["F_Rec"]), 3),
            "Test_AUC": round(test_auc, 3),
            "Test_MCC": round(test_mcc, 3),
            "Test_Formula_Rec": round(test_f_rec, 3),
            "Test_Formula_Prec": round(test_f_prec, 3),
            "Test_Formula_F2": round(test_f2, 3),
            "Best_Params": str(study.best_params),
        })

        # ---- THRESHOLD OPTIMIZATION (reuse OOF probas) ----
        print(f"\n  ⏳ Threshold sweep (reusing OOF probas)...")
        thresholds = np.arange(0.10, 0.55, 0.01)
        best_f2_thr, best_thr = -1, 0.33
        for thr in thresholds:
            yp_oof = apply_threshold(oof_proba, thr, n_classes)
            f2 = formula_f2(y_train, yp_oof)
            if f2 > best_f2_thr:
                best_f2_thr = f2
                best_thr = round(thr, 2)

        yp_test_opt = apply_threshold(ypr_test, best_thr, n_classes)
        opt_mcc = matthews_corrcoef(y_test, yp_test_opt)
        opt_f_rec = recall_score(y_test, yp_test_opt,
                                 labels=[FORMULA_CLASS_IDX],
                                 average="micro", zero_division=0)
        opt_f_prec = precision_score(y_test, yp_test_opt,
                                     labels=[FORMULA_CLASS_IDX],
                                     average="micro", zero_division=0)
        opt_f2 = formula_f2(y_test, yp_test_opt)
        cm = confusion_matrix(y_test, yp_test_opt)

        print(f"  ✓ Threshold={best_thr}: "
              f"F-Rec={opt_f_rec:.3f}  F-Prec={opt_f_prec:.3f}  "
              f"MCC={opt_mcc:.3f}  F2={opt_f2:.3f}")
        print(f"    CM: {cm.tolist()}")

        all_results.append({
            "Config": config_name, "Variant": f"Thr={best_thr}",
            "N_Features": n_feats, "Threshold": best_thr,
            "CV_AUC": round(np.nanmean(cv_m["AUC"]), 3),
            "CV_MCC": round(np.mean(cv_m["MCC"]), 3),
            "CV_Formula_Rec": round(np.mean(cv_m["F_Rec"]), 3),
            "Test_AUC": round(test_auc, 3),
            "Test_MCC": round(opt_mcc, 3),
            "Test_Formula_Rec": round(opt_f_rec, 3),
            "Test_Formula_Prec": round(opt_f_prec, 3),
            "Test_Formula_F2": round(opt_f2, 3),
            "Best_Params": str(study.best_params),
        })

        # Track best (on optimized threshold F2)
        if opt_f2 > best_overall["F2"]:
            best_overall = {
                "F2": opt_f2, "config": config_name,
                "threshold": best_thr, "params": best_params,
                "pipeline": pipe_final, "features": sorted(X.columns.tolist()),
                "num_cols": num_cols, "cat_cols": cat_cols,
            }

        # ---- PICKLE each config ----
        pkl_name = config_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("&", "_")
        pkl_path = OUTPUT_DIR / f"{pkl_name}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({
                "pipeline": pipe_final,
                "threshold": best_thr,
                "params": best_params,
                "features": sorted(X.columns.tolist()),
                "class_labels": {int(i): str(c)
                                 for i, c in enumerate(le.classes_)},
            }, f)
        print(f"  ✓ Pickled: {pkl_name}.pkl")

    # ==================== SUMMARY ============================================
    df_res = pd.DataFrame(all_results)
    excel_path = OUTPUT_DIR / "stage6_final_results.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        df_res.to_excel(writer, sheet_name="Results", index=False)

    print("\n\n" + "=" * 105)
    print("  FULL RESULTS — 6 CONFIGURATIONS × 2 VARIANTS")
    print("=" * 105)
    cols = ["Config", "Variant", "N_Features", "Threshold",
            "CV_AUC", "CV_MCC", "Test_AUC", "Test_MCC",
            "Test_Formula_Rec", "Test_Formula_Prec", "Test_Formula_F2"]
    print(df_res[cols].to_string(index=False))

    print(f"\n  ★ BEST CONFIG: {best_overall['config']}")
    print(f"    Threshold: {best_overall['threshold']}")
    print(f"    Features:  {best_overall['features']}")

    print(f"\n  Saved: {excel_path}")
    print("\n" + "=" * 105)
    print("  DONE!")
    print("=" * 105)


if __name__ == "__main__":
    main()
