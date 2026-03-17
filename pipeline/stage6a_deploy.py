#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Alt-Stage 6: Deployment Optimization
--------------------------------------------
Takes the Optuna-tuned RF from alt-Stage-2pnt5-f2-optimize.py and runs:

  TEST 1: Temporal ablation — baseline, +day1, +day1+2 windows
  TEST 2: COVID removal — best window with vs without covid/epoch

Uses alt-Stage-5-ablation.py architecture for temporal windowing.
Uses alt-Stage-2pnt5 tuned RF (n_estimators=255, max_depth=21, etc.)
with threshold optimization.

Output: ~/Desktop/nicu_deployment/
"""

import pandas as pd
import numpy as np
from pathlib import Path
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

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_stage0_5_cleaned.xlsx"
FEAT_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_selected_features.csv"
OUTPUT_DIR = Path.home() / "Desktop" / "nicu_deployment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
N_FOLDS = 5
FORMULA_CLASS_IDX = 1  # alphabetical: EBF=0, Formula=1, Mixed=2

# ==================== TUNED RF (from alt-Stage-2pnt5 best_params.xlsx) =======

TUNED_RF_PARAMS = dict(
    n_estimators=255,
    max_depth=21,
    min_samples_split=15,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# ==================== TEMPORAL FEATURE GROUPS (from alt-Stage-5) =============

BASELINE_COLS = [
    "anneyasi",
    "dogumagirligi(gram)", "gebelikhaftası", "gebelikhaftagunu",
    "takipilkgün_kilo_gram",
    "eng_weight_per_week",
    "annesutuemzirmeeğitimidurumu",
    "covid19sonrasi", "ikisiarası",      # COVID/Epoch — kept for TEST 1
]

DAY1_COLS = [
    "aldığıannesütü_ilkgün", "aldığımamamiktari1.gün", "kilo1.gun",
    "eng_bm_ratio_d1",
    "ilk_gün_anne_sütü_1111", "ilk_gün_emzirme_111",
]

DAY2_COLS = [
    "beslenmetotali2.gün", "beslenme2.gunannesutucc",
    "beslenmemamamiktarı2.guncc", "kilo2.gun",
    "eng_bm_ratio_d2", "eng_delta_vol_d1_d2",
]

# ==================== TEST CONFIGURATIONS ====================================

COVID_COLS = {"covid19sonrasi", "ikisiarası"}

# TEST 1: Temporal ablation (with COVID)
TEST1_WINDOWS = {
    "Baseline":  BASELINE_COLS,
    "+Day 1":    BASELINE_COLS + DAY1_COLS,
    "+Day 1&2":  BASELINE_COLS + DAY1_COLS + DAY2_COLS,
}

# TEST 2: COVID removal on Day 1+2 window
TEST2_WINDOWS = {
    "+Day 1&2 (with COVID)":    BASELINE_COLS + DAY1_COLS + DAY2_COLS,
    "+Day 1&2 (no COVID)":      [c for c in BASELINE_COLS if c not in COVID_COLS]
                                + DAY1_COLS + DAY2_COLS,
}


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


# ==================== PIPELINE (from alt-Stage-5) ============================

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


# ==================== THRESHOLD OPTIMIZATION =================================

def optimize_threshold(pipeline, X_train, y_train, n_classes):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(y_train), n_classes))
    for ti, vi in cv.split(X_train, y_train):
        p = clone(pipeline)
        p.fit(X_train.iloc[ti], y_train[ti])
        oof_proba[vi] = np.asarray(p.predict_proba(X_train.iloc[vi]))

    thresholds = np.arange(0.10, 0.55, 0.01)
    best_f2, best_thr = -1, 0.33
    sweep = []
    for thr in thresholds:
        yp = apply_threshold(oof_proba, thr, n_classes)
        f2 = formula_f2(y_train, yp)
        rec = recall_score(y_train, yp, labels=[FORMULA_CLASS_IDX],
                           average="micro", zero_division=0)
        prec = precision_score(y_train, yp, labels=[FORMULA_CLASS_IDX],
                               average="micro", zero_division=0)
        mcc = matthews_corrcoef(y_train, yp)
        sweep.append({"Threshold": round(thr, 2), "F2": round(f2, 4),
                       "Recall": round(rec, 4), "Precision": round(prec, 4),
                       "MCC": round(mcc, 4)})
        if f2 > best_f2:
            best_f2 = f2
            best_thr = round(thr, 2)
    return best_thr, sweep


# ==================== EVALUATION =============================================

def evaluate_full(pipeline, X_train, y_train, X_test, y_test, le):
    """CV metrics + holdout, returns (cv_metrics, test_metrics, fitted_pipe, y_proba)."""
    n_classes = len(le.classes_)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_m = {"AUC_ROC": [], "MCC": [], "F1_Macro": [],
            "Formula_Rec": [], "Formula_Prec": []}

    for ti, vi in cv.split(X_train, y_train):
        Xt, yt = X_train.iloc[ti], y_train[ti]
        Xv, yv = X_train.iloc[vi], y_train[vi]
        p = clone(pipeline)
        p.fit(Xt, yt)
        yp = np.ravel(p.predict(Xv))
        ypr = np.asarray(p.predict_proba(Xv))
        cv_m["F1_Macro"].append(f1_score(yv, yp, average="macro"))
        cv_m["MCC"].append(matthews_corrcoef(yv, yp))
        cv_m["Formula_Rec"].append(recall_score(yv, yp, labels=[FORMULA_CLASS_IDX],
                                                average="micro", zero_division=0))
        cv_m["Formula_Prec"].append(precision_score(yv, yp, labels=[FORMULA_CLASS_IDX],
                                                    average="micro", zero_division=0))
        try:
            cv_m["AUC_ROC"].append(roc_auc_score(yv, ypr, multi_class="ovr",
                                                  average="macro"))
        except:
            cv_m["AUC_ROC"].append(np.nan)

    # Holdout
    pipe_full = clone(pipeline)
    pipe_full.fit(X_train, y_train)
    yp_test = np.ravel(pipe_full.predict(X_test))
    ypr_test = np.asarray(pipe_full.predict_proba(X_test))

    test_m = {
        "AUC_ROC": roc_auc_score(y_test, ypr_test, multi_class="ovr", average="macro"),
        "MCC": matthews_corrcoef(y_test, yp_test),
        "F1_Macro": f1_score(y_test, yp_test, average="macro"),
        "Formula_Rec": recall_score(y_test, yp_test, labels=[FORMULA_CLASS_IDX],
                                    average="micro", zero_division=0),
        "Formula_Prec": precision_score(y_test, yp_test, labels=[FORMULA_CLASS_IDX],
                                        average="micro", zero_division=0),
        "Formula_F2": formula_f2(y_test, yp_test),
    }
    return cv_m, test_m, pipe_full, ypr_test


# ==================== MAIN ===================================================

def main():
    print("=" * 70)
    print("  ALT-STAGE 6: DEPLOYMENT OPTIMIZATION")
    print("  (Tuned RF from alt-Stage-2pnt5 × temporal ablation × COVID test)")
    print("=" * 70)

    df, y, selected_raw_cols, le = load_data()
    n_classes = len(le.classes_)

    all_idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  Split: Train={len(train_idx)}, Test={len(test_idx)}\n")

    all_results = []

    def run_test(test_name, windows):
        print(f"\n{'#'*70}")
        print(f"  {test_name}")
        print(f"{'#'*70}")

        for win_name, allowed_cols in windows.items():
            print(f"\n{'='*55}")
            print(f"  Window: {win_name}")
            print(f"{'='*55}")

            X, num_cols, cat_cols = build_feature_set(
                df, allowed_cols, selected_raw_cols)
            n_feats = len(X.columns)
            X_train, X_test_df = X.iloc[train_idx], X.iloc[test_idx]

            print(f"  Features: {n_feats} ({len(num_cols)} num, {len(cat_cols)} cat)")
            print(f"  Columns: {sorted(X.columns.tolist())}")

            # --- Evaluate (default threshold) ---
            model = RandomForestClassifier(**TUNED_RF_PARAMS)
            pipeline = get_pipeline(model, num_cols, cat_cols)

            t0 = time.time()
            cv_m, test_m, fitted_pipe, y_proba = evaluate_full(
                pipeline, X_train, y_train, X_test_df, y_test, le)
            elapsed = time.time() - t0

            print(f"\n  Default threshold:")
            print(f"    CV  AUC: {np.nanmean(cv_m['AUC_ROC']):.3f} ± {np.nanstd(cv_m['AUC_ROC']):.3f}  |  "
                  f"MCC: {np.mean(cv_m['MCC']):.3f}  |  "
                  f"F-Rec: {np.mean(cv_m['Formula_Rec']):.3f}")
            print(f"    Test AUC: {test_m['AUC_ROC']:.3f}  |  "
                  f"MCC: {test_m['MCC']:.3f}  |  "
                  f"F-Rec: {test_m['Formula_Rec']:.3f}  |  "
                  f"F-Prec: {test_m['Formula_Prec']:.3f}")

            row_default = {
                "Test": test_name, "Window": win_name, "Variant": "Default",
                "N_Features": n_feats, "Threshold": 0.33,
                "CV_AUC": f"{np.nanmean(cv_m['AUC_ROC']):.3f} ± {np.nanstd(cv_m['AUC_ROC']):.3f}",
                "CV_MCC": f"{np.mean(cv_m['MCC']):.3f} ± {np.std(cv_m['MCC']):.3f}",
                "CV_Formula_Rec": f"{np.mean(cv_m['Formula_Rec']):.3f} ± {np.std(cv_m['Formula_Rec']):.3f}",
                **{f"Test_{k}": round(v, 3) for k, v in test_m.items()},
                "Time": round(elapsed, 1),
            }
            all_results.append(row_default)

            # --- Threshold optimization ---
            print(f"\n  Threshold optimization...")
            thr_pipeline = get_pipeline(
                RandomForestClassifier(**TUNED_RF_PARAMS), num_cols, cat_cols)
            best_thr, sweep = optimize_threshold(
                thr_pipeline, X_train, y_train, n_classes)

            y_pred_opt = apply_threshold(y_proba, best_thr, n_classes)
            m_opt = {
                "AUC_ROC": test_m["AUC_ROC"],  # AUC doesn't change with threshold
                "MCC": matthews_corrcoef(y_test, y_pred_opt),
                "F1_Macro": f1_score(y_test, y_pred_opt, average="macro"),
                "Formula_Rec": recall_score(y_test, y_pred_opt,
                                            labels=[FORMULA_CLASS_IDX],
                                            average="micro", zero_division=0),
                "Formula_Prec": precision_score(y_test, y_pred_opt,
                                                labels=[FORMULA_CLASS_IDX],
                                                average="micro", zero_division=0),
                "Formula_F2": formula_f2(y_test, y_pred_opt),
            }

            cm = confusion_matrix(y_test, y_pred_opt)
            print(f"    Threshold={best_thr}: "
                  f"F-Rec={m_opt['Formula_Rec']:.3f}  |  "
                  f"F-Prec={m_opt['Formula_Prec']:.3f}  |  "
                  f"MCC={m_opt['MCC']:.3f}  |  "
                  f"F2={m_opt['Formula_F2']:.3f}")
            print(f"    CM: {cm.tolist()}")

            row_opt = {
                "Test": test_name, "Window": win_name, "Variant": f"Thr={best_thr}",
                "N_Features": n_feats, "Threshold": best_thr,
                "CV_AUC": row_default["CV_AUC"],
                "CV_MCC": row_default["CV_MCC"],
                "CV_Formula_Rec": row_default["CV_Formula_Rec"],
                **{f"Test_{k}": round(v, 3) for k, v in m_opt.items()},
                "Time": round(elapsed, 1),
            }
            all_results.append(row_opt)

            # Pickle the best (threshold-tuned) pipeline
            pkl_name = f"{win_name.replace(' ', '_').replace('(', '').replace(')', '').replace('&', '_').lower()}_rf.pkl"
            pkl_path = OUTPUT_DIR / pkl_name
            with open(pkl_path, "wb") as f:
                pickle.dump({
                    "pipeline": fitted_pipe,
                    "threshold": best_thr,
                    "features": sorted(X.columns.tolist()),
                    "class_labels": {int(i): str(c) for i, c in enumerate(le.classes_)},
                }, f)
            print(f"    ✓ Pickled: {pkl_name}")

    # ---- RUN TESTS ----
    run_test("TEST 1: Temporal Ablation", TEST1_WINDOWS)
    run_test("TEST 2: COVID Removal", TEST2_WINDOWS)

    # ==================== SAVE RESULTS =======================================
    df_res = pd.DataFrame(all_results)
    excel_path = OUTPUT_DIR / "stage6_deployment_results.xlsx"

    with pd.ExcelWriter(excel_path) as writer:
        df_res.to_excel(writer, sheet_name="Results", index=False)

    # ==================== SUMMARY ============================================
    print("\n\n" + "=" * 100)
    print("  FULL RESULTS")
    print("=" * 100)
    cols = ["Test", "Window", "Variant", "N_Features", "Threshold",
            "Test_AUC_ROC", "Test_MCC", "Test_Formula_Rec",
            "Test_Formula_Prec", "Test_Formula_F2"]
    print(df_res[cols].to_string(index=False))

    print(f"\n  Saved: {excel_path}")
    print("\n  Pickles:")
    for f in sorted(OUTPUT_DIR.glob("*.pkl")):
        print(f"    {f.name}")

    print("\n" + "=" * 100)
    print("  DONE!")
    print("=" * 100)


if __name__ == "__main__":
    main()
