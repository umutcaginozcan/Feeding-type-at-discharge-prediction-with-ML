#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Alt-Stage 7: 4-Window Deployment Models
----------------------------------------------
Produces 4 deployment-ready RF models, one per temporal window:
  1. Baseline  (admission only)
  2. Day 1     (+ first 24h feeding)
  3. Day 1+2   (+ 24-48h feeding)
  4. Full D1-3 (+ 48-72h feeding)

Pipeline per model:
  1. Optuna F2-tune (50 trials)
  2. Constrained threshold: maximize F2 s.t. Formula Precision ≥ 0.40
  3. BFHI inclusion tested automatically

Metrics reported: Formula Recall/Precision, F1-Macro, MCC, Brier, AUC-ROC

COVID/Epoch variables excluded across all windows.
eng_resilience_index included only in Full (Day 3 feature).

Output: nicu_deployment/{baseline,day1,day1_2,full}_model.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
import optuna
import pickle
import warnings
import time
import copy

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    fbeta_score, matthews_corrcoef, accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, brier_score_loss
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
OUTPUT_DIR = BASE_DIR / "nicu_deployment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 50
FORMULA_CLASS_IDX = 1   # alphabetical: EBF=0, Formula=1, Mixed=2
PRECISION_FLOOR = 0.40  # minimum Formula precision for threshold selection

# ==================== FEATURE GROUPS (no COVID/Epoch) ========================

BASELINE_COLS = [
    "anneyasi",
    "dogumagirligi(gram)", "gebelikhaftası", "gebelikhaftagunu",
    "takipilkgün_kilo_gram",
    "eng_weight_per_week",
    "annesutuemzirmeeğitimidurumu",
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

DAY3_COLS = [
    "beslenmetotali3.gun", "aldıgıannesütü3.gun", "aldıgımamamiktari3.gun",
    "kilo3.gun",
    "eng_bm_ratio_d3", "eng_delta_vol_d2_d3",
    "eng_lactation_momentum",
    "eng_resilience_index",    # d3_total / birth_weight — legitimate Day 3
    "verilisyolu3gun",
]

BFHI_COL = "bebek_dostu_20temmuz2018"

# 4 temporal windows
WINDOWS = {
    "Baseline":  BASELINE_COLS,
    "Day 1":     BASELINE_COLS + DAY1_COLS,
    "Day 1+2":   BASELINE_COLS + DAY1_COLS + DAY2_COLS,
    "Full D1-3": BASELINE_COLS + DAY1_COLS + DAY2_COLS + DAY3_COLS,
}

WINDOW_KEYS = {
    "Baseline":  "baseline",
    "Day 1":     "day1",
    "Day 1+2":   "day1_2",
    "Full D1-3": "full",
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


def compute_brier_formula(y_true, y_proba):
    """Brier score for Formula class (one-vs-rest)."""
    y_binary = (y_true == FORMULA_CLASS_IDX).astype(int)
    return brier_score_loss(y_binary, y_proba[:, FORMULA_CLASS_IDX])


def compute_metrics(y_true, y_pred, y_proba, le):
    """Compute all required metrics."""
    n_classes = len(le.classes_)
    m = {}
    m["Formula_Recall"] = recall_score(
        y_true, y_pred, labels=[FORMULA_CLASS_IDX],
        average="micro", zero_division=0)
    m["Formula_Precision"] = precision_score(
        y_true, y_pred, labels=[FORMULA_CLASS_IDX],
        average="micro", zero_division=0)
    m["Formula_F2"] = formula_f2(y_true, y_pred)
    m["F1_Macro"] = f1_score(y_true, y_pred, average="macro")
    m["MCC"] = matthews_corrcoef(y_true, y_pred)
    m["Brier_Formula"] = compute_brier_formula(y_true, y_proba)
    try:
        m["AUC_ROC"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro")
    except:
        m["AUC_ROC"] = np.nan
    m["Accuracy"] = accuracy_score(y_true, y_pred)
    m["Balanced_Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    return m


def format_cell(scores):
    if isinstance(scores, (float, np.floating, int)):
        return f"{float(scores):.3f}"
    mean_val = np.mean(scores)
    return f"{mean_val:.3f} ± {np.std(scores):.3f}"


# ==================== DATA ===================================================

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    selected_df = pd.read_csv(FEAT_PATH)
    selected_feat_names = set(selected_df["Selected_Features"].tolist())

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    le = LabelEncoder()
    y_enc = le.fit_transform(df[TARGET_COL])

    # Find all raw columns that map to selected features
    all_raw_cols = [c for c in df.columns if c != TARGET_COL]
    selected_raw = set()
    for raw_col in all_raw_cols:
        if raw_col in selected_feat_names:
            selected_raw.add(raw_col)
            continue
        for sel in selected_feat_names:
            if sel.startswith(str(raw_col)):
                selected_raw.add(raw_col)
                break

    # Also add BFHI and any engineered col that exists in df
    all_possible = set()
    for window_cols in WINDOWS.values():
        all_possible.update(window_cols)
    all_possible.add(BFHI_COL)

    available = {c for c in all_possible if c in df.columns}

    X = df[list(available)].copy()

    # Identify types
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in cat_cols:
        X[c] = X[c].astype(str)

    print(f"  {len(df)} samples, {len(available)} total available columns")
    print(f"  Classes: {dict(zip(le.classes_, np.bincount(y_enc)))}")
    return X, y_enc, le


# ==================== PIPELINE ===============================================

def get_pipeline(model, feature_cols, X_all):
    """Build pipeline for a specific feature subset."""
    num_cols = [c for c in feature_cols
                if pd.api.types.is_numeric_dtype(X_all[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

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


# ==================== OPTUNA OBJECTIVE =======================================

def make_objective(X_train, y_train, feature_cols, X_all):
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 5, 35),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features',
                                                       ['sqrt', 'log2']),
            'class_weight': 'balanced',
            'n_jobs': -1,
            'random_state': RANDOM_STATE,
        }
        model = RandomForestClassifier(**param)
        pipeline = get_pipeline(model, feature_cols, X_all)
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)
        scores = []
        X_sub = X_train[feature_cols]
        for ti, vi in cv.split(X_sub, y_train):
            pipe = clone(pipeline)
            pipe.fit(X_sub.iloc[ti], y_train[ti])
            yp = pipe.predict(X_sub.iloc[vi])
            scores.append(formula_f2(y_train[vi], yp))
        return np.mean(scores)
    return objective


# ==================== CONSTRAINED THRESHOLD ==================================

def optimize_threshold_constrained(pipeline, X_train, y_train, feature_cols,
                                    n_classes, precision_floor=0.40):
    """
    Maximize F2 subject to Formula Precision ≥ precision_floor.
    Falls back to max-F2 threshold if NO threshold meets constraint.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(y_train), n_classes))
    X_sub = X_train[feature_cols]

    for ti, vi in cv.split(X_sub, y_train):
        p = clone(pipeline)
        p.fit(X_sub.iloc[ti], y_train[ti])
        oof_proba[vi] = np.asarray(p.predict_proba(X_sub.iloc[vi]))

    thresholds = np.arange(0.10, 0.55, 0.005)
    sweep = []
    best_f2_constrained = -1
    best_thr_constrained = None
    best_f2_unconstrained = -1
    best_thr_unconstrained = None

    for thr in thresholds:
        yp = apply_threshold(oof_proba, thr, n_classes)
        f2 = formula_f2(y_train, yp)
        rec = recall_score(y_train, yp, labels=[FORMULA_CLASS_IDX],
                           average="micro", zero_division=0)
        prec = precision_score(y_train, yp, labels=[FORMULA_CLASS_IDX],
                               average="micro", zero_division=0)
        mcc = matthews_corrcoef(y_train, yp)
        brier = compute_brier_formula(y_train, oof_proba)

        sweep.append({
            "Threshold": round(thr, 3),
            "Formula_F2": round(f2, 4),
            "Formula_Recall": round(rec, 4),
            "Formula_Precision": round(prec, 4),
            "MCC": round(mcc, 4),
            "Brier": round(brier, 4),
        })

        # Unconstrained best
        if f2 > best_f2_unconstrained:
            best_f2_unconstrained = f2
            best_thr_unconstrained = round(thr, 3)

        # Constrained best (precision ≥ floor)
        if prec >= precision_floor and f2 > best_f2_constrained:
            best_f2_constrained = f2
            best_thr_constrained = round(thr, 3)

    if best_thr_constrained is not None:
        print(f"    ✓ Constrained threshold: {best_thr_constrained} "
              f"(F2={best_f2_constrained:.3f}, prec≥{precision_floor})")
        return best_thr_constrained, sweep
    else:
        print(f"    ⚠ No threshold meets precision≥{precision_floor}. "
              f"Falling back to unconstrained: {best_thr_unconstrained}")
        return best_thr_unconstrained, sweep


# ==================== BFHI INCLUSION TEST ====================================

def test_bfhi_inclusion(X_train, y_train, base_cols, X_all):
    """Test if adding BFHI improves CV F2."""
    if BFHI_COL not in X_all.columns:
        print(f"    BFHI column '{BFHI_COL}' not in data. Skipping.")
        return False

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE)

    # Without BFHI
    cols_without = [c for c in base_cols if c in X_all.columns]
    pipe_without = get_pipeline(model, cols_without, X_all)
    scores_without = []
    for ti, vi in cv.split(X_train[cols_without], y_train):
        p = clone(pipe_without)
        p.fit(X_train[cols_without].iloc[ti], y_train[ti])
        yp = p.predict(X_train[cols_without].iloc[vi])
        scores_without.append(formula_f2(y_train[vi], yp))

    # With BFHI
    cols_with = cols_without + [BFHI_COL]
    pipe_with = get_pipeline(model, cols_with, X_all)
    scores_with = []
    for ti, vi in cv.split(X_train[cols_with], y_train):
        p = clone(pipe_with)
        p.fit(X_train[cols_with].iloc[ti], y_train[ti])
        yp = p.predict(X_train[cols_with].iloc[vi])
        scores_with.append(formula_f2(y_train[vi], yp))

    f2_without = np.mean(scores_without)
    f2_with = np.mean(scores_with)
    delta = f2_with - f2_without
    include = delta > 0.005  # meaningful improvement threshold

    print(f"    BFHI test: F2 without={f2_without:.4f}, "
          f"with={f2_with:.4f}, Δ={delta:+.4f} → "
          f"{'INCLUDE' if include else 'EXCLUDE'}")
    return include


# ==================== MAIN ===================================================

def main():
    print("=" * 70)
    print("  ALT-STAGE 7: 4-WINDOW DEPLOYMENT MODELS")
    print("  F2-optimized + constrained threshold (precision ≥ 0.40)")
    print("  COVID excluded | BFHI auto-tested")
    print("=" * 70)

    X_all, y, le = load_data()
    n_classes = len(le.classes_)

    # Global train/test split (same split for all windows)
    all_idx = np.arange(len(X_all))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train_all, X_test_all = X_all.iloc[train_idx], X_all.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  Split: Train={len(train_idx)}, Test={len(test_idx)}\n")

    # Results storage
    all_results = []
    all_sweeps = {}
    all_cms = {}

    for window_name, window_cols_raw in WINDOWS.items():
        print("\n" + "=" * 70)
        print(f"  WINDOW: {window_name}")
        print("=" * 70)

        # Filter to columns that actually exist in data
        feature_cols = [c for c in window_cols_raw if c in X_all.columns]

        # ---- BFHI inclusion test ----
        print("  Testing BFHI inclusion...")
        include_bfhi = test_bfhi_inclusion(
            X_train_all, y_train, feature_cols, X_all)
        if include_bfhi:
            feature_cols = feature_cols + [BFHI_COL]

        print(f"  Final features ({len(feature_cols)}): {sorted(feature_cols)}")

        # ---- Optuna F2-tune ----
        print(f"\n  Optuna F2-tuning ({N_TRIALS} trials)...")
        t0 = time.time()
        study = optuna.create_study(direction="maximize")
        study.optimize(
            make_objective(X_train_all, y_train, feature_cols, X_all),
            n_trials=N_TRIALS)
        elapsed = time.time() - t0

        bp = study.best_params.copy()
        bp.update({"class_weight": "balanced", "n_jobs": -1,
                   "random_state": RANDOM_STATE})
        print(f"    Best CV F2: {study.best_value:.4f} ({elapsed:.0f}s)")
        print(f"    Params: {bp}")

        # ---- Build pipeline with best params ----
        model = RandomForestClassifier(**bp)
        pipeline = get_pipeline(model, feature_cols, X_all)

        # ---- CV evaluation (default threshold) ----
        print("\n  CV evaluation (default threshold)...")
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)
        cv_metrics = []
        X_sub_train = X_train_all[feature_cols]
        X_sub_test = X_test_all[feature_cols]

        for ti, vi in cv.split(X_sub_train, y_train):
            p = clone(pipeline)
            p.fit(X_sub_train.iloc[ti], y_train[ti])
            yp = p.predict(X_sub_train.iloc[vi])
            ypr = np.asarray(p.predict_proba(X_sub_train.iloc[vi]))
            cv_metrics.append(compute_metrics(y_train[vi], yp, ypr, le))

        print("    CV Results (5-fold, default):")
        for k in ["Formula_Recall", "Formula_Precision", "Formula_F2",
                   "F1_Macro", "MCC", "Brier_Formula", "AUC_ROC"]:
            vals = [m[k] for m in cv_metrics]
            print(f"      {k:20s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

        # ---- Constrained threshold optimization ----
        print("\n  Constrained threshold optimization "
              f"(precision ≥ {PRECISION_FLOOR})...")
        thr_pipeline = get_pipeline(
            RandomForestClassifier(**bp), feature_cols, X_all)
        best_thr, sweep = optimize_threshold_constrained(
            thr_pipeline, X_train_all, y_train, feature_cols,
            n_classes, PRECISION_FLOOR)
        all_sweeps[window_name] = sweep

        # ---- Train final pipeline on full training set ----
        pipe_final = clone(pipeline)
        pipe_final.fit(X_sub_train, y_train)

        # ---- Test evaluation (default) ----
        y_proba_test = np.asarray(pipe_final.predict_proba(X_sub_test))
        y_pred_default = np.ravel(pipe_final.predict(X_sub_test))
        m_default = compute_metrics(y_test, y_pred_default, y_proba_test, le)
        cm_default = confusion_matrix(y_test, y_pred_default)

        print(f"\n  Test Set (default threshold):")
        for k in ["Formula_Recall", "Formula_Precision", "Formula_F2",
                   "F1_Macro", "MCC", "Brier_Formula", "AUC_ROC"]:
            print(f"    {k:20s}: {m_default[k]:.3f}")

        # ---- Test evaluation (optimized threshold) ----
        y_pred_opt = apply_threshold(y_proba_test, best_thr, n_classes)
        m_opt = compute_metrics(y_test, y_pred_opt, y_proba_test, le)
        cm_opt = confusion_matrix(y_test, y_pred_opt)

        print(f"\n  Test Set (threshold = {best_thr}):")
        for k in ["Formula_Recall", "Formula_Precision", "Formula_F2",
                   "F1_Macro", "MCC", "Brier_Formula", "AUC_ROC"]:
            print(f"    {k:20s}: {m_opt[k]:.3f}")
        print(f"    Confusion Matrix:\n{cm_opt}")

        all_cms[f"{window_name} (default)"] = cm_default
        all_cms[f"{window_name} (thr={best_thr})"] = cm_opt

        # ---- Store results ----
        row_default = {"Window": window_name, "Variant": "Default",
                       "Threshold": "argmax", "N_Features": len(feature_cols),
                       "BFHI_Included": include_bfhi}
        row_default.update({f"Test_{k}": round(v, 4)
                           for k, v in m_default.items()})
        # CV metrics
        for k in cv_metrics[0].keys():
            vals = [m[k] for m in cv_metrics]
            row_default[f"CV_{k}"] = format_cell(vals)
        all_results.append(row_default)

        row_opt = {"Window": window_name, "Variant": f"Thr={best_thr}",
                   "Threshold": best_thr, "N_Features": len(feature_cols),
                   "BFHI_Included": include_bfhi}
        row_opt.update({f"Test_{k}": round(v, 4) for k, v in m_opt.items()})
        all_results.append(row_opt)

        # ---- Save model bundle ----
        key = WINDOW_KEYS[window_name]
        pkl_path = OUTPUT_DIR / f"{key}_model.pkl"
        bundle = {
            "pipeline": pipe_final,
            "threshold": best_thr,
            "features": sorted(feature_cols),
            "class_labels": {int(i): str(c) for i, c in enumerate(le.classes_)},
            "params": bp,
            "window_name": window_name,
            "bfhi_included": include_bfhi,
            "test_metrics": {k: round(v, 4) for k, v in m_opt.items()},
            "test_metrics_default": {k: round(v, 4)
                                     for k, v in m_default.items()},
            "cv_metrics": {
                k: {"mean": round(np.mean([m[k] for m in cv_metrics]), 4),
                    "std": round(np.std([m[k] for m in cv_metrics]), 4)}
                for k in cv_metrics[0].keys()
            },
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"\n  ✓ Saved: {pkl_path} "
              f"({pkl_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # ==================== SUMMARY TABLE ======================================

    print("\n\n" + "=" * 100)
    print("  SUMMARY: ALL WINDOWS (threshold-optimized)")
    print("=" * 100)

    header = (f"  {'Window':12s} {'Thr':>5s} {'Feat':>4s}  "
              f"{'F_Rec':>6s} {'F_Prec':>6s} {'F_F2':>6s}  "
              f"{'F1-Mac':>6s} {'MCC':>6s} {'Brier':>6s} {'AUC':>6s}")
    print(header)
    print("  " + "-" * 90)

    for row in all_results:
        if row["Variant"] == "Default":
            continue
        thr_str = str(row['Threshold'])
        print(f"  {row['Window']:12s} {thr_str:>5s}   "
              f"{row['N_Features']:>3d}  "
              f"{row['Test_Formula_Recall']:>6.3f} "
              f"{row['Test_Formula_Precision']:>6.3f} "
              f"{row['Test_Formula_F2']:>6.3f}  "
              f"{row['Test_F1_Macro']:>6.3f} "
              f"{row['Test_MCC']:>6.3f} "
              f"{row['Test_Brier_Formula']:>6.3f} "
              f"{row['Test_AUC_ROC']:>6.3f}")

    # ==================== SAVE TO EXCEL ======================================

    excel_path = OUTPUT_DIR / "stage7_4window_results.xlsx"
    print(f"\n  Saving to {excel_path}...")

    with pd.ExcelWriter(excel_path) as writer:
        # Results
        df_res = pd.DataFrame(all_results)
        priority = ["Window", "Variant", "Threshold", "N_Features",
                     "BFHI_Included",
                     "Test_Formula_Recall", "Test_Formula_Precision",
                     "Test_Formula_F2", "Test_F1_Macro", "Test_MCC",
                     "Test_Brier_Formula", "Test_AUC_ROC"]
        other = [c for c in df_res.columns if c not in priority]
        col_order = [c for c in priority if c in df_res.columns] + other
        df_res[col_order].to_excel(writer, sheet_name="Results", index=False)

        # Threshold sweeps
        for wname, sweep in all_sweeps.items():
            sheet = f"Thr_{WINDOW_KEYS.get(wname, wname)}"[:31]
            pd.DataFrame(sweep).to_excel(writer, sheet_name=sheet, index=False)

        # Confusion matrices
        class_names = list(le.classes_)
        for mname, cm_arr in all_cms.items():
            df_cm = pd.DataFrame(
                cm_arr,
                index=[f"Actual: {c}" for c in class_names],
                columns=[f"Predicted: {c}" for c in class_names])
            sheet = f"CM_{mname}"[:31]
            df_cm.to_excel(writer, sheet_name=sheet)

    print(f"  ✓ Saved: {excel_path}")

    print("\n" + "=" * 70)
    print("  DONE! 4 model bundles saved to nicu_deployment/")
    print("=" * 70)


if __name__ == "__main__":
    main()
