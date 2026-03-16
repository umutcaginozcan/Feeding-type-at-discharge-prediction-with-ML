#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Alt-Stage 7: Final Deployment Model
------------------------------------------
Fresh Optuna F2-tune on Day 1+2 features, no COVID, no eng_resilience_index.
Produces a single deployment-ready pickle bundle.

Output: nicu_deployment/final_model.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
import optuna
import pickle
import warnings
import time

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    fbeta_score, matthews_corrcoef, accuracy_score,
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
OUTPUT_DIR = BASE_DIR / "nicu_deployment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 80          # More trials for final model
FORMULA_CLASS_IDX = 1   # alphabetical: EBF=0, Formula=1, Mixed=2

# ==================== FEATURE SET (Day 1+2, no COVID, no eng_resilience_index) =

# Baseline (no COVID/Epoch)
BASELINE_COLS = [
    "anneyasi",
    "dogumagirligi(gram)", "gebelikhaftası", "gebelikhaftagunu",
    "takipilkgün_kilo_gram",
    "eng_weight_per_week",
    "annesutuemzirmeeğitimidurumu",
    # covid19sonrasi, ikisiarası — excluded (redundant per ablation)
]

DAY1_COLS = [
    "aldığıannesütü_ilkgün", "aldığımamamiktari1.gün", "kilo1.gun",
    "eng_bm_ratio_d1",
    # eng_resilience_index — excluded (uses Day 3 data)
    "ilk_gün_anne_sütü_1111", "ilk_gün_emzirme_111",
]

DAY2_COLS = [
    "beslenmetotali2.gün", "beslenme2.gunannesutucc",
    "beslenmemamamiktarı2.guncc", "kilo2.gun",
    "eng_bm_ratio_d2", "eng_delta_vol_d1_d2",
]

ALLOWED_COLS = BASELINE_COLS + DAY1_COLS + DAY2_COLS


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


# ==================== DATA ===================================================

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

    # Intersect with our allowed feature set
    cols = [c for c in ALLOWED_COLS
            if c in selected_raw_cols and c in df.columns]
    X = df[cols].copy()
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in cat_cols:
        X[c] = X[c].astype(str)

    print(f"  {len(df)} samples, {len(cols)} features")
    print(f"  Features: {sorted(cols)}")
    print(f"  Classes: {dict(zip(le.classes_, np.bincount(y_enc)))}")
    return X, y_enc, num_cols, cat_cols, le


# ==================== PIPELINE ===============================================

def get_pipeline(model, num_cols, cat_cols):
    transformers = [("num", SimpleImputer(strategy="median"), num_cols)]
    if cat_cols:
        from sklearn.preprocessing import OneHotEncoder
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False), cat_cols))
    return ImbPipeline([
        ("prep", ColumnTransformer(transformers)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", model),
    ])


# ==================== OPTUNA F2-TUNING =======================================

def objective(trial, X, y, num_cols, cat_cols):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 5, 40),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': RANDOM_STATE,
    }
    model = RandomForestClassifier(**param)
    pipeline = get_pipeline(model, num_cols, cat_cols)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        pipe = clone(pipeline)
        pipe.fit(X.iloc[train_idx], y[train_idx])
        y_pred = pipe.predict(X.iloc[val_idx])
        scores.append(formula_f2(y[val_idx], y_pred))
    return np.mean(scores)


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


# ==================== MAIN ===================================================

def main():
    print("=" * 70)
    print("  ALT-STAGE 7: FINAL DEPLOYMENT MODEL")
    print("  Day 1+2, no COVID, no eng_resilience_index")
    print("  Fresh Optuna F2-optimization")
    print("=" * 70)

    X, y, num_cols, cat_cols, le = load_data()
    n_classes = len(le.classes_)

    all_idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  Split: Train={len(train_idx)}, Test={len(test_idx)}\n")

    # ---- Phase 1: Optuna Tuning ----
    print("=" * 50)
    print(f"  OPTUNA F2-TUNING ({N_TRIALS} trials)")
    print("=" * 50)

    t0 = time.time()
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda t: objective(t, X_train, y_train, num_cols, cat_cols),
        n_trials=N_TRIALS)
    elapsed = time.time() - t0

    bp = study.best_params.copy()
    bp.update({'class_weight': 'balanced', 'n_jobs': -1,
               'random_state': RANDOM_STATE})

    print(f"\n  Best CV F2: {study.best_value:.4f} ({elapsed:.0f}s)")
    print(f"  Best params: {bp}\n")

    # ---- Phase 2: Full evaluation with best params ----
    print("=" * 50)
    print("  EVALUATION (default + optimized threshold)")
    print("=" * 50)

    model = RandomForestClassifier(**bp)
    pipeline = get_pipeline(model, num_cols, cat_cols)

    # CV metrics
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    cv_m = {"AUC_ROC": [], "MCC": [], "F1_Macro": [],
            "Formula_Rec": [], "Formula_Prec": [], "Formula_F2": []}

    for ti, vi in cv.split(X_train, y_train):
        p = clone(pipeline)
        p.fit(X_train.iloc[ti], y_train[ti])
        yp = np.ravel(p.predict(X_train.iloc[vi]))
        ypr = np.asarray(p.predict_proba(X_train.iloc[vi]))
        yv = y_train[vi]

        cv_m["F1_Macro"].append(f1_score(yv, yp, average="macro"))
        cv_m["MCC"].append(matthews_corrcoef(yv, yp))
        cv_m["Formula_Rec"].append(recall_score(yv, yp, labels=[FORMULA_CLASS_IDX],
                                                average="micro", zero_division=0))
        cv_m["Formula_Prec"].append(precision_score(yv, yp, labels=[FORMULA_CLASS_IDX],
                                                    average="micro", zero_division=0))
        cv_m["Formula_F2"].append(formula_f2(yv, yp))
        try:
            cv_m["AUC_ROC"].append(roc_auc_score(yv, ypr, multi_class="ovr",
                                                  average="macro"))
        except:
            cv_m["AUC_ROC"].append(np.nan)

    print("\n  CV Results (5-fold):")
    for k, vals in cv_m.items():
        print(f"    {k:15s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # Holdout (default threshold)
    pipe_final = clone(pipeline)
    pipe_final.fit(X_train, y_train)
    y_pred_default = np.ravel(pipe_final.predict(X_test))
    y_proba_test = np.asarray(pipe_final.predict_proba(X_test))

    print("\n  Test Set (default threshold):")
    print(f"    AUC-ROC:        {roc_auc_score(y_test, y_proba_test, multi_class='ovr', average='macro'):.3f}")
    print(f"    MCC:            {matthews_corrcoef(y_test, y_pred_default):.3f}")
    print(f"    F1-Macro:       {f1_score(y_test, y_pred_default, average='macro'):.3f}")
    print(f"    Formula Recall: {recall_score(y_test, y_pred_default, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0):.3f}")
    print(f"    Formula Prec:   {precision_score(y_test, y_pred_default, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0):.3f}")
    print(f"    Formula F2:     {formula_f2(y_test, y_pred_default):.3f}")

    # ---- Phase 3: Threshold optimization ----
    print("\n  Threshold optimization...")
    thr_pipeline = get_pipeline(RandomForestClassifier(**bp), num_cols, cat_cols)
    best_thr, sweep = optimize_threshold(thr_pipeline, X_train, y_train, n_classes)
    print(f"    Best threshold: {best_thr}")

    # Evaluate with optimized threshold
    y_pred_opt = apply_threshold(y_proba_test, best_thr, n_classes)
    cm = confusion_matrix(y_test, y_pred_opt)

    auc = roc_auc_score(y_test, y_proba_test, multi_class="ovr", average="macro")
    mcc = matthews_corrcoef(y_test, y_pred_opt)
    f1m = f1_score(y_test, y_pred_opt, average="macro")
    f_rec = recall_score(y_test, y_pred_opt, labels=[FORMULA_CLASS_IDX],
                         average="micro", zero_division=0)
    f_prec = precision_score(y_test, y_pred_opt, labels=[FORMULA_CLASS_IDX],
                             average="micro", zero_division=0)
    f_f2 = formula_f2(y_test, y_pred_opt)
    acc = accuracy_score(y_test, y_pred_opt)

    print(f"\n  Test Set (threshold = {best_thr}):")
    print(f"    AUC-ROC:        {auc:.3f}")
    print(f"    MCC:            {mcc:.3f}")
    print(f"    F1-Macro:       {f1m:.3f}")
    print(f"    Accuracy:       {acc:.3f}")
    print(f"    Formula Recall: {f_rec:.3f}")
    print(f"    Formula Prec:   {f_prec:.3f}")
    print(f"    Formula F2:     {f_f2:.3f}")
    print(f"    Confusion Matrix:\n{cm}")

    # ---- Phase 4: Save model bundle ----
    pkl_path = OUTPUT_DIR / "final_model.pkl"
    bundle = {
        "pipeline": pipe_final,
        "threshold": best_thr,
        "features": sorted(X.columns.tolist()),
        "class_labels": {int(i): str(c) for i, c in enumerate(le.classes_)},
        "params": bp,
        "test_metrics": {
            "AUC_ROC": round(auc, 3),
            "MCC": round(mcc, 3),
            "F1_Macro": round(f1m, 3),
            "Accuracy": round(acc, 3),
            "Formula_Recall": round(f_rec, 3),
            "Formula_Precision": round(f_prec, 3),
            "Formula_F2": round(f_f2, 3),
        },
        "cv_metrics": {k: {"mean": round(np.mean(v), 3),
                           "std": round(np.std(v), 3)}
                       for k, v in cv_m.items()},
        "threshold_sweep": sweep,
        "feature_window": "Day 1+2 (no COVID, no eng_resilience_index)",
    }

    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  ✓ Saved model bundle: {pkl_path}")
    print(f"    Size: {pkl_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Save results to Excel
    excel_path = OUTPUT_DIR / "stage7_final_results.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        # CV summary
        cv_df = pd.DataFrame([{
            "Metric": k,
            "Mean": round(np.mean(v), 4),
            "Std": round(np.std(v), 4),
            "Folds": str([round(x, 4) for x in v])
        } for k, v in cv_m.items()])
        cv_df.to_excel(writer, sheet_name="CV Results", index=False)

        # Test results
        test_df = pd.DataFrame([{
            "Variant": "Default (argmax)",
            "AUC_ROC": round(roc_auc_score(y_test, y_proba_test, multi_class="ovr", average="macro"), 3),
            "MCC": round(matthews_corrcoef(y_test, y_pred_default), 3),
            "F1_Macro": round(f1_score(y_test, y_pred_default, average="macro"), 3),
            "Formula_Recall": round(recall_score(y_test, y_pred_default, labels=[FORMULA_CLASS_IDX], average="micro", zero_division=0), 3),
            "Formula_Precision": round(precision_score(y_test, y_pred_default, labels=[FORMULA_CLASS_IDX], average="micro", zero_division=0), 3),
            "Formula_F2": round(formula_f2(y_test, y_pred_default), 3),
        }, {
            "Variant": f"Threshold = {best_thr}",
            "AUC_ROC": round(auc, 3),
            "MCC": round(mcc, 3),
            "F1_Macro": round(f1m, 3),
            "Formula_Recall": round(f_rec, 3),
            "Formula_Precision": round(f_prec, 3),
            "Formula_F2": round(f_f2, 3),
        }])
        test_df.to_excel(writer, sheet_name="Test Results", index=False)

        # Threshold sweep
        pd.DataFrame(sweep).to_excel(writer, sheet_name="Threshold Sweep", index=False)

        # Params
        pd.DataFrame([bp]).to_excel(writer, sheet_name="Params", index=False)

    print(f"  ✓ Saved results: {excel_path}")

    print("\n" + "=" * 70)
    print("  DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
