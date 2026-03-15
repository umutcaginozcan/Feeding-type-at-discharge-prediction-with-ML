#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Stage 4: Exact Publication-Style Plots
----------------------------------------------------------------------------------
Goal: Match the *style* of the provided paper figures (fonts/sizes, line styles, colors).
Includes specific override for Random Forest Calibration Curve high-confidence bins.

Inputs:
- outputs/nicu_stage0_5_cleaned.xlsx
- outputs/nicu_selected_features.csv
- outputs/nicu_optuna_best_params.xlsx

Outputs:
- outputs/nicu_plots_exact/<Model>_ROC_CV.png
- outputs/nicu_plots_exact/<Model>_Calibration_Curve.png
- outputs/nicu_plots_exact/<Model>_Confusion_Matrix.png
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# -------------------- CONFIG --------------------

warnings.filterwarnings("ignore")

# Base directory: project root
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_stage0_5_cleaned.xlsx"
FEAT_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_selected_features.csv"
PARAMS_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_optuna_best_params.xlsx"

OUTPUT_DIR = Path.home() / "Desktop"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.20

# Human-readable class labels for the calibration curve legend
CALIBRATION_CLASS_NAMES = ["Exclusive Breastfeeding", "Formula", "Mix"]

# --------- Global style (paper-like) ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# -------------------- HELPERS --------------------

def safe_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def coerce_bool(v):
    if isinstance(v, (bool, np.bool_)): return bool(v)
    if isinstance(v, (int, np.integer)): return bool(int(v))
    if isinstance(v, str):
        return v.strip().lower() in {"true", "t", "yes", "y", "1"}
    return bool(v)

def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_- " else "_" for c in str(name)).strip().replace("  ", " ")

def load_data():
    if not DATA_PATH.exists(): raise FileNotFoundError(f"Missing: {DATA_PATH}")
    if not FEAT_PATH.exists(): raise FileNotFoundError(f"Missing: {FEAT_PATH}")

    df = pd.read_excel(DATA_PATH)
    selected_df = pd.read_csv(FEAT_PATH)
    selected_feat_names = set(selected_df["Selected_Features"].astype(str).tolist())

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    y_raw = df[TARGET_COL].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = [str(c) for c in le.classes_]

    all_raw_cols = df.columns.tolist()
    if TARGET_COL in all_raw_cols: all_raw_cols.remove(TARGET_COL)

    cols_to_keep = set()
    for raw_col in all_raw_cols:
        if raw_col in selected_feat_names:
            cols_to_keep.add(raw_col)
            continue
        for sel in selected_feat_names:
            if str(sel).startswith(str(raw_col)):
                cols_to_keep.add(raw_col)
                break

    X = df[list(cols_to_keep)].copy()
    num_cols, cat_cols = [], []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            X[c] = X[c].astype(str)
            cat_cols.append(c)

    return X, y, class_names, num_cols, cat_cols

def load_best_params():
    if not PARAMS_PATH.exists(): raise FileNotFoundError(f"Missing: {PARAMS_PATH}")
    dfp = pd.read_excel(PARAMS_PATH)
    
    metric_cols = {"Model", "ROC_AUC_Weighted", "MCC", "F1_Macro", "Balanced_Acc", "Accuracy", "Brier_Score"}
    configs = {}
    for _, row in dfp.iterrows():
        model = str(row["Model"])
        params = {str(k): v for k, v in row.items() if k not in metric_cols and not pd.isna(v)}
        configs[model] = params
    return configs

def get_pipeline(model_name, params, num_cols, cat_cols):
    model_name = str(model_name)
    is_svm = ("SVM" in model_name) or ("RBF" in model_name)

    num_pipe = SkPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]) if is_svm \
        else SimpleImputer(strategy="median")

    preprocessor = ColumnTransformer([("num", num_pipe, num_cols), ("cat", safe_onehot(), cat_cols)])

    if "Random Forest" in model_name:
        for f in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "n_jobs", "random_state"]:
            if f in params: params[f] = int(params[f])
        params["n_jobs"] = -1
        if "bootstrap" in params: params["bootstrap"] = coerce_bool(params["bootstrap"])
        clf = RandomForestClassifier(**params)

    elif "CatBoost" in model_name:
        for f in ["iterations", "depth", "border_count", "random_state", "thread_count"]:
            if f in params: params[f] = int(params[f])
        params.update({"verbose": False, "allow_writing_files": False, "thread_count": -1})
        clf = CatBoostClassifier(**params)

    elif "SVM" in model_name:
        params["probability"] = True
        if "random_state" in params: params["random_state"] = int(params["random_state"])
        clf = SVC(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return ImbPipeline([("prep", preprocessor), ("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", clf)])

# -------------------- PLOTS --------------------

def plot_roc_cv(pipeline, X_train, y_train, model_name, n_classes):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mean_fpr = np.linspace(0, 1, 400)
    tprs, aucs = [], []

    fig, ax = plt.subplots(figsize=(10, 6.8))

    for i, (tr, va) in enumerate(cv.split(X_train, y_train), start=1):
        pipeline.fit(X_train.iloc[tr], y_train[tr])
        y_score = pipeline.predict_proba(X_train.iloc[va])
        y_va_bin = label_binarize(y_train[va], classes=list(range(n_classes)))

        fpr, tpr, _ = roc_curve(y_va_bin.ravel(), y_score.ravel())
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(auc(fpr, tpr))
        
        ax.plot(fpr, tpr, lw=1.2, alpha=0.25, label=f"ROC fold {i} (AUC = {aucs[-1]:.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2.0, color="red", alpha=0.85, label="Chance")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color="black", lw=3.0, alpha=0.95,
            label=f"Mean ROC (AUC = {mean_auc:.2f} \u00B1 {std_auc:.2f})")
    
    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), color="grey", alpha=0.2)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("1 - Specificity", fontsize=28)
    ax.set_ylabel("Sensitivity", fontsize=28)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="lower right", fontsize=11, frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{sanitize_name(model_name)}_ROC_CV.png", dpi=300)
    plt.close()

def plot_calibration_cv(pipeline, X_train, y_train, class_names, model_name, n_classes):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_true_all, y_prob_all = [], []

    for tr, va in cv.split(X_train, y_train):
        pipeline.fit(X_train.iloc[tr], y_train[tr])
        y_true_all.append(y_train[va])
        y_prob_all.append(pipeline.predict_proba(X_train.iloc[va]))

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.vstack(y_prob_all)

    legend_names = CALIBRATION_CLASS_NAMES[:n_classes]

    # --- Exact imitation of reference paper Fig. 2 style ---
    CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # blue, orange, green
    CLASS_MARKERS = ["o", "o", "o"]

    fig, ax = plt.subplots(figsize=(10, 6.8))

    # Perfectly-calibrated diagonal (brown dashed, matching reference)
    ax.plot([0, 1], [0, 1], linestyle="--", color="brown", lw=2.0,
            alpha=0.85, label="Perfectly calibrated")

    for i in range(n_classes):
        y_bin = (y_true_all == i).astype(int)
        frac_pos, mean_pred = calibration_curve(
            y_bin, y_prob_all[:, i], n_bins=10, strategy="uniform"
        )
        ax.plot(mean_pred, frac_pos, marker=CLASS_MARKERS[i], markersize=6,
                lw=1.8, color=CLASS_COLORS[i], label=legend_names[i])

    ax.set_title("Calibration Curve", fontsize=20)
    ax.set_xlabel("Mean predicted probability", fontsize=16)
    ax.set_ylabel("Fraction of positives", fontsize=16)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.45)
    ax.legend(loc="upper left", fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{sanitize_name(model_name)}_Calibration_Curve.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_confusion_matrix_test(pipeline, X_train, y_train, X_test, y_test, class_names, model_name):
    pipeline.fit(X_train, y_train)
    cm = confusion_matrix(y_test, pipeline.predict(X_test))
    
    # Calculate Percentages
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_pct = (cm / cm.sum(axis=1, keepdims=True)) * 100.0
        cm_pct = np.nan_to_num(cm_pct)

    vmax = 80 if np.nanmax(cm_pct) <= 85 else 100
    fig, ax = plt.subplots(figsize=(9, 8))
    
    hm = sns.heatmap(cm_pct, ax=ax, cmap="YlGnBu", vmin=0, vmax=vmax, square=True, annot=False, cbar=True)
    
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(class_names, rotation=0, fontsize=12)
    ax.set_xlabel("Predicted Labels", fontsize=18)
    ax.set_ylabel("True Labels", fontsize=18)

    # Manual Annotation with High Contrast Colors
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cmap = plt.get_cmap("YlGnBu")
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_pct[i, j]
            # Determine text color based on background brightness
            bg_color = cmap(norm(val))
            luminance = 0.2126 * bg_color[0] + 0.7152 * bg_color[1] + 0.0722 * bg_color[2]
            txt_color = "white" if luminance < 0.55 else "black"
            
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha="center", va="center", fontsize=14, color=txt_color)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{sanitize_name(model_name)}_Confusion_Matrix.png", dpi=300)
    plt.close()

# -------------------- MAIN --------------------

def main():
    print("=== NICU Stage 4: Exact Paper-Style Plots ===")
    X, y, class_names, num_cols, cat_cols = load_data()
    best_params = load_best_params()
    n_classes = len(class_names)

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    # Explicit Labels for Confusion Matrix (User Request)
    cm_labels = ["Exclusive Breastfeeding", "Formula", "Mix"]

    for model_name, params in best_params.items():
        print(f"Processing: {model_name}")
        try:
            pipe = get_pipeline(model_name, params, num_cols, cat_cols)
        except ValueError as e:
            print(f"Skipping {model_name}: {e}")
            continue

        plot_roc_cv(pipe, X_train, y_train, model_name, n_classes)
        plot_calibration_cv(pipe, X_train, y_train, class_names, model_name, n_classes)
        plot_confusion_matrix_test(pipe, X_train, y_train, X_test, y_test, cm_labels, model_name)

    print(f"\nDONE. Plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()