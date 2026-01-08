#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Stage 4.5: SHAP Explainability Analysis (FIXED + Fig10/Fig11 Formats)
------------------------------------------------------------------------
Keeps your pipeline + SHAP computation, but formats plots to imitate
Fig.10 / Fig.11 style horizontal bar charts and uses English feature names.

Outputs
-------
- outputs/nicu_shap_analysis/Fig10_SHAP_Grouped.png
- outputs/nicu_shap_analysis/Fig11_SHAP_Overall.png
- (optional) per-class beeswarm + default SHAP bar plots:
    SHAP_Beeswarm_<Class>.png
    SHAP_Bar_<Class>.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import matplotlib.pyplot as plt
import shap

# Scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Imbalanced-learn
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# --- Fig.10 / Fig.11 style (fonts, sizes, clean frame) ---
plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "Arial",
    "axes.labelsize": 22,
    "xtick.labelsize": 12,
    "ytick.labelsize": 22,
    "axes.linewidth": 1.2,
})

# -------------------- CONFIGURATION --------------------

# Base directory: project root
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "outputs" / "nicu_stage0_5_cleaned.xlsx"
FEAT_PATH = BASE_DIR / "outputs" / "nicu_selected_features.csv"
PARAMS_PATH = BASE_DIR / "outputs" / "nicu_optuna_best_params.xlsx"

OUTPUT_DIR = BASE_DIR / "outputs" / "nicu_shap_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42

TOP_K = 10  # change if you want

# -------------------- FORMAT HELPERS --------------------

def sanitize_english(s):
    if not isinstance(s, str):
        s = str(s)
    tr_map = str.maketrans({
        "ç": "c", "Ç": "C",
        "ğ": "g", "Ğ": "G",
        "ı": "i", "İ": "I",
        "ö": "o", "Ö": "O",
        "ş": "s", "Ş": "S",
        "ü": "u", "Ü": "U",
    })
    return s.translate(tr_map)

def pretty_feat_name(n):
    n = sanitize_english(n)
    if "__" in n:
        n = n.split("__", 1)[1]  # drop num__/cat__
    return n

def set_framed_axes(ax):
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.2)

def normalize_shap_multiclass(shap_values, n_classes):
    """
    Return list length=n_classes, each (n_samples, n_features).
    Handles SHAP list OR ndarray (samples, features, classes).
    """
    if isinstance(shap_values, list):
        return shap_values[:n_classes]

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # expected (samples, features, classes)
        return [shap_values[:, :, i] for i in range(n_classes)]

    raise RuntimeError(
        f"Unexpected SHAP format: {type(shap_values)}, shape={getattr(shap_values, 'shape', None)}"
    )

# ---- English feature mapping you requested ----
FEATURE_RENAME = {
    "beslenmemamamiktari2.guncc": "Formula on Day 2",
    "aldigimamamiktari3.gun": "Formula on Day 3",
    "annesutuemzirmeegitimidurumu": "Breastfeeding Education Status",
    "covid19sonrasi": "post-COVID19",
    "verilisyolu3gun": "Day 3 Feeding Route",  
    "ilk_gun_anne_sutu_1111": "MM on First Day (binary)",
    "aldigimamamiktari1.gun": "Formula on Day 1",
    "aldigiannesutu3.gun": "MM on Day 3",
    "ikisiarasi": "BFHI Certificate",
    "eng_bm_ratio_d1" : "MM Ratio on Day 1",
    "eng_bm_ratio_d3": "MM Ratio on Day 3",
}

def rename_feature(n: str) -> str:
    n0 = pretty_feat_name(n)
    if n0 in FEATURE_RENAME:
        return FEATURE_RENAME[n0]
    for k, v in FEATURE_RENAME.items():
        if n0.startswith(k):
            return v
    return n0

# -------------------- FIG10 / FIG11 PLOTS --------------------

def plot_fig10_grouped(features, imp_by_class, class_labels, out_path):
    """
    Fig.10-like grouped horizontal bars (pastel colors).
    imp_by_class: np.array shape (3, K)
    """
    # Pastel palette similar to paper
    colors = ["#E78AC3", "#A6D854", "#8E6BBF"]  # pink, green, purple

    features = [sanitize_english(f) for f in features]
    y = np.arange(len(features))
    h = 0.18
    offs = [-h, 0.0, +h]

    fig = plt.figure(figsize=(16, 9))
    ax = plt.gca()

    for i in range(3):
        ax.barh(
            y + offs[i],
            imp_by_class[i],
            height=h,
            color=colors[i],
            edgecolor="none",
            label=class_labels[i]
        )

    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.6
    )

    xmax = float(np.max(imp_by_class)) * 1.08 if np.max(imp_by_class) > 0 else 1.0
    ax.set_xlim(0.0, xmax)

    set_framed_axes(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_fig11_overall(features, imp_overall, out_path):
    """
    Fig.11-like single horizontal bar chart (teal) with value labels.
    """
    features = [sanitize_english(f) for f in features]
    y = np.arange(len(features))

    fig = plt.figure(figsize=(16, 9))
    ax = plt.gca()

    ax.barh(y, imp_overall, color="#66C2A5", edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")

    xmax = float(np.max(imp_overall)) * 1.08 if np.max(imp_overall) > 0 else 1.0
    ax.set_xlim(0.0, xmax)

    for i, v in enumerate(imp_overall):
        ax.text(float(v) + xmax * 0.005, i, f"{v:.3f}", va="center", ha="left", fontsize=12)

    set_framed_axes(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# -------------------- LOADERS --------------------

def load_data():
    print("Loading data...")
    df = pd.read_excel(DATA_PATH)

    try:
        selected_df = pd.read_csv(FEAT_PATH)
        if "Selected_Features" in selected_df.columns:
            selected_feat_names = set(selected_df["Selected_Features"].astype(str).tolist())
        else:
            selected_feat_names = set(selected_df.iloc[:, 0].astype(str).tolist())
    except Exception as e:
        print("Feature file error:", e)
        sys.exit(1)

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    y = df[TARGET_COL]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)  # original label values (e.g., 1/2/3)

    all_raw_cols = df.columns.tolist()
    if TARGET_COL in all_raw_cols:
        all_raw_cols.remove(TARGET_COL)

    cols_to_keep = set()
    for raw_col in all_raw_cols:
        if raw_col in selected_feat_names:
            cols_to_keep.add(raw_col)
            continue
        for sel in selected_feat_names:
            if str(sel).startswith(str(raw_col)):
                cols_to_keep.add(raw_col)
                break

    if not cols_to_keep:
        print("No columns selected after matching. Check selected features CSV.")
        sys.exit(1)

    X = df[list(cols_to_keep)].copy()

    num_cols, cat_cols = [], []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            X[c] = X[c].astype(str)
            cat_cols.append(c)

    return X, y_enc, num_cols, cat_cols, class_names

def load_rf_params():
    print("Loading RF params...")
    try:
        df_params = pd.read_excel(PARAMS_PATH)
        row = df_params[df_params["Model"] == "Random Forest"].iloc[0]
    except Exception as e:
        print("Params file missing or unreadable:", e)
        sys.exit(1)

    exclude = ["Model", "ROC_AUC_Weighted", "MCC", "F1_Macro", "Balanced_Acc", "Accuracy", "Brier_Score"]
    params = {k: v for k, v in row.items() if k not in exclude and pd.notna(v)}

    if "bootstrap" in params:
        val = params["bootstrap"]
        if isinstance(val, (float, int)):
            params["bootstrap"] = bool(val)

    int_keys = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]
    for key in int_keys:
        if key in params:
            params[key] = int(params[key])

    return params

# -------------------- MAIN --------------------

def main():
    print("--- 🧠 Starting SHAP Analysis (Fixed Version + Fig10/Fig11 Formats) ---")

    X, y, num_cols, cat_cols, class_names = load_data()
    params = load_rf_params()

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Preprocessing
    print("Preprocessing data...")
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Convert to DataFrame to ensure feature names align
    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)

    # Train Model
    print("Training Random Forest...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_df, y_train)

    model = RandomForestClassifier(**params)
    model.set_params(n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced")
    model.fit(X_train_bal, y_train_bal)

    # SHAP Calculations
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df, check_additivity=False)

    # --- FIXED class labels: map by index (0/1/2) ---
    # Raw labels 1/2/3 -> LabelEncoder indices become:
    # index 0 corresponds to label 1 (Exclusive BF)
    # index 1 corresponds to label 2 (Formula)
    # index 2 corresponds to label 3 (Mixed)
    class_labels = ["Exclusive BF", "Formula", "Mixed"]

    n_classes = len(class_labels)
    print(f"Plot labels: {class_labels}")
    print(f"Original class values from LabelEncoder: {class_names}")

    # Robust SHAP output handling
    sv_list = normalize_shap_multiclass(shap_values, n_classes)

    # Mean(|SHAP|) importances
    imp_list = [np.mean(np.abs(sv_list[i]), axis=0) for i in range(n_classes)]
    imp_overall = np.mean(np.stack(imp_list, axis=0), axis=0)

    # English feature names
    features_clean = [rename_feature(n) for n in X_test_df.columns]

    # Top-K by overall importance
    order = np.argsort(-imp_overall)[:min(TOP_K, len(imp_overall))]

    feat_top = [features_clean[j] for j in order]
    imp_by_class_top = np.array([imp_list[i][order] for i in range(n_classes)])
    imp_overall_top = imp_overall[order]

    # Save Fig10/Fig11 style plots
    plot_fig10_grouped(
        feat_top,
        imp_by_class_top,
        class_labels=class_labels,
        out_path=OUTPUT_DIR / "Fig10_SHAP_Grouped.png"
    )
    plot_fig11_overall(
        feat_top,
        imp_overall_top,
        out_path=OUTPUT_DIR / "Fig11_SHAP_Overall.png"
    )

    # OPTIONAL: also save your original per-class SHAP beeswarm + default SHAP bar plots
    for i, class_label in enumerate(class_labels):
        vals = sv_list[i]

        plt.figure()
        shap.summary_plot(
            vals,
            X_test_df,
            show=False,
            max_display=20,
            plot_size=(12, 8)
        )
        plt.title(f"SHAP Summary (Beeswarm): {class_label}", fontsize=16)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"SHAP_Beeswarm_{class_label}.png")
        plt.close()

        plt.figure()
        shap.summary_plot(
            vals,
            X_test_df,
            plot_type="bar",
            show=False,
            max_display=20,
            plot_size=(12, 8)
        )
        plt.title(f"Feature Importance (SHAP): {class_label}", fontsize=16)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"SHAP_Bar_{class_label}.png")
        plt.close()

    print(f"✅ SHAP Analysis Done. Plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
