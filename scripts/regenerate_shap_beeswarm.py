#!/usr/bin/env python3
"""
Regenerate SHAP Beeswarm Plots with Full English Feature Names
Nature Journal Format (600 DPI, Arial, publication-ready)

Re-runs the SHAP pipeline from Stage-5 but applies comprehensive
English feature labels to all beeswarm plots.

Outputs saved to: paper/figures/shap/
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

# ============================================================================
# NATURE JOURNAL STYLE
# ============================================================================

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
})

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_stage0_5_cleaned.xlsx"
FEAT_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_selected_features.csv"
PARAMS_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_optuna_best_params.xlsx"

# Try alternate locations if primary not found
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "outputs" / "nicu_stage0_5_cleaned.xlsx"
if not FEAT_PATH.exists():
    FEAT_PATH = BASE_DIR / "outputs" / "nicu_selected_features.csv"
if not PARAMS_PATH.exists():
    PARAMS_PATH = BASE_DIR / "outputs" / "nicu_optuna_best_params.xlsx"

OUTPUT_DIR = BASE_DIR / "paper" / "figures" / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42

# For significance markers (consistent with other figures)
SIG_COLOR = '#CC78BC'    # Purple
NEUTRAL_COLOR = '#4D4D4D'  # Dark Gray

# ============================================================================
# COMPREHENSIVE ENGLISH FEATURE NAME MAPPING
# ============================================================================
# Maps the raw (sanitized) feature names to professional English labels.
# Covers ALL features from nicu_selected_features.csv

FEATURE_RENAME = {
    # --- Day-by-day feeding volumes ---
    "beslenmemamamiktari2.guncc":       "Formula Volume on Day 2 (mL)",
    "beslenmemamamiktarı2.guncc":       "Formula Volume on Day 2 (mL)",
    "aldigimamamiktari3.gun":           "Formula Volume on Day 3 (mL)",
    "aldıgımamamiktari3.gun":           "Formula Volume on Day 3 (mL)",
    "aldığımamamiktari1.gün":           "Formula Volume on Day 1 (mL)",
    "aldigimamamiktari1.gun":           "Formula Volume on Day 1 (mL)",
    "aldigiannesutu3.gun":              "Breast Milk Volume on Day 3 (mL)",
    "aldıgıannesütü3.gun":             "Breast Milk Volume on Day 3 (mL)",
    "aldığıannesütü_ilkgün":            "Breast Milk Volume on First Day (mL)",
    "aldigiannsutu_ilkgun":             "Breast Milk Volume on First Day (mL)",
    "beslenmetotali2.gun":              "Total Feeding Volume on Day 2 (mL)",
    "beslenmetotali2.gün":              "Total Feeding Volume on Day 2 (mL)",
    "beslenmetotali3.gun":              "Total Feeding Volume on Day 3 (mL)",
    "beslenme2.gunannesutucc":          "Breast Milk Volume on Day 2 (mL)",

    # --- Clinical measures ---
    "dogumagirligi(gram)":              "Birth Weight (g)",
    "anneyasi":                         "Maternal Age (years)",
    "gebelikhaftasi":                   "Gestational Age (weeks)",
    "gebelikhaftagunu":                 "Gestational Age (days)",
    "takipilkgun_kilo_gram":            "First Follow-up Weight (g)",
    "kilo1.gun":                        "Weight on Day 1 (g)",
    "kilo2.gun":                        "Weight on Day 2 (g)",
    "kilo3.gun":                        "Weight on Day 3 (g)",

    # --- Binary/categorical clinical variables ---
    "annesutuemzirmeegitimidurumu":      "Breastfeeding Education Status",
    "annesutuemzirmeeğitimidurumu":      "Breastfeeding Education Status",
    "covid19sonrasi":                    "Post-COVID-19 Period",
    "verilisyolu3gun":                   "Feeding Route on Day 3",
    "ilk_gun_anne_sutu_1111":            "Breast Milk on First Day (binary)",
    "ilk_gün_anne_sütü_1111":            "Breast Milk on First Day (binary)",
    "ikisiarasi":                        "BFHI Epoch",
    "ikisiarası":                        "BFHI Epoch",
    "ilk_gun_emzirme_111":              "Breastfeeding on First Day (binary)",
    "ilk_gün_emzirme_111":              "Breastfeeding on First Day (binary)",

    # --- Engineered features ---
    "eng_bm_ratio_d1":                  "Breast Milk Ratio on Day 1",
    "eng_bm_ratio_d2":                  "Breast Milk Ratio on Day 2",
    "eng_bm_ratio_d3":                  "Breast Milk Ratio on Day 3",
    "eng_delta_vol_d1_d2":              "Volume Change Day 1→2 (mL)",
    "eng_delta_vol_d2_d3":              "Volume Change Day 2→3 (mL)",
    "eng_resilience_index":             "Feeding Resilience Index",
    "eng_lactation_momentum":           "Lactation Momentum Score",
    "eng_weight_per_week":              "Weight Gain per Week (g/wk)",
}


def sanitize_english(s):
    """Remove Turkish special characters"""
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


def rename_feature(name: str) -> str:
    """Convert raw feature name to professional English label."""
    # Strip prefix (num__, cat__)
    clean = name
    if "__" in clean:
        clean = clean.split("__", 1)[1]

    # Try direct match first (with original characters)
    if clean in FEATURE_RENAME:
        return FEATURE_RENAME[clean]

    # Try sanitized match
    sanitized = sanitize_english(clean)
    if sanitized in FEATURE_RENAME:
        return FEATURE_RENAME[sanitized]

    # Try partial/startswith match (for OHE features like covid19sonrasi_0)
    for key, val in FEATURE_RENAME.items():
        key_san = sanitize_english(key)
        if sanitized.startswith(key_san) or sanitized.startswith(key):
            # Append suffix info if present (e.g., _0, _1)
            suffix = sanitized[len(key_san):]
            if suffix and suffix.startswith("_"):
                return f"{val} ({suffix.strip('_')})"
            return val

    # Fallback: clean up the name
    return sanitized


def normalize_shap_multiclass(shap_values, n_classes):
    """Return list of (n_samples, n_features) arrays, one per class."""
    if isinstance(shap_values, list):
        return shap_values[:n_classes]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return [shap_values[:, :, i] for i in range(n_classes)]
    raise RuntimeError(
        f"Unexpected SHAP format: {type(shap_values)}, shape={getattr(shap_values, 'shape', None)}"
    )


# ============================================================================
# DATA LOADING
# ============================================================================

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
    class_names = list(le.classes_)

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
        print("No columns selected after matching.")
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
    print("Loading RF hyperparameters...")
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("REGENERATING SHAP BEESWARM PLOTS WITH ENGLISH LABELS")
    print("Nature Journal Format (600 DPI)")
    print("=" * 70)

    X, y, num_cols, cat_cols, class_names = load_data()
    params = load_rf_params()

    print(f"✓ {len(X)} patients, {len(X.columns)} features")
    print(f"  Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Preprocessing
    print("Preprocessing...")
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    preprocessor.fit(X_train)

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)

    # Train Model
    print("Training Random Forest + SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_df, y_train)

    model = RandomForestClassifier(**params)
    model.set_params(n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced")
    model.fit(X_train_bal, y_train_bal)

    # SHAP
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df, check_additivity=False)

    class_labels = ["Exclusive BF", "Formula", "Mixed"]
    n_classes = len(class_labels)
    sv_list = normalize_shap_multiclass(shap_values, n_classes)

    # ====================================================================
    # RENAME FEATURES TO ENGLISH for beeswarm plots
    # ====================================================================
    english_names = [rename_feature(n) for n in X_test_df.columns]
    X_test_english = X_test_df.copy()
    X_test_english.columns = english_names

    print(f"\nFeature name mapping ({len(english_names)} features):")
    for orig, eng in zip(X_test_df.columns, english_names):
        if orig != eng:
            print(f"  {orig} → {eng}")

    # ====================================================================
    # GENERATE BEESWARM PLOTS (per-class)
    # ====================================================================
    print("\nGenerating beeswarm plots...")

    for i, class_label in enumerate(class_labels):
        vals = sv_list[i]

        plt.figure()
        shap.summary_plot(
            vals,
            X_test_english,
            show=False,
            max_display=20,
            plot_size=(14, 8)
        )
        plt.title(f"SHAP Summary (Beeswarm): {class_label}", fontsize=14, fontweight='bold')
        plt.xlabel("SHAP value (impact on model output)", fontsize=10)
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"beeswarm_{class_label.lower().replace(' ', '_')}.png"
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out_path}")

        # Also save PDF
        out_pdf = OUTPUT_DIR / f"beeswarm_{class_label.lower().replace(' ', '_')}.pdf"
        plt.figure()
        shap.summary_plot(
            vals,
            X_test_english,
            show=False,
            max_display=20,
            plot_size=(14, 8)
        )
        plt.title(f"SHAP Summary (Beeswarm): {class_label}", fontsize=14, fontweight='bold')
        plt.xlabel("SHAP value (impact on model output)", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_pdf, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out_pdf}")

    # ====================================================================
    # GENERATE BAR PLOTS (per-class)
    # ====================================================================
    print("\nGenerating SHAP bar plots...")

    for i, class_label in enumerate(class_labels):
        vals = sv_list[i]

        plt.figure()
        shap.summary_plot(
            vals,
            X_test_english,
            plot_type="bar",
            show=False,
            max_display=20,
            plot_size=(14, 8)
        )
        plt.title(f"Feature Importance (SHAP): {class_label}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"bar_{class_label.lower().replace(' ', '_')}.png"
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out_path}")

    # ====================================================================
    # OVERALL IMPORTANCE
    # ====================================================================
    print("\nGenerating overall importance plot...")

    imp_list = [np.mean(np.abs(sv_list[i]), axis=0) for i in range(n_classes)]
    imp_overall = np.mean(np.stack(imp_list, axis=0), axis=0)

    # Top 20 features
    order = np.argsort(-imp_overall)[:20]
    feat_top = [english_names[j] for j in order]
    imp_top = imp_overall[order]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(feat_top))
    ax.barh(y_pos, imp_top, color='#66C2A5', edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_top)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontweight='bold')
    ax.set_title("Overall Feature Importance (mean |SHAP| across all classes)",
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, v in enumerate(imp_top):
        ax.text(v + imp_top.max() * 0.01, i, f'{v:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overall_importance.png", dpi=600, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "overall_importance.pdf", bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: overall_importance.png / .pdf")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "=" * 70)
    print("✓ ALL SHAP PLOTS REGENERATED WITH ENGLISH LABELS")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("  Beeswarm plots:")
    for cl in class_labels:
        name = cl.lower().replace(' ', '_')
        print(f"    - beeswarm_{name}.png / .pdf")
    print("  Bar plots:")
    for cl in class_labels:
        name = cl.lower().replace(' ', '_')
        print(f"    - bar_{name}.png")
    print("  Overall: overall_importance.png / .pdf")
    print("\nSpecifications:")
    print("  - Resolution: 600 DPI")
    print("  - Font: Arial")
    print("  - All feature names in English")
    print("=" * 70)


if __name__ == "__main__":
    main()
