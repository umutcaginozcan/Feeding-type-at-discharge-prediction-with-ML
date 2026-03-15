#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Alt-Stage 4: SHAP Explainability Analysis
-------------------------------------------------
Generates Nature-quality SHAP plots for the two winning models from
Alt-Stage 3: RF (F2-tuned, winner) and CatBoost (F2-tuned, secondary).

Plots (300 DPI, Arial):
  1. Fig.11-style overall feature importance — teal horizontal bars, annotated
  2. Fig.10-style grouped per-class importance  — pastel horizontal bars
  3. SHAP beeswarm plots — one per class (top-20)
  4. SHAP waterfall plots — representative patient per class

English feature names are imported from src/data/loader.py (RENAME_FOR_PLOT).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import json

import matplotlib.pyplot as plt
import matplotlib as mpl
import shap

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_stage0_5_cleaned.xlsx"
FEAT_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_selected_features.csv"

OUTPUT_DIR = Path.home() / "Desktop" / "nicu_alt_plots" / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
TOP_K = 10

CLASS_LABELS = ["Exclusive BF", "Formula", "Mixed"]

# ==================== ENGLISH FEATURE NAMES ====================
# Imported from src/data/loader.py RENAME_FOR_PLOT

sys.path.insert(0, str(BASE_DIR))
try:
    from src.data.loader import RENAME_FOR_PLOT
    print("  ✓ Loaded RENAME_FOR_PLOT from loader.py")
except ImportError:
    print("  ⚠ Could not import from loader.py, using fallback dict")
    RENAME_FOR_PLOT = {}

# Fallback additions for any engineered or OHE-expanded columns
RENAME_FALLBACK = {
    "beslenmemamamiktari2.guncc": "Formula intake on Day 2 (cc)",
    "aldigimamamiktari3.gun": "Formula intake on Day 3 (cc)",
    "annesutuemzirmeegitimidurumu": "Lactation education status",
    "covid19sonrasi": "COVID-19 Period",
    "verilisyolu3gun": "Feeding route on Day 3",
    "ilk_gun_anne_sutu_1111": "Breast milk present on Day 1",
    "aldigimamamiktari1.gun": "Formula intake on Day 1 (cc)",
    "aldigiannesutu3.gun": "Breast milk intake on Day 3 (cc)",
    "ikisiarasi": "Epoch (COVID × BFHI)",
    "eng_bm_ratio_d1": "Breast milk ratio on Day 1",
    "eng_bm_ratio_d3": "Breast milk ratio on Day 3",
}

# ==================== WINNING MODEL PARAMS ====================
# From Alt-Stage 2.5 v2 (F2-optimized Optuna) — duplicated from alt-Stage-3

RF_PARAMS = {
    'n_estimators': 255,
    'max_depth': 21,
    'min_samples_split': 15,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': RANDOM_STATE
}

CATBOOST_PARAMS = {
    'iterations': 886,
    'learning_rate': 0.01416597475643034,
    'depth': 5,
    'l2_leaf_reg': 2,
    'random_strength': 1.547513550320719,
    'bagging_temperature': 9.2612864041155,
    'border_count': 128,
    'class_weights': {0: 1.0, 1: 3.646662900755656, 2: 1.92025858564541},
    'verbose': False,
    'allow_writing_files': False,
    'random_state': RANDOM_STATE,
    'thread_count': -1
}


# ==================== PAPER STYLE ====================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 11,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ==================== HELPERS ====================

def sanitize_english(s):
    """Remove Turkish characters."""
    if not isinstance(s, str):
        s = str(s)
    tr_map = str.maketrans({
        "ç": "c", "Ç": "C", "ğ": "g", "Ğ": "G",
        "ı": "i", "İ": "I", "ö": "o", "Ö": "O",
        "ş": "s", "Ş": "S", "ü": "u", "Ü": "U",
    })
    return s.translate(tr_map)


def rename_feature(raw_name):
    """
    Translate a raw or OHE-expanded feature name into publication English.
    Priority: RENAME_FOR_PLOT → RENAME_FALLBACK → sanitize.
    Handles num__/cat__ prefixes and cat__col_value OHE expansions.
    """
    n = str(raw_name)

    # Strip sklearn ColumnTransformer prefixes
    if "__" in n:
        n = n.split("__", 1)[1]

    # Direct hit
    if n in RENAME_FOR_PLOT:
        return RENAME_FOR_PLOT[n]

    # Check fallback (handles Turkish-char mismatches)
    n_clean = sanitize_english(n)
    for k, v in RENAME_FALLBACK.items():
        if n_clean == sanitize_english(k) or n_clean.startswith(sanitize_english(k)):
            return v

    # Try partial match on RENAME_FOR_PLOT keys
    for k, v in RENAME_FOR_PLOT.items():
        if sanitize_english(k) == n_clean:
            return v

    # OHE column: e.g. "verilisyolu3gun_4.0" → "Feeding route on Day 3 = 4"
    if "_" in n:
        parts = n.rsplit("_", 1)
        base = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""
        base_en = None
        for k, v in {**RENAME_FOR_PLOT, **RENAME_FALLBACK}.items():
            if sanitize_english(k) == sanitize_english(base):
                base_en = v
                break
        if base_en:
            return f"{base_en} = {suffix}"

    return sanitize_english(n)


def set_framed_axes(ax):
    """All four spines visible, no grid."""
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.2)


def normalize_shap_multiclass(shap_values, n_classes):
    """Handle SHAP list-of-arrays OR 3D-ndarray."""
    if isinstance(shap_values, list):
        return shap_values[:n_classes]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return [shap_values[:, :, i] for i in range(n_classes)]
    raise RuntimeError(
        f"Unexpected SHAP format: {type(shap_values)}, shape={getattr(shap_values, 'shape', None)}"
    )


# ==================== PLOT FUNCTIONS ====================

def plot_fig11_overall(features, imp_overall, model_tag, out_dir):
    """
    Fig.11-style single horizontal bar chart (teal) with value labels.
    """
    y = np.arange(len(features))

    fig, ax = plt.subplots(figsize=(12, max(6, len(features) * 0.55)))
    ax.barh(y, imp_overall, color="#66C2A5", edgecolor="none", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=16)

    xmax = float(np.max(imp_overall)) * 1.12 if np.max(imp_overall) > 0 else 1.0
    ax.set_xlim(0.0, xmax)

    for i, v in enumerate(imp_overall):
        ax.text(float(v) + xmax * 0.008, i, f"{v:.3f}",
                va="center", ha="left", fontsize=12, fontweight="medium")

    set_framed_axes(ax)
    plt.tight_layout()
    fname = f"Fig11_SHAP_Overall_{model_tag}.png"
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {fname}")


def plot_fig10_grouped(features, imp_by_class, class_labels, model_tag, out_dir):
    """
    Fig.10-style grouped horizontal bars (pastel colors), one set per class.
    """
    colors = ["#E78AC3", "#A6D854", "#8E6BBF"]  # pink, green, purple

    y = np.arange(len(features))
    n_cls = len(class_labels)
    h = 0.22
    offsets = [-(n_cls - 1) / 2 * h + i * h for i in range(n_cls)]

    fig, ax = plt.subplots(figsize=(14, max(6, len(features) * 0.65)))

    for i in range(n_cls):
        ax.barh(
            y + offsets[i], imp_by_class[i],
            height=h, color=colors[i], edgecolor="none",
            label=class_labels[i]
        )

    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=16)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.06),
        ncol=n_cls, frameon=False, fontsize=12,
        handlelength=1.8, columnspacing=1.6
    )

    xmax = float(np.max(imp_by_class)) * 1.10 if np.max(imp_by_class) > 0 else 1.0
    ax.set_xlim(0.0, xmax)

    set_framed_axes(ax)
    plt.tight_layout()
    fname = f"Fig10_SHAP_Grouped_{model_tag}.png"
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {fname}")


def plot_beeswarm(shap_vals_class, X_display, class_label, model_tag, out_dir):
    """
    Standard SHAP beeswarm plot, one per class.
    """
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_vals_class, X_display,
        show=False, max_display=20,
        plot_size=None, color_bar_label="Feature value"
    )
    ax = plt.gca()
    ax.set_title(f"SHAP Beeswarm — {class_label}", fontsize=16, pad=12)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=14)
    plt.tight_layout()
    fname = f"SHAP_Beeswarm_{class_label}_{model_tag}.png"
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {fname}")


def plot_waterfall(explainer, shap_vals_class, X_display, class_label,
                   model_tag, out_dir, y_test, class_idx):
    """
    SHAP waterfall for a representative patient from each class.
    Uses the test-set patient closest to the median prediction for that class.
    """
    # Pick patients belonging to this class
    mask = y_test == class_idx
    if mask.sum() == 0:
        print(f"    ⚠ No test patients for {class_label}, skipping waterfall")
        return

    idx_class = np.where(mask)[0]
    # Pick the one with median absolute SHAP contribution
    total_shap = np.abs(shap_vals_class[idx_class]).sum(axis=1)
    median_idx_in_subset = np.argsort(total_shap)[len(total_shap) // 2]
    patient_idx = idx_class[median_idx_in_subset]

    # Build SHAP Explanation object
    exp = shap.Explanation(
        values=shap_vals_class[patient_idx],
        base_values=explainer.expected_value[class_idx] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value,
        data=X_display.iloc[patient_idx].values,
        feature_names=X_display.columns.tolist()
    )

    fig = plt.figure(figsize=(12, 8))
    shap.waterfall_plot(exp, max_display=15, show=False)
    ax = plt.gca()
    ax.set_title(f"SHAP Waterfall — {class_label} (patient #{patient_idx})",
                 fontsize=14, pad=12)
    plt.tight_layout()
    fname = f"SHAP_Waterfall_{class_label}_{model_tag}.png"
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {fname}")


# ==================== DATA LOADING ====================

def load_data():
    print("Loading data...")
    df = pd.read_excel(DATA_PATH)
    selected_df = pd.read_csv(FEAT_PATH)
    selected_feat_names = set(selected_df['Selected_Features'].tolist())

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    y = df[TARGET_COL]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    all_raw_cols = df.columns.tolist()
    if TARGET_COL in all_raw_cols:
        all_raw_cols.remove(TARGET_COL)

    cols_to_keep = set()
    for raw_col in all_raw_cols:
        if raw_col in selected_feat_names:
            cols_to_keep.add(raw_col)
            continue
        for sel in selected_feat_names:
            if sel.startswith(str(raw_col)):
                cols_to_keep.add(raw_col)
                break

    X = df[list(cols_to_keep)]

    num_cols, cat_cols = [], []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            X[c] = X[c].astype(str)
            cat_cols.append(c)

    print(f"  {len(df)} samples, {len(X.columns)} features "
          f"({len(num_cols)} numeric, {len(cat_cols)} categorical)")
    return X, y_enc, num_cols, cat_cols, le


# ==================== SHAP PIPELINE ====================

def run_shap_for_model(model, model_tag, X_train, X_test, y_train, y_test,
                       num_cols, cat_cols):
    """Full SHAP pipeline for one model."""
    print(f"\n{'='*55}")
    print(f"  SHAP — {model_tag}")
    print(f"{'='*55}")

    # --- Preprocessing ---
    numeric_transformer = SkPipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names_raw = preprocessor.get_feature_names_out()
    feature_names_en = [rename_feature(f) for f in feature_names_raw]

    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names_en)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names_en)

    # --- SMOTE + Train ---
    print("  Training with SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_df, y_train)
    model.fit(X_train_bal, y_train_bal)

    # --- SHAP ---
    print("  Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df, check_additivity=False)

    n_classes = len(CLASS_LABELS)
    sv_list = normalize_shap_multiclass(shap_values, n_classes)

    # --- Mean |SHAP| importances ---
    imp_list = [np.mean(np.abs(sv_list[i]), axis=0) for i in range(n_classes)]
    imp_overall = np.mean(np.stack(imp_list, axis=0), axis=0)

    # Top-K indices
    order = np.argsort(-imp_overall)[:min(TOP_K, len(imp_overall))]
    feat_top = [feature_names_en[j] for j in order]
    imp_by_class_top = np.array([imp_list[i][order] for i in range(n_classes)])
    imp_overall_top = imp_overall[order]

    # --- Plot 1: Fig.11 overall bars ---
    print("\n  📊 Fig.11 — Overall feature importance:")
    plot_fig11_overall(feat_top, imp_overall_top, model_tag, OUTPUT_DIR)

    # --- Plot 2: Fig.10 grouped bars ---
    print("  📊 Fig.10 — Per-class feature importance:")
    plot_fig10_grouped(feat_top, imp_by_class_top, CLASS_LABELS, model_tag, OUTPUT_DIR)

    # --- Plot 3: Beeswarm (per-class) ---
    print("  🐝 Beeswarm plots:")
    for i, cl in enumerate(CLASS_LABELS):
        plot_beeswarm(sv_list[i], X_test_df, cl, model_tag, OUTPUT_DIR)

    # --- Plot 4: Waterfall (representative patient per class) ---
    print("  💧 Waterfall plots:")
    for i, cl in enumerate(CLASS_LABELS):
        plot_waterfall(explainer, sv_list[i], X_test_df, cl,
                       model_tag, OUTPUT_DIR, y_test, i)

    # --- Export SHAP summary to CSV ---
    summary_df = pd.DataFrame({
        "Feature": feature_names_en,
        "Overall_mean_abs_SHAP": imp_overall,
        **{f"SHAP_{cl}": imp_list[i] for i, cl in enumerate(CLASS_LABELS)}
    }).sort_values("Overall_mean_abs_SHAP", ascending=False)
    csv_path = OUTPUT_DIR / f"SHAP_summary_{model_tag}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  ✓ CSV: {csv_path}")

    return summary_df


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("  NICU Alt-Stage 4: SHAP Explainability")
    print("=" * 60)

    X, y, num_cols, cat_cols, le = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Split: Train={len(X_train)}, Test={len(X_test)}")

    # ---- 1. RF (winner) ----
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_summary = run_shap_for_model(
        rf_model, "RF", X_train.copy(), X_test.copy(),
        y_train, y_test, num_cols, cat_cols
    )

    # ---- 2. CatBoost (secondary) ----
    cb_model = CatBoostClassifier(**CATBOOST_PARAMS)
    cb_summary = run_shap_for_model(
        cb_model, "CatBoost", X_train.copy(), X_test.copy(),
        y_train, y_test, num_cols, cat_cols
    )

    print(f"\n{'='*60}")
    print(f"  ✅ DONE! All SHAP outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
