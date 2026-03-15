#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Alt-Stage 3: Publication Plots + Model Params Export
----------------------------------------------------------
Produces Nature-quality figures for the winning models from Alt-Stage 2.5 v2:
  - CatBoost (F2-tuned)        — best balanced (Recall 0.857, Precision 0.522)
  - CatBoost (F2-tuned + Thr)  — high-recall clinical mode
  - RF (F2-tuned + Thr)        — competitive alternative

Figures (300 DPI, Arial):
  1. Confusion matrices (row-normalized %, with counts)
  2. Multi-class ROC curves (5-fold CV, mean ± std)
  3. Calibration curves (per-class reliability diagrams)
  4. Precision-Recall curves (per-class, with AP)
  5. Threshold sweep plot (recall/precision/F2/accuracy vs threshold)

Also exports model params as JSON + Excel for Stage-5 (SHAP) and Stage-6 (web deploy).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import json
import copy

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    recall_score, precision_score, fbeta_score, accuracy_score
)
from sklearn.calibration import calibration_curve

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_stage0_5_cleaned.xlsx"
FEAT_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_selected_features.csv"

OUTPUT_DIR = Path.home() / "Desktop" / "nicu_alt_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
N_FOLDS = 5
FORMULA_CLASS_IDX = 1

# Class labels (after LabelEncoder alphabetical sort)
CLASS_LABELS = ["Exclusive BF", "Formula", "Mixed"]

# ==================== WINNING MODEL PARAMS ====================
# From Alt-Stage 2.5 v2 (F2-optimized Optuna)

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
CATBOOST_THRESHOLD = 0.30  # Optimal from threshold sweep

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
RF_THRESHOLD = 0.24

KNN_PARAMS = {
    'n_neighbors': 6,
    'weights': 'distance',
    'metric': 'manhattan',
    'p': 3,
    'n_jobs': -1
}
KNN_THRESHOLD = 0.15

# ==================== PAPER STYLE (matching reference) ====================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 9,
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

# Color palette — matching reference paper
COLORS = {
    'Exclusive BF': '#4C72B0',  # Steel blue
    'Formula':      '#DD8452',  # Warm orange
    'Mixed':        '#55A868',  # Sage green
    'mean':         '#000000',  # Black (mean ROC)
    'chance':       '#C44E52',  # Red dashed (chance)
}
CLASS_COLORS = [COLORS['Exclusive BF'], COLORS['Formula'], COLORS['Mixed']]

# Pastel fold colors (thin, faded — like the reference paper)
FOLD_COLORS = [
    '#AEC7E8', '#FFD699', '#98DF8A', '#FF9896', '#C5B0D5',
    '#C49C94', '#F7B6D2', '#C7C7C7', '#DBDB8D', '#9EDAE5',
]


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

    print(f"  {len(df)} samples, {len(X.columns)} features")
    return X, y_enc, num_cols, cat_cols, le


def get_pipeline(model, num_cols, cat_cols, scale=False):
    steps_num = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps_num.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer([
        ("num", SkPipeline(steps_num), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    return ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", model)
    ])


def apply_threshold(y_proba, threshold, n_classes):
    y_pred = np.zeros(len(y_proba), dtype=int)
    for i in range(len(y_proba)):
        if y_proba[i, FORMULA_CLASS_IDX] >= threshold:
            y_pred[i] = FORMULA_CLASS_IDX
        else:
            probs_adj = y_proba[i].copy()
            probs_adj[FORMULA_CLASS_IDX] = -1
            y_pred[i] = np.argmax(probs_adj)
    return y_pred


def sanitize(name):
    return "".join(c if c.isalnum() or c in "_ " else "_" for c in name).strip()


# ==================== PLOT 1: CONFUSION MATRIX ====================

def plot_confusion_matrix(y_true, y_pred, model_name, le):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(le.classes_)))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cm_pct = np.nan_to_num(cm_pct)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    sns.heatmap(cm_pct, ax=ax, cmap="YlGnBu", vmin=0, vmax=100,
                square=False, annot=False,
                cbar_kws={'shrink': 0.75},
                linewidths=0.8, linecolor='white')

    # Annotate: percentages only (matching reference)
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    cmap_obj = plt.get_cmap("YlGnBu")

    for i in range(len(CLASS_LABELS)):
        for j in range(len(CLASS_LABELS)):
            val_pct = cm_pct[i, j]
            bg = cmap_obj(norm(val_pct))
            lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
            color = "white" if lum < 0.55 else "#333333"

            ax.text(j + 0.5, i + 0.5, f"{val_pct:.2f}",
                    ha="center", va="center", fontsize=15, fontweight="bold", color=color)

    ax.set_xticklabels(CLASS_LABELS, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(CLASS_LABELS, rotation=0, fontsize=12)
    ax.set_xlabel("Predicted Labels", fontsize=16)
    ax.set_ylabel("True Labels", fontsize=16)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"CM_{sanitize(model_name)}.png")
    plt.close()
    print(f"    ✓ CM saved: CM_{sanitize(model_name)}.png")


# ==================== PLOT 2: ROC CURVES (reference paper style) ====================

def plot_roc_curves(pipeline, X_train, y_train, model_name, le, threshold=None):
    """
    Reference paper style: individual fold curves (thin, faded pastels)
    + thick black mean + grey std band + red dashed chance line.
    Uses micro-average ROC across all classes per fold.
    """
    n_classes = len(le.classes_)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mean_fpr = np.linspace(0, 1, 400)
    tprs = []
    aucs = []

    fig, ax = plt.subplots(figsize=(10, 6.8))

    # --- Individual fold curves (thin, faded) ---
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        Xt = X_train.iloc[train_idx]
        yt = y_train[train_idx]
        Xv = X_train.iloc[val_idx]
        yv = y_train[val_idx]

        pipe = clone(pipeline)
        pipe.fit(Xt, yt)
        proba = pipe.predict_proba(Xv)

        # Micro-average ROC: binarize all classes and flatten
        yv_bin = label_binarize(yv, classes=list(range(n_classes)))
        fpr, tpr, _ = roc_curve(yv_bin.ravel(), proba.ravel())
        fold_auc = auc(fpr, tpr)

        # Interpolate onto common FPR grid
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(fold_auc)

        # Plot fold curve — thin, faded
        color = FOLD_COLORS[fold_i % len(FOLD_COLORS)]
        ax.plot(fpr, tpr, lw=1.0, alpha=0.35, color=color,
                label=f"ROC fold {fold_i} (AUC = {fold_auc:.2f})")

    # --- Mean ROC (thick black) ---
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='black', lw=3.0, alpha=0.95,
            label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})")

    # --- Std band (grey fill) ---
    std_tpr = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr,
                     np.maximum(mean_tpr - std_tpr, 0),
                     np.minimum(mean_tpr + std_tpr, 1),
                     color='grey', alpha=0.2)

    # --- Chance line (red dashed) ---
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2.0, color='red', alpha=0.85,
            label='Chance')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("1 - Specificity", fontsize=18)
    ax.set_ylabel("Sensitivity", fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc="lower right", fontsize=10, frameon=True, fancybox=False,
              edgecolor='#CCCCCC')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"ROC_{sanitize(model_name)}.png")
    plt.close()
    print(f"    ✓ ROC saved: ROC_{sanitize(model_name)}.png")


# ==================== PLOT 3: CALIBRATION CURVES (reference paper style) ===========

def plot_calibration(pipeline, X_train, y_train, model_name, le):
    """
    Reference paper style: circle markers, brown dashed diagonal,
    visible grid, title "Calibration Curve".
    """
    n_classes = len(le.classes_)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Collect out-of-fold probabilities
    oof_proba = np.zeros((len(y_train), n_classes))
    for train_idx, val_idx in cv.split(X_train, y_train):
        Xt = X_train.iloc[train_idx]
        yt = y_train[train_idx]
        Xv = X_train.iloc[val_idx]
        pipe = clone(pipeline)
        pipe.fit(Xt, yt)
        oof_proba[val_idx] = pipe.predict_proba(Xv)

    fig, ax = plt.subplots(figsize=(10, 6.8))

    # Perfectly calibrated diagonal (brown dashed, like reference)
    ax.plot([0, 1], [0, 1], linestyle='--', color='brown', lw=2.0, alpha=0.85,
            label='Perfectly calibrated')

    # Per-class calibration with circle markers
    cal_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    for cls_idx in range(n_classes):
        y_bin = (y_train == cls_idx).astype(int)
        frac_pos, mean_pred = calibration_curve(y_bin, oof_proba[:, cls_idx],
                                                 n_bins=10, strategy='uniform')
        ax.plot(mean_pred, frac_pos, 'o-', color=cal_colors[cls_idx],
                lw=1.6, markersize=6, label=CLASS_LABELS[cls_idx])

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Mean predicted probability", fontsize=16)
    ax.set_ylabel("Fraction of positives", fontsize=16)
    ax.set_title("Calibration Curve", fontsize=20)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    ax.grid(True, alpha=0.45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"Calibration_{sanitize(model_name)}.png")
    plt.close()
    print(f"    ✓ Calibration saved: Calibration_{sanitize(model_name)}.png")


# ==================== PLOT 4: PRECISION-RECALL CURVES ====================

def plot_precision_recall(pipeline, X_train, y_train, model_name, le):
    n_classes = len(le.classes_)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Out-of-fold probabilities
    oof_proba = np.zeros((len(y_train), n_classes))
    for train_idx, val_idx in cv.split(X_train, y_train):
        Xt = X_train.iloc[train_idx]
        yt = y_train[train_idx]
        Xv = X_train.iloc[val_idx]
        pipe = clone(pipeline)
        pipe.fit(Xt, yt)
        oof_proba[val_idx] = pipe.predict_proba(Xv)

    fig, ax = plt.subplots(figsize=(7, 6))

    for cls_idx in range(n_classes):
        y_bin = (y_train == cls_idx).astype(int)
        precision, recall, _ = precision_recall_curve(y_bin, oof_proba[:, cls_idx])
        ap = average_precision_score(y_bin, oof_proba[:, cls_idx])

        ax.plot(recall, precision, color=CLASS_COLORS[cls_idx], lw=2.0,
                label=f"{CLASS_LABELS[cls_idx]} (AP = {ap:.3f})")

        # Prevalence baseline
        prevalence = y_bin.mean()
        ax.axhline(y=prevalence, color=CLASS_COLORS[cls_idx], lw=0.8,
                   linestyle=':', alpha=0.4)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall", fontweight='bold')
    ax.set_ylabel("Precision", fontweight='bold')
    ax.set_title(f"Precision-Recall Curves — {model_name}", fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor='#CCCCCC')
    ax.grid(True, alpha=0.3, linestyle='-')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"PR_{sanitize(model_name)}.png")
    plt.close()
    print(f"    ✓ PR saved: PR_{sanitize(model_name)}.png")


# ==================== PLOT 5: THRESHOLD SWEEP ====================

def plot_threshold_sweep(pipeline, X_train, y_train, model_name, le, best_threshold):
    n_classes = len(le.classes_)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Out-of-fold probabilities
    oof_proba = np.zeros((len(y_train), n_classes))
    for train_idx, val_idx in cv.split(X_train, y_train):
        Xt = X_train.iloc[train_idx]
        yt = y_train[train_idx]
        Xv = X_train.iloc[val_idx]
        pipe = clone(pipeline)
        pipe.fit(Xt, yt)
        oof_proba[val_idx] = pipe.predict_proba(Xv)

    thresholds = np.arange(0.10, 0.505, 0.01)
    recalls, precisions, f2s, accs = [], [], [], []

    for thr in thresholds:
        y_pred = apply_threshold(oof_proba, thr, n_classes)
        recalls.append(recall_score(y_train, y_pred, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0))
        precisions.append(precision_score(y_train, y_pred, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0))
        f2s.append(fbeta_score(y_train, y_pred, beta=2, labels=[FORMULA_CLASS_IDX], average='micro', zero_division=0))
        accs.append(accuracy_score(y_train, y_pred))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(thresholds, recalls, '-', color=COLORS['Formula'], lw=2.2, label='Formula Recall')
    ax.plot(thresholds, precisions, '-', color=COLORS['Exclusive BF'], lw=2.2, label='Formula Precision')
    ax.plot(thresholds, f2s, '-', color=COLORS['Mixed'], lw=2.2, label='Formula F₂')
    ax.plot(thresholds, accs, '--', color='#888888', lw=1.5, label='Overall Accuracy')

    # Optimal threshold line
    ax.axvline(x=best_threshold, color=COLORS['chance'], lw=1.5, linestyle='--', alpha=0.8)
    ax.text(best_threshold + 0.01, 0.95, f'Optimal: {best_threshold}',
            color=COLORS['chance'], fontsize=10, fontweight='bold',
            transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes))

    # 1/3 reference (default argmax for 3 classes)
    ax.axvline(x=1/3, color='#AAAAAA', lw=1.0, linestyle=':', alpha=0.6)
    ax.text(1/3 + 0.01, 0.02, 'Default (⅓)',
            color='#999999', fontsize=9,
            transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes))

    ax.set_xlim([0.09, 0.51])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Formula Class Probability Threshold", fontweight='bold')
    ax.set_ylabel("Score", fontweight='bold')
    ax.set_title(f"Threshold Optimization — {model_name}", fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc="center left", frameon=True, fancybox=False, edgecolor='#CCCCCC')
    ax.grid(True, alpha=0.25, linestyle='-')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"Threshold_{sanitize(model_name)}.png")
    plt.close()
    print(f"    ✓ Threshold sweep saved: Threshold_{sanitize(model_name)}.png")


# ==================== MAIN ====================

def main():
    print("=" * 65)
    print("  ALT-STAGE 3: NATURE-QUALITY PLOTS + MODEL EXPORT")
    print("=" * 65)

    X, y, num_cols, cat_cols, le = load_data()
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Split: Train={len(X_train)}, Test={len(X_test)}\n")

    # ---------- Define winners ----------
    models = {
        'CatBoost (F2-tuned)': {
            'model': CatBoostClassifier(**CATBOOST_PARAMS),
            'scale': False,
            'threshold': CATBOOST_THRESHOLD,
            'params': CATBOOST_PARAMS,
        },
        'RF (F2-tuned)': {
            'model': RandomForestClassifier(**RF_PARAMS),
            'scale': False,
            'threshold': RF_THRESHOLD,
            'params': RF_PARAMS,
        },
    }

    for name, info in models.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        pipeline = get_pipeline(info['model'], num_cols, cat_cols, scale=info['scale'])
        thr = info['threshold']

        # --- Fit final model for test predictions ---
        pipeline_full = clone(pipeline)
        pipeline_full.fit(X_train, y_train)

        # Default predictions
        y_pred_default = pipeline_full.predict(X_test)
        # Threshold predictions
        y_proba_test = pipeline_full.predict_proba(X_test)
        y_pred_thr = apply_threshold(y_proba_test, thr, n_classes)

        # --- Plot 1: Confusion Matrices ---
        print("\n  📊 Confusion Matrices:")
        plot_confusion_matrix(y_test, y_pred_default, name, le)
        plot_confusion_matrix(y_test, y_pred_thr, f"{name} + Thr={thr}", le)

        # --- Plot 2: ROC Curves ---
        print("  📈 ROC Curves:")
        plot_roc_curves(pipeline, X_train, y_train, name, le)

        # --- Plot 3: Calibration ---
        print("  📉 Calibration Curves:")
        plot_calibration(pipeline, X_train, y_train, name, le)

        # --- Plot 4: Precision-Recall ---
        print("  🎯 Precision-Recall Curves:")
        plot_precision_recall(pipeline, X_train, y_train, name, le)

        # --- Plot 5: Threshold Sweep ---
        print("  ⚖️  Threshold Sweep:")
        plot_threshold_sweep(pipeline, X_train, y_train, name, le, thr)

    # ==================== MODEL PARAMS EXPORT ====================
    print(f"\n{'='*50}")
    print("  EXPORTING MODEL PARAMETERS")
    print(f"{'='*50}")

    # JSON export
    export_data = {
        "CatBoost": {
            "hyperparameters": {k: v for k, v in CATBOOST_PARAMS.items()
                                if k not in ('verbose', 'allow_writing_files', 'thread_count')},
            "threshold": CATBOOST_THRESHOLD,
            "class_labels": {str(i): CLASS_LABELS[i] for i in range(len(CLASS_LABELS))},
            "formula_class_idx": FORMULA_CLASS_IDX,
        },
        "RandomForest": {
            "hyperparameters": {k: v for k, v in RF_PARAMS.items()
                                if k not in ('n_jobs',)},
            "threshold": RF_THRESHOLD,
            "class_labels": {str(i): CLASS_LABELS[i] for i in range(len(CLASS_LABELS))},
            "formula_class_idx": FORMULA_CLASS_IDX,
        },
        "KNN": {
            "hyperparameters": {k: v for k, v in KNN_PARAMS.items()
                                if k not in ('n_jobs',)},
            "threshold": KNN_THRESHOLD,
        }
    }

    json_path = OUTPUT_DIR / "alt_best_params.json"
    with open(json_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    print(f"  ✓ JSON: {json_path}")

    # Excel export (compatible with Stage-4/5/6 format)
    rows = []
    for model_name, params_dict in [("CatBoost", CATBOOST_PARAMS), ("Random Forest", RF_PARAMS), ("KNN", KNN_PARAMS)]:
        row = {"Model": model_name}
        for k, v in params_dict.items():
            if k == 'class_weights':
                row['w_formula'] = v.get(1, 1.0)
                row['w_mixed'] = v.get(2, 1.0)
            elif k in ('verbose', 'allow_writing_files'):
                continue
            else:
                row[k] = v
        rows.append(row)

    excel_path = OUTPUT_DIR / "alt_stage2pnt5_best_params.xlsx"
    pd.DataFrame(rows).to_excel(excel_path, index=False)
    print(f"  ✓ Excel: {excel_path}")

    print(f"\n{'='*65}")
    print(f"  DONE! All outputs in: {OUTPUT_DIR}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
