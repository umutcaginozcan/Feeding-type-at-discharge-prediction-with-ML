#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Alt-Stage 5c: COVID Sensitivity Analysis
------------------------------------------------
Purpose:
  Assess whether removing COVID-period temporal covariates
  (covid19sonrasi, ikisiarası/Epoch) degrades model performance.
  BFHI status (bebek_dostu_20temmuz2018) is force-included as a
  standalone institutional variable in the COVID-removed model.

Design:
  Two feature configurations, each evaluated with untuned RF and CatBoost:
    1. Full Model  — all RFECV-selected features (status quo)
    2. COVID-Removed — drop covid19sonrasi & ikisiarası, add BFHI standalone

  Untuned models are used (same as temporal ablation) so the comparison
  is not biased by hyperparameters tuned on the COVID-inclusive set.

Statistical Tests:
  1. DeLong's test         — AUC-ROC equivalence (paired on test set)
  2. McNemar's test        — prediction agreement (holdout)
  3. Bootstrap 95% CIs     — metric deltas (10,000 resamples)

Metrics:
  AUC-ROC (macro OVR), MCC, Formula Recall, Formula Precision, F1-Macro

Output:
  ~/Desktop/nicu_covid_sensitivity.xlsx
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time
from scipy import stats as sp_stats

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

N_BOOTSTRAP = 10_000

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_stage0_5_cleaned.xlsx"
FEAT_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_selected_features.csv"
OUTPUT_FILE = Path.home() / "Desktop" / "nicu_covid_sensitivity.xlsx"

TARGET_COL = "taburculuk_beslenmeturu"
RANDOM_STATE = 42
N_FOLDS = 5
FORMULA_CLASS_IDX = 1   # After LabelEncoder (alphabetical): EBF=0, Formula=1, Mixed=2

# ---- COVID-related raw columns to EXCLUDE ----
COVID_RAW_COLS = {"covid19sonrasi", "ikisiarası"}

# ---- BFHI raw column to FORCE-INCLUDE ----
BFHI_RAW_COL = "bebek_dostu_20temmuz2018"

# ==================== DATA LOADING ====================

def load_data():
    """Load data and identify selected features."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    selected_df = pd.read_csv(FEAT_PATH)
    selected_feat_names = set(selected_df["Selected_Features"].tolist())

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    y = df[TARGET_COL]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Map selected OHE feature names back to raw columns
    all_raw_cols = df.columns.tolist()
    if TARGET_COL in all_raw_cols:
        all_raw_cols.remove(TARGET_COL)

    selected_raw_cols = set()
    for raw_col in all_raw_cols:
        if raw_col in selected_feat_names:
            selected_raw_cols.add(raw_col)
            continue
        for sel in selected_feat_names:
            if sel.startswith(str(raw_col)):
                selected_raw_cols.add(raw_col)
                break

    print(f"  {len(df)} samples, {len(selected_raw_cols)} selected raw columns")
    print(f"  Classes: {dict(zip(le.classes_, np.bincount(y_enc)))}")

    return df, y_enc, selected_raw_cols, le


def build_feature_set(df, raw_cols_to_use):
    """
    Build X, num_cols, cat_cols from a specific set of raw column names.
    Only keeps columns that exist in the DataFrame.
    """
    cols = [c for c in raw_cols_to_use if c in df.columns]
    X = df[cols].copy()

    num_cols, cat_cols = [], []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            X[c] = X[c].astype(str)
            cat_cols.append(c)

    return X, num_cols, cat_cols


# ==================== PIPELINE ====================

def get_pipeline(model, num_cols, cat_cols):
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    return ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", model)
    ])


# ==================== EVALUATION ====================

def evaluate(pipeline, X_train, y_train, X_test, y_test, le):
    """
    Full evaluation: 5-fold CV + holdout test.
    Returns cv_metrics, test_metrics, y_pred (holdout), y_proba (holdout).
    """
    n_classes = len(le.classes_)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_metrics = {
        "F1_Macro": [], "MCC": [],
        "Formula_Prec": [], "Formula_Rec": [],
        "AUC_ROC": [],
    }

    for train_idx, val_idx in cv.split(X_train, y_train):
        Xt = X_train.iloc[train_idx]
        yt = y_train[train_idx]
        Xv = X_train.iloc[val_idx]
        yv = y_train[val_idx]

        pipe = clone(pipeline)
        pipe.fit(Xt, yt)
        yp = pipe.predict(Xv)
        yproba = pipe.predict_proba(Xv)

        cv_metrics["F1_Macro"].append(f1_score(yv, yp, average="macro"))
        cv_metrics["MCC"].append(matthews_corrcoef(yv, yp))
        cv_metrics["Formula_Prec"].append(
            precision_score(yv, yp, labels=[FORMULA_CLASS_IDX],
                            average="micro", zero_division=0))
        cv_metrics["Formula_Rec"].append(
            recall_score(yv, yp, labels=[FORMULA_CLASS_IDX],
                         average="micro", zero_division=0))
        try:
            cv_metrics["AUC_ROC"].append(
                roc_auc_score(yv, yproba, multi_class="ovr", average="macro"))
        except Exception:
            cv_metrics["AUC_ROC"].append(np.nan)

    # --- Holdout test ---
    pipe_full = clone(pipeline)
    pipe_full.fit(X_train, y_train)
    y_pred = pipe_full.predict(X_test)
    y_proba = pipe_full.predict_proba(X_test)

    test_metrics = {
        "F1_Macro": f1_score(y_test, y_pred, average="macro"),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Formula_Prec": precision_score(
            y_test, y_pred, labels=[FORMULA_CLASS_IDX],
            average="micro", zero_division=0),
        "Formula_Rec": recall_score(
            y_test, y_pred, labels=[FORMULA_CLASS_IDX],
            average="micro", zero_division=0),
    }
    try:
        test_metrics["AUC_ROC"] = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        test_metrics["AUC_ROC"] = np.nan

    return cv_metrics, test_metrics, y_pred, y_proba


# ==================== STATISTICAL TESTS ====================

def delong_roc_test(y_true, proba_A, proba_B):
    """
    DeLong's test for the difference of two AUC-ROC values.
    Uses the multi-class macro-OVR approach: run per-class binary
    DeLong tests and combine.

    Returns: (z_stat, p_value, auc_A, auc_B)
    """
    classes = np.unique(y_true)
    z_scores = []
    auc_As, auc_Bs = [], []

    for cls in classes:
        y_bin = (y_true == cls).astype(int)
        pA = proba_A[:, cls]
        pB = proba_B[:, cls]

        auc_a = _fast_auc(y_bin, pA)
        auc_b = _fast_auc(y_bin, pB)
        auc_As.append(auc_a)
        auc_Bs.append(auc_b)

        # DeLong variance estimation
        V_A, V_B, V_AB = _delong_covariance(y_bin, pA, pB)
        var_diff = V_A + V_B - 2 * V_AB

        if var_diff <= 0:
            z_scores.append(0.0)
        else:
            z_scores.append((auc_a - auc_b) / np.sqrt(var_diff))

    # Combine per-class z-scores (Stouffer's method)
    z_combined = np.mean(z_scores) * np.sqrt(len(z_scores))
    p_value = 2.0 * sp_stats.norm.sf(np.abs(z_combined))

    return z_combined, p_value, np.mean(auc_As), np.mean(auc_Bs)


def _fast_auc(y_true, y_score):
    """Compute AUC via sorting."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_score)


def _delong_covariance(y_true, scores_A, scores_B):
    """
    Estimate DeLong's covariance components for two ROC curves.
    Based on: DeLong et al., Biometrics, 1988.
    """
    pos = scores_A[y_true == 1]
    neg = scores_A[y_true == 0]
    m = len(pos)  # positives
    n = len(neg)  # negatives

    pos_B = scores_B[y_true == 1]
    neg_B = scores_B[y_true == 0]

    # Placement values
    # V_10(X_i) = (1/n) * Σ_j ψ(X_i, Y_j) for positive sample i
    V10_A = np.array([np.mean((pos[i] > neg).astype(float) +
                               0.5 * (pos[i] == neg).astype(float))
                      for i in range(m)])
    V01_A = np.array([np.mean((pos > neg[j]).astype(float) +
                               0.5 * (pos == neg[j]).astype(float))
                      for j in range(n)])

    V10_B = np.array([np.mean((pos_B[i] > neg_B).astype(float) +
                               0.5 * (pos_B[i] == neg_B).astype(float))
                      for i in range(m)])
    V01_B = np.array([np.mean((pos_B > neg_B[j]).astype(float) +
                               0.5 * (pos_B == neg_B[j]).astype(float))
                      for j in range(n)])

    # Variance components
    S10_A = np.var(V10_A, ddof=1) if m > 1 else 0
    S01_A = np.var(V01_A, ddof=1) if n > 1 else 0
    S10_B = np.var(V10_B, ddof=1) if m > 1 else 0
    S01_B = np.var(V01_B, ddof=1) if n > 1 else 0

    # Cross-covariance
    S10_AB = np.cov(V10_A, V10_B, ddof=1)[0, 1] if m > 1 else 0
    S01_AB = np.cov(V01_A, V01_B, ddof=1)[0, 1] if n > 1 else 0

    V_A = S10_A / m + S01_A / n
    V_B = S10_B / m + S01_B / n
    V_AB = S10_AB / m + S01_AB / n

    return V_A, V_B, V_AB


def mcnemar_test(y_true, pred_A, pred_B):
    """
    McNemar's test: are the two models making systematically different errors?
    Tests whether the off-diagonal counts (A-right-B-wrong vs A-wrong-B-right)
    differ significantly.

    Returns: (chi2_stat, p_value, n_discordant, contingency_table_dict)
    """
    correct_A = (pred_A == y_true)
    correct_B = (pred_B == y_true)

    # 2×2 contingency:  A correct / A wrong  ×  B correct / B wrong
    b = np.sum(correct_A & ~correct_B)   # A right, B wrong
    c = np.sum(~correct_A & correct_B)   # A wrong, B right
    a = np.sum(correct_A & correct_B)    # both right
    d = np.sum(~correct_A & ~correct_B)  # both wrong

    n_discordant = b + c

    # Use exact binomial test if discordant count < 25
    if n_discordant < 25:
        # Exact McNemar: binomial test of b vs (b+c) with p=0.5
        p_value = sp_stats.binom_test(b, n_discordant, 0.5) if n_discordant > 0 else 1.0
        chi2_stat = ((b - c) ** 2) / (b + c) if (b + c) > 0 else 0.0
    else:
        # Corrected McNemar
        chi2_stat = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0
        p_value = 1.0 - sp_stats.chi2.cdf(chi2_stat, df=1)

    table = {"Both_Correct": int(a), "A_only": int(b),
             "B_only": int(c), "Both_Wrong": int(d)}

    return chi2_stat, p_value, n_discordant, table


def bootstrap_metric_ci(y_true, pred_A, proba_A, pred_B, proba_B,
                        metric_name, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    """
    Bootstrap 95% CI for the difference (COVID-Removed − Full) of a metric.
    Returns: (observed_delta, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    def _compute_metric(yt, yp, yproba):
        if metric_name == "AUC_ROC":
            try:
                return roc_auc_score(yt, yproba, multi_class="ovr", average="macro")
            except Exception:
                return np.nan
        elif metric_name == "MCC":
            return matthews_corrcoef(yt, yp)
        elif metric_name == "Formula_Rec":
            return recall_score(yt, yp, labels=[FORMULA_CLASS_IDX],
                                average="micro", zero_division=0)
        elif metric_name == "Formula_Prec":
            return precision_score(yt, yp, labels=[FORMULA_CLASS_IDX],
                                   average="micro", zero_division=0)
        elif metric_name == "F1_Macro":
            return f1_score(yt, yp, average="macro")
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    observed_A = _compute_metric(y_true, pred_A, proba_A)
    observed_B = _compute_metric(y_true, pred_B, proba_B)
    observed_delta = observed_B - observed_A

    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt_b = y_true[idx]
        # Skip if bootstrap sample has <2 classes
        if len(np.unique(yt_b)) < 2:
            continue
        mA = _compute_metric(yt_b, pred_A[idx], proba_A[idx], )
        mB = _compute_metric(yt_b, pred_B[idx], proba_B[idx])
        if np.isnan(mA) or np.isnan(mB):
            continue
        deltas.append(mB - mA)

    deltas = np.array(deltas)
    ci_lo = np.percentile(deltas, 2.5)
    ci_hi = np.percentile(deltas, 97.5)

    return observed_delta, ci_lo, ci_hi


# ==================== MAIN ====================

def main():
    print("=" * 65)
    print("  ALT-STAGE 5c: COVID SENSITIVITY ANALYSIS")
    print("=" * 65)

    df, y, selected_raw_cols, le = load_data()

    # ---- Define the two feature configurations ----

    # 1. Full Model: all RFECV-selected raw columns
    full_cols = selected_raw_cols

    # 2. COVID-Removed: drop COVID-related raw columns, force-add BFHI
    covid_removed_cols = (selected_raw_cols - COVID_RAW_COLS) | {BFHI_RAW_COL}

    print(f"\n  Full Model raw columns ({len(full_cols)}):")
    print(f"    {sorted(full_cols)}")
    print(f"\n  COVID-Removed raw columns ({len(covid_removed_cols)}):")
    print(f"    {sorted(covid_removed_cols)}")

    # Highlight what changed
    dropped = full_cols - covid_removed_cols
    added = covid_removed_cols - full_cols
    print(f"\n  Dropped: {sorted(dropped)}")
    print(f"  Added:   {sorted(added)}")

    # ---- Data split (same seed as all other stages) ----
    all_idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"\n  Split: Train={len(train_idx)}, Test={len(test_idx)}\n")

    # ---- Models ----
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "CatBoost": CatBoostClassifier(
            auto_class_weights="Balanced",
            verbose=False, allow_writing_files=False,
            random_state=RANDOM_STATE, thread_count=-1
        ),
    }

    # ---- Feature configurations ----
    configs = {
        "Full Model": full_cols,
        "COVID-Removed": covid_removed_cols,
    }

    results = []
    # Store holdout predictions/probabilities per (model, config) for stat tests
    holdout_data = {}  # key: (model_name, config_name) → (y_pred, y_proba)

    for config_name, raw_cols in configs.items():
        print(f"\n{'='*55}")
        print(f"  Configuration: {config_name}")
        print(f"{'='*55}")

        X, num_cols, cat_cols = build_feature_set(df, raw_cols)
        n_features = len(X.columns)

        X_train = X.iloc[train_idx]
        X_test_w = X.iloc[test_idx]

        print(f"  Features: {n_features} raw columns "
              f"({len(num_cols)} num, {len(cat_cols)} cat)")
        print(f"  Columns: {sorted(X.columns.tolist())}")

        for model_name, model in models.items():
            print(f"\n  >>> {model_name}...")
            t0 = time.time()

            pipeline = get_pipeline(clone(model), num_cols, cat_cols)
            cv_m, test_m, y_pred_test, y_proba_test = evaluate(
                pipeline, X_train, y_train, X_test_w, y_test, le
            )

            elapsed = time.time() - t0

            # Store holdout data for statistical tests
            holdout_data[(model_name, config_name)] = (y_pred_test, y_proba_test)

            row = {
                "Configuration": config_name,
                "Model": model_name,
                "N_Features": n_features,
                # CV (mean ± std)
                "CV AUC-ROC": f"{np.nanmean(cv_m['AUC_ROC']):.3f} ± {np.nanstd(cv_m['AUC_ROC']):.3f}",
                "CV MCC": f"{np.mean(cv_m['MCC']):.3f} ± {np.std(cv_m['MCC']):.3f}",
                "CV Formula Rec": f"{np.mean(cv_m['Formula_Rec']):.3f} ± {np.std(cv_m['Formula_Rec']):.3f}",
                "CV F1-Macro": f"{np.mean(cv_m['F1_Macro']):.3f} ± {np.std(cv_m['F1_Macro']):.3f}",
                # Test
                "Test AUC-ROC": round(test_m["AUC_ROC"], 3),
                "Test MCC": round(test_m["MCC"], 3),
                "Test Formula Prec": round(test_m["Formula_Prec"], 3),
                "Test Formula Rec": round(test_m["Formula_Rec"], 3),
                "Test F1-Macro": round(test_m["F1_Macro"], 3),
                "Time (s)": round(elapsed, 1),
            }
            results.append(row)

            print(f"      AUC-ROC: {test_m['AUC_ROC']:.3f}  |  "
                  f"MCC: {test_m['MCC']:.3f}  |  "
                  f"Formula Rec: {test_m['Formula_Rec']:.3f}  |  "
                  f"F1: {test_m['F1_Macro']:.3f}  ({elapsed:.1f}s)")

    # ==================== RESULTS TABLE ====================
    df_res = pd.DataFrame(results)

    print("\n\n" + "=" * 100)
    print("  COVID SENSITIVITY — FULL RESULTS")
    print("=" * 100)
    print(df_res.to_string(index=False))

    # ==================== DELTA TABLE ====================
    print("\n\n" + "=" * 70)
    print("  PERFORMANCE DELTA (COVID-Removed − Full Model)")
    print("=" * 70)

    delta_rows = []
    test_metric_names = ["Test AUC-ROC", "Test MCC", "Test Formula Rec",
                         "Test Formula Prec", "Test F1-Macro"]

    for model_name in models.keys():
        full_row = df_res[(df_res["Model"] == model_name)
                          & (df_res["Configuration"] == "Full Model")]
        removed_row = df_res[(df_res["Model"] == model_name)
                             & (df_res["Configuration"] == "COVID-Removed")]

        if full_row.empty or removed_row.empty:
            continue

        delta = {"Model": model_name}
        print(f"\n  {model_name}:")
        for metric in test_metric_names:
            full_val = full_row[metric].values[0]
            rem_val = removed_row[metric].values[0]
            d = rem_val - full_val
            delta[f"Δ{metric.replace('Test ', '')}"] = round(d, 4)
            print(f"    {metric:20s}  Full={full_val:.3f}  "
                  f"Removed={rem_val:.3f}  Δ={d:+.3f}")

        delta_rows.append(delta)

    df_delta = pd.DataFrame(delta_rows)

    # ============================================================
    # STATISTICAL TESTS
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  STATISTICAL TESTS")
    print("=" * 70)

    stat_rows = []

    for model_name in models.keys():
        pred_full, proba_full = holdout_data[(model_name, "Full Model")]
        pred_rem,  proba_rem  = holdout_data[(model_name, "COVID-Removed")]

        print(f"\n  ── {model_name} ──")

        # --- 1. DeLong's Test (AUC-ROC equivalence) ---
        print("\n  1) DeLong's Test (H₀: AUC_Full = AUC_Removed):")
        z_dl, p_dl, auc_full, auc_rem = delong_roc_test(
            y_test, proba_full, proba_rem
        )
        sig_dl = "*" if p_dl < 0.05 else "ns"
        print(f"     AUC Full={auc_full:.4f}, AUC Removed={auc_rem:.4f}")
        print(f"     z = {z_dl:.3f}, p = {p_dl:.4f}  [{sig_dl}]")

        # --- 2. McNemar's Test (prediction agreement) ---
        print("  2) McNemar's Test (H₀: models make same errors):")
        chi2_mn, p_mn, n_disc, table = mcnemar_test(
            y_test, pred_full, pred_rem
        )
        sig_mn = "*" if p_mn < 0.05 else "ns"
        print(f"     Contingency: {table}")
        print(f"     Discordant pairs: {n_disc}")
        exact_note = " (exact binomial)" if n_disc < 25 else " (χ² corrected)"
        print(f"     χ² = {chi2_mn:.3f}, p = {p_mn:.4f}  [{sig_mn}]{exact_note}")

        # --- 3. Bootstrap 95% CIs ---
        print(f"  3) Bootstrap 95% CIs (n={N_BOOTSTRAP:,}):")
        boot_metrics = ["AUC_ROC", "MCC", "Formula_Rec", "Formula_Prec", "F1_Macro"]

        stat_entry = {
            "Model": model_name,
            "DeLong_z": round(z_dl, 3),
            "DeLong_p": round(p_dl, 4),
            "DeLong_sig": sig_dl,
            "McNemar_chi2": round(chi2_mn, 3),
            "McNemar_p": round(p_mn, 4),
            "McNemar_sig": sig_mn,
            "McNemar_discordant": n_disc,
        }

        for bm in boot_metrics:
            obs_d, ci_lo, ci_hi = bootstrap_metric_ci(
                y_test, pred_full, proba_full, pred_rem, proba_rem,
                metric_name=bm
            )
            contains_zero = "contains 0" if ci_lo <= 0 <= ci_hi else "excludes 0"
            print(f"     Δ{bm:15s} = {obs_d:+.4f}  "
                  f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  ({contains_zero})")

            stat_entry[f"Δ{bm}"] = round(obs_d, 4)
            stat_entry[f"Δ{bm}_CI_lo"] = round(ci_lo, 4)
            stat_entry[f"Δ{bm}_CI_hi"] = round(ci_hi, 4)
            stat_entry[f"Δ{bm}_CI_contains_0"] = contains_zero

        stat_rows.append(stat_entry)

    df_stats = pd.DataFrame(stat_rows)

    # ==================== SAVE TO EXCEL ====================
    print(f"\n\nSaving to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        df_res.to_excel(writer, sheet_name="Results", index=False)
        df_delta.to_excel(writer, sheet_name="Delta", index=False)
        df_stats.to_excel(writer, sheet_name="Statistical Tests", index=False)

    # ==================== INTERPRETATION ====================
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)

    all_p_nonsig = all(r["DeLong_p"] > 0.05 and r["McNemar_p"] > 0.05
                       for r in stat_rows)

    if all_p_nonsig:
        print("\n  ✅ Both DeLong and McNemar tests are NON-SIGNIFICANT (p > 0.05)")
        print("     for all models. The Full and COVID-Removed models are")
        print("     statistically equivalent in discriminative performance")
        print("     and prediction patterns.")
        print("\n  Suggested text:")
        for r in stat_rows:
            mn = r["Model"]
            print(f"    {mn}: DeLong p = {r['DeLong_p']:.3f}, "
                  f"McNemar p = {r['McNemar_p']:.3f}")
        print("\n  \"In a sensitivity analysis, removing COVID-related temporal")
        print("  covariates did not significantly alter model discrimination")
        for r in stat_rows:
            mn = r["Model"]
            auc_ci = f"[{r['ΔAUC_ROC_CI_lo']:+.3f}, {r['ΔAUC_ROC_CI_hi']:+.3f}]"
            print(f"  ({mn}: DeLong p = {r['DeLong_p']:.3f}; "
                  f"ΔAUC-ROC 95% CI {auc_ci})")
        print("  or prediction patterns (McNemar p > 0.05 for all models).")
        print("  BFHI status was retained as an active institutional variable.\"")
    else:
        print("\n  ⚠  Some tests are significant — review the results above.")
        for r in stat_rows:
            if r["DeLong_p"] < 0.05:
                print(f"    {r['Model']}: DeLong p = {r['DeLong_p']:.4f} — AUC difference is significant")
            if r["McNemar_p"] < 0.05:
                print(f"    {r['Model']}: McNemar p = {r['McNemar_p']:.4f} — Error patterns differ")

    print("\n" + "=" * 65)
    print("  DONE!")
    print("=" * 65)


if __name__ == "__main__":
    main()
