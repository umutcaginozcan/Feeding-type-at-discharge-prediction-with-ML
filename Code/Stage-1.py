#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Stage 1: The Selector (RFECV)
----------------------------------
Purpose: 
Scientifically determine the optimal subset of features.
Uses Random Forest with Recursive Feature Elimination inside a CV loop.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline

# -------------------- CONFIGURATION --------------------

# Base directory: project root
BASE_DIR = Path(__file__).parent.parent

# POINT THIS TO THE OUTPUT OF STAGE 0
DATA_PATH = BASE_DIR / "outputs" / "nicu_stage0_5_cleaned.xlsx"
OUTPUT_PATH = BASE_DIR / "outputs" / "nicu_selected_features.csv"
PLOT_PATH = BASE_DIR / "outputs" / "nicu_feature_selection_curve.png"
TARGET_COL = "taburculuk_beslenmeturu"

RANDOM_STATE = 42

# --- THE FULL POOL (Original + Engineered) ---

NUMERIC_COLS = [
    # --- Original Clinical Data ---
    "dogumagirligi(gram)", "gebelikhaftası", "anneyasi", 
    "yasayancocuksayisi", "emzirdigicocuksayisi", "bironcekibebegikacayemzirdi",
    "takibegirdigigun", "takipilkgün_kilo_gram", "gebelikhaftagunu",
    "kilo1.gun", "kilo2.gun", "kilo3.gun", "sutdestegivarsakacolcek",
    
    # --- Original Intake Data ---
    "aldığıannesütü_ilkgün", "aldığımamamiktari1.gün",
    "beslenmetotali2.gün", "beslenme2.gunannesutucc", "beslenmemamamiktarı2.guncc",
    "beslenmetotali3.gun", "aldıgıannesütü3.gun", "aldıgımamamiktari3.gun",
    
    # --- NEW: Stage 0 Engineered Features (The heavy hitters) ---
    "eng_weight_per_week",        # Maturity proxy
    "eng_delta_vol_d1_d2",        # Velocity
    "eng_delta_vol_d2_d3",        # Velocity
    "eng_resilience_index",       # Intake relative to weight
    "eng_bm_ratio_d1",            # Early latching success
    "eng_bm_ratio_d2",
    "eng_bm_ratio_d3",            # Trajectory
    "eng_lactation_momentum"      # Is milk supply increasing?
]

CATEGORICAL_COLS = [
    # --- Original ---
    "tanı_gruplu", "dogum_agırlıgı_gruplu", "cinsiyeti", "dogumsekli", "dogumyeri",
    "anne_meslek_grup", "anne_egitim_grup", "anneegitim", "anne_hastalık_grup", "gebeliktipi", "gebelik_34", "annemeslegi", "anne_yaşı_grup",
    "bebek_dostu_20temmuz2018", "covid19sonrasi", "gebelik_tipi_gruplu", "VAR00004", "gebelik_haftası_gruplu", "ilk_gün_anne_sütü_1111", "ikisiarası", "tanısı",
    
    # --- Equipment / Process ---
    "Kullandıgıpompamarkasi", "Kullandıgpompatipi", "galaktokogkullanımı",
    "memesorunuyaşamadurumu", "memesorunuvarsa_tedavidekullanılanlar",
    "baslangictasutdestegi", "annesutuemzirmeeğitimidurumu",
    
    # --- Feeding Routes ---
    "ilkgün_bebeğinannesütüalımı", "ilk_gün_emzirme_111", "beslenmeninilkgunuverilisyolu",
    "verilisyolu2.gun", "verilisyolu3gun", "Kolostrumvarligi",
    
    # --- NEW: Stage 0 Engineered Categoricals ---
    "eng_elbw_flag",              # Extreme prematurity
    "eng_very_preterm",           # <32 weeks
    "eng_severity_score",         # 1/2/3 Scale
    "eng_neuro_barrier",          # HIE/Seizures
    "eng_mat_healthcare_pro",     # Stress factor
    "eng_mat_age_risk",           # <18 or >35
    "eng_pump_used",              # Intervention
    "eng_weaning_success"         # Tube -> Oral transition
]

# -------------------- MAIN EXECUTION --------------------

def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    
    # --- FIX: FORCE DATA TYPES ---
    print("Standardizing data types to prevent Encoder errors...")
    
    # 1. Force Categoricals to String
    # This converts 1 -> "1", NaN -> "nan" (string), preventing mixed-type crashes
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype(str)
    
    # 2. Force Numerics to Float
    # This turns non-numeric garbage into NaNs that the Imputer can handle
    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pd.to_numeric, errors='coerce')

    # 3. Drop rows without target
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]
    
    # Encode Target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # 4. Pipeline Setup
    preprocessor = ColumnTransformer([
        ("num", ImbPipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", MinMaxScaler())
        ]), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)
    ])
    
    # 5. The Selector (Random Forest)
    clf = RandomForestClassifier(
        n_estimators=100, 
        class_weight="balanced", 
        n_jobs=-1, 
        random_state=RANDOM_STATE
    )
    
    rfecv = RFECV(
        estimator=clf, 
        step=1, 
        cv=StratifiedKFold(5), 
        scoring='f1_macro', 
        n_jobs=-1,
        verbose=1
    )
    
    # Create full pipeline
    print("Preprocessing data...")
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names
    try:
        cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_COLS)
        all_feature_names = np.array(NUMERIC_COLS + list(cat_names))
    except:
        all_feature_names = np.array([f"feat_{i}" for i in range(X_processed.shape[1])])
        
    print(f"Starting Recursive Feature Elimination on {X_processed.shape[1]} features...")
    print("This may take a few minutes on M2 Max...")
    
    rfecv.fit(X_processed, y_enc)
    
    # Results
    print(f"\nOptimal number of features: {rfecv.n_features_}")
    print(f"Best F1-Macro Score: {max(rfecv.cv_results_['mean_test_score']):.4f}")
    
    selected_features = all_feature_names[rfecv.support_]
    
    print("\nTop 10 Selected Features (Sample):")
    print(selected_features[:10])
    
    # Save
    pd.DataFrame(selected_features, columns=["Selected_Features"]).to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved selected list to: {OUTPUT_PATH}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (F1 Macro)")
    
    scores = rfecv.cv_results_['mean_test_score']
    
    plt.plot(range(1, len(scores) + 1), scores)
    plt.title(f"RFECV Results (Optimal: {rfecv.n_features_} features)")
    plt.grid(True)
    plt.savefig(PLOT_PATH)
    print(f"Saved plot to: {PLOT_PATH}")

if __name__ == "__main__":
    main()