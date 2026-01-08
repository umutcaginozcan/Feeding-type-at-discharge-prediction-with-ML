#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Stage 0.5: Data Compression & Cleanup
------------------------------------------
Purpose: 
Refine the dataset BEFORE Feature Selection.
1. Convert Binary Categoricals (1/2) to single flags (0/1).
2. Convert Ordinal Categoricals (Low/Mid/High) to numeric scales.
3. Prevent 'OneHotEncoder' explosion in the next stage.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -------------------- CONFIGURATION --------------------

# Base directory: project root
BASE_DIR = Path(__file__).parent.parent

INPUT_PATH = BASE_DIR / "outputs" / "nicu_stage0_engineered.xlsx"
OUTPUT_PATH = BASE_DIR / "outputs" / "nicu_stage0_5_cleaned.xlsx"
TARGET_COL = "taburculuk_beslenmeturu"

# -------------------- MAPPINGS --------------------

def clean_data(df):
    print("--- Starting Cleanup & Compression ---")
    
    # 1. BINARY MAPPING (Collapse 2 columns into 1)
    # ---------------------------------------------
    print("... Compressing Binary Variables")
    
    # Cinsiyet: 1=Female, 2=Male -> 0=Female, 1=Male
    if 'cinsiyeti' in df.columns:
        df['cinsiyeti'] = df['cinsiyeti'].map({1: 0, 2: 1, '1': 0, '2': 1})
        
    # Dogum Sekli: 1=Vaginal, 2=Cesarean -> 0=Vaginal, 1=C-Sec
    if 'dogumsekli' in df.columns:
        df['dogumsekli'] = df['dogumsekli'].map({1: 0, 2: 1, '1': 0, '2': 1})
        
    # Gebelik Tipi: 1=Singleton, 2=Multiple -> 0=Singleton, 1=Multiple
    # Note: 'gebeliktipi' has 3 values (Triplet), but 'gebelik_tipi_gruplu' is binary
    if 'gebelik_tipi_gruplu' in df.columns:
        df['gebelik_tipi_gruplu'] = df['gebelik_tipi_gruplu'].map({1: 0, 2: 1, '1': 0, '2': 1})

    # 2. ORDINAL MAPPING (Preserve 1 < 2 < 3 Logic)
    # ---------------------------------------------
    print("... Restoring Ordinal Scales")
    
    # Education: 1=Illiterate ... 3=HighSchool+
    # We keep these as numbers so the model sees the progression
    ordinal_cols = [
        'anne_egitim_grup', 
        'anne_yaşı_grup', 
        'dogum_agırlıgı_gruplu', 
        'VAR00004' # Gestational age groups
    ]
    
    for col in ordinal_cols:
        if col in df.columns:
            # Force to numeric, coerce errors to NaN (will be imputed later)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. ENGINEERED FLAGS (Protect from OneHot)
    # -----------------------------------------
    print("... Standardizing Engineered Flags")
    # These are already 0/1, but we ensure they are float/int to avoid string encoding
    eng_cols = [c for c in df.columns if c.startswith('eng_')]
    for col in eng_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def generate_new_config(df):
    """
    Generates the LISTS you need to copy-paste into Stage 1.
    """
    all_cols = df.columns.tolist()
    if TARGET_COL in all_cols: all_cols.remove(TARGET_COL)
    
    # Heuristic: 
    # If it's Object/String -> Categorical (Needs OneHot)
    # If it's Numeric -> Numeric (Needs Scaling/Passing)
    
    num_cols = []
    cat_cols = []
    
    for c in all_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
            
    return num_cols, cat_cols

# -------------------- MAIN --------------------

def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_excel(INPUT_PATH)
    
    # Clean
    df_clean = clean_data(df)
    
    # Save
    df_clean.to_excel(OUTPUT_PATH, index=False)
    print(f"\nSAVED cleaned dataset to: {OUTPUT_PATH}")
    
    # Generate Lists for the User
    num_cols, cat_cols = generate_new_config(df_clean)
    
    print("\n" + "="*30)
    print("COPY THESE LISTS INTO STAGE 1 SCRIPT")
    print("="*30)
    print(f"\nNUMERIC_COLS = {num_cols}")
    print(f"\nCATEGORICAL_COLS = {cat_cols}")
    print("\n" + "="*30)

if __name__ == "__main__":
    main()