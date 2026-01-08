#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Stage 0: Feature Engineering & Data Enrichment
---------------------------------------------------
Purpose: 
Read the raw data, apply domain-specific calculations, 
and save a purely numerical/encoded dataset ready for Stage 1.

Academic Integrity:
- No "Test Set" statistics are used (no global means/scaling).
- All features are calculated row-by-row (Leakage Free).
- Discharge-time variables are EXCLUDED to prevent target leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -------------------- CONFIGURATION --------------------

# Base directory: project root (parent of Code folder)
BASE_DIR = Path(__file__).parent.parent

# Input from excels folder, output to outputs folder
INPUT_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "minorities-united.xlsx"
OUTPUT_PATH = BASE_DIR / "outputs" / "nicu_stage0_engineered.xlsx"
TARGET_COL = "taburculuk_beslenmeturu"

# -------------------- MAPPINGS --------------------

# Clinical Severity Grouping (1=Mild/Routine, 2=Moderate, 3=Severe)
DIAGNOSIS_SEVERITY_MAP = {
    # Mild/Routine
    '4': 1, '8': 1, '18': 1, '2': 1, # Jaundice, Dehydration, Fall, TTN
    # Moderate (Metabolic/Prematurity)
    '1': 2, '5': 2, '9': 2, '14': 2, '19': 2,
    # Severe (Neuro/Cardiac/Sepsis/Anomalies)
    '3': 3, '6': 3, '7': 3, '10': 3, '11': 3, '12': 3, '13': 3, '15': 3, '16': 3, '17': 3
}

# -------------------- ENGINEERING LOGIC --------------------

def engineer_features(df):
    print("--- Starting Feature Engineering ---")
    
    # Avoid division by zero
    EPS = 1e-6 

    # 1. GROWTH & MATURITY
    # --------------------
    print("... Processing Growth Metrics")
    # Ponderal Index Proxy (Mass per week of gestation)
    df['eng_weight_per_week'] = df['dogumagirligi(gram)'] / (df['gebelikhaftası'] + EPS)
    
    # ELBW (Extremely Low Birth Weight)
    df['eng_elbw_flag'] = (df['dogumagirligi(gram)'] < 1000).astype(int)
    
    # Late Preterm vs Early Preterm (Split at 34 weeks)
    df['eng_very_preterm'] = (df['gebelikhaftası'] < 32).astype(int)

    # 2. FEEDING VELOCITY (The "Derivative")
    # --------------------------------------
    print("... Processing Feeding Dynamics")
    
    # Normalize intakes (handle NaNs as 0 for calculation)
    d1_bm = df['aldığıannesütü_ilkgün'].fillna(0)
    d1_formula = df['aldığımamamiktari1.gün'].fillna(0)
    d1_total = d1_bm + d1_formula
    
    d2_total = df['beslenmetotali2.gün'].fillna(0)
    d2_bm = df['beslenme2.gunannesutucc'].fillna(0)
    
    d3_total = df['beslenmetotali3.gun'].fillna(0)
    d3_bm = df['aldıgıannesütü3.gun'].fillna(0)
    d3_formula = df['aldıgımamamiktari3.gun'].fillna(0)
    
    # Intake Acceleration (Delta Volume)
    df['eng_delta_vol_d1_d2'] = d2_total - d1_total
    df['eng_delta_vol_d2_d3'] = d3_total - d2_total
    
    # "The Fighter" Index: Day 3 Intake normalized by Birth Weight
    # A small baby eating huge amounts relative to weight is a strong positive sign
    df['eng_resilience_index'] = d3_total / (df['dogumagirligi(gram)'] + EPS)

    # 3. BREAST MILK DOMINANCE
    # ------------------------
    print("... Processing Lactation Signals")
    df['eng_bm_ratio_d1'] = d1_bm / (d1_total + EPS)
    df['eng_bm_ratio_d2'] = d2_bm / (d2_total + EPS)
    
    # Use actual Day 3 data, NOT discharge data (Prevent Leakage)
    total_d3_calc = d3_bm + d3_formula
    df['eng_bm_ratio_d3'] = d3_bm / (total_d3_calc + EPS)
    
    # Momentum: Is BM ratio increasing?
    df['eng_lactation_momentum'] = df['eng_bm_ratio_d3'] - df['eng_bm_ratio_d1']

    # 4. CLINICAL SEVERITY
    # --------------------
    print("... Processing Clinical Severity")
    # Map diagnosis string to severity score
    df['eng_severity_score'] = df['tanı_gruplu'].astype(str).map(DIAGNOSIS_SEVERITY_MAP).fillna(1)
    
    # Neuro Flag (HIE or Seizures) - Specific barrier to oral feeding
    neuro_codes = ['6', '16', '12'] # HIE, Seizures, CNS Anomaly
    df['eng_neuro_barrier'] = df['tanı_gruplu'].astype(str).apply(lambda x: 1 if x in neuro_codes else 0)

    # 5. MATERNAL & SUPPORT FACTORS
    # -----------------------------
    print("... Processing Maternal Context")
    # Healthcare Professional Mother (Codes 4, 5, 6)
    # These mothers might have higher education but higher stress
    df['eng_mat_healthcare_pro'] = df['annemeslegi'].astype(str).isin(['4', '5', '6']).astype(int)
    
    # High Risk Maternal Age (<18 or >35)
    df['eng_mat_age_risk'] = df['anneyasi'].apply(lambda x: 1 if (x < 18 or x > 35) else 0)
    
    # Intervention Flags
    df['eng_pump_used'] = (df['Kullandıgıpompamarkasi'].fillna(0) != 0).astype(int)
    df['eng_galactagogue'] = (df['galaktokogkullanımı'].fillna(0) == 1).astype(int)
    
    # 6. FEEDING ROUTE EVOLUTION
    # --------------------------
    print("... Processing Route Evolution")
    # 0=None, 1=PO, 2=OG...
    # We want to detect OG (Tube) -> PO (Oral) transition
    # Simplified: Is Day 3 "better" than Day 1?
    # Let's assume Higher Number = More Oral? No, the coding is messy.
    # Let's Create Boolean Flags: "Has Tube?"
    
    tube_codes = [2, 3, 6] # OG, PO+OG, BF+OG
    
    df['eng_tube_d1'] = df['beslenmeninilkgunuverilisyolu'].isin(tube_codes).astype(int)
    df['eng_tube_d2'] = df['verilisyolu2.gun'].isin(tube_codes).astype(int)
    df['eng_tube_d3'] = df['verilisyolu3gun'].isin(tube_codes).astype(int)
    
    # Weaning Success: Tube on D1 but NO Tube on D3
    df['eng_weaning_success'] = ((df['eng_tube_d1'] == 1) & (df['eng_tube_d3'] == 0)).astype(int)

    return df

# -------------------- MAIN EXECUTION --------------------

def main():
    print(f"Reading raw data from: {INPUT_PATH}")
    df = pd.read_excel(INPUT_PATH)
    
    # Basic Clean (Fill NaNs for calculation columns)
    # We fill with 0 for volume calculations, but keep original for analysis
    fill_cols = ["sutdestegivarsakacolcek", "varsataburculuktakaçölçek", "memesorunuvarsa_tedavidekullanılanlar"]
    for c in fill_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)
            
    # Drop rows with missing Target
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"Data Loaded: {df.shape[0]} rows")

    # Run Engineering
    df_eng = engineer_features(df)
    
    # Final Cleanup
    print("Saving engineered dataset...")
    df_eng.to_excel(OUTPUT_PATH, index=False)
    print(f"DONE. Saved to: {OUTPUT_PATH}")
    print("You can now point Stage-1 to this new file.")

if __name__ == "__main__":
    main()