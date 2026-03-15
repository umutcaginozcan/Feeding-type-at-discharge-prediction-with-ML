#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Table 1: Sample Characteristics (Nature Format)
---------------------------------------------------------
Creates publication-ready descriptive statistics table for NICU cohort.

Output:
- CSV table with all statistics
- Formatted Excel version
- Individual section DataFrames
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy import stats

# Add src to path
sys.path.append('.')
from src.data.loader import load_nicu_data, get_category_labels, CAT_LABELS_EN

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("paper/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def test_normality(data):
    """Test if data is normally distributed using Shapiro-Wilk test."""
    if len(data.dropna()) < 3:
        return False
    try:
        _, p_value = stats.shapiro(data.dropna())
        return p_value > 0.05
    except:
        return False

def format_continuous(data, name=""):
    """Format continuous variable as mean±SD (range) or median (IQR) [range]."""
    clean = data.dropna()
    if len(clean) == 0:
        return "No data"
    
    mean_val = clean.mean()
    sd_val = clean.std()
    median_val = clean.median()
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    min_val = clean.min()
    max_val = clean.max()
    
    is_normal = test_normality(clean)
    
    if is_normal:
        return f"{mean_val:.1f} ± {sd_val:.1f} ({min_val:.0f}–{max_val:.0f})"
    else:
        return f"{median_val:.1f} ({q1:.1f}–{q3:.1f}) [{min_val:.0f}–{max_val:.0f}]"

def format_categorical(data, labels=None, total_n=None):
    """Format categorical variable as n (%)."""
    if total_n is None:
        total_n = len(data)
    
    counts = data.value_counts().sort_index()
    
    results = []
    for code, count in counts.items():
        pct = (count / total_n) * 100
        label = labels.get(code, str(code)) if labels else str(code)
        results.append({
            'Category': label,
            'n': count,
            '%': pct,
            'Formatted': f"{count} ({pct:.1f}%)"
        })
    
    return pd.DataFrame(results)

def format_binary(data, labels=None, positive_label="Present"):
    """Format binary variable as n (%) for positive cases."""
    total = len(data.dropna())
    if total == 0:
        return "No data"
    
    positive = (data == 1).sum()
    pct = (positive / total) * 100
    
    return f"{positive} ({pct:.1f}%)"

# ============================================================================
# SECTION GENERATORS
# ============================================================================

def generate_maternal_section(df):
    """Generate Section A: Maternal Characteristics."""
    print("\n=== Maternal Characteristics ===")
    
    rows = []
    total_n = len(df)
    
    # Age (continuous)
    rows.append({
        'Variable': 'Maternal age, years',
        'Value': format_continuous(df['anneyasi']),
        'n': len(df['anneyasi'].dropna())
    })
    
    # Age groups
    age_groups = format_categorical(df['anne_yaşı_grup'], CAT_LABELS_EN['anne_yaşı_grup'], total_n)
    for _, row in age_groups.iterrows():
        rows.append({
            'Variable': f"  {row['Category']}",
            'Value': row['Formatted'],
            'n': row['n']
        })
    
    # Education
    rows.append({
        'Variable': 'Maternal education',
        'Value': '',
        'n': ''
    })
    edu = format_categorical(df['anneegitim'], CAT_LABELS_EN['anneegitim'], total_n)
    for _, row in edu.iterrows():
        rows.append({
            'Variable': f"  {row['Category']}",
            'Value': row['Formatted'],
            'n': row['n']
        })
    
    # Occupation
    rows.append({
        'Variable': 'Maternal occupation',
        'Value': '',
        'n': ''
    })
    occ = format_categorical(df['annemeslegi'], CAT_LABELS_EN['annemeslegi'], total_n)
    for _, row in occ.iterrows():
        rows.append({
            'Variable': f"  {row['Category']}",
            'Value': row['Formatted'],
            'n': row['n']
        })
    
    # Delivery method
    rows.append({
        'Variable': 'Delivery method',
        'Value': '',
        'n': ''
    })
    delivery = format_categorical(df['dogumsekli'], CAT_LABELS_EN['dogumsekli'], total_n)
    for _, row in delivery.iterrows():
        rows.append({
            'Variable': f"  {row['Category']}",
            'Value': row['Formatted'],
            'n': row['n']
        })
    
    # Parity
    rows.append({
        'Variable': 'Living children, n',
        'Value': format_continuous(df['yasayancocuksayisi']),
        'n': len(df['yasayancocuksayisi'].dropna())
    })
    
    # Previous breastfeeding
    rows.append({
        'Variable': 'Previously breastfed children, n',
        'Value': format_continuous(df['emzirdigicocuksayisi']),
        'n': len(df['emzirdigicocuksayisi'].dropna())
    })
    
    return pd.DataFrame(rows)

def generate_infant_section(df):
    """Generate Section B: Infant Characteristics."""
    print("\n=== Infant Characteristics ===")
    
    rows = []
    total_n = len(df)
    
    # Birth weight
    rows.append({
        'Variable': 'Birth weight, grams',
        'Value': format_continuous(df['dogumagirligi(gram)']),
        'n': len(df['dogumagirligi(gram)'].dropna())
    })
    
    # Birth weight categories
    bw_cat = format_categorical(df['dogum_agırlıgı_gruplu'], CAT_LABELS_EN['dogum_agırlıgı_gruplu'], total_n)
    for _, row in bw_cat.iterrows():
        rows.append({
            'Variable': f"  {row['Category']}",
            'Value': row['Formatted'],
            'n': row['n']
        })
    
    # Gestational age
    rows.append({
        'Variable': 'Gestational age, weeks',
        'Value': format_continuous(df['gebelikhaftası']),
        'n': len(df['gebelikhaftası'].dropna())
    })
    
    # GA categories
    ga_cat = format_categorical(df['VAR00004'], CAT_LABELS_EN['VAR00004'], total_n)
    for _, row in ga_cat.iterrows():
        rows.append({
            'Variable': f"  {row['Category']}",
            'Value': row['Formatted'],
            'n': row['n']
        })
    
    # Sex
    rows.append({
        'Variable': 'Sex',
        'Value': '',
        'n': ''
    })
    sex = format_categorical(df['cinsiyeti'], CAT_LABELS_EN['cinsiyeti'], total_n)
    for _, row in sex.iterrows():
        rows.append({
            'Variable': f"  {row['Category']}",
            'Value': row['Formatted'],
            'n': row['n']
        })
    
    # Pregnancy type
    rows.append({
        'Variable': 'Pregnancy type',
        'Value': '',
        'n': ''
    })
    preg = format_categorical(df['gebeliktipi'], CAT_LABELS_EN['gebeliktipi'], total_n)
    for _, row in preg.iterrows():
        rows.append({
            'Variable': f"  {row['Category']}",
            'Value': row['Formatted'],
            'n': row['n']
        })
    
    # All diagnoses (expanded per reviewer request)
    rows.append({
        'Variable': 'Primary diagnosis',
        'Value': '',
        'n': ''
    })
    
    diag_counts = df['tanı_gruplu'].value_counts()
    diag_labels = CAT_LABELS_EN['tanı_gruplu']
    for code, count in diag_counts.items():
        pct = (count / total_n) * 100
        label = diag_labels.get(code, f"Code {code}")
        rows.append({
            'Variable': f"  {label}",
            'Value': f"{count} ({pct:.1f}%)",
            'n': count
        })
    
    return pd.DataFrame(rows)

def generate_clinical_section(df):
    """Generate Section C: Clinical Course."""
    print("\n=== Clinical Course ===")
    
    rows = []
    
    # Length of stay
    rows.append({
        'Variable': 'Length of stay, days',
        'Value': format_continuous(df['takiptekacgun']),
        'n': len(df['takiptekacgun'].dropna())
    })
    
    # Day 1 breastfeeding — split into direct and expressed (per reviewer)
    rows.append({
        'Variable': 'Day 1 direct breastfeeding initiation',
        'Value': format_binary(df['ilk_gün_emzirme_111']),
        'n': len(df['ilk_gün_emzirme_111'].dropna())
    })
    
    rows.append({
        'Variable': 'Day 1 breast milk received (any route)†',
        'Value': format_binary(df['ilk_gün_anne_sütü_1111']),
        'n': len(df['ilk_gün_anne_sütü_1111'].dropna())
    })
    
    # Colostrum present
    rows.append({
        'Variable': 'Colostrum present',
        'Value': format_binary(df['Kolostrumvarligi']),
        'n': len(df['Kolostrumvarligi'].dropna())
    })
    
    # Initial feeding route — show ALL categories (per reviewer)
    rows.append({
        'Variable': 'Initial feeding route on Day 1',
        'Value': '',
        'n': ''
    })
    
    route_labels = CAT_LABELS_EN['beslenmeninilkgunuverilisyolu']
    route_data = format_categorical(df['beslenmeninilkgunuverilisyolu'], route_labels)
    
    for idx, row in route_data.iterrows():
        rows.append({
            'Variable': f"  {row['Category']}",
            'Value': row['Formatted'],
            'n': row['n']
        })
    
    return pd.DataFrame(rows)

def generate_interventions_section(df):
    """Generate Section D: Interventions & Support."""
    print("\n=== Interventions & Support ===")
    
    rows = []
    
    # Breast pump used
    pump_used = (df['Kullandıgıpompamarkasi'] != 0) & (df['Kullandıgıpompamarkasi'].notna())
    total = len(df['Kullandıgıpompamarkasi'].dropna())
    count = pump_used.sum()
    pct = (count / total) * 100 if total > 0 else 0
    
    rows.append({
        'Variable': 'Breast pump used',
        'Value': f"{count} ({pct:.1f}%)",
        'n': count
    })
    
    # Lactation support
    rows.append({
        'Variable': 'Lactation support received',
        'Value': format_binary(df['baslangictasutdestegi']),
        'n': len(df['baslangictasutdestegi'].dropna())
    })
    
    # Breastfeeding education
    rows.append({
        'Variable': 'Breastfeeding education received',
        'Value': format_binary(df['annesutuemzirmeeğitimidurumu']),
        'n': len(df['annesutuemzirmeeğitimidurumu'].dropna())
    })
    
    # Galactagogue use
    rows.append({
        'Variable': 'Galactagogue use',
        'Value': format_binary(df['galaktokogkullanımı']),
        'n': len(df['galaktokogkullanımı'].dropna())
    })
    
    # Breast problems
    rows.append({
        'Variable': 'Breast problems experienced',
        'Value': format_binary(df['memesorunuyaşamadurumu']),
        'n': len(df['memesorunuyaşamadurumu'].dropna())
    })
    
    return pd.DataFrame(rows)

def generate_context_section(df):
    """Generate Section E: Study Context."""
    print("\n=== Study Context ===")
    
    rows = []
    total_n = len(df)
    
    # Study epoch
    rows.append({
        'Variable': 'Study epoch',
        'Value': '',
        'n': ''
    })
    
    epoch = format_categorical(df['ikisiarası'], CAT_LABELS_EN['ikisiarası'], total_n)
    for _, row in epoch.iterrows():
        rows.append({
            'Variable': f"  {row['Category']}",
            'Value': row['Formatted'],
            'n': row['n']
        })
    
    return pd.DataFrame(rows)

def generate_outcome_section(df):
    """Generate Section F: Primary Outcome."""
    print("\n=== Primary Outcome ===")
    
    rows = []
    total_n = len(df)
    
    rows.append({
        'Variable': 'Feeding type at discharge',
        'Value': '',
        'n': ''
    })
    
    outcome = format_categorical(df['taburculuk_beslenmeturu'], CAT_LABELS_EN['taburculuk_beslenmeturu'], total_n)
    for _, row in outcome.iterrows():
        rows.append({
            'Variable': f"  {row['Category']}",
            'Value': row['Formatted'],
            'n': row['n']
        })
    
    return pd.DataFrame(rows)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Generate complete Table 1."""
    print("=" * 70)
    print("GENERATING TABLE 1: SAMPLE CHARACTERISTICS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df = load_nicu_data(clean=False)
    print(f"Total sample: n = {len(df)}")
    
    # Generate all sections
    sections = {
        'A_Maternal': generate_maternal_section(df),
        'B_Infant': generate_infant_section(df),
        'C_Clinical': generate_clinical_section(df),
        'D_Interventions': generate_interventions_section(df),
        'E_Context': generate_context_section(df),
        'F_Outcome': generate_outcome_section(df)
    }
    
    # Combine all sections
    print("\n" + "=" * 70)
    print("COMBINING ALL SECTIONS")
    print("=" * 70)
    
    all_rows = []
    
    # Add header row
    all_rows.append({
        'Variable': f'Table 1. Baseline Characteristics of Study Population (n={len(df)})',
        'Value': '',
        'n': ''
    })
    all_rows.append({'Variable': '', 'Value': '', 'n': ''})  # Blank line
    
    section_names = {
        'A_Maternal': 'Maternal Characteristics',
        'B_Infant': 'Infant Characteristics',
        'C_Clinical': 'Clinical Course',
        'D_Interventions': 'Interventions & Support',
        'E_Context': 'Study Context',
        'F_Outcome': 'Primary Outcome'
    }
    
    for section_key, section_df in sections.items():
        # Add section header
        all_rows.append({
            'Variable': f'**{section_names[section_key]}**',
            'Value': '',
            'n': ''
        })
        
        # Add section rows
        for _, row in section_df.iterrows():
            all_rows.append(row.to_dict())
        
        # Add blank line between sections
        all_rows.append({'Variable': '', 'Value': '', 'n': ''})
    
    # Create final DataFrame
    table1 = pd.DataFrame(all_rows)
    
    # Add footnote
    footnote = ("Data are presented as mean ± SD (range) for normally distributed continuous variables, "
                "median (IQR) [range] for non-normally distributed continuous variables, or n (%) for "
                "categorical variables. †Includes both direct breastfeeding and expressed breast milk "
                "delivered via oral (PO) or orogastric (OG) routes. 'None' in initial feeding route "
                "indicates NPO (nil per os) infants who were too clinically unstable to receive enteral "
                "feeding on Day 1. EBF, exclusive breastfeeding; BF, direct breastfeeding; BFHI, "
                "Baby-Friendly Hospital Initiative; ELBW, extremely low birth weight; VLBW, very low "
                "birth weight; LBW, low birth weight; NPO, nil per os; PO, oral; OG, orogastric.")
    
    table1 = pd.concat([
        table1,
        pd.DataFrame([
            {'Variable': '', 'Value': '', 'n': ''},
            {'Variable': footnote, 'Value': '', 'n': ''}
        ])
    ], ignore_index=True)
    
    # Save outputs
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)
    
    # CSV version
    csv_path = OUTPUT_DIR / "table1_sample_characteristics.csv"
    table1.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    
    # Excel version (formatted)
    excel_path = OUTPUT_DIR / "table1_sample_characteristics.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        table1.to_excel(writer, sheet_name='Table 1', index=False)
        
        # Also save individual sections
        for section_key, section_df in sections.items():
            section_df.to_excel(writer, sheet_name=section_names[section_key][:31], index=False)
    
    print(f"✓ Saved Excel: {excel_path}")
    
    # Print preview
    print("\n" + "=" * 70)
    print("TABLE 1 PREVIEW (First 30 rows)")
    print("=" * 70)
    print(table1.head(30).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✓ TABLE 1 GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("  - table1_sample_characteristics.csv")
    print("  - table1_sample_characteristics.xlsx (with individual section tabs)")

if __name__ == "__main__":
    main()
