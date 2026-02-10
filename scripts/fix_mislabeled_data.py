"""
Data cleaning script to fix mislabeled data issues identified in review.

This script addresses three issues:
1. ✓ gebelik_tipi_gruplu: Fixed in loader.py (encoding 0,1 instead of 1,2)
2. anne_hastalık_grup: Recode value 761 → 11 (Other)
3. ilk_gün_anne_sütü_1111: Recode value 172 → 1 (Present)
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src/data')
from loader import CAT_LABELS_EN, load_nicu_data

def fix_mislabeled_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Fix identified mislabeled data issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Print details of fixes
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe
    """
    df_cleaned = df.copy()
    fixes_applied = []
    
    # Issue #2: Fix invalid category code in anne_hastalık_grup
    if 'anne_hastalık_grup' in df_cleaned.columns:
        mask = df_cleaned['anne_hastalık_grup'] == 761
        if mask.any():
            n_fixed = mask.sum()
            df_cleaned.loc[mask, 'anne_hastalık_grup'] = 11  # Recode to "Other"
            fixes_applied.append(f"  • anne_hastalık_grup: Recoded 761 → 11 (n={n_fixed})")
    
    # Issue #3: Fix numeric value in binary variable
    if 'ilk_gün_anne_sütü_1111' in df_cleaned.columns:
        mask = df_cleaned['ilk_gün_anne_sütü_1111'] == 172
        if mask.any():
            n_fixed = mask.sum()
            df_cleaned.loc[mask, 'ilk_gün_anne_sütü_1111'] = 1  # Recode to "Present"
            fixes_applied.append(f"  • ilk_gün_anne_sütü_1111: Recoded 172 → 1 (n={n_fixed})")
    
    if verbose:
        if fixes_applied:
            print("✓ Data cleaning completed:")
            for fix in fixes_applied:
                print(fix)
        else:
            print("✓ No mislabeled data found (may have been previously fixed)")
    
    return df_cleaned


def verify_all_encodings(df: pd.DataFrame) -> dict:
    """
    Verify that all categorical variables have complete label definitions.
    
    Returns
    -------
    dict
        Results with 'all_valid' flag and list of any issues
    """
    issues = []
    
    for var, labels in CAT_LABELS_EN.items():
        if var not in df.columns:
            continue
            
        # Get unique values in data
        unique_vals = df[var].dropna().unique()
        unique_vals_cleaned = []
        
        for x in unique_vals:
            if pd.notnull(x):
                try:
                    unique_vals_cleaned.append(int(float(x)))
                except:
                    unique_vals_cleaned.append(x)
        
        unique_vals_set = set(unique_vals_cleaned)
        defined_labels_set = set(labels.keys())
        
        # Check for missing definitions
        missing = unique_vals_set - defined_labels_set
        if missing:
            issues.append({
                'variable': var,
                'type': 'missing_labels',
                'values': sorted(missing),
                'n_affected': sum(df[var].isin(missing))
            })
    
    return {
        'all_valid': len(issues) == 0,
        'issues': issues,
        'n_variables_checked': len([v for v in CAT_LABELS_EN.keys() if v in df.columns])
    }


def main():
    print("="*70)
    print("DATA CLEANING AND ENCODING VERIFICATION")
    print("="*70)
    print()
    
    # Load data
    print("Loading data...")
    df_raw = pd.read_excel('excels-NICU-breatsfeeding-data/nicu_stage0_5_cleaned.xlsx')
    print(f"  Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")
    print()
    
    # Verify encodings BEFORE cleaning
    print("STEP 1: Verify encodings before cleaning")
    print("-" * 70)
    results_before = verify_all_encodings(df_raw)
    
    if results_before['all_valid']:
        print("✓ All encodings valid!")
    else:
        print(f"⚠️  Found {len(results_before['issues'])} issue(s):")
        for issue in results_before['issues']:
            print(f"  • {issue['variable']}: Missing labels for {issue['values']} (n={issue['n_affected']})")
    print()
    
    # Clean data
    print("STEP 2: Apply data cleaning fixes")
    print("-" * 70)
    df_cleaned = fix_mislabeled_data(df_raw, verbose=True)
    print()
    
    # Verify encodings AFTER cleaning
    print("STEP 3: Verify encodings after cleaning")
    print("-" * 70)
    results_after = verify_all_encodings(df_cleaned)
    
    if results_after['all_valid']:
        print(f"✅ SUCCESS! All {results_after['n_variables_checked']} categorical variables have complete label definitions")
    else:
        print(f"❌ FAILED: Still have {len(results_after['issues'])} issue(s):")
        for issue in results_after['issues']:
            print(f"  • {issue['variable']}: Missing labels for {issue['values']} (n={issue['n_affected']})")
    print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    issues_fixed = len(results_before['issues']) - len(results_after['issues'])
    print(f"Issues before cleaning: {len(results_before['issues'])}")
    print(f"Issues after cleaning:  {len(results_after['issues'])}")
    print(f"Issues fixed:           {issues_fixed}")
    print()
    
    if results_after['all_valid']:
        print("✅ READY FOR PUBLICATION")
        print("   All categorical encodings verified and complete.")
    else:
        print("⚠️  ADDITIONAL REVIEW REQUIRED")
        print(f"   {len(results_after['issues'])} encoding issue(s) remain.")
    print()
    
    # Save cleaned data
    output_path = 'excels-NICU-breatsfeeding-data/nicu_stage0_5_cleaned_fixed.xlsx'
    df_cleaned.to_excel(output_path, index=False)
    print(f"💾 Cleaned data saved to: {output_path}")
    
    return df_cleaned, results_after


if __name__ == "__main__":
    df_cleaned, results = main()
