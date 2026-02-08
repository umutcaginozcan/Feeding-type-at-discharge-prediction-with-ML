#!/usr/bin/env python3
"""
Statistical Analysis: Association between Exclusive Breastfeeding and Time Epochs

Research Question: 
Is there an association between feeding type at discharge and different time periods
(pre/post COVID-19, pre/post Baby-Friendly Hospital Initiative certification)?

Variables:
- Outcome: taburculuk_beslenmeturu (0=Exclusive BF, 1=Formula, 2=Mixed)
- Predictors:
  * ikisiarası (epoch: 0=pre-COVID+pre-BFHI, 1=pre-COVID+post-BFHI, 2=post-COVID)
  * covid19sonrasi (0=pre-COVID, 1=post-COVID)
  * bebek_dostu_20temmuz2018 (0=pre-BFHI, 1=post-BFHI)

Statistical Methods:
1. Chi-square test of independence (tests if two categorical variables are independent)
2. Cramér's V (measures strength of association, 0-1 scale)
3. Post-hoc pairwise comparisons with Bonferroni correction
4. Visualizations (contingency tables, mosaic plots, proportions)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
import os
output_dir = 'outputs/statistics/Epochs x EBF'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("STATISTICAL ANALYSIS: BREASTFEEDING OUTCOMES × TIME EPOCHS")
print("=" * 80)
print()

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("STEP 1: Loading data...")
print("-" * 80)

df = pd.read_excel('excels-NICU-breatsfeeding-data/nicu_stage0_5_cleaned.xlsx')
print(f"✓ Loaded {len(df)} patients")
print()

# Define outcome and predictor variables
outcome = 'taburculuk_beslenmeturu'
predictors = {
    'ikisiarası': 'Epoch (COVID × BFHI)',
    'covid19sonrasi': 'COVID-19 Period',
    'bebek_dostu_20temmuz2018': 'Baby-Friendly Hospital'
}

# Clean data: remove missing values
df_clean = df[[outcome] + list(predictors.keys())].dropna()
print(f"✓ After removing missing values: {len(df_clean)} patients")
print(f"  (Dropped {len(df) - len(df_clean)} patients with missing data)")
print()

# ============================================================================
# STEP 2: DESCRIPTIVE STATISTICS
# ============================================================================
print("STEP 2: Descriptive Statistics")
print("-" * 80)
print()

# Overall feeding type distribution
feeding_labels = {0: 'Exclusive BF', 1: 'Formula', 2: 'Mixed'}
print("Overall Feeding Type Distribution:")
print(df_clean[outcome].value_counts().sort_index().rename(feeding_labels))
print()

# ============================================================================
# STATISTICAL TEST EXPLANATION
# ============================================================================
print("=" * 80)
print("📊 UNDERSTANDING THE STATISTICAL TESTS")
print("=" * 80)
print()
print("1. CHI-SQUARE TEST (χ²)")
print("   Purpose: Tests if two categorical variables are independent")
print("   ")
print("   How it works:")
print("   • Compares observed frequencies vs. expected frequencies")
print("   • Expected = what we'd see if there was NO association")
print("   • Large differences → significant association")
print()
print("   Hypotheses:")
print("   • H₀ (null): Feeding type is INDEPENDENT of time period")
print("   • H₁ (alternative): Feeding type is ASSOCIATED with time period")
print()
print("   Interpretation:")
print("   • p < 0.05 → Reject H₀ → There IS an association ✓")
print("   • p ≥ 0.05 → Fail to reject H₀ → No significant association")
print()
print("2. CRAMÉR'S V (Effect Size)")
print("   Purpose: Measures HOW STRONG the association is")
print("   ")
print("   Scale: 0 to 1")
print("   • 0.00-0.10: Negligible association")
print("   • 0.10-0.30: Weak association")
print("   • 0.30-0.50: Moderate association")
print("   • 0.50+:     Strong association")
print()
print("3. POST-HOC PAIRWISE COMPARISONS")
print("   Purpose: If overall test is significant, find WHICH groups differ")
print("   • Bonferroni correction: Adjusts p-values for multiple comparisons")
print("   • Prevents false positives when doing many tests")
print()
print("=" * 80)
print()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cramers_v(contingency_table):
    """
    Calculate Cramér's V statistic for contingency table
    
    Cramér's V = sqrt(χ² / (n × (min(r,c) - 1)))
    where:
    - χ² = chi-square statistic
    - n = sample size
    - r = number of rows
    - c = number of columns
    """
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, c = contingency_table.shape
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))

def interpret_cramers_v(v):
    """Interpret Cramér's V effect size"""
    if v < 0.1:
        return "Negligible"
    elif v < 0.3:
        return "Weak"
    elif v < 0.5:
        return "Moderate"
    else:
        return "Strong"

def chi_square_test(data, outcome_var, predictor_var, predictor_name):
    """
    Perform chi-square test and display results
    """
    print(f"\n{'=' * 80}")
    print(f"ANALYSIS: {predictor_name}")
    print(f"{'=' * 80}\n")
    
    # Create contingency table
    contingency = pd.crosstab(data[outcome_var], data[predictor_var])
    
    # Define labels
    feeding_labels = {0: 'Exclusive BF', 1: 'Formula', 2: 'Mixed'}
    contingency.index = [feeding_labels.get(i, i) for i in contingency.index]
    
    print("CONTINGENCY TABLE (Observed Frequencies):")
    print(contingency)
    print()
    
    # Add row and column totals
    contingency_with_totals = contingency.copy()
    contingency_with_totals['Total'] = contingency_with_totals.sum(axis=1)
    contingency_with_totals.loc['Total'] = contingency_with_totals.sum(axis=0)
    
    print("CONTINGENCY TABLE WITH TOTALS:")
    print(contingency_with_totals)
    print()
    
    # Calculate proportions
    proportions = contingency.div(contingency.sum(axis=1), axis=0) * 100
    print("PROPORTIONS (% within each feeding type):")
    print(proportions.round(1))
    print()
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # Calculate Cramér's V
    v = cramers_v(contingency)
    v_interpretation = interpret_cramers_v(v)
    
    # Display results
    print("STATISTICAL TEST RESULTS:")
    print(f"  χ² statistic: {chi2:.4f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Cramér's V: {v:.4f} ({v_interpretation} association)")
    print()
    
    # Interpretation
    print("INTERPRETATION:")
    if p_value < 0.001:
        print(f"  ✓ HIGHLY SIGNIFICANT association (p < 0.001)")
    elif p_value < 0.01:
        print(f"  ✓ VERY SIGNIFICANT association (p < 0.01)")
    elif p_value < 0.05:
        print(f"  ✓ SIGNIFICANT association (p < 0.05)")
    else:
        print(f"  ✗ NO significant association (p ≥ 0.05)")
    
    print(f"  The strength of association is {v_interpretation.lower()} (V = {v:.3f})")
    print()
    
    # Expected frequencies
    expected_df = pd.DataFrame(expected, 
                               index=contingency.index, 
                               columns=contingency.columns)
    print("EXPECTED FREQUENCIES (if there was NO association):")
    print(expected_df.round(1))
    print()
    
    # Check assumptions
    print("ASSUMPTIONS CHECK:")
    min_expected = expected.min()
    pct_below_5 = (expected < 5).sum() / expected.size * 100
    print(f"  Minimum expected frequency: {min_expected:.2f}")
    print(f"  Cells with expected < 5: {pct_below_5:.1f}%")
    if min_expected >= 5 and pct_below_5 < 20:
        print("  ✓ Chi-square test is appropriate (all assumptions met)")
    else:
        print("  ⚠ Warning: Some expected frequencies < 5. Consider Fisher's exact test.")
    print()
    
    # Save results
    results = {
        'predictor': predictor_name,
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': v,
        'interpretation': v_interpretation,
        'significant': 'Yes' if p_value < 0.05 else 'No'
    }
    
    # Save contingency table
    filename = f"{predictor_var}_contingency_table.csv"
    contingency_with_totals.to_csv(f"{output_dir}/{filename}")
    print(f"  ✓ Saved: {filename}")
    
    return results, contingency, proportions

# ============================================================================
# STEP 3: RUN TESTS FOR EACH PREDICTOR
# ============================================================================
print("\nSTEP 3: Statistical Tests")
print("-" * 80)

all_results = []

for pred_var, pred_name in predictors.items():
    results, contingency, proportions = chi_square_test(df_clean, outcome, pred_var, pred_name)
    all_results.append(results)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Contingency table heatmap
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title(f'Observed Frequencies\n{pred_name}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Period')
    axes[0].set_ylabel('Feeding Type at Discharge')
    
    # Plot 2: Proportions stacked bar chart
    proportions.T.plot(kind='bar', stacked=True, ax=axes[1], 
                       color=['#2ecc71', '#e74c3c', '#3498db'])
    axes[1].set_title(f'Feeding Distribution by Period\n{pred_name}', 
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time Period')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].legend(title='Feeding Type', bbox_to_anchor=(1.05, 1))
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    # Plot 3: Side-by-side bar chart
    proportions.plot(kind='bar', ax=axes[2], color=['#2ecc71', '#e74c3c', '#3498db'])
    axes[2].set_title(f'Feeding Type Proportions\n{pred_name}', 
                      fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Feeding Type at Discharge')
    axes[2].set_ylabel('Percentage (%)')
    axes[2].legend(title='Time Period', bbox_to_anchor=(1.05, 1))
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{pred_var}_analysis.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {pred_var}_analysis.png")
    plt.close()

# ============================================================================
# STEP 4: SUMMARY TABLE
# ============================================================================
print(f"\n{'=' * 80}")
print("SUMMARY OF ALL TESTS")
print(f"{'=' * 80}\n")

summary_df = pd.DataFrame(all_results)
summary_df = summary_df[['predictor', 'chi2', 'p_value', 'cramers_v', 
                         'interpretation', 'significant']]

print(summary_df.to_string(index=False))
print()

# Save summary
summary_df.to_csv(f"{output_dir}/summary_all_tests.csv", index=False)
print(f"✓ Saved: summary_all_tests.csv")
print()

# ============================================================================
# STEP 5: POST-HOC ANALYSIS (if significant)
# ============================================================================
print(f"\n{'=' * 80}")
print("STEP 4: Post-Hoc Pairwise Comparisons")
print(f"{'=' * 80}\n")
print("(Only for multi-level predictors with significant overall test)")
print()

# For ikisiarası (3 levels), do pairwise comparisons if significant
if summary_df[summary_df['predictor'] == 'Epoch (COVID × BFHI)']['significant'].values[0] == 'Yes':
    print("PAIRWISE COMPARISONS: Epoch (COVID × BFHI)")
    print("-" * 80)
    
    epochs = df_clean['ikisiarası'].unique()
    epoch_labels = {0: 'Pre-COVID + Pre-BFHI', 
                   1: 'Pre-COVID + Post-BFHI', 
                   2: 'Post-COVID'}
    
    pairwise_results = []
    
    for i in range(len(epochs)):
        for j in range(i+1, len(epochs)):
            epoch_i, epoch_j = epochs[i], epochs[j]
            
            # Subset data
            subset = df_clean[df_clean['ikisiarası'].isin([epoch_i, epoch_j])]
            contingency = pd.crosstab(subset[outcome], subset['ikisiarası'])
            
            # Chi-square test
            chi2, p_value, dof, _ = chi2_contingency(contingency)
            
            # Bonferroni correction (3 pairwise comparisons)
            p_adjusted = min(p_value * 3, 1.0)
            
            pairwise_results.append({
                'Comparison': f"{epoch_labels[epoch_i]} vs {epoch_labels[epoch_j]}",
                'χ²': f"{chi2:.4f}",
                'p-value': f"{p_value:.6f}",
                'p-adjusted': f"{p_adjusted:.6f}",
                'Significant': 'Yes' if p_adjusted < 0.05 else 'No'
            })
    
    pairwise_df = pd.DataFrame(pairwise_results)
    print(pairwise_df.to_string(index=False))
    print()
    
    pairwise_df.to_csv(f"{output_dir}/pairwise_comparisons.csv", index=False)
    print(f"✓ Saved: pairwise_comparisons.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n{'=' * 80}")
print("🎯 KEY FINDINGS")
print(f"{'=' * 80}\n")

for _, row in summary_df.iterrows():
    print(f"{row['predictor']}:")
    if row['significant'] == 'Yes':
        print(f"  ✓ Significant association found (p = {row['p_value']:.6f})")
        print(f"  • Effect size: {row['interpretation']} (Cramér's V = {row['cramers_v']:.3f})")
    else:
        print(f"  ✗ No significant association (p = {row['p_value']:.3f})")
    print()

print(f"{'=' * 80}")
print(f"✓ Analysis complete! All outputs saved to: {output_dir}/")
print(f"{'=' * 80}\n")
