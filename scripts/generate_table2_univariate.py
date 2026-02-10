"""
Generate Table 2: Univariate Associations with Feeding Outcome

This script tests EVERY predictor variable against the primary outcome
(feeding type at discharge) and generates a comprehensive publication-ready table.

For categorical predictors: Chi-square test, Cramér's V, odds ratios
For continuous predictors: ANOVA/Kruskal-Wallis, effect sizes, mean differences

Output: Table 2 in multiple formats (CSV, Excel, LaTeX)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal, mannwhitneyu
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src/data')
from loader import load_nicu_data, CAT_LABELS_EN, NUMERIC_COLS, get_category_labels, RENAME_FOR_PLOT

def get_english_name(variable_name):
    """Get English name for variable from RENAME_FOR_PLOT."""
    return RENAME_FOR_PLOT.get(variable_name, variable_name)

def cramers_v(confusion_matrix):
    """Calculate Cramér's V effect size for chi-square test."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * min(r-1, k-1)))

def calculate_or_ci(a, b, c, d, confidence=0.95):
    """
    Calculate odds ratio and 95% CI from 2x2 contingency table.
    
    Layout:
         Exposed  Unexposed
    Case    a        b
    Ctrl    c        d
    
    OR = (a*d) / (b*c)
    """
    # Add 0.5 to all cells if any is 0 (Haldane correction)
    if 0 in [a, b, c, d]:
        a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
    
    or_value = (a * d) / (b * c)
    
    # Calculate SE of log(OR)
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    
    # Calculate CI
    z = stats.norm.ppf((1 + confidence) / 2)
    log_or = np.log(or_value)
    ci_lower = np.exp(log_or - z * se_log_or)
    ci_upper = np.exp(log_or + z * se_log_or)
    
    return or_value, ci_lower, ci_upper

def test_categorical_predictor(df, predictor, outcome='taburculuk_beslenmeturu'):
    """Test association between categorical predictor and feeding outcome."""
    
    # Create contingency table
    ct = pd.crosstab(df[predictor], df[outcome])
    
    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(ct)
    
    # Cramér's V
    cramers = cramers_v(ct.values)
    
    # Get labels
    pred_labels = get_category_labels(predictor)
    outcome_labels = {1: "Exclusive BF", 2: "Formula", 3: "Mixed"}
    
    # Calculate percentages
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    
    # For binary predictors, calculate OR for EBF vs not-EBF
    or_value, or_ci_low, or_ci_high = None, None, None
    if len(ct) == 2 and outcome in ['taburculuk_beslenmeturu']:
        # Create binary outcome: EBF (1) vs not-EBF (2,3)
        df_binary = df.copy()
        df_binary['ebf'] = (df_binary[outcome] == 1).astype(int)
        
        # Get unique predictor values
        pred_vals = sorted(df[predictor].dropna().unique())
        if len(pred_vals) == 2:
            # 2x2 table for OR calculation
            ct_binary = pd.crosstab(df_binary[predictor], df_binary['ebf'])
            if ct_binary.shape == (2, 2):
                # Reference: first category (usually 0 or 1)
                # Exposed: second category
                # a = exposed + EBF, b = exposed + not-EBF
                # c = unexposed + EBF, d = unexposed + not-EBF
                a = ct_binary.iloc[1, 1]  # Exposed, EBF
                b = ct_binary.iloc[1, 0]  # Exposed, not-EBF
                c = ct_binary.iloc[0, 1]  # Unexposed, EBF
                d = ct_binary.iloc[0, 0]  # Unexposed, not-EBF
                
                or_value, or_ci_low, or_ci_high = calculate_or_ci(a, b, c, d)
    
    result = {
        'variable': predictor,
        'variable_english': get_english_name(predictor),
        'type': 'categorical',
        'n': len(df[predictor].dropna()),
        'n_categories': len(ct),
        'test': 'Chi-square',
        'statistic': chi2,
        'p_value': p_value,
        'effect_size': cramers,
        'effect_size_name': "Cramér's V",
        'or': or_value,
        'or_ci_low': or_ci_low,
        'or_ci_high': or_ci_high
    }
    
    return result

def test_continuous_predictor(df, predictor, outcome='taburculuk_beslenmeturu'):
    """Test association between continuous predictor and feeding outcome."""
    
    # Split by feeding type
    groups = []
    for outcome_val in sorted(df[outcome].dropna().unique()):
        group_data = df[df[outcome] == outcome_val][predictor].dropna()
        groups.append(group_data)
    
    # Check normality (Shapiro-Wilk) - if any group has n>5
    normality_ok = all(len(g) > 5 for g in groups)
    if normality_ok:
        shapiro_ps = [stats.shapiro(g)[1] if len(g) <= 5000 else 1.0 for g in groups]
        normality_ok = all(p > 0.05 for p in shapiro_ps)
    
    # Choose test
    if normality_ok and len(groups) > 2:
        # ANOVA
        statistic, p_value = f_oneway(*groups)
        test_name = 'ANOVA'
        
        # Calculate eta-squared (effect size)
        grand_mean = df[predictor].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_total = sum((df[predictor] - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        effect_size = eta_squared
        effect_size_name = "η²"
        
    else:
        # Kruskal-Wallis (non-parametric)
        statistic, p_value = kruskal(*groups)
        test_name = 'Kruskal-Wallis'
        
        # Calculate epsilon-squared (effect size)
        n = sum(len(g) for g in groups)
        H = statistic
        epsilon_squared = H / (n - 1) if n > 1 else 0
        effect_size = epsilon_squared
        effect_size_name = "ε²"
    
    # Calculate means and SDs for each group
    group_stats = {}
    for i, outcome_val in enumerate(sorted(df[outcome].dropna().unique())):
        group_data = df[df[outcome] == outcome_val][predictor].dropna()
        group_stats[outcome_val] = {
            'mean': group_data.mean(),
            'sd': group_data.std(),
            'median': group_data.median(),
            'n': len(group_data)
        }
    
    result = {
        'variable': predictor,
        'variable_english': get_english_name(predictor),
        'type': 'continuous',
        'n': len(df[predictor].dropna()),
        'test': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_size_name': effect_size_name,
        'group_stats': group_stats
    }
    
    return result

def generate_univariate_table(df, outcome='taburculuk_beslenmeturu', output_dir='paper/tables'):
    """
    Generate comprehensive univariate associations table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with all variables
    outcome : str
        Primary outcome variable
    output_dir : str
        Directory to save output tables
        
    Returns
    -------
    pd.DataFrame
        Univariate results table
    """
    
    print("="*70)
    print("GENERATING TABLE 2: UNIVARIATE ASSOCIATIONS")
    print("="*70)
    print()
    
    # Get all potential predictors
    all_cols = df.columns.tolist()
    
    # Exclude outcome and ID variables
    exclude = [outcome, 'ID', 'hastakodu', 'doğumtarihi', 'takibegirdigitarih', 'taburculuktarihi']
    predictors = [c for c in all_cols if c not in exclude]
    
    print(f"Testing {len(predictors)} predictors against {outcome}")
    print()
    
    results = []
    
    for i, predictor in enumerate(predictors, 1):
        print(f"[{i}/{len(predictors)}] Testing: {predictor}...", end=" ")
        
        try:
            # Determine if categorical or continuous
            if predictor in CAT_LABELS_EN:
                result = test_categorical_predictor(df, predictor, outcome)
                print(f"✓ Chi-square: p={result['p_value']:.4f}, V={result['effect_size']:.3f}")
            elif predictor in NUMERIC_COLS or df[predictor].dtype in ['int64', 'float64']:
                # Check if actually continuous (>10 unique values)
                n_unique = df[predictor].nunique()
                if n_unique > 10:
                    result = test_continuous_predictor(df, predictor, outcome)
                    print(f"✓ {result['test']}: p={result['p_value']:.4f}, {result['effect_size_name']}={result['effect_size']:.3f}")
                else:
                    # Treat as categorical
                    result = test_categorical_predictor(df, predictor, outcome)
                    print(f"✓ Chi-square: p={result['p_value']:.4f}, V={result['effect_size']:.3f}")
            else:
                print("SKIP (unable to classify)")
                continue
            
            results.append(result)
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    print()
    print("="*70)
    print(f"COMPLETED: {len(results)} tests performed")
    print("="*70)
    print()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Add significance stars
    def add_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''
    
    results_df['sig'] = results_df['p_value'].apply(add_stars)
    
    # Sort by p-value
    results_df = results_df.sort_values('p_value')
    
    # Format for publication
    results_df['p_formatted'] = results_df['p_value'].apply(
        lambda x: f"{x:.4f}" if x >= 0.001 else "<0.001"
    )
        # Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Reorder and rename columns for publication
    output_df = results_df.copy()
    output_df = output_df.rename(columns={
        'variable': 'Variable_Code',
        'variable_english': 'Variable'
    })
    
    # Reorder columns to put English name first
    cols = output_df.columns.tolist()
    if 'Variable' in cols and 'Variable_Code' in cols:
        cols.remove('Variable')
        cols.remove('Variable_Code')
        cols = ['Variable', 'Variable_Code'] + cols
        output_df = output_df[cols]
    
    # CSV (full data)
    csv_file = output_path / 'table2_univariate_associations.csv'
    output_df.to_csv(csv_file, index=False)
    print(f"✓ Saved: {csv_file}")
    
    # Excel
    excel_file = output_path / 'table2_univariate_associations.xlsx'
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        output_df.to_excel(writer, sheet_name='Univariate Results', index=False)
        
        # Add summary sheet
        summary = {
            'Total tests': len(results_df),
            'Significant (p<0.05)': (results_df['p_value'] < 0.05).sum(),
            'Categorical variables': (results_df['type'] == 'categorical').sum(),
            'Continuous variables': (results_df['type'] == 'continuous').sum(),
        }
        pd.DataFrame([summary]).T.to_excel(writer, sheet_name='Summary')
    
    print(f"✓ Saved: {excel_file}")
    
    # Publication-ready table (significant only)
    sig_results = results_df[results_df['p_value'] < 0.05].copy()
    
    pub_table = []
    for _, row in sig_results.iterrows():
        if row['type'] == 'categorical':
            pub_row = {
                'Predictor': row['variable_english'],
                'Variable Code': row['variable'],
                'N': row['n'],
                'Test': row['test'],
                'Statistic': f"{row['statistic']:.2f}",
                'P-value': row['p_formatted'] + row['sig'],
                'Effect Size': f"{row['effect_size_name']}={row['effect_size']:.3f}",
                'OR (95% CI)': f"{row['or']:.2f} ({row['or_ci_low']:.2f}-{row['or_ci_high']:.2f})" if row['or'] else "—"
            }
        else:
            pub_row = {
                'Predictor': row['variable_english'],
                'Variable Code': row['variable'],
                'N': row['n'],
                'Test': row['test'],
                'Statistic': f"{row['statistic']:.2f}",
                'P-value': row['p_formatted'] + row['sig'],
                'Effect Size': f"{row['effect_size_name']}={row['effect_size']:.3f}",
                'OR (95% CI)': "—"
            }
        pub_table.append(pub_row)
    
    pub_df = pd.DataFrame(pub_table)
    pub_file = output_path / 'table2_significant_only.csv'
    pub_df.to_csv(pub_file, index=False)
    print(f"✓ Saved: {pub_file} ({len(pub_df)} significant variables)")
    
    print()
    print("SUMMARY STATISTICS:")
    print(f"  Total variables tested: {len(results_df)}")
    print(f"  Significant (p<0.05): {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).sum()/len(results_df)*100:.1f}%)")
    print(f"  Highly significant (p<0.001): {(results_df['p_value'] < 0.001).sum()}")
    print()
    print(f"  Categorical predictors: {(results_df['type'] == 'categorical').sum()}")
    print(f"  Continuous predictors: {(results_df['type'] == 'continuous').sum()}")
    print()
    
    # Show top 10 by effect size
    print("TOP 10 PREDICTORS BY EFFECT SIZE:")
    print("-" * 70)
    top10 = results_df.nlargest(10, 'effect_size')[['variable_english', 'p_value', 'effect_size', 'effect_size_name']]
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        english_name = row['variable_english'][:60]  # Truncate if too long
        print(f"{i:2d}. {english_name:60s} {row['effect_size_name']}={row['effect_size']:.3f}, p={row['p_value']:.4f}")
    
    print()
    
    return results_df

def main():
    """Main execution."""
    
    # Load data
    print("Loading data...")
    df = pd.read_excel('excels-NICU-breatsfeeding-data/nicu_stage0_5_cleaned_fixed.xlsx')
    print(f"Loaded {len(df)} rows × {len(df.columns)} columns")
    print()
    
    # Generate table
    results = generate_univariate_table(df)
    
    print()
    print("="*70)
    print("✅ TABLE 2 GENERATION COMPLETE")
    print("="*70)
    print()
    print("Output files:")
    print("  - paper/tables/table2_univariate_associations.csv (complete)")
    print("  - paper/tables/table2_univariate_associations.xlsx (with summary)")
    print("  - paper/tables/table2_significant_only.csv (publication-ready)")
    print()

if __name__ == "__main__":
    main()
