"""
Statistical tests for categorical data.

Includes chi-square test, Fisher's exact test, effect sizes, and post-hoc comparisons.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from pathlib import Path


def cramers_v(contingency_table):
    """
    Calculate Cramér's V effect size for contingency table.
    
    Cramér's V measures the strength of association between two categorical variables.
    Range: 0 (no association) to 1 (perfect association)
    
    Parameters
    ----------
    contingency_table : pd.DataFrame or np.ndarray
        Contingency table
    
    Returns
    -------
    float
        Cramér's V statistic
    
    References
    ----------
    Cramér, H. (1946). Mathematical Methods of Statistics. Princeton University Press.
    """
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum() if isinstance(contingency_table, pd.DataFrame) else contingency_table.sum()
    
    if isinstance(contingency_table, pd.DataFrame):
        r, c = contingency_table.shape
    else:
        r, c = contingency_table.shape
    
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))


def interpret_cramers_v(v):
    """
    Interpret Cramér's V effect size.
    
    Parameters
    ----------
    v : float
        Cramér's V value
    
    Returns
    -------
    str
        Interpretation (Negligible, Weak, Moderate, or Strong)
    """
    if v < 0.1:
        return "Negligible"
    elif v < 0.3:
        return "Weak"
    elif v < 0.5:
        return "Moderate"
    else:
        return "Strong"


def chi_square_test(data, outcome_var, predictor_var, 
                    outcome_labels=None, predictor_labels=None,
                    output_dir=None, alpha=0.05):
    """
    Perform chi-square test of independence.
    
    Tests whether two categorical variables are independent.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data containing the variables
    outcome_var : str
        Name of outcome variable
    predictor_var : str
        Name of predictor variable
    outcome_labels : dict, optional
        Mapping from outcome codes to labels
    predictor_labels : dict, optional
        Mapping from predictor codes to labels
    output_dir : str or Path, optional
        Directory to save results
    alpha : float, default=0.05
        Significance level
    
    Returns
    -------
    dict
        Results including:
        - contingency: Contingency table
        - chi2: Chi-square statistic
        - p_value: P-value
        - dof: Degrees of freedom
        - cramers_v: Cramér's V effect size
        - interpretation: Effect size interpretation
        - significant: Boolean indicating significance
        - expected: Expected frequencies
        - proportions: Proportions table
    
    Examples
    --------
    >>> from src.data import load_nicu_data
    >>> df = load_nicu_data(variables=['taburculuk_beslenmeturu', 'ikisiarası'])
    >>> results = chi_square_test(df, 'taburculuk_beslenmeturu', 'ikisiarası')
    >>> print(f"p-value: {results['p_value']:.4f}")
    """
    # Create contingency table
    contingency = pd.crosstab(data[outcome_var], data[predictor_var])
    
    # Apply labels if provided
    if outcome_labels:
        contingency.index = [outcome_labels.get(i, i) for i in contingency.index]
    if predictor_labels:
        contingency.columns = [predictor_labels.get(i, i) for i in contingency.columns]
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # Calculate effect size
    v = cramers_v(contingency)
    v_interpretation = interpret_cramers_v(v)
    
    # Calculate proportions
    proportions = contingency.div(contingency.sum(axis=1), axis=0) * 100
    
    # Create expected frequencies DataFrame
    expected_df = pd.DataFrame(expected, 
                               index=contingency.index, 
                               columns=contingency.columns)
    
    # Prepare results
    results = {
        'contingency': contingency,
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': v,
        'interpretation': v_interpretation,
        'significant': p_value < alpha,
        'expected': expected_df,
        'proportions': proportions,
        'min_expected': expected.min(),
        'pct_below_5': (expected < 5).sum() / expected.size * 100
    }
    
    # Save results if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save contingency table
        contingency_with_totals = contingency.copy()
        contingency_with_totals['Total'] = contingency_with_totals.sum(axis=1)
        contingency_with_totals.loc['Total'] = contingency_with_totals.sum(axis=0)
        contingency_with_totals.to_csv(output_dir / f'{predictor_var}_contingency.csv')
        
        # Save proportions
        proportions.to_csv(output_dir / f'{predictor_var}_proportions.csv')
        
        # Save summary
        summary = pd.DataFrame([{
            'predictor': predictor_var,
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'cramers_v': v,
            'interpretation': v_interpretation,
            'significant': 'Yes' if p_value < alpha else 'No'
        }])
        summary.to_csv(output_dir / f'{predictor_var}_summary.csv', index=False)
    
    return results


def pairwise_comparisons(data, outcome_var, predictor_var, 
                        predictor_levels=None, predictor_labels=None,
                        correction='bonferroni', alpha=0.05):
    """
    Perform pairwise chi-square tests with multiple comparison correction.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data
    outcome_var : str
        Outcome variable
    predictor_var : str
        Predictor variable (must have 3+ levels)
    predictor_levels : list, optional
        Specific levels to compare. If None, uses all unique values.
    predictor_labels : dict, optional
        Labels for predictor levels
    correction : str, default='bonferroni'
        Multiple comparison correction method ('bonferroni' or 'none')
    alpha : float, default=0.05
        Significance level
    
    Returns
    -------
    pd.DataFrame
        Pairwise comparison results
    """
    if predictor_levels is None:
        predictor_levels = sorted(data[predictor_var].unique())
    
    if len(predictor_levels) < 2:
        raise ValueError("Need at least 2 levels for pairwise comparisons")
    
    results = []
    n_comparisons = len(predictor_levels) * (len(predictor_levels) - 1) // 2
    
    for i in range(len(predictor_levels)):
        for j in range(i + 1, len(predictor_levels)):
            level_i, level_j = predictor_levels[i], predictor_levels[j]
            
            # Subset data
            subset = data[data[predictor_var].isin([level_i, level_j])]
            contingency = pd.crosstab(subset[outcome_var], subset[predictor_var])
            
            # Chi-square test
            chi2, p_value, dof, _ = chi2_contingency(contingency)
            
            # Bonferroni correction
            if correction == 'bonferroni':
                p_adjusted = min(p_value * n_comparisons, 1.0)
            else:
                p_adjusted = p_value
            
            # Labels
            label_i = predictor_labels.get(level_i, level_i) if predictor_labels else level_i
            label_j = predictor_labels.get(level_j, level_j) if predictor_labels else level_j
            
            results.append({
                'Comparison': f"{label_i} vs {label_j}",
                'Level 1': level_i,
                'Level 2': level_j,
                'χ²': chi2,
                'p-value': p_value,
                'p-adjusted': p_adjusted,
                'Significant': 'Yes' if p_adjusted < alpha else 'No'
            })
    
    return pd.DataFrame(results)


def print_chi_square_results(results, predictor_name='Predictor'):
    """
    Print formatted chi-square test results.
    
    Parameters
    ----------
    results : dict
        Results from chi_square_test()
    predictor_name : str
        Name of predictor for display  
    """
    print(f"\n{'='*80}")
    print(f"CHI-SQUARE TEST: {predictor_name}")
    print(f"{'='*80}\n")
    
    print("Contingency Table (Observed Frequencies):")
    print(results['contingency'])
    print()
    
    print("Proportions (% within each row):")
    print(results['proportions'].round(1))
    print()
    
    print("Statistical Results:")
    print(f"  χ² = {results['chi2']:.4f}")
    print(f"  df = {results['dof']}")
    print(f"  p-value = {results['p_value']:.6f}")
    print(f"  Cramér's V = {results['cramers_v']:.4f} ({results['interpretation']})")
    print()
    
    if results['significant']:
        print(f"  ✓ SIGNIFICANT association (p < 0.05)")
    else:
        print(f"  ✗ NOT significant (p ≥ 0.05)")
    print()
    
    print("Assumptions Check:")
    print(f"  Minimum expected frequency: {results['min_expected']:.2f}")
    print(f"  Cells with expected < 5: {results['pct_below_5']:.1f}%")
    if results['min_expected'] >= 5 and results['pct_below_5'] < 20:
        print("  ✓ All assumptions met")
    else:
        print("  ⚠ Warning: Some expected frequencies < 5")
    print()
