"""
Analysis: Association between Breastfeeding Outcomes and Time Epochs

Refactored version using src package modules for reusability.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data import load_nicu_data, get_variable_label, get_category_labels
from src.statistics import chi_square_test, pairwise_comparisons, print_chi_square_results
from src.visualization import create_analysis_figure

import pandas as pd

# Configuration
CONFIG = {
    'outcome_var': 'taburculuk_beslenmeturu',
    'predictors': {
        'ikisiarası': 'Epoch (COVID × BFHI)',
        'covid19sonrasi': 'COVID-19 Period',
        'bebek_dostu_20temmuz2018': 'Baby-Friendly Hospital Initiative',
    },
    'output_dir': 'outputs/statistics/Epochs_x_EBF',
    'alpha': 0.05,
}

def main():
    """Run the analysis."""
    print("=" * 80)
    print("ANALYSIS: Breastfeeding Outcomes × Time Epochs")
    print("=" * 80)
    print()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    variables = [CONFIG['outcome_var']] + list(CONFIG['predictors'].keys())
    df = load_nicu_data(clean=True, variables=variables)
    print(f"✓ Loaded {len(df)} patients\n")
    
    # Step 2: Descriptive statistics
    print("Step 2: Descriptive Statistics")
    print("-" * 80)
    outcome_labels = get_category_labels(CONFIG['outcome_var'])
    print(f"\n{get_variable_label(CONFIG['outcome_var'])} Distribution:")
    print(df[CONFIG['outcome_var']].value_counts().sort_index().rename(outcome_labels))
    print()
    
    # Step 3: Run analyses for each predictor
    print("\nStep 3: Statistical Tests")
    print("-" * 80)
    
    all_results = []
    
    for pred_var, pred_name in CONFIG['predictors'].items():
        # Get labels
        outcome_labels = get_category_labels(CONFIG['outcome_var'])
        predictor_labels = get_category_labels(pred_var)
        
        # Run chi-square test
        results = chi_square_test(
            df,
            outcome_var=CONFIG['outcome_var'],
            predictor_var=pred_var,
            outcome_labels=outcome_labels,
            predictor_labels=predictor_labels,
            output_dir=CONFIG['output_dir'],
            alpha=CONFIG['alpha']
        )
        
        # Print results
        print_chi_square_results(results, pred_name)
        
        # Create visualization
        output_path = Path(CONFIG['output_dir']) / f"{pred_var}_analysis.png"
        create_analysis_figure(
            results['contingency'],
            results['proportions'],
            pred_name,
            output_path=output_path
        )
        
        # Store summary
        all_results.append({
            'predictor': pred_name,
            'variable': pred_var,
            'chi2': results['chi2'],
            'p_value': results['p_value'],
            'cramers_v': results['cramers_v'],
            'interpretation': results['interpretation'],
            'significant': 'Yes' if results['significant'] else 'No'
        })
    
    # Step 4: Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'=' * 80}\n")
    
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    print()
    
    # Save summary
    summary_path = Path(CONFIG['output_dir']) / 'summary_all_tests.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved: {summary_path}\n")
    
    # Step 5: Post-hoc analysis for multi-level predictors
    print(f"\n{'=' * 80}")
    print("Step 4: Post-Hoc Pairwise Comparisons")
    print(f"{'=' * 80}\n")
    
    # For ikisiarası (3 levels)
    if summary_df[summary_df['variable'] == 'ikisiarası']['significant'].values[0] == 'Yes':
        print("Pairwise Comparisons: Epoch (COVID × BFHI)")
        print("-" * 80)
        
        predictor_labels = get_category_labels('ikisiarası')
        pairwise_results = pairwise_comparisons(
            df,
            outcome_var=CONFIG['outcome_var'],
            predictor_var='ikisiarası',
            predictor_labels=predictor_labels,
            correction='bonferroni',
            alpha=CONFIG['alpha']
        )
        
        print(pairwise_results.to_string(index=False))
        print()
        
        # Save
        pairwise_path = Path(CONFIG['output_dir']) / 'pairwise_comparisons.csv'
        pairwise_results.to_csv(pairwise_path, index=False)
        print(f"✓ Saved: {pairwise_path}\n")
    
    # Final summary
    print(f"\n{'=' * 80}")
    print("🎯 KEY FINDINGS")
    print(f"{'=' * 80}\n")
    
    for _, row in summary_df.iterrows():
        print(f"{row['predictor']}:")
        if row['significant'] == 'Yes':
            print(f"  ✓ Significant association (p = {row['p_value']:.6f})")
            print(f"  • Effect size: {row['interpretation']} (Cramér's V = {row['cramers_v']:.3f})")
        else:
            print(f"  ✗ No significant association (p = {row['p_value']:.3f})")
        print()
    
    print(f"{'=' * 80}")
    print(f"✓ Analysis complete! All outputs saved to: {CONFIG['output_dir']}/")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
