"""
Analysis Template

Copy this file and modify for your specific analysis.
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

# ============================================================================
# CONFIGURATION - Modify this section for your analysis
# ============================================================================

CONFIG = {
    # Outcome variable
    'outcome_var': 'taburculuk_beslenmeturu',
    
    # Predictor variables and their names
    'predictors': {
        'your_predictor_var': 'Your Predictor Name',
    },
    
    # Output directory
    'output_dir': 'outputs/statistics/Your_Analysis_Name',
    
    # Statistical parameters
    'alpha': 0.05,
}

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run the analysis."""
    print("=" * 80)
    print("YOUR ANALYSIS TITLE")
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
    
    # Step 3: Run analyses
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
    
    # Step 4: Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}\n")
    
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    print()
    
    summary_path = Path(CONFIG['output_dir']) / 'summary_all_tests.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved: {summary_path}\n")
    
    print(f"{'=' * 80}")
    print(f"✓ Analysis complete! All outputs saved to: {CONFIG['output_dir']}/")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
