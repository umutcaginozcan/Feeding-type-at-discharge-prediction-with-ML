# Analysis Guide

## Creating a New Analysis

### 1. Copy the Template

```bash
cp -r analyses/template analyses/XX_your_analysis_name
cd analyses/XX_your_analysis_name
```

### 2. Edit the Configuration

Edit `config.yaml` to specify your variables and parameters.

### 3. Modify the Analysis Script

Update `analyze.py` to use your specified variables and add any custom analysis steps.

### 4. Run the Analysis

```bash
python analyze.py
```

## Using the `src` Package

The `src` package contains reusable modules for systematic analysis.

### Data Loading

```python
from src.data import load_nicu_data, get_variable_label, get_category_labels

# Load specific variables
df = load_nicu_data(
    clean=True,  # Remove missing values
    variables=['taburculuk_beslenmeturu', 'covid19sonrasi']
)

# Get human-readable labels
label = get_variable_label('taburculuk_beslenmeturu')  # "Feeding Type at Discharge"
categories = get_category_labels('covid19sonrasi')  # {0: 'Pre-COVID', 1: 'Post-COVID'}
```

### Statistical Tests

```python
from src.statistics import chi_square_test, pairwise_comparisons

# Chi-square test
results = chi_square_test(
    df,
    outcome_var='taburculuk_beslenmeturu',
    predictor_var='covid19sonrasi',
    outcome_labels={0: 'Exclusive BF', 1: 'Formula', 2: 'Mixed'},
    predictor_labels={0: 'Pre-COVID', 1: 'Post-COVID'},
    output_dir='outputs/my_analysis',
    alpha=0.05
)

# Access results
print(f"p-value: {results['p_value']}")
print(f"Cramér's V: {results['cramers_v']}")
print(f"Significant: {results['significant']}")

# Post-hoc pairwise comparisons
pairwise = pairwise_comparisons(
    df,
    outcome_var='taburculuk_beslenmeturu',
    predictor_var='ikisiarası',
    predictor_labels={0: 'Epoch 0', 1: 'Epoch 1', 2: 'Epoch 2'},
    correction='bonferroni'
)
```

### Visualization

```python
from src.visualization import create_analysis_figure

# Create comprehensive 3-panel figure
fig = create_analysis_figure(
    contingency=results['contingency'],
    proportions=results['proportions'],
    predictor_name='COVID-19 Period',
    output_path='outputs/my_analysis/figure.png'
)
```

## Analysis Output Structure

Each analysis should save outputs to:

```
outputs/statistics/Your_Analysis_Name/
├── {predictor}_contingency.csv    # Contingency table
├── {predictor}_proportions.csv    # Proportions table
├── {predictor}_summary.csv        # Test summary
├── {predictor}_analysis.png       # Visualization
├── pairwise_comparisons.csv       # Post-hoc tests (if applicable)
└── summary_all_tests.csv          # Overall summary
```

## Best Practices

1. **Self-contained analyses**: Each analysis folder should be independent
2. **Configuration files**: Use config files to track parameters
3. **Documentation**: Always include a README.md describing the analysis
4. **Reuse modules**: Use `src` package functions instead of duplicating code
5. **Version control**: Commit analysis scripts and configurations (not large output files)

## Example Analyses

- `01_epochs_ebf/` - Association between feeding outcomes and time epochs
- `template/` - Template for new analyses

## Adding New Statistical Methods

To add a new statistical method:

1. Add functions to appropriate module in `src/statistics/`
2. Export in `src/statistics/__init__.py`
3. Use in your analysis scripts
4. Document in `docs/API_REFERENCE.md`

## Generating Paper-Ready Outputs

For publication-ready tables and figures:

```python
# Save tables in LaTeX format
results['contingency'].to_latex('paper/tables/table1.tex')

# Save high-resolution figures
fig.savefig('paper/figures/figure1.pdf', dpi=600, format='pdf')
fig.savefig('paper/figures/figure1.png', dpi=300, format='png')
```

## Troubleshooting

**Import errors:**
- Ensure you're running from project root
- Check that `src` package is in Python path

**Missing data:**
- Use `check_missing_data()` to identify missing values
- Set `clean=True` in `load_nicu_data()` to remove missing values

**Visualization issues:**
- Check that output directory exists
- Verify category labels match your data

## Questions?

See `docs/API_REFERENCE.md` for detailed function documentation.
