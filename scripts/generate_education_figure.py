#!/usr/bin/env python3
"""
Publication-Quality Analysis: Breastfeeding Education × COVID-19 Period
Nature Journal Format

This script creates a publication-ready figure following Nature journal guidelines:
- 600 DPI resolution
- Colorblind-friendly palettes
- Professional fonts (Arial 8-10pt)
- Statistical annotations with p-values
- Hardcoded heatmap annotations for visibility
- Handles only non-empty categories

Author: NICU Breastfeeding Research Team
Date: February 16, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('.')
from src.data.loader import load_nicu_data, CAT_LABELS_EN

# ============================================================================
# NATURE JOURNAL STYLE CONFIGURATION
# ============================================================================

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})

# For significance markers
SIG_COLOR = '#CC78BC'    # Purple => POST-COVID
NEUTRAL_COLOR = '#4D4D4D'  # Dark Gray => PRE-COVID

# Education status colors
EDUCATION_COLORS = {
    'Present': '#029E73',   # Green
    'Absent': '#DE8F05',    # Orange
}

# Output directories
OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("paper/figures/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_DPI = 600
FIGURE_FORMAT = ['png', 'pdf']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cramers_v(contingency_table):
    """Calculate Cramér's V effect size"""
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, c = contingency_table.shape
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))

def format_p_value(p):
    """Format p-value for display"""
    if p < 0.001:
        return 'p < 0.001'
    elif p < 0.01:
        return f'p = {p:.3f}'
    else:
        return f'p = {p:.2f}'

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("=" * 70)
print("GENERATING BREASTFEEDING EDUCATION STATUS FIGURE")
print("Nature Journal Format")
print("=" * 70)

print("\nLoading data...")
df = load_nicu_data(clean=False)

# Map education status
education_labels = CAT_LABELS_EN['annesutuemzirmeeğitimidurumu']
df['Education Status'] = df['annesutuemzirmeeğitimidurumu'].map(education_labels)

# Map time period
df['Time Period'] = df['covid19sonrasi'].map({0: 'Pre-COVID-19', 1: 'Post-COVID-19'})

# Map feeding outcome
feeding_labels = CAT_LABELS_EN['taburculuk_beslenmeturu']
df['Feeding Type'] = df['taburculuk_beslenmeturu'].map(feeding_labels)

# Clean data
df_clean = df[['Education Status', 'Time Period', 'Feeding Type']].dropna()

print(f"✓ Loaded {len(df_clean)} patients")
print(f"\nEducation Status distribution:")
print(df_clean['Education Status'].value_counts())
print(f"\nTime Period distribution:")
print(df_clean['Time Period'].value_counts())

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

print("\nPerforming statistical analysis...")

# Create contingency table: Education × Time Period
ct_education_period = pd.crosstab(df_clean['Education Status'],
                                   df_clean['Time Period'],
                                   margins=True)

print("\n=== Education Status × Time Period ===")
print(ct_education_period)

# Chi-square test
ct_no_margins = ct_education_period.iloc[:-1, :-1]
chi2, p_value, dof, expected = chi2_contingency(ct_no_margins)

# Cramér's V (effect size)
v = cramers_v(ct_no_margins)

print(f"\nChi-square test: χ²({dof}) = {chi2:.3f}, {format_p_value(p_value)}")
print(f"Cramér's V = {v:.3f}")

# Calculate percentages for each time period
pct_by_period = pd.crosstab(df_clean['Education Status'],
                             df_clean['Time Period'],
                             normalize='columns') * 100

print("\n=== Education Status by Time Period (%) ===")
print(pct_by_period)

# Calculate percentages for each education status
pct_by_education = pd.crosstab(df_clean['Time Period'],
                                df_clean['Education Status'],
                                normalize='columns') * 100

print("\n=== Time Period by Education Status (%) ===")
print(pct_by_education)

# ============================================================================
# CREATE 3-PANEL FIGURE (NATURE STYLE)
# ============================================================================

print("\nCreating 3-panel figure (Nature style)...")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('Figure. Breastfeeding Education Status by COVID-19 Period',
             fontweight='bold', y=1.02)

# ============================================================================
# PANEL A: OBSERVED FREQUENCIES (HEATMAP)
# ============================================================================

ax = axes[0]

# Prepare heatmap data - rows: Education Status, columns: Time Period
# Order: Pre-COVID-19 (left), Post-COVID-19 (right)
heatmap_data = ct_no_margins[['Pre-COVID-19', 'Post-COVID-19']]

# Draw heatmap WITHOUT annotations first
sns.heatmap(heatmap_data, annot=False, cmap='YlOrRd',
            ax=ax, cbar_kws={'label': 'Count'}, linewidths=0.5,
            linecolor='white', square=False, vmin=0)

# Manually add text annotations to ensure visibility
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        value = heatmap_data.iloc[i, j]
        ax.text(j + 0.5, i + 0.5, str(int(value)),
               ha='center', va='center',
               fontsize=11, fontweight='bold',
               color='white' if value > heatmap_data.max().max() * 0.6 else 'black')

ax.set_title('A. Observed Frequencies', fontweight='bold', pad=10)
ax.set_xlabel('Time Period')
ax.set_ylabel('Education Status')

# ============================================================================
# PANEL B: EDUCATION DISTRIBUTION BY TIME PERIOD (STACKED %)
# ============================================================================

ax = axes[1]

# Order: Pre-COVID-19 (left), Post-COVID-19 (right)
periods = ['Pre-COVID-19', 'Post-COVID-19']
x = np.arange(len(periods))
width = 0.6
bottom = np.zeros(len(periods))

for status in ['Present', 'Absent']:
    values = [pct_by_period.loc[status, p] for p in periods]
    ax.bar(x, values, width, bottom=bottom, label=status,
           color=EDUCATION_COLORS[status], edgecolor='white', linewidth=0.8)

    # Add percentage labels
    for i, val in enumerate(values):
        if val > 5:
            ax.text(x[i], bottom[i] + val/2, f'{val:.1f}%',
                   ha='center', va='center', fontsize=8, color='white',
                   fontweight='bold')

    bottom += values

ax.set_title('B. Education Distribution (%)', fontweight='bold', pad=10)
ax.set_xlabel('Time Period')
ax.set_ylabel('Percentage (%)')
ax.set_xticks(x)
ax.set_xticklabels(periods, rotation=0)
ax.set_ylim(0, 100)
ax.legend(title='Education Status', frameon=True, fancybox=False, edgecolor='black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ============================================================================
# PANEL C: TIME PERIOD DISTRIBUTION BY EDUCATION STATUS
# ============================================================================

ax = axes[2]

education_statuses = ['Absent', 'Present']
x = np.arange(len(education_statuses))
n_periods = 2
bar_width = 0.8 / n_periods

# Pre-COVID on LEFT, Post-COVID on RIGHT
for i, (period, color) in enumerate([('Pre-COVID-19', NEUTRAL_COLOR),
                                       ('Post-COVID-19', SIG_COLOR)]):
    offset = (i - n_periods/2 + 0.5) * bar_width
    values = [pct_by_education.loc[period, status] for status in education_statuses]
    bars = ax.bar(x + offset, values, bar_width,
                 label=period, color=color,
                 edgecolor='black', linewidth=0.8)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

ax.set_title(f'C. Time Period Distribution\n{format_p_value(p_value)}, Cramér\'s V = {v:.3f}',
             fontweight='bold', pad=10, fontsize=10)
ax.set_xlabel('Education Status')
ax.set_ylabel('Percentage (%)')
ax.set_xticks(x)
ax.set_xticklabels(education_statuses, rotation=0)
ax.legend(title='Time Period', frameon=True, fancybox=False, edgecolor='black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ============================================================================
# SAVE FIGURE
# ============================================================================

plt.tight_layout()

for fmt in FIGURE_FORMAT:
    filepath = OUTPUT_DIR / f"figure_education_status.{fmt}"
    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

plt.close()

# ============================================================================
# EXPORT STATISTICS TO EXCEL
# ============================================================================

print("\nAppending to all_statistics.xlsx...")

excel_path = DATA_DIR / 'all_statistics.xlsx'

# Prepare tables
desc_data = []
for period in ['Pre-COVID-19', 'Post-COVID-19']:
    for status in ['Absent', 'Present']:
        count = ct_education_period.loc[status, period]
        pct = pct_by_period.loc[status, period]
        desc_data.append({
            'Time Period': period,
            'Education Status': status,
            'Count': count,
            'Percentage': f'{pct:.1f}%'
        })

desc_df = pd.DataFrame(desc_data)

test_results = pd.DataFrame([
    {
        'Test': 'Chi-square',
        'Statistic': f'χ²({dof}) = {chi2:.3f}',
        'p-value': format_p_value(p_value),
        'Interpretation': 'Significant' if p_value < 0.05 else 'Not significant'
    },
    {
        'Test': 'Effect size (Cramér\'s V)',
        'Statistic': f'{v:.3f}',
        'p-value': 'N/A',
        'Interpretation': 'Negligible' if v < 0.1 else 'Small' if v < 0.3 else 'Medium' if v < 0.5 else 'Large'
    }
])

# Write to Excel
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    desc_df.to_excel(writer, sheet_name='Education_Descriptive', index=False)
    test_results.to_excel(writer, sheet_name='Education_Statistical_Tests', index=False)
    ct_education_period.to_excel(writer, sheet_name='Education_Crosstab')

print(f"  ✓ Appended to: {excel_path}")

# Save standalone CSV
desc_df.to_csv(DATA_DIR / 'education_descriptive_table.csv', index=False)
test_results.to_csv(DATA_DIR / 'education_test_results.csv', index=False)

print(f"  ✓ Saved: {DATA_DIR / 'education_descriptive_table.csv'}")
print(f"  ✓ Saved: {DATA_DIR / 'education_test_results.csv'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("✓ BREASTFEEDING EDUCATION ANALYSIS COMPLETE")
print("=" * 70)

print("\nKey Findings:")
print(f"  Pre-COVID-19:")
print(f"    - Education Present: {pct_by_period.loc['Present', 'Pre-COVID-19']:.1f}%")
print(f"    - Education Absent: {pct_by_period.loc['Absent', 'Pre-COVID-19']:.1f}%")
print(f"  Post-COVID-19:")
print(f"    - Education Present: {pct_by_period.loc['Present', 'Post-COVID-19']:.1f}%")
print(f"    - Education Absent: {pct_by_period.loc['Absent', 'Post-COVID-19']:.1f}%")

print(f"\nStatistical Summary:")
print(f"  - Chi-square: χ²({dof}) = {chi2:.3f}, {format_p_value(p_value)}")
print(f"  - Effect size: Cramér's V = {v:.3f}")

if p_value < 0.05:
    print("  - Conclusion: Significant association between time period and education status")
else:
    print("  - Conclusion: No significant association")

print("\nFigure specifications:")
print("  - Resolution: 600 DPI (publication-ready)")
print("  - Color scheme: Colorblind-friendly Nature palette")
print("  - Font: Arial 8-10pt")
print("  - Statistical annotations included")

print("\n" + "=" * 70)
