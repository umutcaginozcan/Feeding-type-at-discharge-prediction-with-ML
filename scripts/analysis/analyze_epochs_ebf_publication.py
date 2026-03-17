#!/usr/bin/env python3
"""
Publication-Quality Analysis: Exclusive Breastfeeding × Time Epochs
Nature Journal Format

This script creates publication-ready figures following Nature journal guidelines:
- 600 DPI resolution
- Colorblind-friendly palettes
- Professional fonts (Arial 8-10pt)
- Statistical annotations with p-values
- Comprehensive figure legends
- Handles only non-empty categories
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NATURE JOURNAL STYLE CONFIGURATION
# ============================================================================

# Set publication-quality style
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

# Nature colorblind-friendly palette
# Map by confirmed data values: 1=Exclusive BF, 2=Formula, 3=Mixed
NATURE_COLORS_MAP = {
    1: '#0173B2',   # Blue - Exclusive BF (category 1, n=747)
    2: '#DE8F05',   # Orange - Formula (category 2, n=280)
    3: '#029E73',   # Green - Mixed (category 3, n=37)
}

# For significance markers
SIG_COLOR = '#CC78BC'  # Purple
NEUTRAL_COLOR = '#4D4D4D'  # Dark Gray

# Output directory
import os
output_dir = 'outputs/statistics/Epochs x EBF'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("PUBLICATION-QUALITY ANALYSIS: BREASTFEEDING × TIME EPOCHS")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")
df = pd.read_excel('excels-NICU-breatsfeeding-data/nicu_stage0_5_cleaned.xlsx')
print(f"✓ Loaded {len(df)} patients\n")

# Define variables
outcome = 'taburculuk_beslenmeturu'
predictors = {
    'covid19sonrasi': 'COVID-19 Period',
    'bebek_dostu_20temmuz2018': 'Baby-Friendly Hospital Initiative',
    'ikisiarası': 'Epoch (COVID × BFHI)'
}

# Clean data
df_clean = df[[outcome] + list(predictors.keys())].dropna()
print(f"Working with {len(df_clean)} patients after removing missing values\n")

# Check actual feeding categories in data
actual_categories = sorted(df_clean[outcome].unique())
print(f"Feeding categories in data: {actual_categories}")
print(f"Value counts:\n{df_clean[outcome].value_counts().sort_index()}\n")

# Define labels matching CONFIRMED encoding from loader.py
# 1 = Exclusive BF (~700 patients)
# 2 = Formula (~200 patients)
# 3 = Mixed (~37 patients)
feeding_labels = {
    1: 'Exclusive BF',
    2: 'Formula', 
    3: 'Mixed'
}

predictor_labels = {
    'covid19sonrasi': {0: 'Pre-COVID-19', 1: 'Post-COVID-19'},
    'bebek_dostu_20temmuz2018': {0: 'Pre-BFHI', 1: 'Post-BFHI'},
    'ikisiarası': {0: 'Pre-COVID + Pre-BFHI', 1: 'Pre-COVID + Post-BFHI', 2: 'Post-COVID'}
}

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

def create_figure(pred_var, pred_name, pred_lab, figure_num, colormap_name):
    """
    Create publication-quality 3-panel figure
    
    Parameters
    ----------
    pred_var : str
        Predictor variable name
    pred_name : str
        Full name of predictor
    pred_lab : dict
        Labels for predictor categories
    figure_num : int
        Figure number
    colormap_name : str
        Name of colormap for heatmap ('YlOrRd', 'Blues', 'Greens')
    """
    print(f"Creating Figure {figure_num}: {pred_name}...")
    
    # Create contingency table
    contingency = pd.crosstab(df_clean[outcome], df_clean[pred_var])
    
    # Apply labels
    contingency.index = [feeding_labels[i] for i in contingency.index]
    contingency.columns = [pred_lab[i] for i in contingency.columns]
    
    # Calculate statistics
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    v = cramers_v(contingency)
    
    # Calculate proportions
    props = contingency.div(contingency.sum(axis=0), axis=1) * 100
    
    # Get colors for available feeding types (in order they appear)
    feeding_colors = [NATURE_COLORS_MAP[cat] for cat in sorted(df_clean[outcome].unique())]
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'Figure {figure_num}. Feeding Outcomes by {pred_name}', 
                 fontweight='bold', y=1.02)
    
    # Panel A: Observed frequencies heatmap
    ax = axes[0]
    # Draw heatmap WITHOUT annotations first
    sns.heatmap(contingency, annot=False, fmt='d', cmap=colormap_name, 
                ax=ax, cbar_kws={'label': 'Count'}, linewidths=0.5,
                linecolor='white', square=False, vmin=0)
    
    # Manually add text annotations to ensure visibility
    for i in range(len(contingency.index)):
        for j in range(len(contingency.columns)):
            value = contingency.iloc[i, j]
            ax.text(j + 0.5, i + 0.5, str(int(value)),
                   ha='center', va='center',
                   fontsize=11, fontweight='bold',
                   color='white' if value > contingency.max().max() * 0.6 else 'black')
    
    ax.set_title('A. Observed Frequencies', fontweight='bold', pad=10)
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Feeding Type at Discharge')
    
    # Panel B: Stacked proportions
    ax = axes[1]
    x = np.arange(len(contingency.columns))
    width = 0.6
    bottom = np.zeros(len(contingency.columns))
    
    for idx, (feeding_type, row) in enumerate(props.iterrows()):
        ax.bar(x, row, width, bottom=bottom, label=feeding_type, 
               color=feeding_colors[idx], edgecolor='white', linewidth=0.8)
        
        # Add percentage labels
        for i, val in enumerate(row):
            if val > 5:  # Only show if segment is large enough
                ax.text(x[i], bottom[i] + val/2, f'{val:.1f}%', 
                       ha='center', va='center', fontsize=8, color='white',
                       fontweight='bold')
        
        bottom += row
    
    ax.set_title('B. Feeding Distribution (%)', fontweight='bold', pad=10)
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Percentage (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(contingency.columns, rotation=0)
    ax.set_ylim(0, 100)
    ax.legend(title='Feeding Type', frameon=True, fancybox=False, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel C: Grouped bar chart by feeding type
    ax = axes[2]
    x = np.arange(len(props.index))
    n_periods = len(props.columns)
    width = 0.8 / n_periods
    
    bars_list = []
    for i, col in enumerate(props.columns):
        offset = (i - n_periods/2 + 0.5) * width
        bars = ax.bar(x + offset, props[col], width, 
                     label=col, color=SIG_COLOR if i == 1 else NEUTRAL_COLOR,
                     edgecolor='black', linewidth=0.8)
        bars_list.append(bars)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_title(f'C. Time Period Distribution\n{format_p_value(p_value)}, Cramér\'s V = {v:.3f}', 
                 fontweight='bold', pad=10, fontsize=10)
    ax.set_xlabel('Feeding Type at Discharge')
    ax.set_ylabel('Percentage (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(props.index, rotation=0)
    ax.legend(title='Time Period', frameon=True, fancybox=False, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Figure_{figure_num}.png", dpi=600, bbox_inches='tight')
    print(f"✓ Saved: Figure_{figure_num}.png")
    plt.close()
    
    return chi2, p_value, v

# ============================================================================
# FIGURE 1: COVID-19 PERIOD ANALYSIS
# ============================================================================

chi2_1, p_1, v_1 = create_figure(
    'covid19sonrasi',
    predictors['covid19sonrasi'],
    predictor_labels['covid19sonrasi'],
    1,
    'YlOrRd'
)

# ============================================================================
# FIGURE 2: BABY-FRIENDLY HOSPITAL INITIATIVE
# ============================================================================

chi2_2, p_2, v_2 = create_figure(
    'bebek_dostu_20temmuz2018',
    predictors['bebek_dostu_20temmuz2018'],
    predictor_labels['bebek_dostu_20temmuz2018'],
    2,
    'Blues'
)

# ============================================================================
# FIGURE 3: COMBINED EPOCHS ANALYSIS
# ============================================================================

print("Creating Figure 3: Combined Epochs Analysis...")

pred_var = 'ikisiarası'
pred_name = predictors[pred_var]
pred_lab = predictor_labels[pred_var]

# Create contingency table
contingency = pd.crosstab(df_clean[outcome], df_clean[pred_var])
contingency.index = [feeding_labels[i] for i in contingency.index]
contingency.columns = [pred_lab[i] for i in contingency.columns]

# Calculate statistics
chi2, p_value, dof, expected = chi2_contingency(contingency)
v = cramers_v(contingency)

# Calculate proportions
props = contingency.div(contingency.sum(axis=0), axis=1) * 100

# Pairwise comparisons
epochs = [0, 1, 2]
pairwise_results = []

for i in range(len(epochs)):
    for j in range(i+1, len(epochs)):
        subset = df_clean[df_clean[pred_var].isin([epochs[i], epochs[j]])]
        cont = pd.crosstab(subset[outcome], subset[pred_var])
        chi2_pair, p_pair, _, _ = chi2_contingency(cont)
        p_adjusted = min(p_pair * 3, 1.0)  # Bonferroni
        
        pairwise_results.append({
            'comparison': f'{pred_lab[epochs[i]]} vs\n{pred_lab[epochs[j]]}',
            'p': p_pair,
            'p_adj': p_adjusted,
            'sig': p_adjusted < 0.05
        })

# Get colors for available feeding types
feeding_colors = [NATURE_COLORS_MAP[cat] for cat in sorted(df_clean[outcome].unique())]

# Create figure
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

fig.suptitle('Figure 3. Feeding Outcomes by Time Epoch (COVID-19 × BFHI)', 
             fontweight='bold', y=0.98, fontsize=13)

# Panel A: Heatmap of observed frequencies
ax1 = fig.add_subplot(gs[0, 0])
# Draw heatmap WITHOUT annotations first
sns.heatmap(contingency, annot=False, fmt='d', cmap='Greens', 
            ax=ax1, cbar_kws={'label': 'Count'}, linewidths=0.5,
            linecolor='white', square=False, vmin=0)

# Manually add text annotations
for i in range(len(contingency.index)):
    for j in range(len(contingency.columns)):
        value = contingency.iloc[i, j]
        ax1.text(j + 0.5, i + 0.5, str(int(value)),
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white' if value > contingency.max().max() * 0.6 else 'black')

ax1.set_title('A. Observed Frequencies', fontweight='bold', pad=10)
ax1.set_xlabel('Time Epoch')
ax1.set_ylabel('Feeding Type at Discharge')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15, ha='right')

# Panel B: Stacked bar chart
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(len(contingency.columns))
width = 0.65
bottom = np.zeros(len(contingency.columns))

for idx, (feeding_type, row) in enumerate(props.iterrows()):
    ax2.bar(x, row, width, bottom=bottom, label=feeding_type, 
           color=feeding_colors[idx], edgecolor='white', linewidth=0.8)
    
    for i, val in enumerate(row):
        if val > 4:
            ax2.text(x[i], bottom[i] + val/2, f'{val:.1f}%', 
                   ha='center', va='center', fontsize=8, color='white',
                   fontweight='bold')
    
    bottom += row

ax2.set_title(f'B. Feeding Distribution by Epoch\n{format_p_value(p_value)}, Cramér\'s V = {v:.3f}', 
             fontweight='bold', pad=10, fontsize=10)
ax2.set_xlabel('Time Epoch')
ax2.set_ylabel('Percentage (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(contingency.columns, rotation=15, ha='right')
ax2.set_ylim(0, 100)
ax2.legend(title='Feeding Type', frameon=True, fancybox=False, edgecolor='black')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Panel C: Grouped bar chart across epochs
ax3 = fig.add_subplot(gs[1, 0])
x_pos = np.arange(len(contingency.columns))
width = 0.25
n_feeding_types = len(props.index)

for idx, feeding_type in enumerate(props.index):
    offset = (idx - n_feeding_types/2 + 0.5) * width
    values = props.loc[feeding_type]
    bars = ax3.bar(x_pos + offset, values, width,
                  label=feeding_type, color=feeding_colors[idx],
                  edgecolor='black', linewidth=0.8)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=7)

ax3.set_title('C. Feeding Type Comparison Across Epochs', fontweight='bold', pad=10)
ax3.set_xlabel('Time Epoch')
ax3.set_ylabel('Percentage (%)')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(contingency.columns, rotation=15, ha='right')
ax3.legend(title='Feeding Type', frameon=True, fancybox=False, edgecolor='black')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Panel D: Pairwise comparisons
ax4 = fig.add_subplot(gs[1, 1])
comparisons = [r['comparison'] for r in pairwise_results]
p_values = [-np.log10(r['p_adj']) for r in pairwise_results]
colors_sig = [SIG_COLOR if r['sig'] else NEUTRAL_COLOR for r in pairwise_results]

bars = ax4.barh(comparisons, p_values, color=colors_sig, edgecolor='black', linewidth=0.8)
ax4.axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=1, 
           label='Significance threshold (p=0.05)')
ax4.set_xlabel('-log₁₀(p-value, Bonferroni-corrected)', fontweight='bold')
ax4.set_title('D. Post-hoc Pairwise Comparisons', fontweight='bold', pad=10)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.legend(frameon=True, fancybox=False, edgecolor='black', loc='lower right')

# Add p-value labels
for idx, (bar, result) in enumerate(zip(bars, pairwise_results)):
    width_val = bar.get_width()
    label = f"p_adj = {result['p_adj']:.4f}" if result['p_adj'] >= 0.001 else "p_adj < 0.001"
    ax4.text(width_val + 0.1, idx, label, ha='left', va='center', fontsize=8)

plt.savefig(f"{output_dir}/Figure_3.png", dpi=600, bbox_inches='tight')
print(f"✓ Saved: Figure_3.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✓ All publication figures created successfully!")
print("=" * 80)
print(f"\nOutputs saved to: {output_dir}/")
print("  - Figure_1.png: COVID-19 Period Analysis")
print(f"    χ² = {chi2_1:.2f}, p {format_p_value(p_1)}, V = {v_1:.3f}")
print("  - Figure_2.png: Baby-Friendly Hospital Initiative")
print(f"    χ² = {chi2_2:.2f}, p = {format_p_value(p_2)}, V = {v_2:.3f}")
print("  - Figure_3.png: Combined Epochs Analysis")
print(f"    χ² = {chi2:.2f}, p = {format_p_value(p_value)}, V = {v:.3f}")
print("\nFigure specifications:")
print("  - Resolution: 600 DPI (publication-ready)")
print("  - Color scheme: Colorblind-friendly Nature palette")
print("  - Font: Arial 8-10pt")
print("  - Only non-empty categories shown")
print("  - Statistical annotations included")
print("\n" + "=" * 80)
