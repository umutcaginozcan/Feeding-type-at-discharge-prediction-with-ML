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
NATURE_COLORS = {
    'Formula': '#DE8F05',   # Orange - Formula
    'Mixed': '#029E73',     # Green - Mixed
    'Other': '#949494',     # Gray - Other
    'sig': '#CC78BC',       # Purple - Significance markers
    'neutral': '#4D4D4D'    # Dark Gray
}

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

# Define labels matching actual data (1=Formula, 2=Mixed, 3=Other based on value_counts)
feeding_labels_short = {1: 'Formula', 2: 'Mixed', 3: 'Other'}
feeding_labels_full = {1: 'Formula Feeding', 2: 'Mixed Feeding', 3: 'Other'}

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

def add_significance_bracket(ax, x1, x2, y, p_value, height=0.02):
    """Add significance bracket to plot"""
    # Determine significance level
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'
    
    # Draw bracket
    ax.plot([x1, x1, x2, x2], [y, y+height, y+height, y], 
            linewidth=1, color='black')
    ax.text((x1+x2)/2, y+height, sig_text, ha='center', va='bottom', fontsize=9)

def format_p_value(p):
    """Format p-value for display"""
    if p < 0.001:
        return 'p < 0.001'
    elif p < 0.01:
        return f'p = {p:.3f}'
    else:
        return f'p = {p:.2f}'

# ============================================================================
# FIGURE 1: COVID-19 PERIOD ANALYSIS
# ============================================================================

print("Creating Figure 1: COVID-19 Period Analysis...")

pred_var = 'covid19sonrasi'
pred_name = predictors[pred_var]
pred_lab = predictor_labels[pred_var]

# Create contingency table
contingency = pd.crosstab(df_clean[outcome], df_clean[pred_var])
contingency.index = [feeding_labels_short[i] for i in contingency.index]
contingency.columns = [pred_lab[i] for i in contingency.columns]

# Calculate statistics
chi2, p_value, dof, expected = chi2_contingency(contingency)
v = cramers_v(contingency)

# Calculate proportions
props = contingency.div(contingency.sum(axis=0), axis=1) * 100

# Create figure with 3 panels
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('Figure 1. Feeding Outcomes by COVID-19 Period', 
             fontweight='bold', y=1.02)

# Panel A: Observed frequencies heatmap
ax = axes[0]
sns.heatmap(contingency, annot=True, fmt='d', cmap='YlOrRd', 
            ax=ax, cbar_kws={'label': 'Count'}, linewidths=0.5,
            linecolor='white', square=True)
ax.set_title('A. Observed Frequencies', fontweight='bold', pad=10)
ax.set_xlabel('Time Period')
ax.set_ylabel('Feeding Type at Discharge')

# Panel B: Stacked proportions
ax = axes[1]
x = np.arange(len(contingency.columns))
width = 0.6
bottom = np.zeros(len(contingency.columns))

colors = [NATURE_COLORS['Formula'], NATURE_COLORS['Mixed'], NATURE_COLORS['Other']]

for idx, (feeding_type, row) in enumerate(props.iterrows()):
    ax.bar(x, row, width, bottom=bottom, label=feeding_type, 
           color=colors[idx], edgecolor='white', linewidth=0.8)
    
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
width = 0.35

bars1 = ax.bar(x - width/2, props.iloc[:, 0], width, 
               label=contingency.columns[0], color=NATURE_COLORS['neutral'],
               edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, props.iloc[:, 1], width,
               label=contingency.columns[1], color=NATURE_COLORS['sig'],
               edgecolor='black', linewidth=0.8)

ax.set_title(f'C. Time Period Distribution\n{format_p_value(p_value)}, Cramér\'s V = {v:.3f}', 
             fontweight='bold', pad=10, fontsize=10)
ax.set_xlabel('Feeding Type at Discharge')
ax.set_ylabel('Percentage (%)')
ax.set_xticks(x)
ax.set_xticklabels(props.index, rotation=0)
ax.legend(title='Time Period', frameon=True, fancybox=False, edgecolor='black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f"{output_dir}/Figure_1.png", dpi=600, bbox_inches='tight')
print(f"✓ Saved: Figure_1.png")
plt.close()

# ============================================================================
# FIGURE 2: BABY-FRIENDLY HOSPITAL INITIATIVE
# ============================================================================

print("Creating Figure 2: Baby-Friendly Hospital Initiative...")

pred_var = 'bebek_dostu_20temmuz2018'
pred_name = predictors[pred_var]
pred_lab = predictor_labels[pred_var]

# Create contingency table
contingency = pd.crosstab(df_clean[outcome], df_clean[pred_var])
contingency.index = [feeding_labels_short[i] for i in contingency.index]
contingency.columns = [pred_lab[i] for i in contingency.columns]

# Calculate statistics
chi2, p_value, dof, expected = chi2_contingency(contingency)
v = cramers_v(contingency)

# Calculate proportions
props = contingency.div(contingency.sum(axis=0), axis=1) * 100

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('Figure 2. Feeding Outcomes by Baby-Friendly Hospital Initiative', 
             fontweight='bold', y=1.02)

# Panel A: Observed frequencies
ax = axes[0]
sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', 
            ax=ax, cbar_kws={'label': 'Count'}, linewidths=0.5,
            linecolor='white', square=True)
ax.set_title('A. Observed Frequencies', fontweight='bold', pad=10)
ax.set_xlabel('BFHI Period')
ax.set_ylabel('Feeding Type at Discharge')

# Panel B: Stacked proportions
ax = axes[1]
x = np.arange(len(contingency.columns))
width = 0.6
bottom = np.zeros(len(contingency.columns))

for idx, (feeding_type, row) in enumerate(props.iterrows()):
    ax.bar(x, row, width, bottom=bottom, label=feeding_type, 
           color=colors[idx], edgecolor='white', linewidth=0.8)
    
    for i, val in enumerate(row):
        if val > 5:
            ax.text(x[i], bottom[i] + val/2, f'{val:.1f}%', 
                   ha='center', va='center', fontsize=8, color='white',
                   fontweight='bold')
    
    bottom += row

ax.set_title('B. Feeding Distribution (%)', fontweight='bold', pad=10)
ax.set_xlabel('BFHI Period')
ax.set_ylabel('Percentage (%)')
ax.set_xticks(x)
ax.set_xticklabels(contingency.columns, rotation=0)
ax.set_ylim(0, 100)
ax.legend(title='Feeding Type', frameon=True, fancybox=False, edgecolor='black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel C: Grouped bar chart
ax = axes[2]
x = np.arange(len(props.index))
width = 0.35

bars1 = ax.bar(x - width/2, props.iloc[:, 0], width, 
               label=contingency.columns[0], color=NATURE_COLORS['neutral'],
               edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, props.iloc[:, 1], width,
               label=contingency.columns[1], color=NATURE_COLORS['Formula'],
               edgecolor='black', linewidth=0.8)

ax.set_title(f'C. BFHI Period Distribution\n{format_p_value(p_value)}, Cramér\'s V = {v:.3f}', 
             fontweight='bold', pad=10, fontsize=10)
ax.set_xlabel('Feeding Type at Discharge')
ax.set_ylabel('Percentage (%)')
ax.set_xticks(x)
ax.set_xticklabels(props.index, rotation=0)
ax.legend(title='BFHI Period', frameon=True, fancybox=False, edgecolor='black',
         loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f"{output_dir}/Figure_2.png", dpi=600, bbox_inches='tight')
print(f"✓ Saved: Figure_2.png")
plt.close()

# ============================================================================
# FIGURE 3: COMBINED EPOCHS ANALYSIS
# ============================================================================

print("Creating Figure 3: Combined Epochs Analysis...")

pred_var = 'ikisiarası'
pred_name = predictors[pred_var]
pred_lab = predictor_labels[pred_var]

# Create contingency table
contingency = pd.crosstab(df_clean[outcome], df_clean[pred_var])

# Handle missing EBF category - check what categories exist
existing_feeding_types = contingency.index.tolist()
print(f"  Note: Available feeding types in data: {existing_feeding_types}")

# Use only available categories
available_labels = {k: v for k, v in feeding_labels_short.items() if k in existing_feeding_types}
contingency.index = [available_labels[i] for i in contingency.index]
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

# Create figure
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

fig.suptitle('Figure 3. Feeding Outcomes by Time Epoch (COVID-19 × BFHI)', 
             fontweight='bold', y=0.98, fontsize=13)

# Panel A: Heatmap of observed frequencies
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(contingency, annot=True, fmt='d', cmap='Greens', 
            ax=ax1, cbar_kws={'label': 'Count'}, linewidths=0.5,
            linecolor='white', square=False)
ax1.set_title('A. Observed Frequencies', fontweight='bold', pad=10)
ax1.set_xlabel('Time Epoch')
ax1.set_ylabel('Feeding Type at Discharge')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15, ha='right')

# Panel B: Stacked bar chart
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(len(contingency.columns))
width = 0.65
bottom = np.zeros(len(contingency.columns))

# Use colors for all available feeding types
available_colors = [NATURE_COLORS['Formula'], NATURE_COLORS['Mixed'], NATURE_COLORS['Other']]

for idx, (feeding_type, row) in enumerate(props.iterrows()):
    ax2.bar(x, row, width, bottom=bottom, label=feeding_type, 
           color=available_colors[idx], edgecolor='white', linewidth=0.8)
    
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

# Panel C: Grouped bar chart - Formula feeding across epochs
ax3 = fig.add_subplot(gs[1, 0])
formula_props = props.loc['Formula'] if 'Formula' in props.index else props.iloc[0]
mixed_props = props.loc['Mixed'] if 'Mixed' in props.index else props.iloc[1]

x = np.arange(len(contingency.columns))
width = 0.35

bars1 = ax3.bar(x - width/2, formula_props, width, 
               label='Formula', color=NATURE_COLORS['Formula'],
               edgecolor='black', linewidth=0.8)
bars2 = ax3.bar(x + width/2, mixed_props, width,
               label='Mixed', color=NATURE_COLORS['Mixed'],
               edgecolor='black', linewidth=0.8)

ax3.set_title('C. Feeding Type Comparison Across Epochs', fontweight='bold', pad=10)
ax3.set_xlabel('Time Epoch')
ax3.set_ylabel('Percentage (%)')
ax3.set_xticks(x)
ax3.set_xticklabels(contingency.columns, rotation=15, ha='right')
ax3.legend(frameon=True, fancybox=False, edgecolor='black')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

# Panel D: Pairwise comparisons
ax4 = fig.add_subplot(gs[1, 1])
comparisons = [r['comparison'] for r in pairwise_results]
p_values = [-np.log10(r['p_adj']) for r in pairwise_results]
colors_sig = [NATURE_COLORS['sig'] if r['sig'] else NATURE_COLORS['neutral'] 
              for r in pairwise_results]

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
    width = bar.get_width()
    label = f"p_adj = {result['p_adj']:.4f}" if result['p_adj'] >= 0.001 else "p_adj < 0.001"
    ax4.text(width + 0.1, idx, label, ha='left', va='center', fontsize=8)

plt.savefig(f"{output_dir}/Figure_3.png", dpi=600, bbox_inches='tight')
print(f"✓ Saved: Figure_3.png")
plt.close()

# ============================================================================
# CREATE METHODOLOGY FILE
# ============================================================================

print("\n" + "=" * 80)
print("✓ All publication figures created successfully!")
print("=" * 80)
print(f"\nOutputs saved to: {output_dir}/")
print("  - Figure_1.png: COVID-19 Period Analysis")
print("  - Figure_2.png: Baby-Friendly Hospital Initiative")
print("  - Figure_3.png: Combined Epochs Analysis")
print("\nFigure specifications:")
print("  - Resolution: 600 DPI (publication-ready)")
print("  - Color scheme: Colorblind-friendly Nature palette")
print("  - Font: Arial 8-10pt")
print("  - Format: PNG with transparent background option")
print("\n" + "=" * 80)
