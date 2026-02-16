#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Length of Stay (LOS) Figure + Append to All Statistics Table
----------------------------------------------------------------------
Creates:
1. LOS distribution figure by feeding outcome (matching existing format)
2. Appends comprehensive statistics to all_statistics.xlsx

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
from openpyxl import load_workbook

# Add src to path
sys.path.append('.')
from src.data.loader import load_nicu_data, CAT_LABELS_EN

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("paper/figures/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Nature journal style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.3)
sns.set_palette("Set2")

FIGURE_DPI = 300
FIGURE_FORMAT = ['png', 'pdf']

# Colors matching existing figures
COLORS = ['#06D6A0', '#EF476F', '#FFD166']

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("="*70)
print("GENERATING LENGTH OF STAY (LOS) FIGURE")
print("="*70)

print("\nLoading data...")
df = load_nicu_data(clean=False)

# LOS variable is 'takiptekacgun' (days in follow-up)
df_clean = df[['takiptekacgun', 'taburculuk_beslenmeturu']].dropna()
feeding_labels = CAT_LABELS_EN['taburculuk_beslenmeturu']
df_clean['Feeding Type'] = df_clean['taburculuk_beslenmeturu'].map(feeding_labels)

print(f"Total sample: n = {len(df_clean)}")

# Check for outliers (LOS > 60 days is very long for NICU)
print(f"LOS range: {df_clean['takiptekacgun'].min():.0f} - {df_clean['takiptekacgun'].max():.0f} days")
print(f"Patients with LOS > 60 days: {len(df_clean[df_clean['takiptekacgun'] > 60])}")

# ============================================================================
# COMPREHENSIVE STATISTICAL ANALYSIS
# ============================================================================

print("\nPerforming comprehensive statistical analysis...")

# Prepare data
ebf_data = df_clean[df_clean['Feeding Type'] == 'Exclusive BF']['takiptekacgun']
formula_data = df_clean[df_clean['Feeding Type'] == 'Formula']['takiptekacgun']
mixed_data = df_clean[df_clean['Feeding Type'] == 'Mixed']['takiptekacgun']

# 1. Descriptive statistics
stats_list = []
for label, data in [('Exclusive BF', ebf_data), ('Formula', formula_data), ('Mixed', mixed_data)]:
    stats_dict = {
        'Feeding Type': label,
        'n': len(data),
        'Mean': data.mean(),
        'SD': data.std(),
        'Median': data.median(),
        'Q1': data.quantile(0.25),
        'Q3': data.quantile(0.75),
        'IQR': data.quantile(0.75) - data.quantile(0.25),
        'Min': data.min(),
        'Max': data.max(),
        'Range': f"{data.min():.0f}–{data.max():.0f}"
    }
    stats_list.append(stats_dict)
    print(f"\n{label} (n={len(data)}):")
    print(f"  Mean ± SD: {data.mean():.1f} ± {data.std():.1f} days")
    print(f"  Median (IQR): {data.median():.1f} ({data.quantile(0.25):.1f}–{data.quantile(0.75):.1f}) days")

# 2. Normality tests
normality_results = []
for label, data in [('Exclusive BF', ebf_data), ('Formula', formula_data), ('Mixed', mixed_data)]:
    stat, p = stats.shapiro(data)
    normality_results.append({
        'Group': label,
        'Test': 'Shapiro-Wilk',
        'Statistic': stat,
        'p-value': p,
        'Normal': 'Yes' if p > 0.05 else 'No'
    })

# 3. Variance homogeneity
levene_stat, levene_p = stats.levene(ebf_data, formula_data, mixed_data)

# 4. ANOVA
f_stat, p_anova = stats.f_oneway(ebf_data, formula_data, mixed_data)

# 5. Kruskal-Wallis (non-parametric - likely more appropriate for LOS)
h_stat, p_kruskal = stats.kruskal(ebf_data, formula_data, mixed_data)

# 6. Effect size (eta-squared)
grand_mean = df_clean['takiptekacgun'].mean()
ss_between = sum([len(data) * (data.mean() - grand_mean)**2 
                  for data in [ebf_data, formula_data, mixed_data]])
ss_total = sum((df_clean['takiptekacgun'] - grand_mean)**2)
eta_squared = ss_between / ss_total

print(f"\n  ANOVA: F(2,{len(df_clean)-3}) = {f_stat:.3f}, p = {p_anova:.4f}")
print(f"  Kruskal-Wallis: H = {h_stat:.3f}, p = {p_kruskal:.4f}")
print(f"  Effect size: η² = {eta_squared:.4f}")

# 7. Post-hoc pairwise comparisons
pairwise_results = []
comparisons = [
    ('Exclusive BF vs Formula', ebf_data, formula_data),
    ('Exclusive BF vs Mixed', ebf_data, mixed_data),
    ('Formula vs Mixed', formula_data, mixed_data)
]

for name, data1, data2 in comparisons:
    # Mann-Whitney U test (non-parametric)
    u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    p_bonferroni = min(p_val * 3, 1.0)
    # Effect size: rank-biserial correlation
    n1, n2 = len(data1), len(data2)
    r = 1 - (2*u_stat) / (n1 * n2)
    pairwise_results.append({
        'Comparison': name,
        'Mann-Whitney U': u_stat,
        'p-value (unadjusted)': p_val,
        'p-value (Bonferroni)': p_bonferroni,
        'Effect size (r)': r
    })

# ============================================================================
# CREATE FIGURE WITH STATISTICAL ANNOTATION
# ============================================================================

print("\nCreating figure...")

fig, ax = plt.subplots(figsize=(10, 6))

# Violin plot
parts = ax.violinplot(
    [df_clean[df_clean['Feeding Type'] == label]['takiptekacgun'].values 
     for label in ['Exclusive BF', 'Formula', 'Mixed']],
    positions=[1, 2, 3],
    showmeans=True,
    showmedians=True,
    widths=0.7
)

# Color the violins
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(COLORS[i])
    pc.set_alpha(0.6)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

# Overlay box plots
bp = ax.boxplot(
    [df_clean[df_clean['Feeding Type'] == label]['takiptekacgun'].values 
     for label in ['Exclusive BF', 'Formula', 'Mixed']],
    positions=[1, 2, 3],
    widths=0.3,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.8),
    whiskerprops=dict(color='black', linewidth=1.5),
    capprops=dict(color='black', linewidth=1.5),
    medianprops=dict(color='#A81E1E', linewidth=2.5)
)

# Styling
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Exclusive BF', 'Formula', 'Mixed'], fontsize=11, fontweight='bold')
ax.set_ylabel('Length of Stay (days)', fontsize=12, fontweight='bold')
ax.set_title('Length of Stay Distribution by Feeding Type at Discharge',
             fontsize=14, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add sample sizes
for i, label in enumerate(['Exclusive BF', 'Formula', 'Mixed'], 1):
    n = len(df_clean[df_clean['Feeding Type'] == label])
    ax.text(i, ax.get_ylim()[0] + 0.5, f'n={n}',
            ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Add statistical annotation
if p_kruskal < 0.001:
    p_text = "p<0.001"
elif p_kruskal < 0.01:
    p_text = f"p={p_kruskal:.3f}"
else:
    p_text = f"p={p_kruskal:.2f}"

stat_text = f"Kruskal-Wallis: H={h_stat:.2f}, {p_text}\nη²={eta_squared:.4f}"
ax.text(0.98, 0.98, stat_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', 
                 edgecolor='gray', alpha=0.9, linewidth=1))

plt.tight_layout()

# Save figure
for fmt in FIGURE_FORMAT:
    filepath = OUTPUT_DIR / f"figure_los_by_outcome.{fmt}"
    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

plt.close()

# ============================================================================
# APPEND TO ALL_STATISTICS.XLSX
# ============================================================================

print("\nAppending to all_statistics.xlsx...")

excel_path = DATA_DIR / 'all_statistics.xlsx'

# Load existing workbook and append new sheets
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    # Append LOS sheets
    # Sheet: LOS Descriptive Statistics
    desc_df = pd.DataFrame(stats_list)
    desc_df['Mean ± SD'] = desc_df.apply(lambda x: f"{x['Mean']:.1f} ± {x['SD']:.1f}", axis=1)
    desc_df['Median (IQR)'] = desc_df.apply(lambda x: f"{x['Median']:.1f} ({x['Q1']:.1f}–{x['Q3']:.1f})", axis=1)
    desc_table = desc_df[['Feeding Type', 'n', 'Mean ± SD', 'Median (IQR)', 'Range']]
    desc_table.to_excel(writer, sheet_name='LOS_Descriptive', index=False)

    # Sheet: LOS Statistical Tests
    test_results = pd.DataFrame([
        {'Test': 'One-way ANOVA', 'Statistic': f'F(2,{len(df_clean)-3})={f_stat:.3f}', 
         'p-value': f'{p_anova:.4f}', 'Interpretation': 'Not significant' if p_anova >= 0.05 else 'Significant'},
        {'Test': 'Kruskal-Wallis', 'Statistic': f'H={h_stat:.3f}', 
         'p-value': f'{p_kruskal:.4f}', 'Interpretation': 'Not significant' if p_kruskal >= 0.05 else 'Significant'},
        {'Test': 'Levene (variance)', 'Statistic': f'F={levene_stat:.3f}', 
         'p-value': f'{levene_p:.4f}', 'Interpretation': 'Equal variances' if levene_p >= 0.05 else 'Unequal variances'},
        {'Test': 'Effect size (η²)', 'Statistic': f'{eta_squared:.4f}', 
         'p-value': 'N/A', 'Interpretation': 'Negligible' if eta_squared < 0.01 else 'Small' if eta_squared < 0.06 else 'Medium'}
    ])
    test_results.to_excel(writer, sheet_name='LOS_Statistical_Tests', index=False)

    # Sheet: LOS Normality Tests
    normality_df = pd.DataFrame(normality_results)
    normality_df.to_excel(writer, sheet_name='LOS_Normality', index=False)

    # Sheet: LOS Pairwise Comparisons
    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_df.to_excel(writer, sheet_name='LOS_Pairwise', index=False)

print(f"  ✓ Appended LOS statistics to: {excel_path}")
print(f"  ✓ Added sheets: LOS_Descriptive, LOS_Statistical_Tests, LOS_Normality, LOS_Pairwise")

# Also save standalone CSV
desc_table.to_csv(DATA_DIR / 'los_descriptive_table.csv', index=False)
test_results.to_csv(DATA_DIR / 'los_test_results.csv', index=False)

print(f"  ✓ Saved: {DATA_DIR / 'los_descriptive_table.csv'}")
print(f"  ✓ Saved: {DATA_DIR / 'los_test_results.csv'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✓ LENGTH OF STAY ANALYSIS COMPLETE")
print("="*70)
print("\nFigure:")
print("  - paper/figures/figure_los_by_outcome.png/pdf")
print("\nStatistics appended to:")
print("  - paper/figures/data/all_statistics.xlsx")
print(f"\nStatistical Summary:")
print(f"  - Kruskal-Wallis: H = {h_stat:.2f}, p = {p_kruskal:.3f}")
print(f"  - Effect size: η² = {eta_squared:.4f}")
if p_kruskal < 0.05:
    print("  - Conclusion: Significant difference between groups")
else:
    print("  - Conclusion: No significant difference between groups")
print("\n" + "="*70)
