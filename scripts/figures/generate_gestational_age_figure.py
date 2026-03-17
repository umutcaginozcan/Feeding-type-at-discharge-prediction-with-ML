#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Gestational Age Figure - Matching Existing Format
-----------------------------------------------------------
Creates Nature-quality figure matching the style of figure1b_birthweight_by_outcome.

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

# Nature journal style - MATCHING EXISTING FIGURES
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.3)
sns.set_palette("Set2")

FIGURE_DPI = 300
FIGURE_FORMAT = ['png', 'pdf']

# Colors matching birth weight figure
COLORS = ['#06D6A0', '#EF476F', '#FFD166']  # Teal, Pink, Yellow

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("="*70)
print("GENERATING GESTATIONAL AGE FIGURE")
print("="*70)

print("\nLoading data...")
df = load_nicu_data(clean=False)

# Prepare data
df_clean = df[['gebelikhaftası', 'taburculuk_beslenmeturu']].dropna()

# Map labels
feeding_labels = CAT_LABELS_EN['taburculuk_beslenmeturu']
df_clean['Feeding Type'] = df_clean['taburculuk_beslenmeturu'].map(feeding_labels)

print(f"Total sample: n = {len(df_clean)}")

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

print("\nPerforming statistical analysis...")

# Prepare data for tests
ebf_data = df_clean[df_clean['Feeding Type'] == 'Exclusive BF']['gebelikhaftası']
formula_data = df_clean[df_clean['Feeding Type'] == 'Formula']['gebelikhaftası']
mixed_data = df_clean[df_clean['Feeding Type'] == 'Mixed']['gebelikhaftası']

# One-way ANOVA
f_stat, p_anova = stats.f_oneway(ebf_data, formula_data, mixed_data)

# Kruskal-Wallis (non-parametric)
h_stat, p_kruskal = stats.kruskal(ebf_data, formula_data, mixed_data)

print(f"  ANOVA: F({2},{len(df_clean)-3}) = {f_stat:.4f}, p = {p_anova:.4f}")
print(f"  Kruskal-Wallis: H = {h_stat:.4f}, p = {p_kruskal:.4f}")

# Descriptive statistics
for label in ['Exclusive BF', 'Formula', 'Mixed']:
    data = df_clean[df_clean['Feeding Type'] == label]['gebelikhaftası']
    print(f"\n{label} (n={len(data)}):")
    print(f"  Mean ± SD: {data.mean():.1f} ± {data.std():.1f} weeks")
    print(f"  Median (IQR): {data.median():.1f} ({data.quantile(0.25):.1f}–{data.quantile(0.75):.1f}) weeks")

# ============================================================================
# CREATE FIGURE - MATCHING EXISTING FORMAT
# ============================================================================

print("\nCreating figure...")

fig, ax = plt.subplots(figsize=(10, 6))

# Violin plot with box plot inside (EXACTLY like birth weight figure)
parts = ax.violinplot(
    [df_clean[df_clean['Feeding Type'] == label]['gebelikhaftası'].values 
     for label in ['Exclusive BF', 'Formula', 'Mixed']],
    positions=[1, 2, 3],
    showmeans=True,
    showmedians=True,
    widths=0.7
)

# Color the violins (MATCHING birth weight colors)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(COLORS[i])
    pc.set_alpha(0.6)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

# Overlay box plots (EXACTLY like birth weight figure)
bp = ax.boxplot(
    [df_clean[df_clean['Feeding Type'] == label]['gebelikhaftası'].values 
     for label in ['Exclusive BF', 'Formula', 'Mixed']],
    positions=[1, 2, 3],
    widths=0.3,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.8),
    whiskerprops=dict(color='black', linewidth=1.5),
    capprops=dict(color='black', linewidth=1.5),
    medianprops=dict(color='#A81E1E', linewidth=2.5)  # Red median line
)

# Styling (MATCHING birth weight figure)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Exclusive BF', 'Formula', 'Mixed'], fontsize=11, fontweight='bold')
ax.set_ylabel('Gestational Age (weeks)', fontsize=12, fontweight='bold')
ax.set_title('Gestational Age Distribution by Feeding Type at Discharge',
             fontsize=14, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add sample sizes (MATCHING birth weight figure format)
for i, label in enumerate(['Exclusive BF', 'Formula', 'Mixed'], 1):
    n = len(df_clean[df_clean['Feeding Type'] == label])
    ax.text(i, ax.get_ylim()[0] + 0.5, f'n={n}',
            ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()

# Save
for fmt in FIGURE_FORMAT:
    filepath = OUTPUT_DIR / f"ffigure1d_gestational_age_by_outcome.{fmt}"
    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  ✓ Saved: {filepath}")

plt.close()

# ============================================================================
# SAVE SUMMARY STATISTICS
# ============================================================================

print("\nSaving summary statistics...")

summary_data = []
for label in ['Exclusive BF', 'Formula', 'Mixed']:
    data = df_clean[df_clean['Feeding Type'] == label]['gebelikhaftası']
    summary_data.append({
        'Feeding Type': label,
        'n': len(data),
        'Mean': data.mean(),
        'SD': data.std(),
        'Median': data.median(),
        'Q1': data.quantile(0.25),
        'Q3': data.quantile(0.75),
        'Min': data.min(),
        'Max': data.max()
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(DATA_DIR / 'gestational_age_by_feeding.csv', index=False)
print(f"  ✓ Saved: {DATA_DIR / 'gestational_age_by_feeding.csv'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✓ GESTATIONAL AGE FIGURE COMPLETE")
print("="*70)
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print("  - figure1d_gestational_age_by_outcome.png/pdf")
print(f"\nStatistical result: F({2},{len(df_clean)-3}) = {f_stat:.2f}, p = {p_anova:.3f}")
if p_anova < 0.05:
    print("  → Significant difference between groups")
else:
    print("  → No significant difference between groups")
print("\n" + "="*70)
