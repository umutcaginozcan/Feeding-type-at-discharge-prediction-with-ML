#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Visualizations for Table 1
-------------------------------------
Creates Nature-quality figures to complement descriptive statistics.

Outputs:
- Figure 1A: Maternal age distribution
- Figure 1B: Birth weight by feeding outcome
- Figure 1C: Study timeline (epoch distribution)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append('.')
from src.data.loader import load_nicu_data, CAT_LABELS_EN

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Nature journal style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.3)
sns.set_palette("Set2")

FIGURE_DPI = 300
FIGURE_FORMAT = ['png', 'pdf']

# ============================================================================
# FIGURE 1A: MATERNAL AGE DISTRIBUTION
# ============================================================================

def create_figure1a(df):
    """Maternal age distribution with normal curve overlay."""
    print("\nCreating Figure 1A: Maternal Age Distribution...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Remove NaN
    age_data = df['anneyasi'].dropna()
    
    # Histogram
    n, bins, patches = ax.hist(age_data, bins=25, density=True, alpha=0.7,
                                color='#2E86AB', edgecolor='black', linewidth=0.5)
    
    # Normal distribution overlay
    mu, sigma = age_data.mean(), age_data.std()
    x = np.linspace(age_data.min(), age_data.max(), 100)
    ax.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2)),
            linewidth=2, color='#A23B72', label=f'Normal dist.\n(μ={mu:.1f}, σ={sigma:.1f})')
    
    # Styling
    ax.set_xlabel('Maternal Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Maternal Age', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add sample size
    ax.text(0.02, 0.98, f'n = {len(age_data)}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save
    for fmt in FIGURE_FORMAT:
        filepath = OUTPUT_DIR / f"figure1a_maternal_age_distribution.{fmt}"
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  ✓ Saved: {filepath}")
    
    plt.close()

# ============================================================================
# FIGURE 1B: BIRTH WEIGHT BY FEEDING OUTCOME
# ============================================================================

def create_figure1b(df):
    """Birth weight distribution by feeding type at discharge."""
    print("\nCreating Figure 1B: Birth Weight by Feeding Outcome...")
    
    # Prepare data
    df_clean = df[['dogumagirligi(gram)', 'taburculuk_beslenmeturu']].dropna()
    
    # Map labels
    feeding_labels = CAT_LABELS_EN['taburculuk_beslenmeturu']
    df_clean['Feeding Type'] = df_clean['taburculuk_beslenmeturu'].map(feeding_labels)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Violin plot with box plot inside
    parts = ax.violinplot(
        [df_clean[df_clean['Feeding Type'] == label]['dogumagirligi(gram)'].values 
         for label in ['Exclusive BF', 'Formula', 'Mixed']],
        positions=[1, 2, 3],
        showmeans=True,
        showmedians=True,
        widths=0.7
    )
    
    # Color the violins
    colors = ['#06D6A0', '#EF476F', '#FFD166']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Overlay box plots
    bp = ax.boxplot(
        [df_clean[df_clean['Feeding Type'] == label]['dogumagirligi(gram)'].values 
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
    ax.set_ylabel('Birth Weight (grams)', fontsize=12, fontweight='bold')
    ax.set_title('Birth Weight Distribution by Feeding Type at Discharge',
                 fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add sample sizes
    for i, label in enumerate(['Exclusive BF', 'Formula', 'Mixed'], 1):
        n = len(df_clean[df_clean['Feeding Type'] == label])
        ax.text(i, ax.get_ylim()[0] + 100, f'n={n}',
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    for fmt in FIGURE_FORMAT:
        filepath = OUTPUT_DIR / f"figure1b_birthweight_by_outcome.{fmt}"
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  ✓ Saved: {filepath}")
    
    plt.close()

# ============================================================================
# FIGURE 1C: STUDY TIMELINE (EPOCH DISTRIBUTION)
# ============================================================================

def create_figure1c(df):
    """Study timeline showing epoch distribution."""
    print("\nCreating Figure 1C: Study Timeline...")
    
    # Count by epoch
    epoch_labels = CAT_LABELS_EN['ikisiarası']
    epoch_counts = df['ikisiarası'].value_counts().sort_index()
    
    # Short labels for plot
    short_labels = {
        0: 'Pre-COVID +\nPre-BFHI',
        1: 'Pre-COVID +\nPost-BFHI',
        2: 'Post-COVID'
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar chart
    colors = ['#118AB2', '#06D6A0', '#EF476F']
    bars = ax.bar(range(len(epoch_counts)), epoch_counts.values, color=colors,
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, epoch_counts.values)):
        height = bar.get_height()
        pct = (count / len(df)) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Styling
    ax.set_xticks(range(len(epoch_counts)))
    ax.set_xticklabels([short_labels[i] for i in epoch_counts.index],
                        fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax.set_title('Study Population Distribution by Epoch',
                 fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add total
    ax.text(0.98, 0.98, f'Total n = {len(df)}',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#FFD166', alpha=0.7,
                     edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    
    # Save
    for fmt in FIGURE_FORMAT:
        filepath = OUTPUT_DIR / f"figure1c_study_timeline.{fmt}"
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  ✓ Saved: {filepath}")
    
    plt.close()

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Generate all Table 1 figures."""
    print("=" * 70)
    print("GENERATING TABLE 1 VISUALIZATIONS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df = load_nicu_data(clean=False)
    print(f"Total sample: n = {len(df)}")
    
    # Create figures
    create_figure1a(df)
    create_figure1b(df)
    create_figure1c(df)
    
    print("\n" + "=" * 70)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("  - figure1a_maternal_age_distribution.png/pdf")
    print("  - figure1b_birthweight_by_outcome.png/pdf")
    print("  - figure1c_study_timeline.png/pdf")

if __name__ == "__main__":
    main()
