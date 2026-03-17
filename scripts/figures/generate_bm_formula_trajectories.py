#!/usr/bin/env python3
"""
Generate Breast Milk & Formula Trajectories Figure

This script creates a comprehensive 4-panel figure showing:
- Panel A: Breast milk trajectories by feeding outcome
- Panel B: Formula trajectories by feeding outcome  
- Panel C: Mixed group composition over time
- Panel D: Mixed group breast milk contribution percentage

All volumes are in cubic centimeters (cc).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_nicu_data, CAT_LABELS_EN

# Set style matching reference figure
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.5

# Vibrant color palette matching reference figure
COLORS = {
    'Exclusive BF': '#1abc9c',  # Bright teal/cyan
    'Formula': '#e74c3c',        # Coral/red
    'Mixed': '#f39c12'           # Golden yellow
}
BM_COLOR = '#3498db'      # Bright blue for breast milk
FORMULA_COLOR = '#e67e22'  # Orange for formula

def calculate_trajectories(df):
    """Calculate breast milk and formula trajectories for each feeding group."""
    
    # Define column mappings
    bm_cols = {
        'Day 1': 'aldığıannesütü_ilkgün',
        'Day 2': 'beslenme2.gunannesutucc',
        'Day 3': 'aldıgıannesütü3.gun',
        'Discharge': 'aldığıannesütü_taburculuk'
    }
    
    formula_cols = {
        'Day 1': 'aldığımamamiktari1.gün',
        'Day 2': 'beslenmemamamiktarı2.guncc',
        'Day 3': 'aldıgımamamiktari3.gun',
        'Discharge': 'taburculuktamamamiktari'
    }
    
    # Get feeding type labels
    feeding_labels = CAT_LABELS_EN['taburculuk_beslenmeturu']
    df['Feeding Type'] = df['taburculuk_beslenmeturu'].map(feeding_labels)
    
    # Calculate breast milk trajectories
    bm_results = []
    for feeding_type in ['Exclusive BF', 'Formula', 'Mixed']:
        df_group = df[df['Feeding Type'] == feeding_type]
        
        for day_label, col in bm_cols.items():
            data = df_group[col].dropna()
            
            bm_results.append({
                'Feeding Type': feeding_type,
                'Timepoint': day_label,
                'Mean': data.mean(),
                'Median': data.median(),
                'SD': data.std(),
                'SE': data.sem(),
                'N': len(data)
            })
    
    # Calculate formula trajectories
    formula_results = []
    for feeding_type in ['Exclusive BF', 'Formula', 'Mixed']:
        df_group = df[df['Feeding Type'] == feeding_type]
        
        for day_label, col in formula_cols.items():
            data = df_group[col].dropna()
            
            formula_results.append({
                'Feeding Type': feeding_type,
                'Timepoint': day_label,
                'Mean': data.mean(),
                'Median': data.median(),
                'SD': data.std(),
                'SE': data.sem(),
                'N': len(data)
            })
    
    return pd.DataFrame(bm_results), pd.DataFrame(formula_results)

def print_trajectories(bm_df, formula_df):
    """Print trajectory statistics."""
    
    print("\nBREAST MILK Trajectories (Mean ± SD):")
    timepoints = ['Day 1', 'Day 2', 'Day 3', 'Discharge']
    
    for feeding_type in ['Exclusive BF', 'Formula', 'Mixed']:
        print(f"\n{feeding_type}:")
        for day in timepoints:
            row = bm_df[(bm_df['Feeding Type'] == feeding_type) & (bm_df['Timepoint'] == day)]
            if len(row) > 0:
                mean = row['Mean'].values[0]
                sd = row['SD'].values[0]
                print(f"  {day}: {mean:.1f} ± {sd:.1f} cc")
    
    print("\nFORMULA Trajectories (Mean ± SD):")
    
    for feeding_type in ['Exclusive BF', 'Formula', 'Mixed']:
        print(f"\n{feeding_type}:")
        for day in timepoints:
            row = formula_df[(formula_df['Feeding Type'] == feeding_type) & (formula_df['Timepoint'] == day)]
            if len(row) > 0:
                mean = row['Mean'].values[0]
                sd = row['SD'].values[0]
                print(f"  {day}: {mean:.1f} ± {sd:.1f} cc")

def create_comprehensive_figure(bm_df, formula_df):
    """Create comprehensive 4-panel figure."""
    
    # Create figure with 2x2 grid and purple background
    fig = plt.figure(figsize=(18, 14), facecolor='#f0f0ff')
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    timepoints = ['Day 1', 'Day 2', 'Day 3', 'Discharge']
    x_pos = np.arange(len(timepoints))
    
    # ============================================================
    # PANEL A: Breast Milk Trajectories
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    for feeding_type in ['Exclusive BF', 'Formula', 'Mixed']:
        means = []
        sems = []
        
        for day in timepoints:
            row = bm_df[(bm_df['Feeding Type'] == feeding_type) & (bm_df['Timepoint'] == day)]
            if len(row) > 0:
                means.append(row['Mean'].values[0])
                sems.append(row['SE'].values[0])
            else:
                means.append(np.nan)
                sems.append(np.nan)
        
        # All solid lines, bold markers
        ax1.plot(x_pos, means, marker='o', linewidth=3.5, markersize=12,
                label=feeding_type, color=COLORS[feeding_type], 
                linestyle='-', alpha=0.95, markeredgewidth=0)
        ax1.errorbar(x_pos, means, yerr=sems, fmt='none', 
                    ecolor=COLORS[feeding_type], alpha=0.4, linewidth=2.5, capsize=6)
    
    ax1.set_facecolor('#f0f0ff')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(timepoints, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Timepoint', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Breast Milk Volume (mL)', fontsize=13, fontweight='bold')
    ax1.set_title('Breast Milk Trajectories by Feeding Outcome',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=11, frameon=True, shadow=True, fancybox=True)
    ax1.grid(True, color='white', linewidth=1.2, alpha=0.7)
    ax1.set_ylim(bottom=0)
    
    # ============================================================
    # PANEL B: Formula Trajectories
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    for feeding_type in ['Exclusive BF', 'Formula', 'Mixed']:
        means = []
        sems = []
        
        for day in timepoints:
            row = formula_df[(formula_df['Feeding Type'] == feeding_type) & (formula_df['Timepoint'] == day)]
            if len(row) > 0:
                means.append(row['Mean'].values[0])
                sems.append(row['SE'].values[0])
            else:
                means.append(np.nan)
                sems.append(np.nan)
        
        # All solid lines, bold markers
        ax2.plot(x_pos, means, marker='s', linewidth=3.5, markersize=12,
                label=feeding_type, color=COLORS[feeding_type], 
                linestyle='-', alpha=0.95, markeredgewidth=0)
        ax2.errorbar(x_pos, means, yerr=sems, fmt='none', 
                    ecolor=COLORS[feeding_type], alpha=0.4, linewidth=2.5, capsize=6)
    
    ax2.set_facecolor('#f0f0ff')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(timepoints, fontsize=12, fontweight='bold')
    ax2.set_xlabel('Timepoint', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Formula Volume (mL)', fontsize=13, fontweight='bold')
    ax2.set_title('Formula Trajectories by Feeding Outcome',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', fontsize=11, frameon=True, shadow=True, fancybox=True)
    ax2.grid(True, color='white', linewidth=1.2, alpha=0.7)
    ax2.set_ylim(bottom=0)
    
    # ============================================================
    # PANEL C: Mixed Group Composition (Stacked Area)
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Get Mixed group data
    mixed_bm = []
    mixed_formula = []
    
    for day in timepoints:
        bm_row = bm_df[(bm_df['Feeding Type'] == 'Mixed') & (bm_df['Timepoint'] == day)]
        formula_row = formula_df[(formula_df['Feeding Type'] == 'Mixed') & (formula_df['Timepoint'] == day)]
        
        mixed_bm.append(bm_row['Mean'].values[0] if len(bm_row) > 0 else 0)
        mixed_formula.append(formula_row['Mean'].values[0] if len(formula_row) > 0 else 0)
    
    # Stacked area plot with vibrant colors
    ax3.fill_between(x_pos, 0, mixed_bm, color=BM_COLOR, alpha=0.7, label='Breast Milk', linewidth=0)
    ax3.fill_between(x_pos, mixed_bm, np.array(mixed_bm) + np.array(mixed_formula), 
                     color=FORMULA_COLOR, alpha=0.7, label='Formula', linewidth=0)
    
    # Add total line - bold and dark
    total = np.array(mixed_bm) + np.array(mixed_formula)
    ax3.plot(x_pos, total, color='#2c3e50', linewidth=3.5, marker='s', markersize=11, 
            label='Total', alpha=0.95, markeredgewidth=0)
    
    ax3.set_facecolor('#f0f0ff')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(timepoints, fontsize=12, fontweight='bold')
    ax3.set_xlabel('Timepoint', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Volume (mL)', fontsize=13, fontweight='bold')
    ax3.set_title('Mixed Group - Composition Over Time',
                 fontsize=14, fontweight='bold', pad=15)
    ax3.legend(loc='upper left', fontsize=11, frameon=True, shadow=True, fancybox=True)
    ax3.grid(True, color='white', linewidth=1.2, alpha=0.7)
    ax3.set_ylim(bottom=0)
    
    # ============================================================
    # PANEL D: Mixed Group - Breast Milk Percentage
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate percentages
    bm_percentages = []
    for i in range(len(timepoints)):
        total_vol = mixed_bm[i] + mixed_formula[i]
        if total_vol > 0:
            bm_pct = (mixed_bm[i] / total_vol) * 100
        else:
            bm_pct = 0
        bm_percentages.append(bm_pct)
    
    # Plot with bold line and markers
    ax4.plot(x_pos, bm_percentages, marker='o', linewidth=3.5, markersize=12,
            color=COLORS['Mixed'], alpha=0.95, markeredgewidth=0)
    
    # Add 50% threshold line
    ax4.axhline(y=50, color='#7f8c8d', linestyle='--', linewidth=2, alpha=0.6, 
               label='50% threshold')
    
    # Add percentage labels on points - larger font
    for i, (x, y) in enumerate(zip(x_pos, bm_percentages)):
        ax4.annotate(f'{y:.1f}%', xy=(x, y), xytext=(0, 12), 
                    textcoords='offset points', ha='center', fontsize=11, 
                    fontweight='bold', color='#2c3e50')
    
    ax4.set_facecolor('#f0f0ff')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(timepoints, fontsize=12, fontweight='bold')
    ax4.set_xlabel('Timepoint', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Breast Milk Percentage (%)', fontsize=13, fontweight='bold')
    ax4.set_title('Mixed Group - Breast Milk Contribution',
                 fontsize=14, fontweight='bold', pad=15)
    ax4.legend(loc='upper right', fontsize=11, frameon=True, shadow=True, fancybox=True)
    ax4.grid(True, color='white', linewidth=1.2, alpha=0.7)
    ax4.set_ylim(0, 100)
    
    return fig, (mixed_bm, mixed_formula, bm_percentages)

def save_outputs(bm_df, formula_df, mixed_data):
    """Save all outputs."""
    
    # Create output directories
    figures_dir = project_root / 'paper' / 'figures'
    data_dir = figures_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    fig, _ = create_comprehensive_figure(bm_df, formula_df)
    
    png_path = figures_dir / 'figure_bm_formula_trajectories.png'
    pdf_path = figures_dir / 'figure_bm_formula_trajectories.pdf'
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"\nCreating comprehensive trajectory figure...")
    print(f"  ✓ Saved: {png_path.relative_to(project_root)}")
    print(f"  ✓ Saved: {pdf_path.relative_to(project_root)}")
    
    plt.close()
    
    # Prepare data for Excel
    timepoints = ['Day 1', 'Day 2', 'Day 3', 'Discharge']
    
    # Format breast milk trajectories table
    bm_table = bm_df.pivot(index='Timepoint', columns='Feeding Type', values='Mean')
    bm_table = bm_table.reindex(timepoints)
    bm_table.columns = [f'{col} (cc)' for col in bm_table.columns]
    
    # Format formula trajectories table
    formula_table = formula_df.pivot(index='Timepoint', columns='Feeding Type', values='Mean')
    formula_table = formula_table.reindex(timepoints)
    formula_table.columns = [f'{col} (cc)' for col in formula_table.columns]
    
    # Create mixed composition table
    mixed_bm, mixed_formula, bm_percentages = mixed_data
    mixed_composition = pd.DataFrame({
        'Timepoint': timepoints,
        'Breast Milk (cc)': mixed_bm,
        'Formula (cc)': mixed_formula,
        'Total (cc)': [bm + fm for bm, fm in zip(mixed_bm, mixed_formula)],
        'BM Percentage (%)': bm_percentages
    })
    mixed_composition.set_index('Timepoint', inplace=True)
    
    # Append to all_statistics.xlsx
    excel_path = data_dir / 'all_statistics.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a' if excel_path.exists() else 'w', 
                       if_sheet_exists='replace' if excel_path.exists() else None) as writer:
        bm_table.to_excel(writer, sheet_name='BM_Trajectories')
        formula_table.to_excel(writer, sheet_name='Formula_Trajectories')
        mixed_composition.to_excel(writer, sheet_name='Mixed_Composition')
    
    print(f"\nAppending to all_statistics.xlsx...")
    print(f"  ✓ Appended to: {excel_path.relative_to(project_root)}")
    print(f"  ✓ Added sheets: BM_Trajectories, Formula_Trajectories, Mixed_Composition")
    
    # Save individual CSV files
    bm_csv = data_dir / 'bm_trajectories_table.csv'
    mixed_csv = data_dir / 'mixed_composition_table.csv'
    
    bm_table.to_csv(bm_csv)
    mixed_composition.to_csv(mixed_csv)
    
    print(f"  ✓ Saved: {bm_csv.relative_to(project_root)}")
    print(f"  ✓ Saved: {mixed_csv.relative_to(project_root)}")
    
    return mixed_bm, mixed_formula, bm_percentages

def main():
    """Main execution."""
    
    print("=" * 70)
    print("GENERATING BREAST MILK & FORMULA TRAJECTORIES")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df = load_nicu_data(clean=False)
    print(f"Total sample: n = {len(df)}")
    
    # Calculate trajectories
    print("\nCalculating breast milk and formula trajectories...")
    bm_df, formula_df = calculate_trajectories(df)
    
    # Print results
    print_trajectories(bm_df, formula_df)
    
    # Create and save figure
    fig, mixed_data = create_comprehensive_figure(bm_df, formula_df)
    mixed_bm, mixed_formula, bm_percentages = save_outputs(bm_df, formula_df, mixed_data)
    
    # Print key findings
    print("\n" + "=" * 70)
    print("✓ BREAST MILK & FORMULA TRAJECTORIES COMPLETE")
    print("=" * 70)
    
    print("\nKey Findings:")
    print("\nMixed Group Breast Milk Contribution:")
    timepoints = ['Day 1', 'Day 2', 'Day 3', 'Discharge']
    for i, day in enumerate(timepoints):
        total = mixed_bm[i] + mixed_formula[i]
        bm_pct = bm_percentages[i]
        formula_pct = 100 - bm_pct
        print(f"  {day}: {bm_pct:.1f}% breast milk, {formula_pct:.1f}% formula")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
