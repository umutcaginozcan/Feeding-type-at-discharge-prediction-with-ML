"""
Visualization functions for association analyses.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_contingency_heatmap(contingency, title='Contingency Table', ax=None):
    """
    Plot contingency table as a heatmap.
    
    Parameters
    ----------
    contingency : pd.DataFrame
        Contingency table
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    
    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predictor')
    ax.set_ylabel('Outcome')
    
    return ax


def plot_stacked_bars(proportions, title='Distribution by Predictor', ax=None):
    """
    Plot stacked bar chart of proportions.
    
    Parameters
    ----------
    proportions : pd.DataFrame
        Proportions table (rows=outcome, columns=predictor)
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    
    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    proportions.T.plot(kind='bar', stacked=True, ax=ax,
                       color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predictor Level')
    ax.set_ylabel('Percentage (%)')
    ax.legend(title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    return ax


def plot_side_by_side_bars(proportions, title='Predictor Distribution by Outcome', ax=None):
    """
    Plot side-by-side bar chart.
    
    Parameters
    ----------
    proportions : pd.DataFrame
        Proportions table
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    
    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    proportions.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Percentage (%)')
    ax.legend(title='Predictor', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    return ax


def create_analysis_figure(contingency, proportions, predictor_name, output_path=None):
    """
    Create comprehensive 3-panel analysis figure.
    
    Parameters
    ----------
    contingency : pd.DataFrame
        Contingency table
    proportions : pd.DataFrame
        Proportions table
    predictor_name : str
        Name of predictor for titles
    output_path : str or Path, optional
        Path to save figure
    
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Heatmap
    plot_contingency_heatmap(
        contingency,
        title=f'Observed Frequencies\n{predictor_name}',
        ax=axes[0]
    )
    
    # Panel 2: Stacked bars
    plot_stacked_bars(
        proportions,
        title=f'Distribution by {predictor_name}',
        ax=axes[1]
    )
    
    # Panel 3: Side-by-side bars
    plot_side_by_side_bars(
        proportions,
        title=f'{predictor_name} by Outcome',
        ax=axes[2]
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure: {output_path}")
    
    return fig
