"""
NICU Breastfeeding Research Analysis Package

Reusable modules for systematic statistical analysis and visualization.
"""

__version__ = '1.0.0'
__author__ = 'NICU Breastfeeding Research Team'

# Make key modules easily accessible
from src.data import load_nicu_data
from src.statistics import chi_square_test, cramers_v
from src.visualization import create_analysis_figure

__all__ = [
    'load_nicu_data',
    'chi_square_test',
    'cramers_v',
    'create_analysis_figure',
]
