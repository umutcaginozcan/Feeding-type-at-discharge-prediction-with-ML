"""
Data loading and preprocessing utilities for NICU breastfeeding research.
"""

import pandas as pd
from pathlib import Path

# Data path
DATA_PATH = Path('excels-NICU-breatsfeeding-data/nicu_stage0_5_cleaned.xlsx')

# Variable labels for human-readable output
VARIABLE_LABELS = {
    'taburculuk_beslenmeturu': 'Feeding Type at Discharge',
    'ikisiarası': 'Epoch (COVID × BFHI)',
    'covid19sonrasi': 'COVID-19 Period',
    'bebek_dostu_20temmuz2018': 'Baby-Friendly Hospital Initiative',
    'anneyasi': 'Maternal Age',
    'dogumagirligi(gram)': 'Birth Weight (g)',
    'gebelikhaftası': 'Gestational Age (weeks)',
}

# Category labels
FEEDING_TYPE_LABELS = {
    0: 'Exclusive BF',
    1: 'Formula',
    2: 'Mixed',
    3: 'Other'
}

EPOCH_LABELS = {
    0: 'Pre-COVID + Pre-BFHI',
    1: 'Pre-COVID + Post-BFHI',
    2: 'Post-COVID'
}

COVID_LABELS = {
    0: 'Pre-COVID',
    1: 'Post-COVID'
}

BFHI_LABELS = {
    0: 'Pre-BFHI',
    1: 'Post-BFHI'
}


def load_nicu_data(clean=True, variables=None):
    """
    Load NICU breastfeeding data.
    
    Parameters
    ----------
    clean : bool, default=True
        If True, remove rows with missing values in specified variables
    variables : list of str, optional
        Variables to include. If None, loads all variables.
        If clean=True, removes rows with missing values in these variables.
    
    Returns
    -------
    pd.DataFrame
        Loaded data
    
    Examples
    --------
    >>> df = load_nicu_data()
    >>> df_subset = load_nicu_data(variables=['taburculuk_beslenmeturu', 'ikisiarası'])
    """
    df = pd.read_excel(DATA_PATH)
    
    if variables is not None:
        # Check if all variables exist
        missing_vars = set(variables) - set(df.columns)
        if missing_vars:
            raise ValueError(f"Variables not found in data: {missing_vars}")
        
        df = df[variables]
    
    if clean and variables is not None:
        original_len = len(df)
        df = df.dropna()
        dropped = original_len - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with missing values ({dropped/original_len*100:.1f}%)")
    
    return df


def get_variable_label(variable_name):
    """
    Get human-readable label for a variable.
    
    Parameters
    ----------
    variable_name : str
        Variable name in the dataset
    
    Returns
    -------
    str
        Human-readable label
    """
    return VARIABLE_LABELS.get(variable_name, variable_name)


def get_category_labels(variable_name):
    """
    Get category labels for a categorical variable.
    
    Parameters
    ----------
    variable_name : str
        Variable name
    
    Returns
    -------
    dict
        Mapping from category codes to labels
    """
    label_maps = {
        'taburculuk_beslenmeturu': FEEDING_TYPE_LABELS,
        'ikisiarası': EPOCH_LABELS,
        'covid19sonrasi': COVID_LABELS,
        'bebek_dostu_20temmuz2018': BFHI_LABELS,
    }
    
    return label_maps.get(variable_name, {})


def describe_variable(df, variable):
    """
    Get descriptive statistics for a variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data
    variable : str
        Variable name
    
    Returns
    -------
    pd.Series or pd.DataFrame
        Descriptive statistics
    """
    if df[variable].dtype in ['int64', 'float64']:
        return df[variable].describe()
    else:
        counts = df[variable].value_counts()
        labels = get_category_labels(variable)
        if labels:
            counts.index = counts.index.map(lambda x: labels.get(x, x))
        return counts


def check_missing_data(df, variables=None):
    """
    Check for missing data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data
    variables : list of str, optional
        Variables to check. If None, checks all variables.
    
    Returns
    -------
    pd.DataFrame
        Missing data report
    """
    if variables is None:
        variables = df.columns
    
    missing_counts = df[variables].isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    
    report = pd.DataFrame({
        'Variable': variables,
        'Missing Count': missing_counts.values,
        'Missing %': missing_pct.values
    })
    
    return report[report['Missing Count'] > 0].sort_values('Missing %', ascending=False)
