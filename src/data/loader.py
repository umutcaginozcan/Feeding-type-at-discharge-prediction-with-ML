"""
Data loading and preprocessing utilities for NICU breastfeeding research.

This module provides comprehensive label encodings and data loading functions
for systematic analysis of NICU breastfeeding outcomes.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

# ============================================================================
# DATA PATH
# ============================================================================

DATA_PATH = Path('excels-NICU-breatsfeeding-data/nicu_stage0_5_cleaned.xlsx')

# ============================================================================
# COMPREHENSIVE CATEGORICAL LABELS (Professional English)
# ============================================================================

CAT_LABELS_EN: Dict[str, Dict[Any, str]] = {
    "taburculuk_beslenmeturu": {
        1: "Exclusive BF",
        2: "Formula", 
        3: "Mixed"
    },
    "tanı_gruplu": {
        1: "Prematurity",
        2: "Transient tachypnea of the newborn (TTN)",
        3: "Congenital heart disease",
        4: "Indirect (unconjugated) hyperbilirubinemia",
        5: "Hypoglycemia",
        6: "Hypoxic–ischemic encephalopathy (HIE)",
        7: "Neonatal infection/sepsis",
        8: "Dehydration",
        9: "Polycythemia",
        10: "Hydrops fetalis",
        11: "Congenital gastrointestinal anomaly",
        12: "Central nervous system anomaly",
        13: "Hematologic anomaly",
        14: "Endocrine anomaly",
        15: "Inborn error of metabolism",
        16: "Neonatal seizures",
        17: "Other congenital anomaly",
        18: "Traumatic fall",
        19: "Perinatal hypoxia"
    },
    "dogum_agırlıgı_gruplu": {
        1: "<1000 g (ELBW)",
        2: "1000–1499 g (VLBW)",
        3: "1500–2500 g (LBW)",
        4: "2501–4000 g",
        5: ">4000 g (Macrosomia)"
    },
    "bebek_dostu_20temmuz2018": {
        0: "Before July 20, 2018", 
        1: "After July 20, 2018"
    },
    "cinsiyeti": {
        0: "Female", 
        1: "Male"
    },
    "gebelik_tipi_gruplu": {
        0: "Singleton", 
        1: "Multiple"
    },
    "gebeliktipi": {
        1: "Singleton", 
        2: "Twin", 
        3: "Triplet"
    },
    "gebelik_34": {
        0: "≥34 weeks", 
        1: "<34 weeks"
    },
    "VAR00004": {
        0: "<28 weeks",
        1: "28–31+6/7 weeks",
        2: "32–33+6/7 weeks",
        3: "34–36+6/7 weeks",
        4: "≥37 weeks"
    },
    "anne_meslek_grup": {
        1: "Homemaker", 
        2: "Healthcare personnel", 
        3: "Other"
    },
    "annemeslegi": {
        1: "Homemaker", 
        2: "Civil servant", 
        3: "Teacher", 
        4: "Nurse/Midwife",
        5: "Allied health personnel", 
        6: "Physician", 
        7: "Engineer",
        8: "Legal professional", 
        9: "Other"
    },
    "anne_yaşı_grup": {
        1: "<18 years", 
        2: "18–35 years", 
        3: ">35 years"
    },
    "anne_egitim_grup": {
        1: "Illiterate", 
        2: "Primary/Secondary school", 
        3: "High school+"
    },
    "anneegitim": {
        1: "Illiterate", 
        2: "Primary school", 
        3: "Middle school", 
        4: "High school",
        5: "Associate degree", 
        6: "Bachelor's degree", 
        7: "Graduate+"
    },
    "dogumsekli": {
        0: "Vaginal delivery", 
        1: "Cesarean section"
    },
    "anne_hastalık_grup": {
        0: "None", 
        1: "Hypothyroidism", 
        2: "Diabetes", 
        3: "Hypertension",
        4: "Asthma/Allergy/Urticaria", 
        5: "Cardiac disease", 
        6: "Renal disease",
        7: "Hematologic disorder", 
        8: "Neurologic disorder", 
        9: "Rheumatologic disorder",
        10: "Hepatitis B/C", 
        11: "Other"
    },
    "Kullandıgıpompamarkasi": {
        0: "None", 
        1: "Medela", 
        2: "Avent", 
        3: "Lansinoh", 
        4: "Ameda",
        5: "Mamajoo", 
        6: "Kraft", 
        7: "Chicco", 
        8: "Other"
    },
    "Kullandıgpompatipi": {
        1: "Electric pump", 
        2: "Manual pump", 
        3: "Hand expression"
    },
    "Kolostrumvarligi": {
        0: "Absent", 
        1: "Present"
    },
    "ilk_gün_anne_sütü_1111": {
        0: "Absent", 
        1: "Present"
    },
    "ilkgün_bebeğinannesütüalımı": {
        0: "Absent", 
        1: "Present"
    },
    "beslenmeninilkgunuverilisyolu": {
        0: "None", 
        1: "Oral (PO)", 
        2: "Orogastric (OG)", 
        3: "PO + OG",
        4: "Breastfeeding", 
        5: "Breastfeeding + PO", 
        6: "Breastfeeding + OG",
        7: "Bottle feeding", 
        8: "Bottle + Breastfeeding"
    },
    "ilk_gün_emzirme_111": {
        0: "Absent", 
        1: "Present"
    },
    "verilisyolu2.gun": {
        0: "None", 
        1: "Oral (PO)", 
        2: "Orogastric (OG)", 
        3: "PO + OG",
        4: "Breastfeeding", 
        5: "Breastfeeding + PO", 
        6: "Breastfeeding + OG",
        7: "Bottle feeding", 
        8: "Bottle + Breastfeeding",
        " ": "None"  # Data entry as space for missing
    },
    "verilisyolu3gun": {
        0: "None", 
        1: "Oral (PO)", 
        2: "Orogastric (OG)", 
        3: "PO + OG",
        4: "Breastfeeding", 
        5: "Breastfeeding + PO", 
        6: "Breastfeeding + OG",
        7: "Bottle feeding", 
        8: "Bottle + Breastfeeding"
    },
    "taburculukta_annesutu_111": {
        0: "Absent", 
        1: "Present"
    },
    "taburculuktanasılbeslenmeyolu": {
        1: "Breast milk", 
        2: "Breast milk + Formula", 
        3: "Formula",
        4: "Specialized formula", 
        5: "Breast milk + Specialized formula",
        6: "Formula + Specialized formula",
        8: "Other combination"
    },
    "taburculuktaogvarmiyokmu": {
        0: "Absent", 
        1: "Present"
    },
    "baslangictasutdestegi": {
        0: "Absent", 
        1: "Present"
    },
    "taburculuktadestekcesidi": {
        0: "None",
        1: "Euproten", 
        2: "Supplement", 
        3: "Euproten + Supplement", 
        4: "Fantomalt"
    },
    "annesutuemzirmeeğitimidurumu": {
        0: "Absent", 
        1: "Present"
    },
    "galaktokogkullanımı": {
        0: "Absent", 
        1: "Present"
    },
    "memesorunuyaşamadurumu": {
        0: "Absent", 
        1: "Present"
    },
    "memesorunuvarsa_tedavidekullanılanlar": {
        0: "None",
        1: "Massage", 
        2: "Hand expression", 
        3: "General surgery consult",
        4: "All", 
        5: "Massage + Hand expression"
    },
    "taburculukrtasutdestegivarmı": {
        0: "Absent", 
        1: "Present",
        " ": "Unknown"  # Data entry as space for missing
    },
    "Ataburculuktaannesutu": {
        0: "Absent", 
        1: "Present"
    },
    "emzirme_Taburculuk": {
        0: "Absent", 
        1: "Present"
    },
    "covid19sonrasi": {
        0: "No", 
        1: "Yes"
    },
    # Custom epoch variable
    "ikisiarası": {
        0: "Pre-COVID + Pre-BFHI",
        1: "Pre-COVID + Post-BFHI",
        2: "Post-COVID"
    },
}

# ============================================================================
# NUMERIC COLUMNS
# ============================================================================

NUMERIC_COLS: List[str] = [
    "Postnatalgunemzirme", "kacıncıgundesutdestegibaslandı",
    "varsataburculuktakaçölçek", "sutdestegivarsakacolcek",
    "kacgunogkullandi", "taburculuktamamamiktari", "beslenmetotalitaburculuk",
    "aldıgımamamiktari3.gun", "beslenmetotali3.gun", "beslenme2.gunannesutucc",
    "beslenmemamamiktarı2.guncc", "beslenmetotali2.gün",
    "aldığıannesütü_ilkgün", "aldığımamamiktari1.gün",
    "bironcekibebegikacayemzirdi", "anneyasi", "gebelikhaftası",
    "dogumagirligi(gram)"
]

# ============================================================================
# PUBLICATION-FRIENDLY DISPLAY NAMES
# ============================================================================

RENAME_FOR_PLOT: Dict[str, str] = {
    # Primary outcome
    "taburculuk_beslenmeturu": "Feeding type at discharge",
    # Infant characteristics
    "tanı_gruplu": "Diagnosis (grouped)",
    "dogum_agırlıgı_gruplu": "Birth weight (grouped)",
    "gebelik_haftası_gruplu": "Gestational age (grouped)",
    "dogumagirligi(gram)": "Birth weight (g)",
    "takipilkgün_kilo_gram": "Weight on first follow-up day (g)",
    "kilo1.gun": "Weight on Day 1 (g)",
    "kilo2.gun": "Weight on Day 2 (g)",
    "kilo3.gun": "Weight on Day 3 (g)",
    "takibegirdigigun": "Day of admission to follow-up",
    "gebelikhaftası": "Gestational age (weeks)",
    "gebelikhaftagunu": "Gestational day within week",
    "anneyasi": "Maternal age",
    "yasayancocuksayisi": "Number of living children",
    "emzirdigicocuksayisi": "Number of breastfed children",
    "bironcekibebegikacayemzirdi": "Breastfeeding duration of previous child (months)",
    "aldığımamamiktari1.gün": "Formula intake on Day 1 (cc)",
    "aldığıannesütü_ilkgün": "Breast milk intake on Day 1 (cc)",
    "beslenmetotali2.gün": "Total feeding volume on Day 2 (cc)",
    "beslenmemamamiktarı2.guncc": "Formula intake on Day 2 (cc)",
    "beslenme2.gunannesutucc": "Breast milk intake on Day 2 (cc)",
    "beslenmetotali3.gun": "Total feeding volume on Day 3 (cc)",
    "aldıgımamamiktari3.gun": "Formula intake on Day 3 (cc)",
    "aldıgıannesütü3.gun": "Breast milk intake on Day 3 (cc)",
    "sutdestegivarsakacolcek": "Lactation support volume (cc)",
    # Discharge feeding variables
    "taburculuktamamamiktari": "Formula intake at discharge (cc)",
    "taburculukta_annesutu_111": "Breast milk present at discharge",
    "Ataburculuktaannesutu": "Breast milk at discharge (binary)",
    "taburculuktanasılbeslenmeyolu": "Feeding route at discharge",
    "emzirme_Taburculuk": "Breastfeeding at discharge",
    "aldığıannesütü_taburculuk": "Breast milk volume at discharge (cc)",
    "beslenmetotalitaburculuk": "Total feeding volume at discharge (cc)",
    "taburculuktaogvarmiyokmu": "Orogastric tube at discharge",
    
    # Day 1 feeding variables
    "ilk_gün_anne_sütü_1111": "Breast milk present on Day 1",
    "ilkgün_bebeğinannesütüalımı": "Breast milk intake on Day 1 (binary)",
    "ilk_gün_emzirme_111": "Breastfeeding on Day 1",
    "beslenmeninilkgunuverilisyolu": "Initial feeding route on Day 1",
    
    # Day 2-3 feeding routes
    "verilisyolu2.gun": "Feeding route on Day 2",
    "verilisyolu3gun": "Feeding route on Day 3",
    
    # Lactation support variables
    "annesutuemzirmeeğitimidurumu": "Lactation education status",
    "baslangictasutdestegi": "Lactation support at initiation",
    "sutdestegivarsakacolcek": "Lactation support volume (cc)",
    "varsataburculuktakaçölçek": "Lactation support at discharge (servings)",
    "kacıncıgundesutdestegibaslandı": "Day lactation support initiated",
    "taburculuktadestekcesidi": "Type of lactation support at discharge",
    "taburculukrtasutdestegivarmı": "Lactation support present at discharge",
    
    # Breast pump variables
    "Kullandıgıpompamarkasi": "Breast pump brand",
    "Kullandıgpompatipi": "Breast pump type",
    
    # Colostrum and breast problems
    "Kolostrumvarligi": "Colostrum present",
    "memesorunuyaşamadurumu": "Breast problem experienced",
    "memesorunuvarsa_tedavidekullanılanlar": "Breast problem treatment",
    "galaktokogkullanımı": "Galactagogue use",
    
    # Birth and infant characteristics
    "dogumyeri": "Place of birth",
    "dogumsekli": "Delivery method",
    "kacgunogkullandi": "Days of orogastric tube use",
    
    # Maternal variables
    "emzirdigicocuksayisi": "Number of previously breastfed children",
    "bironcekibebegikacayemzirdi": "Breastfeeding duration of previous child (months)",
    "yasayancocuksayisi": "Number of living children",
    "anne_hastalık_grup": "Maternal medical condition (grouped)",
    "anne_egitim_grup": "Maternal education (grouped)",
    "anneegitim": "Maternal education level",
    "anne_yaşı_grup": "Maternal age (grouped)",
    "anne_meslek_grup": "Maternal occupation (grouped)",
    "annemeslegi": "Maternal occupation",
    "bebekdostuoncesonra": "Baby-Friendly status (binary)",
    "gebeliktipi": "Pregnancy type",
    "gebelik_tipi_gruplu": "Pregnancy type (grouped)",
    "gebelik_34": "Gestational age category",
    
    # Postnatal timing
    "pntaburculuktarihi": "Postnatal discharge date (days)",
    "pntakibegirdigitarih": "Postnatal admission date (days)",
    "Postnatalgunemzirme": "Postnatal day of breastfeeding initiation",
    "takiptekacgun": "Days in follow-up",
    
    # Epoch and COVID variables
    "ikisiarası": "Epoch (COVID × BFHI)",
    "covid19sonrasi": "COVID-19 Period",
    "bebek_dostu_20temmuz2018": "Baby-Friendly Hospital Initiative",
    
    # Engineered features
    "eng_weight_per_week": "Weight gain per week (g/week)",
    "eng_elbw_flag": "Extremely low birth weight (<1000g)",
    "eng_very_preterm": "Very preterm (<32 weeks)",
    "eng_delta_vol_d1_d2": "Volume change Day 1→2 (cc)",
    "eng_delta_vol_d2_d3": "Volume change Day 2→3 (cc)",
    "eng_resilience_index": "Feeding resilience index",
    "eng_bm_ratio_d1": "Breast milk ratio on Day 1",
    "eng_bm_ratio_d2": "Breast milk ratio on Day 2",
    "eng_bm_ratio_d3": "Breast milk ratio on Day 3",
    "eng_lactation_momentum": "Lactation momentum score",
    "eng_severity_score": "Clinical severity score",
    "eng_neuro_barrier": "Neurological barrier to feeding",
    "eng_mat_healthcare_pro": "Maternal healthcare professional",
    "eng_mat_age_risk": "Maternal age risk category",
    "eng_pump_used": "Breast pump utilized",
    "eng_galactagogue": "Galactagogue used",
    "eng_tube_d1": "Tube feeding on Day 1",
    "eng_tube_d2": "Tube feeding on Day 2",
    "eng_tube_d3": "Tube feeding on Day 3",
    "eng_weaning_success": "Successful weaning from tube",
    
    # Variable codes
    "VAR00002": "Gestational age category (detailed)",
    "VAR00004": "Gestational age groups",
    "VAR00003": "Additional variable 3",
}

# List of all categorical columns
CAT_COLS: List[str] = list(CAT_LABELS_EN.keys())

# ============================================================================
# LEGACY COMPATIBILITY (Simplified labels for common variables)
# ============================================================================

# Keep these for backward compatibility with existing code
VARIABLE_LABELS = RENAME_FOR_PLOT.copy()
FEEDING_TYPE_LABELS = CAT_LABELS_EN["taburculuk_beslenmeturu"]
EPOCH_LABELS = CAT_LABELS_EN["ikisiarası"]
COVID_LABELS = CAT_LABELS_EN["covid19sonrasi"]
BFHI_LABELS = CAT_LABELS_EN["bebek_dostu_20temmuz2018"]

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_nicu_data(clean: bool = True, variables: Optional[List[str]] = None) -> pd.DataFrame:
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


def get_variable_label(variable_name: str) -> str:
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
    
    Examples
    --------
    >>> get_variable_label('taburculuk_beslenmeturu')
    'Feeding type at discharge'
    """
    return RENAME_FOR_PLOT.get(variable_name, variable_name)


def get_category_labels(variable_name: str) -> Dict[Any, str]:
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
    
    Examples
    --------
    >>> get_category_labels('covid19sonrasi')
    {0: 'No', 1: 'Yes'}
    >>> get_category_labels('anneegitim')
    {1: 'Illiterate', 2: 'Primary school', ...}
    """
    return CAT_LABELS_EN.get(variable_name, {})


def describe_variable(df: pd.DataFrame, variable: str):
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
    if df[variable].dtype in ['int64', 'float64'] and variable not in CAT_COLS:
        return df[variable].describe()
    else:
        counts = df[variable].value_counts().sort_index()
        labels = get_category_labels(variable)
        if labels:
            counts.index = counts.index.map(lambda x: labels.get(x, x))
        return counts


def check_missing_data(df: pd.DataFrame, variables: Optional[List[str]] = None) -> pd.DataFrame:
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
        variables = df.columns.tolist()
    
    missing_counts = df[variables].isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    
    report = pd.DataFrame({
        'Variable': variables,
        'Missing Count': missing_counts.values,
        'Missing %': missing_pct.values
    })
    
    return report[report['Missing Count'] > 0].sort_values('Missing %', ascending=False)


def is_categorical(variable: str) -> bool:
    """
    Check if a variable is categorical.
    
    Parameters
    ----------
    variable : str
        Variable name
    
    Returns
    -------
    bool
        True if variable is categorical
    """
    return variable in CAT_COLS


def is_numeric(variable: str) -> bool:
    """
    Check if a variable is numeric.
    
    Parameters
    ----------
    variable : str
        Variable name
    
    Returns
    -------
    bool
        True if variable is numeric
    """
    return variable in NUMERIC_COLS
