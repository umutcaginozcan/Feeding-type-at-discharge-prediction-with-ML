# Recommended Statistical Analyses

## Overview

This document suggests appropriate statistical analyses for the NICU breastfeeding dataset based on variable types and research relevance. Use this as a guide for exploring associations and predictive models.

**Dataset:** 1,064 patients, 107 variables

---

## 🎯 Primary Outcome: Feeding Type at Discharge

`taburculuk_beslenmeturu` (0=Exclusive BF, 1=Formula, 2=Mixed)

All analyses below examine associations with this outcome unless otherwise specified.

---

## 📊 Categorical × Categorical Analyses (Chi-Square Tests)

### 1. Maternal Demographic Factors

| Variable | Label | Categories | Research Question |
|----------|-------|------------|-------------------|
| **anne_yaşı_grup** | Maternal age group | \u003c18, 18-35, \u003e35 years | Do younger/older mothers have different feeding outcomes? |
| **anne_egitim_grup** | Maternal education (grouped) | Illiterate, Primary/Secondary, High school+ | Does education level affect breastfeeding success? |
| **anneegitim** | Maternal education (detailed) | 7 levels (Illiterate → Graduate+) | More granular education analysis |
| **anne_meslek_grup** | Maternal occupation (grouped) | Homemaker, Healthcare, Other | Does occupation type influence feeding? |
| **annemeslegi** | Maternal occupation (detailed) | 9 categories | Which specific occupations support breastfeeding? |

**Clinical Relevance:** Identifying demographic risk factors for suboptimal feeding outcomes

---

### 2. Maternal Health & Obstetric Factors

| Variable | Label | Categories | Research Question |
|----------|-------|------------|-------------------|
| **anne_hastalık_grup** | Maternal health conditions | None, Hypothyroidism, Diabetes, HTN, etc. (11 categories) | Do maternal conditions affect feeding type? |
| **dogumsekli** | Delivery method | Vaginal, C-section | Does delivery mode impact breastfeeding? |
| **gebelik_tipi_gruplu** | Pregnancy type (grouped) | Singleton, Multiple | Do multiples have different feeding patterns? |
| **gebeliktipi** | Pregnancy type (detailed) | Singleton, Twin, Triplet | Specific multiple pregnancy analysis |

**Clinical Relevance:** Understanding how maternal health and delivery factors influence feeding

---

### 3. Infant Clinical Factors

| Variable | Label | Categories | Research Question |
|----------|-------|------------|-------------------|
| **tanı_gruplu** | Diagnosis group | 19 diagnostic categories | Does neonatal diagnosis affect feeding outcome? |
| **dogum_agırlıgı_gruplu** | Birth weight group | \u003c1000g (ELBW), 1000-1499g (VLBW), 1500-2500g (LBW), 2501-4000g, \u003e4000g | How does birth weight category relate to feeding? |
| **gebelik_34** | Gestational age | \u003c34 weeks, ≥34 weeks | Is prematurity (\u003c34 weeks) associated with feeding type? |
| **VAR00004** | Gestational age (detailed) | 5 categories (\u003c28 → ≥37 weeks) | More granular gestational age analysis |
| **gebelik_haftası_gruplu** | GA grouped | Categories based on prematurity | Alternative GA grouping |
| **cinsiyeti** | Infant sex | Female, Male | Does sex affect feeding outcomes? |

**Clinical Relevance:** Most important infant factors predicting feeding success

**Priority:** HIGH - these are core clinical predictors

---

### 4. Temporal & Policy Factors ✅ *Already Analyzed*

| Variable | Label | Status |
|----------|-------|--------|
| **covid19sonrasi** | COVID-19 period | ✅ Done - Strong association (V=0.44) |
| **bebek_dostu_20temmuz2018** | BFHI certification | ✅ Done - Weak association (V=0.18) |
| **ikisiarası** | Combined epochs | ✅ Done - Moderate association (V=0.31) |

---

### 5. Early Feeding Practice Factors (Days 1-3)

**Critical Variables - Strong Predictors Expected**

| Variable | Label | Categories | Research Question |
|----------|-------|------------|-------------------|
| **Kolostrumvarligi** | Colostrum presence | Absent, Present | Is early colostrum associated with EBF at discharge? |
| **ilk_gün_anne_sütü_1111** | Day 1 breast milk | Absent, Present | Does day 1 breast milk predict discharge feeding? |
| **ilk_gün_emzirme_111** | Day 1 breastfeeding | Absent, Present | KEY: Is day 1 breastfeeding critical for EBF? |
| **ilkgün_bebeğinannesütüalımı** | Day 1 infant breast milk intake | Absent, Present | Alternative day 1 BF variable |
| **beslenmeninilkgunuverilisyolu** | Day 1 feeding route | 9 categories (None, PO, OG, BF, etc.) | Which early feeding route predicts EBF? |
| **verilisyolu2.gun** | Day 2 feeding route | 9 categories | Does day 2 route matter? |
| **verilisyolu3gun** | Day 3 feeding route | 9 categories | Trajectory of feeding route changes |

**Priority:** VERY HIGH - these are modifiable early interventions

**Hypothesis:** Early breastfeeding initiation should strongly predict discharge EBF

---

### 6. Lactation Support Factors

**Potentially Modifiable Interventions**

| Variable | Label | Categories | Research Question |
|----------|-------|------------|-------------------|
| **annesutuemzirmeeğitimidurumu** | Lactation education | Absent, Present | Does education improve EBF rates? |
| **galaktokogkullanımı** | Galactagogue use | No, Yes | Is galactagogue use associated with feeding type? |
| **Kullandıgpompatipi** | Pump type | Electric, Manual, Hand expression | Does pump type affect outcomes? |
| **Kullandıgıpompamarkasi** | Pump brand | 8 brands | Are certain pumps associated with better outcomes? |
| **memesorunuyaşamadurumu** | Breast problems | No, Yes | How do breast problems affect feeding? |
| **memesorunuvarsa_tedavidekullanılanlar** | Breast problem treatment | 5 treatment types | Which treatments support continued BF? |
| **baslangictasutdestegi** | Initial lactation support | No, Yes | Does initial support affect discharge feeding? |
| **taburculuktadestekcesidi** | Type of discharge support | 4 types | Which support type is most effective? |

**Priority:** HIGH - these identify effective interventions

---

### 7. Discharge & Follow-up Factors

| Variable | Label | Categories | Research Question |
|----------|-------|------------|-------------------|
| **taburculukta_annesutu_111** | Breast milk at discharge | Absent, Present | Verification variable |
| **emzirme_Taburculuk** | Breastfeeding at discharge | Absent, Present | Direct breastfeeding vs expressed milk |
| **taburculuktanasılbeslenmeyolu** | Feeding method at discharge | 5 categories | Detailed feeding method |
| **taburculuktaogvarmiyokmu** | OG tube at discharge | No, Yes | Is OG feeding associated with outcomes? |

---

## 📈 Numeric × Categorical Analyses (t-tests / ANOVA)

### Compare numeric variables across feeding type groups

### 1. Birth & Growth Metrics

**Research Question:** Do feeding type groups differ in birth characteristics or growth patterns?

| Variable | Description | Test | Expected Finding |
|----------|-------------|------|------------------|
| **dogumagirligi(gram)** | Birth weight | ANOVA | EBF infants may have higher birth weights |
| **gebelikhaftası** | Gestational age | ANOVA | EBF associated with higher GA |
| **kilo1.gun** | Weight day 1 | ANOVA | Growth trajectory differences |
| **kilo2.gun** | Weight day 2 | ANOVA | Monitor weight loss patterns |
| **kilo3.gun** | Weight day 3 | ANOVA | Weight recovery by feeding type |
| **taburculukta_kilo_gram** | Discharge weight | ANOVA | Final weight outcomes |

**Priority:** HIGH - these are objective clinical outcomes

**Analysis Approach:**
1. One-way ANOVA for each variable
2. If significant, post-hoc pairwise comparisons (Bonferroni)
3. Report means ± SD for each feeding group
4. Calculate effect sizes (eta-squared)

---

### 2. Maternal Age & Parity

| Variable | Description | Test | Expected Finding |
|----------|-------------|------|------------------|
| **anneyasi** | Maternal age | ANOVA | Older mothers may have higher EBF rates |
| **yasayancocuksayisi** | Number of living children | ANOVA or K-W | Parity effects on feeding |
| **emzirdigicocuksayisi** | Number of BF children | ANOVA or K-W | Prior BF experience predicts success |
| **bironcekibebegikacayemzirdi** | Prior BF duration (months) | ANOVA or K-W | Longer prior BF → Higher EBF |

**Priority:** MEDIUM

---

### 3. Feeding Volumes (Days 1-3)

**Critical for understanding feeding patterns**

| Variable | Description | Test | Expected Finding |
|----------|-------------|------|------------------|
| **aldığıannesütü_ilkgün** | Breast milk intake day 1 (cc) | ANOVA | EBF group should have highest volumes |
| **aldığımamamiktari1.gün** | Formula intake day 1 (cc) | ANOVA | Formula group highest, EBF zero |
| **beslenme2.gunannesutucc** | Breast milk intake day 2 (cc) | ANOVA | Increasing BF volumes |
| **beslenmemamamiktarı2.guncc** | Formula intake day 2 (cc) | ANOVA | Trajectory analysis |
| **beslenmetotali2.gün** | Total feeding day 2 (cc) | ANOVA | Total intake comparison |
| **aldıgıannesütü3.gun** | Breast milk intake day 3 (cc) | ANOVA | Establishment of feeding |
| **aldıgımamamiktari3.gun** | Formula intake day 3 (cc) | ANOVA | Formula reduction patterns |
| **beslenmetotali3.gun** | Total feeding day 3 (cc) | ANOVA | Adequate intake verification |
| **beslenmetotalitaburculuk** | Total feeding at discharge (cc) | ANOVA | Final feeding volumes |

**Priority:** VERY HIGH - these show feeding establishment patterns

**Visualization:** Create line plots showing volume trajectories day 1→2→3 by feeding group

---

### 4. Hospital Stay Duration

| Variable | Calculation | Test | Expected Finding |
|----------|-------------|------|------------------|
| **Length of Stay** | taburculuktarihi - doğumtarihi (or takibegirdigitarih) | ANOVA or K-W | EBF infants may have shorter stays |

**Priority:** HIGH - important outcome and cost measure

---

### 5. Lactation Support Timing

| Variable | Description | Test |
|----------|-------------|------|
| **kacıncıgundesutdestegibaslandı** | Day lactation support started | ANOVA or K-W |
| **Postnatalgunemzirme** | Postnatal day breastfeeding started | ANOVA or K-W |

---

## 🔗 Numeric × Numeric Analyses (Correlations)

### 1. Birth Metrics Correlations

**Question:** How are birth characteristics related?

| Variable 1 | Variable 2 | Test | Expected |
|------------|------------|------|----------|
| Birth weight | Gestational age | Pearson | Strong positive (r ≈ 0.7-0.8) |
| Birth weight | Maternal age | Pearson or Spearman | Weak/moderate positive |
| Gestational age | Maternal age | Pearson or Spearman | Weak correlation |

---

### 2. Feeding Volume Trajectories

**Question:** Are day 1-3 volumes correlated (consistency)?

| Variable 1 | Variable 2 | Test |
|------------|------------|------|
| BF volume day 1 | BF volume day 2 | Pearson |
| BF volume day 2 | BF volume day 3 | Pearson |
| Formula day 1 | Formula day 2 | Pearson |
| BF volume day 3 | Discharge BF volume | Pearson |

**Use:** Create trajectory plots showing volume progressions

---

### 3. Weight Changes & Feeding Volumes

| Variable 1 | Variable 2 | Test |
|------------|------------|------|
| Total feeding day 1 | Weight change day 1→2 | Pearson |
| Total feeding day 2 | Weight change day 2→3 | Pearson |
| Cumulative BF volume | Discharge weight | Pearson |

---

## 🎯 Predictive Models (Regression)

### 1. **Primary Model: Predicting Exclusive Breastfeeding**

**Logistic Regression** (Binary outcome: EBF vs. Not EBF)

**Candidate Predictors:**

**Maternal Factors:**
- Maternal age (continuous)
- Maternal education (ordinal or dummy)
- Maternal occupation (dummy variables)
- Delivery method (vaginal vs C-section)

**Infant Factors:**
- Birth weight (continuous)
- Gestational age (continuous)
- Sex (male vs female)
- Diagnosis group (major categories)

**Temporal Factors:**
- COVID-19 period ✓
- BFHI certification ✓

**Early Feeding Practices (MOST IMPORTANT):**
- Day 1 breastfeeding (yes/no)
- Day 1 breast milk present (yes/no)
- Colostrum presence (yes/no)
- Day 1 feeding route

**Lactation Support:**
- Lactation education (yes/no)
- Galactagogue use (yes/no)
- Pump use (yes/no)

**Model Building Approach:**
1. Univariable screening: Test each predictor individually
2. Check for multicollinearity (VIF \u003c 5)
3. Build multivariable model with significant predictors
4. Stepwise selection or LASSO regularization
5. Validate with train/test split or cross-validation
6. Report AUC, calibration plot, Hosmer-Lemeshow test

**Priority:** VERY HIGH - this is your main predictive model

---

### 2. **Alternative Model: Multinomial Regression**

**Predicting 3-category outcome:** EBF vs Formula vs Mixed

**Same predictors as above, but:**
- Use multinomial logistic regression
- Interpret relative risk ratios for each category comparison
- More complex but retains full outcome information

---

### 3. **Linear Regression: Predicting Day 3 Breast Milk Volume**

**Outcome:** Breast milk intake on day 3 (continuous)

**Predictors:**
- Day 1 breast milk volume
- Day 2 breast milk volume
- Colostrum presence
- Day 1 breastfeeding
- Maternal education
- Lactation support
- Birth weight
- Gestational age

**Purpose:** Identify early predictors of adequate breast milk volumes

---

### 4. **Survival Analysis: Time to First Breastfeeding**

**Outcome:** Postnatalgunemzirme (day of first breastfeeding)

**Analysis:** Cox proportional hazards or Kaplan-Meier

**Predictors:**
- Maternal factors
- Infant clinical status
- BFHI period
- COVID period

**Purpose:** Identify factors promoting early breastfeeding initiation

---

## 📋 Analysis Priority Ranking

### Tier 1: Essential Analyses (Do First)

1. ✅ **Temporal factors** (COVID, BFHI, Epochs) - Already done
2. **Early feeding practices** × Feeding outcome (Chi-square)
   - Day 1 breastfeeding, colostrum, feeding routes
3. **Birth metrics** × Feeding outcome (ANOVA)
   - Birth weight, gestational age by feeding type
4. **Logistic regression:** Predicting exclusive breastfeeding

### Tier 2: Important Analyses (Do Second)

5. **Maternal demographics** × Feeding outcome
   - Education, occupation, age group
6. **Infant diagnosis** × Feeding outcome
7. **Lactation support factors** × Feeding outcome
8. **Feeding volume trajectories** (ANOVA + visualizations)
9. **Birth weight vs gestational age** (correlation)

### Tier 3: Exploratory Analyses (Do Third)

10. **Delivery method** × Feeding outcome
11. **Maternal health conditions** × Feeding outcome
12. **Pregnancy type** × Feeding outcome
13. **Length of stay** × Feeding type
14. **Pump type/brand** × Feeding outcome
15. **Multinomial regression** (3-category outcome)

### Tier 4: Advanced/Specialized

16. **Survival analysis** for time to first BF
17. **Mediation analysis:** Does lactation support mediate education effect?
18. **Interaction effects:** COVID × Education, BFHI × Diagnosis, etc.
19. **Propensity score matching:** Control for confounders

---

## 🔬 Analysis Templates

### Template 1: Categorical × Categorical

```python
from src.data import load_nicu_data, get_category_labels
from src.statistics import chi_square_test

df = load_nicu_data(clean=True, variables=['taburculuk_beslenmeturu', 'predictor_variable'])

results = chi_square_test(
    df,
    outcome_var='taburculuk_beslenmeturu',
    predictor_var='predictor_variable',
    outcome_labels={0: 'Exclusive BF', 1: 'Formula', 2: 'Mixed'},
    predictor_labels=get_category_labels('predictor_variable'),
    output_dir='outputs/statistics/Analysis_Name'
)
```

### Template 2: Numeric × Categorical

```python
from scipy.stats import f_oneway
from src.data import load_nicu_data

df = load_nicu_data(clean=True, variables=['taburculuk_beslenmeturu', 'numeric_variable'])

# Split by feeding type
ebf = df[df['taburculuk_beslenmeturu'] == 0]['numeric_variable']
formula = df[df['taburculuk_beslenmeturu'] == 1]['numeric_variable']
mixed = df[df['taburculuk_beslenmeturu'] == 2]['numeric_variable']

# ANOVA
f_stat, p_value = f_oneway(ebf, formula, mixed)

# Report means ± SD
print(f"EBF: {ebf.mean():.1f} ± {ebf.std():.1f}")
print(f"Formula: {formula.mean():.1f} ± {formula.std():.1f}")
print(f"Mixed: {mixed.mean():.1f} ± {mixed.std():.1f}")
print(f"F({2},{len(df)-3}) = {f_stat:.2f}, p = {p_value:.4f}")
```

### Template 3: Logistic Regression

```python
import pandas as pd
import statsmodels.api as sm
from src.data import load_nicu_data

# Load data
df = load_nicu_data(clean=True, variables=[...])

# Create binary outcome (EBF vs not)
df['ebf'] = (df['taburculuk_beslenmeturu'] == 0).astype(int)

# Prepare predictors
X = pd.get_dummies(df[predictor_columns], drop_first=True)
X = sm.add_constant(X)
y = df['ebf']

# Fit model
model = sm.Logit(y, X).fit()
print(model.summary())

# Odds ratios
odds_ratios = np.exp(model.params)
conf_int = np.exp(model.conf_int())
```

---

## 📊 Visualization Recommendations

### Publication Figures

**Figure 1: Temporal Trends** ✅ Already created
- COVID period, BFHI certification, Combined epochs

**Figure 2: Early Feeding Practices**
- Day 1 breastfeeding × Outcome
- Colostrum × Outcome
- Feeding route trajectories

**Figure 3: Birth Characteristics** 
- Birth weight distributions by feeding type (violin plots)
- Gestational age by feeding type (box plots)
- Scatter: Birth weight vs GA, colored by feeding type

**Figure 4: Feeding Volume Trajectories**
- Line plots: Day 1→2→3 volumes by feeding group
- Separate panels for breast milk, formula, total

**Figure 5: Multivariable Model**
- Forest plot of odds ratios from logistic regression
- ROC curve showing model performance

---

## ✅ Next Steps

1. **Start with Tier 1 analyses** using the templates above
2. **Document each analysis** with methodology and interpretation
3. **Create publication figures** following Nature format guidelines
4. **Build toward multivariable models** after univariable screening
5. **Prepare results for manuscript** tables and figures

---

**Questions? Consult `STATISTICS_GUIDE.md` for test selection and interpretation help!**
