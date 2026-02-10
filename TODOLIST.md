# NICU Breastfeeding Paper - Week Plan (Feb 10-17, 2026)

**Goal:** Complete all analyses and be ready to write the paper by Feb 17

---

## 📊 Phase 1: Descriptive Analyses (Days 1-2)

### [ ] 1.1 Sample Characteristics Table (Table 1)
- [ ] Overall cohort demographics (n=1,064)
  - [ ] Maternal age: mean ± SD, range
  - [ ] Maternal education: n (%) by category
  - [ ] Maternal occupation: n (%) by category
  - [ ] Gravida/parity: median (IQR)
- [ ] Infant characteristics
  - [ ] Birth weight: mean ± SD (grams)
  - [ ] Gestational age: mean ± SD (weeks), categories n (%)
  - [ ] Sex: n (%) male/female
  - [ ] Delivery method: n (%) vaginal/C-section
  - [ ] Diagnosis groups: n (%) by category
- [ ] Outcomes
  - [ ] Feeding at discharge: n (%) EBF/Mixed/Formula
  - [ ] Length of stay: median (IQR) days
  - [ ] BFHI success rate: n (%)

### [ ] 1.2 Epoch Comparison Table
- [ ] Characteristics by epoch (Pre-COVID-1, Pre-COVID-2, Post-COVID)
- [ ] Test epoch differences (ANOVA/Kruskal-Wallis for continuous, chi-square for categorical)

### [ ] 1.3 Feeding Volume Trajectories
- [ ] Day 1, 2, 3 volumes: mean ± SD by feeding type
- [ ] Breast vs. formula volumes over time

---

## 🔬 Phase 2: Statistical Analyses (Days 3-5) - PRIORITIZED

### **Priority 1: PRIMARY OUTCOMES** 

#### [ ] 2.1 Epochs × Feeding Type (MAIN FINDING)
- [ ] Chi-square test for association
- [ ] Cramér's V effect size
- [ ] Post-hoc pairwise comparisons (Bonferroni corrected)
- [ ] Create Figure 1: Heatmap with percentages
- [ ] Write results paragraph with proper reporting

#### [ ] 2.2 Epochs × EBF Success (BINARY OUTCOME)
- [ ] Chi-square test
- [ ] Odds ratios with 95% CI by epoch
- [ ] Create Figure 2: Bar chart with error bars
- [ ] Statistical markers (*, **, ***)

### **Priority 2: MATERNAL FACTORS**

#### [ ] 2.3 Maternal Education × EBF
- [ ] Chi-square test
- [ ] Trend test (if ordinal)
- [ ] Create contingency table
- [ ] Effect size (Cramér's V)

#### [ ] 2.4 Maternal Occupation × EBF
- [ ] Chi-square test
- [ ] Post-hoc comparisons
- [ ] Create visualization

#### [ ] 2.5 Maternal Age × EBF
- [ ] t-test or Mann-Whitney U (check normality)
- [ ] Cohen's d effect size
- [ ] Create box plot with p-value

### **Priority 3: SECONDARY ASSOCIATIONS**

#### [ ] 2.6 Length of Stay × Feeding Type
- [ ] Kruskal-Wallis test (non-normal distribution expected)
- [ ] Dunn's post-hoc tests
- [ ] Create box plot
- [ ] Report median (IQR) by group

#### [ ] 2.7 Length of Stay × Epochs
- [ ] Test differences across epochs
- [ ] Control for confounders (gestational age, birth weight)

#### [ ] 2.8 Birth Weight × Feeding Outcome
- [ ] One-way ANOVA
- [ ] Post-hoc comparisons
- [ ] Effect size (η²)

#### [ ] 2.9 Gestational Age × Feeding Outcome
- [ ] One-way ANOVA or Kruskal-Wallis
- [ ] Post-hoc tests

### **Priority 4: MULTIVARIATE ANALYSES**

#### [ ] 2.10 Logistic Regression: Predicting EBF
- [ ] Independent variables:
  - [ ] Epoch (categorical)
  - [ ] Maternal education (ordinal)
  - [ ] Maternal age (continuous)
  - [ ] Gestational age (continuous)
  - [ ] Day 1 breastfeeding initiation (binary)
- [ ] Report: OR, 95% CI, p-values
- [ ] Model fit: Pseudo-R², AUC-ROC
- [ ] Create Figure: Forest plot of ORs

#### [ ] 2.11 Multiple Comparisons Correction
- [ ] Apply Bonferroni or FDR correction
- [ ] Document corrected p-values
- [ ] Update significance markers

---

## 📝 Phase 3: Results Synthesis (Day 6)

### [ ] 3.1 Create All Tables
- [ ] Table 1: Sample characteristics
- [ ] Table 2: Epochs × Feeding outcomes
- [ ] Table 3: Maternal factors × EBF
- [ ] Table 4: Logistic regression results
- [ ] Format in medical journal style (mean ± SD, n (%), p-values)

### [ ] 3.2 Create All Figures
- [ ] Figure 1: Epochs × Feeding heatmap
- [ ] Figure 2: EBF rates by epoch (bar chart)
- [ ] Figure 3: Maternal education × EBF
- [ ] Figure 4: Length of stay by feeding type
- [ ] Figure 5: Logistic regression forest plot
- [ ] Ensure: 300 DPI, clear labels, statistical markers

### [ ] 3.3 Organize Outputs
- [ ] Move all final figures to `/paper/figures/`
- [ ] Move all final tables to `/paper/tables/`
- [ ] Create supplementary materials folder

---

## 📖 Phase 4: Narrative Structure (Day 7)

### [ ] 4.1 Results Section Outline
- [ ] **4.1.1 Sample Characteristics**
  - Opening: "A total of 1,064 NICU infants were included..."
  - Key demographics in narrative form
  - Reference Table 1
  
- [ ] **4.1.2 Primary Outcome: COVID-19 Impact**
  - Lead with main finding: "Feeding patterns differed significantly across epochs..."
  - Statistical results: χ², p-value, Cramér's V
  - Specific comparisons: Pre vs. Post COVID
  - Reference Figure 1
  
- [ ] **4.1.3 Maternal Factors**
  - Education, occupation, age effects
  - Statistical results for each
  - Reference Table 3, relevant figures
  
- [ ] **4.1.4 Secondary Outcomes**
  - Length of stay findings
  - Birth weight/gestational age effects
  - Any unexpected findings
  
- [ ] **4.1.5 Multivariate Analysis**
  - Independent predictors of EBF
  - Adjusted ORs with interpretation
  - Model performance
  - Reference Table 4, Figure 5

### [ ] 4.2 Methods Section (Statistical Analysis subsection)
- [ ] Describe all tests used and why
- [ ] State significance level (α = 0.05)
- [ ] Describe multiple testing corrections
- [ ] Software used (Python, scipy, statsmodels)
- [ ] Effect size measures reported

### [ ] 4.3 Discussion Section Framework
- [ ] **4.3.1 Principal Findings**
  - COVID-19 impact on feeding practices
  - Maternal factors importance
  
- [ ] **4.3.2 Context with Literature**
  - How findings compare to other studies
  - Unique contributions
  
- [ ] **4.3.3 Clinical Implications**
  - What this means for NICU practice
  - Policy recommendations
  
- [ ] **4.3.4 Strengths & Limitations**
  - Sample size, data quality
  - Generalizability concerns
  - Unmeasured confounders
  
- [ ] **4.3.5 Future Research**
  - Longitudinal follow-up
  - Mechanistic studies

### [ ] 4.4 Abstract (250 words)
- [ ] Background (1-2 sentences)
- [ ] Objectives (1 sentence)
- [ ] Methods (2-3 sentences)
- [ ] Results (3-4 sentences with key statistics)
- [ ] Conclusions (1-2 sentences)

---

## ✅ Pre-writing Checklist

### [ ] 5.1 Quality Checks
- [ ] All p-values exact (not just p<0.05)
- [ ] All effect sizes calculated
- [ ] All confidence intervals reported
- [ ] Figures high resolution (≥300 DPI)
- [ ] Tables properly formatted
- [ ] Statistical assumptions tested and documented

### [ ] 5.2 Organization
- [ ] `/paper/sections/` folder created with:
  - [ ] `01_abstract.md`
  - [ ] `02_introduction.md`
  - [ ] `03_methods.md`
  - [ ] `04_results.md`
  - [ ] `05_discussion.md`
  - [ ] `06_conclusion.md`
- [ ] All figures in `/paper/figures/`
- [ ] All tables in `/paper/tables/`

### [ ] 5.3 Documentation
- [ ] Analysis scripts commented
- [ ] Reproducibility notes
- [ ] Data dictionary updated

---

## 📅 Daily Schedule

- **Day 1 (Feb 10):** Descriptive analyses (1.1-1.3)
- **Day 2 (Feb 11):** Priority 1 stats (2.1-2.2)
- **Day 3 (Feb 12):** Priority 2 stats (2.3-2.5)
- **Day 4 (Feb 13):** Priority 3 stats (2.6-2.9)
- **Day 5 (Feb 14):** Priority 4 stats (2.10-2.11)
- **Day 6 (Feb 15):** Tables & figures finalization (3.1-3.3)
- **Day 7 (Feb 16):** Narrative structure (4.1-4.4)
- **Day 8 (Feb 17):** Quality checks & ready to write! (5.1-5.3)

---

**Next Action:** Start with descriptive analyses (Section 1.1)
