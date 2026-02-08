# Statistical Analysis Tutorial: Breastfeeding × Time Epochs

## 📚 What We Tested

**Research Question:**  
Is there an association between feeding type at discharge (Exclusive BF, Formula, Mixed) and different time periods (pre/post COVID-19, pre/post Baby-Friendly Hospital certification)?

---

## 🔍 Understanding the Statistics (In Plain Language)

### The Chi-Square Test (χ²)

**Think of it like this:**

Imagine you have a bag of marbles with 3 colors (feeding types) and you're wondering if the color distribution changes depending on when you draw them (time period).

**The chi-square test asks:**  
"Is the pattern of feeding types DIFFERENT between time periods, or is it just random chance?"

**How it works:**

1. **Observed frequencies** = What we actually see in the data
   - Example: 325 formula-fed babies in epoch 0

2. **Expected frequencies** = What we'd expect if there was NO relationship
   - Calculated based on overall proportions
   - Example: If 70% are formula-fed overall, we'd expect 70% in each epoch

3. **Chi-square statistic (χ²)** = Sum of all differences between observed and expected
   - Large χ² = Big differences = Likely a real association
   - Small χ² = Small differences = Probably just chance

4. **P-value** = Probability that the differences are due to random chance
   - p < 0.05 = Less than 5% chance it's random → **SIGNIFICANT** ✓
   - p ≥ 0.05 = More than 5% chance it's random → Not significant

---

### Cramér's V (Effect Size)

**Chi-square tells us IF there's an association, but not HOW STRONG it is.**

Cramér's V answers: "How strong is this relationship?"

**Scale:**
- 0.00-0.10: Negligible (basically no relationship)
- 0.10-0.30: Weak (small effect)
- 0.30-0.50: Moderate (medium effect)  
- 0.50+: Strong (large effect)

**Analogy:**  
- Chi-square is like a smoke detector (detects IF there's smoke)
- Cramér's V is like measuring how much smoke (HOW MUCH smoke)

---

### Post-Hoc Pairwise Comparisons

When you have more than 2 groups (like our 3 epochs), and the overall test is significant, you need to ask:

**"Which specific pairs of groups are different?"**

**Bonferroni Correction:**
- When doing multiple comparisons, we adjust the p-value threshold
- Prevents "false positives" from testing many pairs
- Formula: Adjusted p = Original p × Number of comparisons

**Example:**
- 3 epochs → 3 pairwise comparisons (0 vs 1, 0 vs 2, 1 vs 2)
- Multiplying p-value by 3 makes it harder to reach significance
- More conservative, more reliable

---

## 📊 Our Results

### 1. **Epoch (COVID × BFHI) Analysis**

**Finding:** HIGHLY SIGNIFICANT association (p < 0.001)  
**Effect Size:** Moderate (V = 0.309)

**What the numbers tell us:**

| Feeding Type | Pre-COVID + Pre-BFHI | Pre-COVID + Post-BFHI | Post-COVID |
|--------------|---------------------|----------------------|------------|
| Formula      | 325 (43.5%)         | 335 (44.8%)          | 87 (11.6%) |
| Mixed        | 74 (26.4%)          | 60 (21.4%)           | 146 (52.1%)|

**Key Insights:**
- **Post-COVID period is VERY different** from pre-COVID periods
- Formula feeding: Dropped from ~44% to 12% in post-COVID
- Mixed feeding: Jumped from ~24% to 52% in post-COVID
- Pre-COVID periods (with/without BFHI): Not significantly different from each other

**Why this matters:**
The COVID-19 pandemic appears to have shifted feeding patterns substantially, while BFHI certification alone (pre-COVID) didn't create a big change.

---

### 2. **COVID-19 Period Analysis**

**Finding:** HIGHLY SIGNIFICANT association (p < 0.001)  
**Effect Size:** Moderate (V = 0.438) — **Strongest effect!**

| Feeding Type | Pre-COVID | Post-COVID |
|--------------|-----------|------------|
| Formula      | 660 (88.4%) | 87 (11.6%) |
| Mixed        | 134 (47.9%) | 146 (52.1%) |

**Key Insights:**
- **Massive shift** in feeding patterns after COVID-19
- Formula feeding: Predominant pre-COVID (88%) → Minority post-COVID (12%)
- Mixed feeding: Nearly balanced pre/post COVID
- This is the **strongest association** we found (V = 0.438)

**Possible explanations:**
- COVID-19 → More mothers staying with babies → More breastfeeding support
- Policy changes during pandemic
- Increased awareness of breastfeeding benefits during health crisis

---

### 3. **Baby-Friendly Hospital Certification**

**Finding:** HIGHLY SIGNIFICANT association (p < 0.001)  
**Effect Size:** Weak (V = 0.176)

| Feeding Type | Pre-BFHI | Post-BFHI |
|--------------|----------|-----------|
| Formula      | 325 (43.5%) | 422 (56.5%) |
| Mixed        | 74 (26.4%) | 206 (73.6%) |

**Key Insights:**
- BFHI certification **does** have an effect, but it's weaker than COVID-19
- Mixed feeding increased after BFHI (27% → 74%)
- The association exists, but it's not as dramatic

---

## 🧮 Statistical Formulas (For Reference)

### Chi-Square Statistic

```
χ² = Σ [(Observed - Expected)² / Expected]
```

Where:
- Observed = Actual count in each cell
- Expected = (Row total × Column total) / Grand total

### Cramér's V

```
V = √(χ² / (n × (min(r,c) - 1)))
```

Where:
- n = Total sample size
- r = Number of rows
- c = Number of columns

### Degrees of Freedom

```
df = (r - 1) × (c - 1)
```

---

## ✅ Assumptions Check

Chi-square test requires:

1. **Independence:** Each observation is independent
   - ✓ Each patient appears once

2. **Expected frequency:** All expected cells should be ≥ 5
   - ✓ All our expected frequencies were > 5
   - ✓ Chi-square test is valid

3. **Categorical data:** Variables are categorical
   - ✓ All variables are categorical (feeding type, time periods)

---

## 📈 Visualizations Created

For each analysis, we generated 3 plots:

1. **Heatmap:** Shows observed frequencies in each cell
2. **Stacked bar chart:** Shows proportion breakdown by time period
3. **Side-by-side bar chart:** Shows time period distribution for each feeding type

All saved to: `outputs/statistics/Epochs x EBF/`

---

## 🎯 Clinical Interpretation

### Main Takeaway:

**The COVID-19 pandemic had a much stronger effect on feeding patterns than the Baby-Friendly Hospital Initiative certification alone.**

### Recommendations:

1. **Investigate what changed during COVID-19**
   - What policies were implemented?
   - How did maternal-infant contact protocols change?
   - Were there staffing or support changes?

2. **Learn from the post-COVID period**
   - Identify successful practices
   - Consider maintaining beneficial changes post-pandemic

3. **Strengthen BFHI implementation**
   - While BFHI showed an effect, it's weaker than COVID changes
   - Opportunity to enhance BFHI practices to match COVID-era improvements

---

## 📝 How to Report These Results

**Example (in a paper):**

> "We examined the association between feeding type at discharge and three time-related variables using chi-square tests of independence. A highly significant association was found between feeding outcomes and the COVID-19 pandemic period (χ² = 203.67, p < 0.001, Cramér's V = 0.438), indicating a moderate-to-strong effect. Post-COVID feeding patterns differed substantially from pre-COVID patterns, with formula feeding decreasing from 88.4% to 11.6%. Baby-Friendly Hospital Initiative certification also showed a significant but weaker association (χ² = 32.96, p < 0.001, Cramér's V = 0.176). Post-hoc pairwise comparisons with Bonferroni correction revealed that the post-COVID epoch differed significantly from both pre-COVID epochs (p < 0.001), while the two pre-COVID epochs did not significantly differ from each other (p = 0.494)."

---

## 🔧 Files Generated

1. **Summary table:** `summary_all_tests.csv`
2. **Contingency tables:** One for each predictor
3. **Visualizations:** 3 plots for each predictor
4. **Pairwise comparisons:** `pairwise_comparisons.csv`

All saved to: `outputs/statistics/Epochs x EBF/`

---

## 📚 Further Reading

If you want to dive deeper:

- **Chi-square test:** Pearson, K. (1900). On the criterion that a given system of deviations...
- **Cramér's V:** Cramér, H. (1946). Mathematical Methods of Statistics
- **Bonferroni correction:** Dunn, O. J. (1961). Multiple comparisons among means

---

**Questions? Run the script again or modify it to test other associations!**
