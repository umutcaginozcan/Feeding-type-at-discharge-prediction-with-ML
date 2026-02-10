# Analysis: Breastfeeding Outcomes × Time Epochs

## Research Question

Is there an association between feeding type at discharge (Exclusive BF, Formula, Mixed) and different time periods (pre/post COVID-19, pre/post Baby-Friendly Hospital Initiative certification)?

## Variables

**Outcome:**
- `taburculuk_beslenmeturu` - Feeding type at discharge (0=Exclusive BF, 1=Formula, 2=Mixed, 3=Other)

**Predictors:**
- `ikisiarası` - Combined epoch (0=Pre-COVID+Pre-BFHI, 1=Pre-COVID+Post-BFHI, 2=Post-COVID)
- `covid19sonrasi` - COVID-19 period (0=Pre-COVID, 1=Post-COVID)
- `bebek_dostu_20temmuz2018` - Baby-Friendly Hospital Initiative (0=Pre-BFHI, 1=Post-BFHI)

## Methods

- **Statistical Test:** Chi-square test of independence
- **Effect Size:** Cramér's V
- **Post-hoc:** Pairwise comparisons with Bonferroni correction
- **Significance Level:** α = 0.05

## How to Run

```bash
python analyze.py
```

## Outputs

All results saved to: `outputs/statistics/Epochs_x_EBF/`

- Contingency tables (CSV)
- Proportions tables (CSV)
- Summary statistics (CSV)
- Pairwise comparisons (CSV)
- Visualization figures (PNG)

## Key Findings

1. **COVID-19 Period**: Strongest association (V=0.438)
   - Massive shift in feeding patterns post-COVID
   - Formula feeding: 88% → 12%

2. **Combined Epochs**: Moderate association (V=0.309)
   - Post-COVID epoch differs significantly from both pre-COVID epochs
   - Pre-COVID epochs (with/without BFHI) not significantly different from each other

3. **BFHI Certification**: Weak but significant association (V=0.176)
   - Mixed feeding increased after BFHI
   - Weaker effect than COVID-19

## Clinical Implications

The COVID-19 pandemic appears to have had a transformative effect on breastfeeding practices, while BFHI certification alone showed a weaker effect. This suggests investigating what practices changed during COVID-19 and considering maintaining beneficial changes.
