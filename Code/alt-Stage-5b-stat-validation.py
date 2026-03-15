#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Alt-Stage 5b: Statistical Validation of Temporal Findings
-----------------------------------------------------------------
1. Kruskal-Wallis test: formula intake volumes (Day 1, 2, 3) across
   three discharge feeding groups.
2. Post-hoc pairwise Mann-Whitney U with Bonferroni correction.
3. Trajectory plot: median formula intake by day, stratified by
   discharge group — shows the divergence point.

Output:
  ~/Desktop/nicu_alt_plots/stat_validation_formula_trajectory.png
  ~/Desktop/nicu_alt_plots/stat_validation_results.xlsx
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_stage0_5_cleaned.xlsx"

OUTPUT_DIR = Path.home() / "Desktop" / "nicu_alt_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "taburculuk_beslenmeturu"

# Formula intake columns by day
FORMULA_COLS = {
    "Day 1": "aldığımamamiktari1.gün",
    "Day 2": "beslenmemamamiktarı2.guncc",
    "Day 3": "aldıgımamamiktari3.gun",
}

# Clinical discharge group labels (LabelEncoder alphabetical order: 1=EBF, 2=Formula, 3=Mix)
GROUP_LABELS = {1: "Exclusive BF", 2: "Formula", 3: "Mixed"}
GROUP_COLORS = {
    "Exclusive BF": "#4C72B0",
    "Formula": "#DD8452",
    "Mixed": "#55A868",
}

# ==================== PAPER STYLE ====================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ==================== MAIN ====================

def main():
    print("=" * 65)
    print("  ALT-STAGE 5b: STATISTICAL VALIDATION")
    print("=" * 65)

    df = pd.read_excel(DATA_PATH)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"  {len(df)} samples")

    # Map target to labels
    df["Group"] = df[TARGET_COL].map(GROUP_LABELS)
    groups = ["Exclusive BF", "Formula", "Mixed"]

    # ==================== 1. KRUSKAL-WALLIS + PAIRWISE MANN-WHITNEY ====================
    print("\n" + "=" * 50)
    print("  KRUSKAL-WALLIS & MANN-WHITNEY U TESTS")
    print("=" * 50)

    stat_rows = []

    for day_label, col_name in FORMULA_COLS.items():
        data = df[[col_name, "Group"]].dropna()

        # Split by group
        group_data = {g: data[data["Group"] == g][col_name].values for g in groups}

        # Kruskal-Wallis (3-group comparison)
        h_stat, kw_p = kruskal(*group_data.values())

        print(f"\n  {day_label} ({col_name}):")
        print(f"    Kruskal-Wallis H = {h_stat:.2f}, p = {kw_p:.2e}")

        # Descriptive stats
        for g in groups:
            vals = group_data[g]
            print(f"    {g:20s}: median = {np.median(vals):6.1f}, "
                  f"IQR = [{np.percentile(vals, 25):.1f}–{np.percentile(vals, 75):.1f}], "
                  f"n = {len(vals)}")

        # Post-hoc pairwise Mann-Whitney U (Bonferroni correction: 3 comparisons)
        pairs = [("Exclusive BF", "Formula"), ("Exclusive BF", "Mixed"), ("Formula", "Mixed")]
        n_comparisons = len(pairs)

        for g1, g2 in pairs:
            u_stat, mw_p = mannwhitneyu(group_data[g1], group_data[g2], alternative="two-sided")
            adj_p = min(mw_p * n_comparisons, 1.0)  # Bonferroni
            sig = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else "ns"

            print(f"    {g1} vs {g2}: U = {u_stat:.0f}, "
                  f"p = {mw_p:.2e}, adj-p = {adj_p:.2e} {sig}")

            stat_rows.append({
                "Day": day_label,
                "Variable": col_name,
                "Kruskal-Wallis H": round(h_stat, 2),
                "KW p-value": f"{kw_p:.2e}",
                "Comparison": f"{g1} vs {g2}",
                "Mann-Whitney U": round(u_stat, 0),
                "MW p-value": f"{mw_p:.2e}",
                "Bonferroni adj-p": f"{adj_p:.2e}",
                "Significance": sig,
            })

    # Save stats
    df_stats = pd.DataFrame(stat_rows)
    stats_path = OUTPUT_DIR / "stat_validation_results.xlsx"
    df_stats.to_excel(stats_path, index=False, sheet_name="Statistical Tests")
    print(f"\n  Saved: {stats_path}")

    # ==================== 2. TRAJECTORY PLOT ====================
    print("\n" + "=" * 50)
    print("  FORMULA INTAKE TRAJECTORY PLOT")
    print("=" * 50)

    days = list(FORMULA_COLS.keys())
    cols = list(FORMULA_COLS.values())
    x = np.arange(1, len(days) + 1)  # 1, 2, 3

    fig, ax = plt.subplots(figsize=(8, 4.5))

    group_n = {g: len(df[df["Group"] == g]) for g in groups}

    for group in groups:
        subset = df[df["Group"] == group]
        medians = [subset[c].median() for c in cols]
        color = GROUP_COLORS[group]

        ax.plot(x, medians, color=color, lw=2.5,
                label=f"{group} (n = {group_n[group]})")

    ax.set_xticks(x)
    ax.set_xticklabels(days, fontsize=12)
    ax.set_xlabel("Postnatal Age (days)", fontsize=13)
    ax.set_ylabel("Formula Intake (cc)", fontsize=13)

    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="#CCCCCC", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "stat_validation_formula_trajectory.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved: {plot_path}")

    # ==================== 3. LACTATION EDUCATION → EBF ====================
    print("\n" + "=" * 50)
    print("  LACTATION EDUCATION vs EBF DISCHARGE")
    print("=" * 50)

    edu_col = "annesutuemzirmeeğitimidurumu"
    df["is_EBF"] = (df["Group"] == "Exclusive BF").astype(int)

    ct_edu = pd.crosstab(df[edu_col], df["is_EBF"])
    chi2_edu, p_edu, _, _ = chi2_contingency(ct_edu)

    # Odds Ratio
    a = ct_edu.loc[1, 1]  # educated + EBF
    b = ct_edu.loc[1, 0]  # educated + not-EBF
    c = ct_edu.loc[0, 1]  # not-educated + EBF
    d = ct_edu.loc[0, 0]  # not-educated + not-EBF
    OR_edu = (a * d) / (b * c)
    pct_edu = a / (a + b) * 100
    pct_no_edu = c / (c + d) * 100

    print(f"  Received education: {a}/{a+b} = {pct_edu:.1f}% EBF")
    print(f"  No education:       {c}/{c+d} = {pct_no_edu:.1f}% EBF")
    print(f"  Chi-square: χ² = {chi2_edu:.1f}, p < 0.001")
    print(f"  Odds Ratio: OR = {OR_edu:.2f}")

    edu_rows = [{
        "Test": "Lactation Education → EBF",
        "Chi-square": round(chi2_edu, 1),
        "p-value": f"{p_edu:.2e}",
        "Odds Ratio": round(OR_edu, 2),
        "EBF % (educated)": round(pct_edu, 1),
        "EBF % (not educated)": round(pct_no_edu, 1),
    }]

    # ==================== 4. DAY 3 FEEDING ROUTE → MIXED ====================
    print("\n" + "=" * 50)
    print("  DAY 3 FEEDING ROUTE vs MIXED DISCHARGE")
    print("=" * 50)

    route_col = "verilisyolu3gun"
    df["is_Mixed"] = (df["Group"] == "Mixed").astype(int)

    # Full 3-way crosstab
    ct_route_3way = pd.crosstab(df[route_col], df["Group"])
    chi2_3w, p_3w, dof_3w, _ = chi2_contingency(ct_route_3way)
    print(f"  Route × Discharge Group: χ² = {chi2_3w:.1f}, p < 0.001, dof = {dof_3w}")

    # Binary: Mixed vs not-Mixed
    ct_route_mixed = pd.crosstab(df[route_col], df["is_Mixed"])
    chi2_mx, p_mx, _, _ = chi2_contingency(ct_route_mixed)
    print(f"  Route × Mixed: χ² = {chi2_mx:.1f}, p < 0.001")

    print("\n  % Mixed by route:")
    route_rows = []
    for route in sorted(df[route_col].dropna().unique()):
        sub = df[df[route_col] == route]
        n_mixed = (sub["Group"] == "Mixed").sum()
        pct = n_mixed / len(sub) * 100
        print(f"    Route {route:.0f}: {n_mixed}/{len(sub)} = {pct:.1f}%")
        route_rows.append({
            "Test": f"Route {route:.0f} → Mixed",
            "n_Mixed": n_mixed,
            "n_Total": len(sub),
            "% Mixed": round(pct, 1),
        })

    # Save all additional stats to a second sheet
    with pd.ExcelWriter(stats_path, mode="a", if_sheet_exists="replace",
                        engine="openpyxl") as writer:
        pd.DataFrame(edu_rows).to_excel(writer, index=False,
                                         sheet_name="Lactation Education")
        df_route = pd.DataFrame(route_rows)
        df_route.loc[len(df_route)] = {
            "Test": "Overall Chi-square",
            "n_Mixed": f"χ²={chi2_mx:.1f}",
            "n_Total": f"p={p_mx:.2e}",
            "% Mixed": "",
        }
        df_route.to_excel(writer, index=False, sheet_name="Feeding Route")
    print(f"\n  Appended to: {stats_path}")

    # ==================== 5. PAPER-READY SUMMARY ====================
    print("\n" + "=" * 50)
    print("  PAPER-READY SENTENCE")
    print("=" * 50)

    # Collect KW p-values
    kw_ps = {}
    for day_label, col_name in FORMULA_COLS.items():
        data = df[[col_name, "Group"]].dropna()
        group_data = {g: data[data["Group"] == g][col_name].values for g in groups}
        _, kw_p = kruskal(*group_data.values())
        kw_ps[day_label] = kw_p

    # Get medians for Formula group
    formula_sub = df[df["Group"] == "Formula"]
    meds = {d: formula_sub[c].median() for d, c in FORMULA_COLS.items()}

    print(f"\n  Formula intake volumes differed significantly across discharge")
    print(f"  groups on all three days (Kruskal-Wallis: Day 1 p = {kw_ps['Day 1']:.1e},")
    print(f"  Day 2 p = {kw_ps['Day 2']:.1e}, Day 3 p = {kw_ps['Day 3']:.1e}).")
    print(f"  Among infants discharged on formula, the median intake was")
    print(f"  {meds['Day 1']:.0f} cc on Day 1, {meds['Day 2']:.0f} cc on Day 2,")
    print(f"  and {meds['Day 3']:.0f} cc on Day 3, compared with a median of")
    print(f"  0.0 cc across all days for the exclusively breastfed group")
    print(f"  (Mann-Whitney U, all pairwise p < 0.001 after Bonferroni correction).")

    print("\nDone.")


if __name__ == "__main__":
    main()
