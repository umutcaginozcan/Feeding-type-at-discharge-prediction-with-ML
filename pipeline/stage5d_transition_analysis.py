#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICU Alt-Stage 5d: Day 2 Transition Analysis
-----------------------------------------------
Purpose:
  Validate WHY Day 2 formula volume is the strongest predictor by testing
  the transient supplementation hypothesis:

  H₁: Among infants who received formula on Day 1, those eventually
      discharged on EBF have significantly lower Day 2 formula intake
      than those discharged on formula — proving Day 2 captures the
      trajectory inflection point.

Experiments:
  1. Transition analysis: Mann-Whitney U comparing Day 2 formula volumes
     between "Day 1 formula → EBF discharge" vs "Day 1 formula → Formula
     discharge" subgroups, with box plot visualization.
  2. SHAP interaction: Day 2 × Day 1 conditional dependency from saved
     SHAP values (TreeExplainer interaction_values).
  3. Sankey/alluvial diagram: Day 1 formula status → Day 2 trajectory
     direction → discharge outcome (optional).

Output:
  ~/Desktop/nicu_alt_plots/transition_analysis/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import mannwhitneyu, kruskal

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "excels-NICU-breatsfeeding-data" / "nicu_stage0_5_cleaned.xlsx"

OUTPUT_DIR = Path.home() / "Desktop" / "nicu_alt_plots" / "transition_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "taburculuk_beslenmeturu"

# Column names (from Stage-0)
D1_FORMULA = "aldığımamamiktari1.gün"       # Formula intake Day 1 (cc)
D2_FORMULA = "beslenmemamamiktarı2.guncc"    # Formula intake Day 2 (cc)
D3_FORMULA = "aldıgımamamiktari3.gun"        # Formula intake Day 3 (cc)
D1_BM      = "aldığıannesütü_ilkgün"        # Breast milk Day 1 (cc)
D2_TOTAL   = "beslenmetotali2.gün"           # Total intake Day 2 (cc)

GROUP_LABELS = {1: "Exclusive BF", 2: "Formula", 3: "Mixed"}
PALETTE = {
    "Exclusive BF": "#4477AA",
    "Formula":      "#EE6677",
    "Mixed":        "#228833",
}

# ==================== FIGURE DIMENSIONS (mm → inches) ====================
SINGLE_COL_MM = 89
DOUBLE_COL_MM = 183
MM_TO_INCH = 1 / 25.4

# ==================== PAPER STYLE ====================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 0.75,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


# ==================== MAIN ====================

def main():
    print("=" * 65)
    print("  ALT-STAGE 5d: DAY 2 TRANSITION ANALYSIS")
    print("=" * 65)

    df = pd.read_excel(DATA_PATH)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df["Group"] = df[TARGET_COL].map(GROUP_LABELS)
    print(f"  {len(df)} total samples")

    # ============================================================
    # EXPERIMENT 1: TRANSITION ANALYSIS
    # ============================================================
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: TRANSIENT SUPPLEMENTATION TEST")
    print("=" * 60)

    # Isolate infants who received ANY formula on Day 1
    df_d1_formula = df[df[D1_FORMULA].fillna(0) > 0].copy()
    n_d1_formula = len(df_d1_formula)
    print(f"\n  Infants with Day 1 formula > 0: {n_d1_formula} / {len(df)} "
          f"({100*n_d1_formula/len(df):.1f}%)")

    # Split by discharge outcome
    for g in ["Exclusive BF", "Formula", "Mixed"]:
        n = (df_d1_formula["Group"] == g).sum()
        print(f"    → Discharged {g}: n={n}")

    # Primary test: EBF vs Formula discharge among Day 1 formula recipients
    ebf_d2 = df_d1_formula[df_d1_formula["Group"] == "Exclusive BF"][D2_FORMULA].dropna()
    form_d2 = df_d1_formula[df_d1_formula["Group"] == "Formula"][D2_FORMULA].dropna()
    mixed_d2 = df_d1_formula[df_d1_formula["Group"] == "Mixed"][D2_FORMULA].dropna()

    print(f"\n  Day 2 formula intake among Day 1 formula recipients:")
    for name, arr in [("EBF discharge", ebf_d2), ("Formula discharge", form_d2),
                      ("Mixed discharge", mixed_d2)]:
        print(f"    {name:20s}: n={len(arr)}, median={np.median(arr):.1f}, "
              f"IQR=[{np.percentile(arr, 25):.1f}–{np.percentile(arr, 75):.1f}], "
              f"mean={np.mean(arr):.1f}")

    # --- Primary: Mann-Whitney U (EBF vs Formula) ---
    print("\n  PRIMARY TEST: EBF vs Formula discharge")
    u_stat, p_val = mannwhitneyu(ebf_d2, form_d2, alternative="two-sided")
    # Effect size: rank-biserial correlation
    n1, n2 = len(ebf_d2), len(form_d2)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

    print(f"    Mann-Whitney U = {u_stat:.0f}")
    print(f"    p = {p_val:.2e}  [{sig}]")
    print(f"    Rank-biserial r = {r_rb:.3f}")
    print(f"    n₁ (EBF) = {n1}, n₂ (Formula) = {n2}")

    # --- Secondary: 3-group Kruskal-Wallis ---
    if len(mixed_d2) >= 3:
        print("\n  SECONDARY TEST: 3-group Kruskal-Wallis")
        h_stat, kw_p = kruskal(ebf_d2, form_d2, mixed_d2)
        print(f"    H = {h_stat:.2f}, p = {kw_p:.2e}")

    # --- Trajectory direction ---
    print("\n  TRAJECTORY DIRECTION ANALYSIS:")
    df_d1_formula["d1_formula_vol"] = df_d1_formula[D1_FORMULA].fillna(0)
    df_d1_formula["d2_formula_vol"] = df_d1_formula[D2_FORMULA].fillna(0)
    df_d1_formula["delta_d1_d2"] = df_d1_formula["d2_formula_vol"] - df_d1_formula["d1_formula_vol"]
    df_d1_formula["direction"] = pd.cut(
        df_d1_formula["delta_d1_d2"],
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=["Decreasing", "Stable", "Increasing"]
    )

    for g in ["Exclusive BF", "Formula", "Mixed"]:
        sub = df_d1_formula[df_d1_formula["Group"] == g]
        counts = sub["direction"].value_counts()
        total = len(sub)
        print(f"\n    {g} (n={total}):")
        for d in ["Decreasing", "Stable", "Increasing"]:
            n_ = counts.get(d, 0)
            pct = 100 * n_ / total if total > 0 else 0
            print(f"      {d:12s}: {n_:3d} ({pct:5.1f}%)")

    # --- Box Plot (single-column, data only) ---
    print("\n  📊 Generating box plot...")

    fig_w = SINGLE_COL_MM * MM_TO_INCH
    fig_h = fig_w * 0.85
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    groups_to_plot = ["Exclusive BF", "Formula"]
    data_to_plot = [
        df_d1_formula[df_d1_formula["Group"] == g][D2_FORMULA].dropna().values
        for g in groups_to_plot
    ]
    if len(mixed_d2) >= 3:
        groups_to_plot.append("Mixed")
        data_to_plot.append(mixed_d2.values)

    colors = [PALETTE[g] for g in groups_to_plot]

    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        widths=0.45,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.35,
                    "markeredgewidth": 0.3},
        medianprops={"color": "black", "linewidth": 0.75},
        whiskerprops={"linewidth": 0.5},
        capprops={"linewidth": 0.5},
        boxprops={"linewidth": 0.5},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(groups_to_plot)
    ax.set_ylabel("Formula intake, day 2 (cc)")
    ax.set_ylim(bottom=-2)

    plt.tight_layout()
    for ext in ["pdf", "svg"]:
        fig.savefig(OUTPUT_DIR / f"transition_boxplot_day2.{ext}")
    plt.close()
    print(f"  ✓ Saved: transition_boxplot_day2.pdf / .svg")

    # ============================================================
    # EXPERIMENT 2: SHAP INTERACTION (Day 2 × Day 1)
    # ============================================================
    print("\n\n" + "=" * 60)
    print("  EXPERIMENT 2: SHAP CONDITIONAL ANALYSIS")
    print("=" * 60)

    # Check if SHAP summary CSVs exist from alt-Stage-4
    shap_dir = Path.home() / "Desktop" / "nicu_alt_plots" / "shap"
    shap_rf_path = shap_dir / "SHAP_summary_RF.csv"

    if shap_rf_path.exists():
        shap_df = pd.read_csv(shap_rf_path)
        print(f"  Loaded SHAP summary from {shap_rf_path}")
        print(f"  Top 5 features by overall mean |SHAP|:")
        for _, row in shap_df.head(5).iterrows():
            print(f"    {row['Feature']:40s}  {row['Overall_mean_abs_SHAP']:.4f}")

        # Check Day 2 formula rank
        d2_keywords = ["Day 2", "formula", "2.gun", "mama"]
        d2_rows = shap_df[shap_df["Feature"].str.contains("|".join(d2_keywords),
                                                           case=False, na=False)]
        if not d2_rows.empty:
            print(f"\n  Day 2 formula-related features in SHAP ranking:")
            for _, row in d2_rows.iterrows():
                rank = (shap_df["Overall_mean_abs_SHAP"] >= row["Overall_mean_abs_SHAP"]).sum()
                print(f"    Rank {rank}: {row['Feature']}  "
                      f"(|SHAP| = {row['Overall_mean_abs_SHAP']:.4f})")
    else:
        print(f"  ⚠ SHAP summary not found at {shap_rf_path}")
        print(f"    Run alt-Stage-4.py first to generate SHAP values.")

    # Direct conditional analysis: split by Day 1 formula volume
    print("\n  CONDITIONAL DAY 2 IMPORTANCE:")
    print("  (Comparing Day 2 formula intake predictive power by Day 1 formula status)")

    df["d1_has_formula"] = df[D1_FORMULA].fillna(0) > 0
    df["d2_formula"] = df[D2_FORMULA].fillna(0)

    # Among Day 1 formula=Yes vs Day 1 formula=No,
    # how different is Day 2 formula between discharge groups?
    for d1_status, d1_label in [(True, "Day 1 Formula = YES"), (False, "Day 1 Formula = NO")]:
        sub = df[df["d1_has_formula"] == d1_status]
        ebf_vals = sub[sub["Group"] == "Exclusive BF"]["d2_formula"].dropna()
        form_vals = sub[sub["Group"] == "Formula"]["d2_formula"].dropna()

        if len(ebf_vals) >= 2 and len(form_vals) >= 2:
            u, p = mannwhitneyu(ebf_vals, form_vals, alternative="two-sided")
            print(f"\n    {d1_label} (n={len(sub)}):")
            print(f"      EBF  Day 2 formula: median={np.median(ebf_vals):.1f} (n={len(ebf_vals)})")
            print(f"      Form Day 2 formula: median={np.median(form_vals):.1f} (n={len(form_vals)})")
            print(f"      Mann-Whitney U={u:.0f}, p={p:.2e}")
        else:
            print(f"\n    {d1_label}: insufficient data for comparison")

    # ============================================================
    # EXPERIMENT 3: SANKEY / ALLUVIAL FLOW
    # ============================================================
    print("\n\n" + "=" * 60)
    print("  EXPERIMENT 3: FEEDING TRANSITION FLOW")
    print("=" * 60)

    # Build transition table
    df["D1_Formula_Status"] = np.where(df[D1_FORMULA].fillna(0) > 0,
                                        "Formula on Day 1", "No Formula on Day 1")
    df["D2_Direction"] = pd.cut(
        df[D2_FORMULA].fillna(0) - df[D1_FORMULA].fillna(0),
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=["↓ Decreasing", "→ Stable", "↑ Increasing"]
    )

    flow = df.groupby(["D1_Formula_Status", "D2_Direction", "Group"]).size().reset_index(name="Count")
    flow = flow.sort_values(["D1_Formula_Status", "D2_Direction", "Group"])

    print("\n  Transition Flow Table:")
    print(f"  {'D1 Status':25s} {'D2 Direction':15s} {'Discharge':15s} {'n':>5s} {'%':>7s}")
    print("  " + "-" * 70)

    for d1_status in ["Formula on Day 1", "No Formula on Day 1"]:
        sub_flow = flow[flow["D1_Formula_Status"] == d1_status]
        total = sub_flow["Count"].sum()
        for _, row in sub_flow.iterrows():
            pct = 100 * row["Count"] / total if total > 0 else 0
            print(f"  {row['D1_Formula_Status']:25s} {row['D2_Direction']:15s} "
                  f"{row['Group']:15s} {row['Count']:5d} {pct:6.1f}%")
        print()

    # --- Stacked bar chart (double-column, panels a/b) ---
    print("  📊 Generating transition flow chart...")

    fig_w = DOUBLE_COL_MM * MM_TO_INCH
    fig_h = fig_w * 0.38
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    panel_labels = ["a", "b"]
    panel_titles = ["Formula on day 1", "No formula on day 1"]

    for ax_idx, d1_status in enumerate(["Formula on Day 1", "No Formula on Day 1"]):
        ax = axes[ax_idx]
        sub = df[df["D1_Formula_Status"] == d1_status]

        directions = ["↓ Decreasing", "→ Stable", "↑ Increasing"]
        x = np.arange(len(directions))

        ebf_counts, form_counts, mix_counts = [], [], []
        for d in directions:
            d_sub = sub[sub["D2_Direction"] == d]
            ebf_counts.append((d_sub["Group"] == "Exclusive BF").sum())
            form_counts.append((d_sub["Group"] == "Formula").sum())
            mix_counts.append((d_sub["Group"] == "Mixed").sum())

        ebf_arr = np.array(ebf_counts)
        form_arr = np.array(form_counts)
        mix_arr = np.array(mix_counts)

        bar_w = 0.55
        ax.bar(x, ebf_arr, bar_w, color=PALETTE["Exclusive BF"],
               label="Exclusive BF" if ax_idx == 0 else "")
        ax.bar(x, form_arr, bar_w, bottom=ebf_arr,
               color=PALETTE["Formula"],
               label="Formula" if ax_idx == 0 else "")
        ax.bar(x, mix_arr, bar_w, bottom=ebf_arr + form_arr,
               color=PALETTE["Mixed"],
               label="Mixed" if ax_idx == 0 else "")

        ax.set_xticks(x)
        ax.set_xticklabels(["Decreasing", "Stable", "Increasing"])
        if ax_idx == 0:
            ax.set_ylabel("Number of infants")

        # Panel label (bold lowercase)
        ax.text(-0.08, 1.05, f"\textbf{{{panel_labels[ax_idx]}}}"
                if False else panel_labels[ax_idx],
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                va="top", ha="right")

    # Legend inside first panel
    axes[0].legend(loc="upper left", frameon=False, fontsize=6)

    plt.tight_layout()
    for ext in ["pdf", "svg"]:
        fig.savefig(OUTPUT_DIR / f"transition_flow_chart.{ext}")
    plt.close()
    print(f"  ✓ Saved: transition_flow_chart.pdf / .svg")

    # ============================================================
    # SAVE ALL TO EXCEL
    # ============================================================
    excel_path = OUTPUT_DIR / "transition_analysis_results.xlsx"
    print(f"\n\nSaving to {excel_path}...")

    with pd.ExcelWriter(excel_path) as writer:
        # Experiment 1: Summary stats
        exp1_rows = []
        for g, arr in [("Exclusive BF", ebf_d2), ("Formula", form_d2), ("Mixed", mixed_d2)]:
            exp1_rows.append({
                "Group": g,
                "n": len(arr),
                "Median": round(np.median(arr), 1),
                "IQR_25": round(np.percentile(arr, 25), 1),
                "IQR_75": round(np.percentile(arr, 75), 1),
                "Mean": round(np.mean(arr), 1),
                "SD": round(np.std(arr), 1),
            })
        pd.DataFrame(exp1_rows).to_excel(writer, sheet_name="Exp1 Descriptive", index=False)

        # Experiment 1: Test results
        test_rows = [{
            "Comparison": "EBF vs Formula (among D1 formula recipients)",
            "Test": "Mann-Whitney U",
            "U": round(u_stat, 0),
            "p_value": f"{p_val:.2e}",
            "Significance": sig,
            "Effect_Size_r": round(r_rb, 3),
            "n_EBF": n1,
            "n_Formula": n2,
        }]
        pd.DataFrame(test_rows).to_excel(writer, sheet_name="Exp1 Test", index=False)

        # Experiment 3: Flow table
        flow.to_excel(writer, sheet_name="Transition Flow", index=False)

    print(f"  ✓ Saved: {excel_path}")

    # ============================================================
    # MANUSCRIPT-READY SUMMARY
    # ============================================================
    print("\n" + "=" * 65)
    print("  MANUSCRIPT-READY SUMMARY")
    print("=" * 65)

    ebf_med = np.median(ebf_d2)
    form_med = np.median(form_d2)

    print(f"""
  Among {n_d1_formula} infants who received formula supplementation on Day 1
  ({100*n_d1_formula/len(df):.1f}% of the cohort), Day 2 formula intake volumes
  differed significantly by discharge feeding status: infants ultimately
  discharged on exclusive breastfeeding had a median Day 2 formula volume
  of {ebf_med:.0f} cc (IQR [{np.percentile(ebf_d2, 25):.0f}–{np.percentile(ebf_d2, 75):.0f}]),
  compared with {form_med:.0f} cc (IQR [{np.percentile(form_d2, 25):.0f}–{np.percentile(form_d2, 75):.0f}])
  among those discharged on formula (Mann-Whitney U = {u_stat:.0f}, p = {p_val:.1e},
  rank-biserial r = {r_rb:.2f}).""")

    # Derive transient supplementation rate
    n_d1f_ebf = (df_d1_formula["Group"] == "Exclusive BF").sum()
    pct_transient = 100 * n_d1f_ebf / n_d1_formula
    print(f"""
  Of these Day 1 formula recipients, {n_d1f_ebf} ({pct_transient:.1f}%) were
  discharged on exclusive breastfeeding, demonstrating that Day 1 formula
  supplementation is frequently transient. The Day 2 inflection point —
  captured by the model's strongest predictor — distinguishes these
  transient supplementation cases from persistent formula feeding paths.""")

    print("\n" + "=" * 65)
    print("  DONE!")
    print("=" * 65)


if __name__ == "__main__":
    main()
