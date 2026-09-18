import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep

def analyze_ab_test(results_df: pd.DataFrame, group_col: str = "group",
                     outcome_col: str = "converted") -> dict:
    """
    Run a two-proportion z-test comparing treatment vs control conversion rates.
    Returns a dictionary summarizing lift, significance, and confidence interval.
    """
    treatment = results_df[results_df[group_col] == "treatment"][outcome_col]
    control = results_df[results_df[group_col] == "control"][outcome_col]

    n_treatment = len(treatment)
    n_control = len(control)
    conversions_treatment = treatment.sum()
    conversions_control = control.sum()

    rate_treatment = conversions_treatment / n_treatment
    rate_control = conversions_control / n_control

    # --- Lift ---
    absolute_lift = rate_treatment - rate_control
    relative_lift = absolute_lift / rate_control

    # --- Two-proportion z-test ---
    # This tests: "Is the difference between these two rates likely real,
    # or could it easily be random chance?"
    count = np.array([conversions_treatment, conversions_control])
    nobs = np.array([n_treatment, n_control])
    pooled_rate = count.sum() / nobs.sum()
    se_pooled = np.sqrt(pooled_rate * (1 - pooled_rate) * (1 / n_treatment + 1 / n_control))
    z_score = absolute_lift / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # two-tailed test

    # --- 95% Confidence interval for the DIFFERENCE in proportions ---
    ci_low, ci_high = confint_proportions_2indep(
        conversions_treatment, n_treatment,
        conversions_control, n_control,
        method="wald"
    )

    is_significant = p_value < 0.05

    return {
        "n_treatment": n_treatment,
        "n_control": n_control,
        "rate_treatment": rate_treatment,
        "rate_control": rate_control,
        "absolute_lift": absolute_lift,
        "relative_lift": relative_lift,
        "z_score": z_score,
        "p_value": p_value,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "is_significant": is_significant,
        "decision": "SHIP" if is_significant and absolute_lift > 0 else "DO NOT SHIP",
    }


def print_report(result: dict):
    print("=" * 40)
    print("A/B TEST REPORT")
    print("=" * 40)
    print(f"Control:   n={result['n_control']:,}  rate={result['rate_control']:.2%}")
    print(f"Treatment: n={result['n_treatment']:,}  rate={result['rate_treatment']:.2%}")
    print(f"\nAbsolute lift: {result['absolute_lift']:.2%}")
    print(f"Relative lift: {result['relative_lift']:.2%}")
    print(f"95% CI on absolute lift: [{result['ci_95_low']:.2%}, {result['ci_95_high']:.2%}]")
    print(f"Z-score: {result['z_score']:.3f}")
    p = result['p_value']
    p_display = "< 0.0001" if p < 0.0001 else f"{p:.4f}"
    print(f"P-value: {p_display}")
    print(f"\nStatistically significant (p<0.05)? {result['is_significant']}")
    print(f"Decision: {result['decision']}")
    print("=" * 40)


if __name__ == "__main__":
    df = pd.read_csv("data/experiment_results.csv")
    result = analyze_ab_test(df)
    print_report(result)