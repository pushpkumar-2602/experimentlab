import pandas as pd
import numpy as np
from scipy import stats

def analyze_continuous_metric(results_df: pd.DataFrame, group_col: str = "group",
                                outcome_col: str = "revenue") -> dict:
    """
    Compares a continuous metric (e.g. revenue) between treatment and control
    using Welch's t-test (does not assume equal variances between groups).
    """
    treatment = results_df[results_df[group_col] == "treatment"][outcome_col]
    control = results_df[results_df[group_col] == "control"][outcome_col]

    n_treatment = len(treatment)
    n_control = len(control)
    mean_treatment = treatment.mean()
    mean_control = control.mean()

    absolute_lift = mean_treatment - mean_control
    relative_lift = absolute_lift / mean_control

    # Welch's t-test: compares two means, allows unequal variances.
    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

    # 95% CI for the DIFFERENCE in means, using the same Welch-style
    # standard error the t-test itself is based on.
    se_treatment = treatment.var(ddof=1) / n_treatment
    se_control = control.var(ddof=1) / n_control
    se_diff = np.sqrt(se_treatment + se_control)

    # Welch-Satterthwaite degrees of freedom (needed for an accurate CI,
    # matching what ttest_ind uses internally).
    df_welch = (se_treatment + se_control) ** 2 / (
        (se_treatment ** 2) / (n_treatment - 1) + (se_control ** 2) / (n_control - 1)
    )

    t_critical = stats.t.ppf(0.975, df_welch)
    ci_low = absolute_lift - t_critical * se_diff
    ci_high = absolute_lift + t_critical * se_diff

    is_significant = p_value < 0.05

    return {
        "n_treatment": n_treatment,
        "n_control": n_control,
        "mean_treatment": mean_treatment,
        "mean_control": mean_control,
        "absolute_lift": absolute_lift,
        "relative_lift": relative_lift,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "is_significant": is_significant,
        "decision": "SHIP" if is_significant and absolute_lift > 0 else "DO NOT SHIP",
    }


def print_continuous_report(result: dict, unit: str = "$"):
    p = result['p_value']
    p_display = "< 0.0001" if p < 0.0001 else f"{p:.4f}"

    print("=" * 45)
    print("CONTINUOUS METRIC TEST REPORT (Welch's t-test)")
    print("=" * 45)
    print(f"Control:   n={result['n_control']:,}  mean={unit}{result['mean_control']:.4f}")
    print(f"Treatment: n={result['n_treatment']:,}  mean={unit}{result['mean_treatment']:.4f}")
    print(f"\nAbsolute lift: {unit}{result['absolute_lift']:.4f}")
    print(f"Relative lift: {result['relative_lift']:.2%}")
    print(f"95% CI on absolute lift: [{unit}{result['ci_95_low']:.4f}, {unit}{result['ci_95_high']:.4f}]")
    print(f"T-statistic: {result['t_stat']:.3f}")
    print(f"P-value: {p_display}")
    print(f"\nStatistically significant (p<0.05)? {result['is_significant']}")
    print(f"Decision: {result['decision']}")
    print("=" * 45)


if __name__ == "__main__":
    df = pd.read_csv("data/experiment_results.csv")
    result = analyze_continuous_metric(df, outcome_col="revenue")
    print_continuous_report(result)