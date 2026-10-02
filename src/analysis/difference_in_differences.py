import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def compute_naive_before_after(df: pd.DataFrame, region_col: str = "region",
                                  treated_region: str = "west") -> dict:
    """
    The NAIVE approach: just compare the treated region's rate before vs after,
    ignoring whatever happened in the control region. This is what DiD corrects.
    """
    treated = df[df[region_col] == treated_region]

    before = treated[~treated["is_post_rollout"]]["conversion_rate"].mean()
    after = treated[treated["is_post_rollout"]]["conversion_rate"].mean()

    return {
        "before": before,
        "after": after,
        "naive_estimate": after - before,
    }


def compute_difference_in_differences(df: pd.DataFrame, region_col: str = "region",
                                        treated_region: str = "west",
                                        control_region: str = "east") -> dict:
    """
    The DiD approach: (treated_after - treated_before) - (control_after - control_before)
    """
    treated = df[df[region_col] == treated_region]
    control = df[df[region_col] == control_region]

    treated_before = treated[~treated["is_post_rollout"]]["conversion_rate"].mean()
    treated_after = treated[treated["is_post_rollout"]]["conversion_rate"].mean()
    control_before = control[~control["is_post_rollout"]]["conversion_rate"].mean()
    control_after = control[control["is_post_rollout"]]["conversion_rate"].mean()

    treated_change = treated_after - treated_before
    control_change = control_after - control_before
    did_estimate = treated_change - control_change

    return {
        "treated_before": treated_before,
        "treated_after": treated_after,
        "control_before": control_before,
        "control_after": control_after,
        "treated_change": treated_change,
        "control_change": control_change,
        "did_estimate": did_estimate,
    }


def check_parallel_trends(df: pd.DataFrame, region_col: str = "region",
                            treated_region: str = "west", control_region: str = "east") -> dict:
    """
    A basic parallel-trends sanity check: compares the PRE-ROLLOUT trend slope
    of each region. If they were already diverging before treatment started,
    DiD's core assumption is questionable.
    """
    pre_period = df[~df["is_post_rollout"]]

    trends = {}
    for region in [treated_region, control_region]:
        region_data = pre_period[pre_period[region_col] == region]
        slope = np.polyfit(region_data["day"], region_data["conversion_rate"], 1)[0]
        trends[region] = slope

    slope_difference = abs(trends[treated_region] - trends[control_region])

    return {
        "treated_pretrend_slope": trends[treated_region],
        "control_pretrend_slope": trends[control_region],
        "slope_difference": slope_difference,
        "trends_look_parallel": slope_difference < 0.001,
    }


def regression_based_did(df: pd.DataFrame) -> dict:
    """
    Estimates the DiD effect using OLS regression with an interaction term,
    which gives us a proper standard error, confidence interval, and p-value
    -- something our simple four-average approach couldn't provide.
    """
    model = smf.ols(
        "conversion_rate ~ is_treated_region * is_post_rollout",
        data=df
    ).fit()

    interaction_term = "is_treated_region[T.True]:is_post_rollout[T.True]"

    coef = model.params[interaction_term]
    ci_low, ci_high = model.conf_int().loc[interaction_term]
    p_value = model.pvalues[interaction_term]

    return {
        "did_estimate": coef,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "model_summary": model,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/did_data.csv")
    true_effect = 0.03

    print("=" * 50)
    print("PARALLEL TRENDS CHECK (pre-rollout period only)")
    print("=" * 50)
    trends = check_parallel_trends(df)
    print(f"West pre-trend slope:  {trends['treated_pretrend_slope']:.5f} per day")
    print(f"East pre-trend slope:  {trends['control_pretrend_slope']:.5f} per day")
    print(f"Slope difference:      {trends['slope_difference']:.5f}")
    print(f"Trends look parallel?  {trends['trends_look_parallel']}")

    print("\n" + "=" * 50)
    print("NAIVE vs. DIFFERENCE-IN-DIFFERENCES (simple averages)")
    print("=" * 50)
    naive = compute_naive_before_after(df)
    did = compute_difference_in_differences(df)

    print(f"\nTRUE effect (known, simulated): {true_effect:.4f}")
    print(f"\nNaive estimate (West before vs after only): {naive['naive_estimate']:.4f}  (bias: {naive['naive_estimate'] - true_effect:+.4f})")
    print(f"DiD estimate:                                 {did['did_estimate']:.4f}  (bias: {did['did_estimate'] - true_effect:+.4f})")

    naive_bias = abs(naive['naive_estimate'] - true_effect)
    did_bias = abs(did['did_estimate'] - true_effect)
    improvement = (1 - did_bias / naive_bias) * 100
    print(f"\nBias reduction from DiD: {improvement:.1f}%")

    print("\n" + "=" * 50)
    print("REGRESSION-BASED DiD (with proper statistics)")
    print("=" * 50)
    reg_result = regression_based_did(df)
    print(f"DiD estimate: {reg_result['did_estimate']:.4f}")
    print(f"95% CI: [{reg_result['ci_95_low']:.4f}, {reg_result['ci_95_high']:.4f}]")
    print(f"P-value: {reg_result['p_value']:.4f}")
    print(f"Statistically significant? {reg_result['is_significant']}")
    print(f"Bias vs true effect: {reg_result['did_estimate'] - true_effect:+.4f}")
