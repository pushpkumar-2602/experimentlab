import pandas as pd
import numpy as np


def compute_ipw_weights(df: pd.DataFrame, treatment_col: str = "received_email",
                          propensity_col: str = "estimated_propensity") -> pd.DataFrame:
    """
    Computes inverse probability weights for each user based on their
    propensity score and actual treatment status.
    """
    df = df.copy()

    is_treated = df[treatment_col] == 1
    df["ipw_weight"] = np.where(
        is_treated,
        1 / df[propensity_col],
        1 / (1 - df[propensity_col]),
    )
    return df


def estimate_ipw_effect(df: pd.DataFrame, treatment_col: str = "received_email",
                          outcome_col: str = "converted") -> dict:
    """
    Estimates the treatment effect using IPW-weighted averages,
    instead of plain (naive) group averages.
    """
    df = compute_ipw_weights(df, treatment_col)

    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    # Weighted average: sum(outcome * weight) / sum(weight)
    # This is the IPW estimator for each group's "as if randomized" conversion rate.
    weighted_rate_treated = (treated[outcome_col] * treated["ipw_weight"]).sum() / treated["ipw_weight"].sum()
    weighted_rate_control = (control[outcome_col] * control["ipw_weight"]).sum() / control["ipw_weight"].sum()

    ipw_lift = weighted_rate_treated - weighted_rate_control

    naive_rate_treated = treated[outcome_col].mean()
    naive_rate_control = control[outcome_col].mean()
    naive_lift = naive_rate_treated - naive_rate_control

    return {
        "naive_rate_treated": naive_rate_treated,
        "naive_rate_control": naive_rate_control,
        "naive_lift": naive_lift,
        "ipw_rate_treated": weighted_rate_treated,
        "ipw_rate_control": weighted_rate_control,
        "ipw_lift": ipw_lift,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/confounded_results_with_propensity.csv")
    result = estimate_ipw_effect(df)

    true_effect = 0.04

    print("=" * 50)
    print("PROPENSITY SCORE / IPW CORRECTION REPORT")
    print("=" * 50)
    print(f"\nTRUE effect (known, since this is simulated data): {true_effect:.4f}")
    print(f"\nNaive lift (no correction):     {result['naive_lift']:.4f}   (bias: {result['naive_lift'] - true_effect:+.4f})")
    print(f"IPW-corrected lift:              {result['ipw_lift']:.4f}   (bias: {result['ipw_lift'] - true_effect:+.4f})")

    naive_bias = abs(result['naive_lift'] - true_effect)
    ipw_bias = abs(result['ipw_lift'] - true_effect)
    improvement = (1 - ipw_bias / naive_bias) * 100
    print(f"\nBias reduction from IPW correction: {improvement:.1f}%")