import pandas as pd
import numpy as np
from src.analysis.ab_test_stats import analyze_ab_test, print_report

def apply_cuped(df: pd.DataFrame, outcome_col: str = "converted",
                covariate_col: str = "pre_experiment_purchases") -> pd.DataFrame:
    """
    Applies CUPED variance reduction to the outcome column.
    Adds a new column: '{outcome_col}_cuped'
    """
    df = df.copy()

    covariate_mean = df[covariate_col].mean()

    # theta = covariance(outcome, covariate) / variance(covariate)
    covariance = np.cov(df[outcome_col], df[covariate_col])[0, 1]
    variance = np.var(df[covariate_col])
    theta = covariance / variance

    df[f"{outcome_col}_cuped"] = df[outcome_col] - theta * (df[covariate_col] - covariate_mean)

    return df, theta


if __name__ == "__main__":
    from src.analysis.continuous_test import analyze_continuous_metric, print_continuous_report

    df = pd.read_csv("data/experiment_results.csv")

    df_cuped, theta = apply_cuped(
        df,
        outcome_col="session_duration_minutes",
        covariate_col="pre_experiment_avg_session_minutes",
    )
    print(f"Theta (variance reduction coefficient): {theta:.5f}\n")

    print("### WITHOUT CUPED (session duration) ###")
    result_raw = analyze_continuous_metric(df_cuped, outcome_col="session_duration_minutes")
    print_continuous_report(result_raw, unit="")

    print("\n### WITH CUPED (session duration) ###")
    result_cuped = analyze_continuous_metric(df_cuped, outcome_col="session_duration_minutes_cuped")
    print_continuous_report(result_cuped, unit="")

    ci_width_raw = result_raw['ci_95_high'] - result_raw['ci_95_low']
    ci_width_cuped = result_cuped['ci_95_high'] - result_cuped['ci_95_low']
    reduction = (1 - ci_width_cuped / ci_width_raw) * 100

    print(f"\nCI width WITHOUT CUPED: {ci_width_raw:.4f} min")
    print(f"CI width WITH CUPED:    {ci_width_cuped:.4f} min")
    print(f"CI width reduction: {reduction:.1f}%")
