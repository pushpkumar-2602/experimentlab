import pandas as pd
import numpy as np

def simulate_conversions(assigned_df: pd.DataFrame, seed: int = 456) -> pd.DataFrame:
    """
    Simulate whether each user converted (purchased), with a probability
    that depends on their group, whether they're new, and their value segment.
    """
    rng = np.random.default_rng(seed)
    df = assigned_df.copy()

    base_rate = 0.07
    treatment_boost = 0.02
    new_user_extra_boost = 0.03

    segment_boost = df["user_value_segment"].map({
        "low": 0.0,
        "medium": 0.015,
        "high": 0.03,
    })

    prob = base_rate + segment_boost.copy()

    is_treatment = df["group"] == "treatment"
    prob = prob + np.where(is_treatment, treatment_boost, 0.0)

    is_treatment_new = is_treatment & df["is_new_user"]
    prob = prob + np.where(is_treatment_new, new_user_extra_boost, 0.0)

    prob = prob.clip(0, 1)

    df["converted"] = rng.binomial(n=1, p=prob)

    return df


def simulate_revenue(df: pd.DataFrame, seed: int = 789) -> pd.DataFrame:
    """
    Simulate revenue per user, only for users who converted.
    Revenue scales with pre-experiment purchase history plus random noise,
    with a small extra boost for the treatment group.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    base_order_value = 30 + (df["pre_experiment_purchases"] * 2)
    treatment_boost = np.where(df["group"] == "treatment", 5.0, 0.0)
    noise = rng.normal(loc=0, scale=8, size=len(df))

    order_value = base_order_value + treatment_boost + noise
    order_value = order_value.clip(lower=5)

    df["revenue"] = np.where(df["converted"] == 1, order_value, 0.0)

    return df


def simulate_session_duration(df: pd.DataFrame, seed: int = 321) -> pd.DataFrame:
    """
    Simulate session duration (minutes) for EVERY user during the experiment.
    Unlike revenue, this is dense (no zero-inflation) — everyone has a session.
    Correlates with pre-experiment session behavior, plus a small treatment boost.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # Baseline is mostly a continuation of past behavior.
    base_duration = df["pre_experiment_avg_session_minutes"] * 1.0

    # Treatment nudges people to browse slightly longer (better recommendations).
    treatment_boost = np.where(df["group"] == "treatment", 1.2, 0.0)

    noise = rng.normal(loc=0, scale=1.5, size=len(df))

    duration = base_duration + treatment_boost + noise
    df["session_duration_minutes"] = duration.clip(lower=0.2)

    return df


if __name__ == "__main__":
    assigned = pd.read_csv("data/users_assigned.csv")
    result = simulate_conversions(assigned)
    result = simulate_revenue(result)
    result = simulate_session_duration(result)
    result.to_csv("data/experiment_results.csv", index=False)

    print("Overall conversion by group:")
    print(result.groupby("group")["converted"].mean())

    print("\nConversion by group AND new-user status:")
    print(result.groupby(["group", "is_new_user"])["converted"].mean())

    print("\nAverage revenue per user by group (includes $0 for non-converters):")
    print(result.groupby("group")["revenue"].mean())

    print("\nAverage session duration (minutes) by group:")
    print(result.groupby("group")["session_duration_minutes"].mean())
