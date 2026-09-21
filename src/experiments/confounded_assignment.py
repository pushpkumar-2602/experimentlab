import pandas as pd
import numpy as np

def assign_confounded_treatment(users_df: pd.DataFrame, seed: int = 999) -> pd.DataFrame:
    """
    Simulates a NON-random treatment assignment: users with higher
    pre-experiment engagement (purchases + session minutes) are MORE LIKELY
    to receive the email campaign — mirroring real marketing targeting.
    This deliberately creates confounding for us to later correct.
    """
    rng = np.random.default_rng(seed)
    df = users_df.copy()

    # Normalize the two engagement signals to a comparable 0-1ish scale,
    # so neither dominates just because of its raw units (minutes vs purchase count).
    purchases_norm = df["pre_experiment_purchases"] / df["pre_experiment_purchases"].max()
    session_norm = df["pre_experiment_avg_session_minutes"] / df["pre_experiment_avg_session_minutes"].max()

    engagement_score = 0.5 * purchases_norm + 0.5 * session_norm

    # Base 20% chance of being emailed, PLUS up to +60% more for highly engaged users.
    # This is the confounding: propensity to be treated depends on pre-existing behavior.
    propensity_true = 0.20 + 0.60 * engagement_score
    propensity_true = propensity_true.clip(0.02, 0.98)  # keep some randomness at every level

    df["received_email"] = rng.binomial(n=1, p=propensity_true)
    df["true_propensity"] = propensity_true  # we keep this ONLY to validate our model later

    return df


if __name__ == "__main__":
    users = pd.read_csv("data/users.csv")
    result = assign_confounded_treatment(users)
    result.to_csv("data/confounded_assignment.csv", index=False)

    print("Email received by engagement level:")
    print(result.groupby("user_value_segment")["received_email"].mean())