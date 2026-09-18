import pandas as pd
import numpy as np

def simulate_conversions(assigned_df: pd.DataFrame, seed: int = 456) -> pd.DataFrame:
    """
    Simulate whether each user converted (purchased), with a probability
    that depends on their group, whether they're new, and their value segment.
    """
    rng = np.random.default_rng(seed)
    df = assigned_df.copy()

    # Baseline conversion rate for EVERYONE, regardless of group.
    base_rate = 0.07

    # Extra lift just for being in treatment (the "average" effect).
    treatment_boost = 0.02

    # New users in treatment get an ADDITIONAL boost on top of the above.
    # This is what creates the "New users +7.8%" effect from the doc.
    new_user_extra_boost = 0.03

    # High-value users tend to convert more regardless of group —
    # this is a REAL WORLD confounder we're baking in on purpose,
    # which our causal inference methods will need to account for later.
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

    # Safety clip: probabilities must stay between 0 and 1.
    prob = prob.clip(0, 1)

    # Flip a weighted coin per user using their personal probability.
    df["converted"] = rng.binomial(n=1, p=prob)

    return df

if __name__ == "__main__":
    assigned = pd.read_csv("data/users_assigned.csv")
    result = simulate_conversions(assigned)
    result.to_csv("data/experiment_results.csv", index=False)

    print("Overall conversion by group:")
    print(result.groupby("group")["converted"].mean())

    print("\nConversion by group AND new-user status:")
    print(result.groupby(["group", "is_new_user"])["converted"].mean())