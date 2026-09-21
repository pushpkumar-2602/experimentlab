import pandas as pd
import numpy as np

def simulate_confounded_conversions(df: pd.DataFrame, seed: int = 111) -> pd.DataFrame:
    """
    Simulates conversion for the confounded email campaign dataset.
    TRUE effect of the email itself is a flat +4 percentage points —
    we know this because WE set it, and we'll check later whether our
    propensity-score-corrected estimate can recover something close to it.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # Same segment-based baseline as our earlier experiments.
    segment_boost = df["user_value_segment"].map({"low": 0.0, "medium": 0.02, "high": 0.05})
    base_rate = 0.06 + segment_boost

    true_email_effect = 0.04  # <-- the TRUE causal effect we're trying to recover

    prob = base_rate + np.where(df["received_email"] == 1, true_email_effect, 0.0)
    prob = prob.clip(0, 1)

    df["converted"] = rng.binomial(n=1, p=prob)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/confounded_assignment.csv")
    result = simulate_confounded_conversions(df)
    result.to_csv("data/confounded_results.csv", index=False)

    print("### NAIVE comparison (ignores confounding) ###")
    naive_rates = result.groupby("received_email")["converted"].mean()
    print(naive_rates)

    naive_lift = naive_rates[1] - naive_rates[0]
    print(f"\nNaive estimated lift: {naive_lift:.4f}")
    print(f"TRUE lift (what we actually built in): 0.0400")
    print(f"Naive bias: {naive_lift - 0.04:+.4f}")