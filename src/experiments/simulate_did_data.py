import pandas as pd
import numpy as np

def simulate_did_data(n_days: int = 60, rollout_day: int = 30, seed: int = 555) -> pd.DataFrame:
    """
    Simulates daily conversion data for two regions over time:
    - 'west': receives the new search ranking starting at rollout_day
    - 'east': never receives it (control region)

    Includes a shared background trend (affects both regions equally) PLUS
    a true treatment effect (only affects west, only after rollout_day).
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Each region has a slightly different baseline conversion rate,
    # which is normal and doesn't bias DiD (DiD only cares about CHANGE over time).
    baseline_rate = {"west": 0.08, "east": 0.07}

    true_treatment_effect = 0.03  # <-- the TRUE effect we're trying to recover

    for day in range(n_days):
        # Shared background trend: conversion drifts upward slightly over time,
        # for BOTH regions equally (e.g. general seasonality) -- this is exactly
        # the kind of confound a naive before/after comparison would wrongly
        # attribute to the treatment.
        background_trend = 0.0008 * day

        for region in ["west", "east"]:
            is_treated_region = region == "west"
            is_post_rollout = day >= rollout_day

            rate = baseline_rate[region] + background_trend
            if is_treated_region and is_post_rollout:
                rate += true_treatment_effect

            # Small daily random noise on top of the "true" rate.
            rate = np.clip(rate + rng.normal(0, 0.005), 0, 1)

            n_visitors = 1000
            conversions = rng.binomial(n_visitors, rate)

            rows.append({
                "day": day,
                "region": region,
                "is_treated_region": is_treated_region,
                "is_post_rollout": is_post_rollout,
                "n_visitors": n_visitors,
                "conversions": conversions,
                "conversion_rate": conversions / n_visitors,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = simulate_did_data()
    df.to_csv("data/did_data.csv", index=False)

    print("Average conversion rate by region and period:")
    print(df.groupby(["region", "is_post_rollout"])["conversion_rate"].mean())