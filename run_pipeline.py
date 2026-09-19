"""
Runs the full ExperimentLab data pipeline end-to-end, in the correct order:
1. Generate users
2. Assign treatment/control groups
3. Simulate conversion, revenue, and session duration outcomes

Run this any time you want fresh data, instead of running each script manually.
"""
import pandas as pd

from src.data_generation.generate_users import generate_users
from src.experiments.assign_groups import assign_groups
from src.experiments.simulate_outcomes import (
    simulate_conversions,
    simulate_revenue,
    simulate_session_duration,
)


def run_pipeline(n_users: int = 10000):
    print("[1/3] Generating users...")
    users = generate_users(n_users)
    users.to_csv("data/users.csv", index=False)
    print(f"      -> {len(users):,} users saved to data/users.csv")

    print("[2/3] Assigning treatment/control groups...")
    assigned = assign_groups(users)
    assigned.to_csv("data/users_assigned.csv", index=False)
    print(f"      -> saved to data/users_assigned.csv")

    print("[3/3] Simulating outcomes...")
    result = simulate_conversions(assigned)
    result = simulate_revenue(result)
    result = simulate_session_duration(result)
    result.to_csv("data/experiment_results.csv", index=False)
    print(f"      -> saved to data/experiment_results.csv")

    print("\nPipeline complete. Summary:")
    print(result.groupby("group")[["converted", "revenue", "session_duration_minutes"]].mean())

    return result


if __name__ == "__main__":
    run_pipeline()