import pandas as pd
from faker import Faker
import numpy as np

# Setting a "seed" makes our random data reproducible —
# every time we run this script, we get the SAME fake data.
# This matters for debugging: if numbers change every run, you can never
# tell if a bug is a real bug or just different random luck.
np.random.seed(42)
fake = Faker()
Faker.seed(42)

def generate_users(n_users: int) -> pd.DataFrame:
    """Create a fake table of n_users users with basic profile info."""
    segment_to_avg_purchases = {"low": 1.0, "medium": 3.0, "high": 7.0}
    segment_to_avg_session_minutes = {"low": 4.0, "medium": 7.0, "high": 11.0}

    users = []
    for user_id in range(1, n_users + 1):
        segment = np.random.choice(["low", "medium", "high"], p=[0.5, 0.35, 0.15])

        users.append({
            "user_id": user_id,
            "name": fake.name(),
            "email": fake.email(),
            "signup_date": fake.date_between(start_date="-2y", end_date="today"),
            "is_new_user": np.random.choice([True, False], p=[0.3, 0.7]),
            "user_value_segment": segment,
            "pre_experiment_purchases": np.random.poisson(
                lam=segment_to_avg_purchases[segment]
            ),
            "pre_experiment_avg_session_minutes": np.random.normal(
                loc=segment_to_avg_session_minutes[segment], scale=2.0
            ),
        })
    df = pd.DataFrame(users)
    # Clip lives HERE now — inside the function — so every caller
    # (this file's __main__, run_pipeline.py, tests, anything future)
    # automatically gets clean, valid data without repeating this line.
    df["pre_experiment_avg_session_minutes"] = df["pre_experiment_avg_session_minutes"].clip(lower=0.5)
    return df

if __name__ == "__main__":
    df = generate_users(10000)
    df.to_csv("data/users.csv", index=False)
    print(f"Generated {len(df)} users")
    print(df.head())
