import pandas as pd
import numpy as np

def assign_groups(users_df: pd.DataFrame, treatment_ratio: float = 0.5, seed: int = 123) -> pd.DataFrame:
    """
    Randomly assign each user to 'treatment' or 'control'.
    treatment_ratio: fraction of users who should land in treatment (0.5 = 50/50 split)
    """
    rng = np.random.default_rng(seed)
    users_df = users_df.copy()
    users_df["group"] = rng.choice(
        ["treatment", "control"],
        size=len(users_df),
        p=[treatment_ratio, 1 - treatment_ratio],
    )
    return users_df

if __name__ == "__main__":
    users = pd.read_csv("data/users.csv")
    assigned = assign_groups(users)
    assigned.to_csv("data/users_assigned.csv", index=False)

    print(assigned["group"].value_counts())
    print(assigned["group"].value_counts(normalize=True))