import pandas as pd
from src.experiments.assign_groups import assign_groups


def make_fake_users(n=1000):
    """Helper: creates a minimal fake users dataframe for testing,
    without needing to run the real generate_users() function."""
    return pd.DataFrame({"user_id": range(1, n + 1)})


def test_assign_groups_returns_only_treatment_and_control():
    users = make_fake_users()
    result = assign_groups(users)

    unique_groups = set(result["group"].unique())
    assert unique_groups == {"treatment", "control"}


def test_assign_groups_respects_ratio_approximately():
    users = make_fake_users(n=10000)
    result = assign_groups(users, treatment_ratio=0.5)

    treatment_fraction = (result["group"] == "treatment").mean()

    # We allow a small tolerance since it's random —
    # exact 50.00% every time would actually be suspicious, not correct.
    assert 0.45 < treatment_fraction < 0.55


def test_assign_groups_is_reproducible_with_same_seed():
    users = make_fake_users()
    result1 = assign_groups(users, seed=42)
    result2 = assign_groups(users, seed=42)

    # Same seed should produce IDENTICAL assignments, every time.
    assert (result1["group"] == result2["group"]).all()


def test_assign_groups_does_not_modify_original_dataframe():
    users = make_fake_users()
    original_columns = list(users.columns)

    assign_groups(users)

    # The original dataframe passed in should be untouched —
    # this tests that our earlier .copy() actually works as intended.
    assert list(users.columns) == original_columns