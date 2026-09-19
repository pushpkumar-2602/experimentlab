import pandas as pd
from src.data_generation.generate_users import generate_users


def test_generate_users_returns_correct_row_count():
    df = generate_users(500)
    assert len(df) == 500


def test_generate_users_has_expected_columns():
    df = generate_users(100)
    expected_columns = {
        "user_id", "name", "email", "signup_date", "is_new_user",
        "user_value_segment", "pre_experiment_purchases",
        "pre_experiment_avg_session_minutes",
    }
    assert expected_columns.issubset(set(df.columns))


def test_user_ids_are_unique():
    df = generate_users(1000)
    assert df["user_id"].nunique() == len(df)


def test_value_segments_are_valid():
    df = generate_users(1000)
    valid_segments = {"low", "medium", "high"}
    assert set(df["user_value_segment"].unique()).issubset(valid_segments)


def test_session_minutes_never_negative_or_zero():
    # This specifically tests the .clip(lower=0.5) logic —
    # without it, rare random draws could go negative or to zero.
    df = generate_users(5000)
    assert (df["pre_experiment_avg_session_minutes"] >= 0.5).all()


def test_high_value_segment_has_more_purchases_than_low_on_average():
    # This tests the CORRELATION we deliberately built in Step 13 —
    # not just "does the column exist" but "does it behave the way we designed it to."
    df = generate_users(5000)
    avg_by_segment = df.groupby("user_value_segment")["pre_experiment_purchases"].mean()
    assert avg_by_segment["high"] > avg_by_segment["medium"] > avg_by_segment["low"]