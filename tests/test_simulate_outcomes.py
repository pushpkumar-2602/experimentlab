import pandas as pd
import numpy as np
from src.experiments.simulate_outcomes import (
    simulate_conversions,
    simulate_revenue,
    simulate_session_duration,
)


def make_fake_assigned_users(n=5000):
    """Helper: builds a minimal assigned-users dataframe for testing,
    with a clean 50/50 treatment/control split and realistic segment values."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "user_id": range(1, n + 1),
        "group": rng.choice(["treatment", "control"], size=n),
        "is_new_user": rng.choice([True, False], size=n),
        "user_value_segment": rng.choice(["low", "medium", "high"], size=n),
        "pre_experiment_purchases": rng.poisson(3, size=n),
        "pre_experiment_avg_session_minutes": rng.normal(7, 2, size=n).clip(min=0.5),
    })


def test_conversions_are_only_zero_or_one():
    df = make_fake_assigned_users()
    result = simulate_conversions(df)
    assert set(result["converted"].unique()).issubset({0, 1})


def test_treatment_converts_higher_than_control_on_average():
    # This tests our DESIGNED effect, not just random behavior.
    # With 5000+ users, treatment should reliably beat control here.
    df = make_fake_assigned_users(n=20000)
    result = simulate_conversions(df)
    rate_by_group = result.groupby("group")["converted"].mean()
    assert rate_by_group["treatment"] > rate_by_group["control"]


def test_non_converters_have_zero_revenue():
    df = make_fake_assigned_users()
    result = simulate_conversions(df)
    result = simulate_revenue(result)

    non_converters = result[result["converted"] == 0]
    assert (non_converters["revenue"] == 0).all()


def test_converters_have_positive_revenue():
    df = make_fake_assigned_users()
    result = simulate_conversions(df)
    result = simulate_revenue(result)

    converters = result[result["converted"] == 1]
    assert (converters["revenue"] > 0).all()


def test_session_duration_correlates_with_pre_experiment_behavior():
    # Tests the core assumption CUPED depends on.
    df = make_fake_assigned_users(n=20000)
    result = simulate_session_duration(df)

    correlation = result["session_duration_minutes"].corr(
        result["pre_experiment_avg_session_minutes"]
    )
    # We expect a strong positive correlation (we designed it to be ~1:1).
    assert correlation > 0.7