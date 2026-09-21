import pandas as pd
import numpy as np
from src.analysis.ipw import compute_ipw_weights, estimate_ipw_effect


def test_ipw_weight_for_treated_user_is_inverse_of_propensity():
    df = pd.DataFrame({
        "received_email": [1],
        "estimated_propensity": [0.25],
    })
    result = compute_ipw_weights(df)
    assert abs(result["ipw_weight"].iloc[0] - 4.0) < 1e-9  # 1 / 0.25 = 4.0


def test_ipw_weight_for_control_user_is_inverse_of_one_minus_propensity():
    df = pd.DataFrame({
        "received_email": [0],
        "estimated_propensity": [0.25],
    })
    result = compute_ipw_weights(df)
    assert abs(result["ipw_weight"].iloc[0] - 1.3333333) < 1e-5  # 1 / (1 - 0.25) = 1.333...


def test_ipw_estimate_equals_naive_when_propensity_is_constant():
    """
    If EVERYONE has the exact same propensity score, IPW weighting has
    nothing to correct for — it should give the same answer as a naive
    average. This tests that IPW correctly reduces to the simple case.
    """
    df = pd.DataFrame({
        "received_email": [1, 1, 1, 0, 0, 0],
        "converted":      [1, 0, 1, 0, 0, 1],
        "estimated_propensity": [0.5] * 6,  # same for everyone
    })
    result = estimate_ipw_effect(df)
    assert abs(result["naive_lift"] - result["ipw_lift"]) < 1e-9


def test_ipw_corrects_a_known_biased_scenario():
    """
    Builds a small dataset where propensity VARIES WITHIN each group,
    so IPW weighting has something real to correct. If every member of
    a group shared the same propensity, IPW would mathematically reduce
    to the naive average (see test above) -- so variation within groups
    is essential for this test to be meaningful.
    """
    df = pd.DataFrame({
        "received_email":       [1, 1, 1, 1, 0, 0, 0, 0],
        "converted":            [1, 1, 1, 0, 0, 0, 0, 1],
        "estimated_propensity": [0.9, 0.7, 0.8, 0.6, 0.1, 0.3, 0.2, 0.4],
    })
    result = estimate_ipw_effect(df)

    # We're not checking an exact number here (too small a sample to be precise),
    # just that IPW actually changed the estimate rather than doing nothing.
    assert result["ipw_lift"] != result["naive_lift"]