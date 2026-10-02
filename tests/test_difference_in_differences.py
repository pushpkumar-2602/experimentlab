import pandas as pd
from src.analysis.difference_in_differences import (
    compute_naive_before_after,
    compute_difference_in_differences,
    check_parallel_trends,
    regression_based_did,
)


def make_known_panel_data():
    """
    Builds a small panel dataset with KNOWN before/after values per region,
    with tiny variation added so regression can compute valid standard errors.
    Known values:
        West before: ~0.10, West after: ~0.16  (change = +0.06)
        East before: ~0.08, East after: ~0.10  (change = +0.02)
        True DiD = 0.06 - 0.02 = 0.04
    """
    rows = []
    # 5 days before rollout, 5 days after -- for each region.
    for day in range(5):
        rows.append({"day": day, "region": "west", "is_treated_region": True,
                     "is_post_rollout": False, "conversion_rate": 0.10 + 0.001 * day})
        rows.append({"day": day, "region": "east", "is_treated_region": False,
                     "is_post_rollout": False, "conversion_rate": 0.08 + 0.001 * day})
    for day in range(5, 10):
        rows.append({"day": day, "region": "west", "is_treated_region": True,
                     "is_post_rollout": True, "conversion_rate": 0.16 + 0.001 * day})
        rows.append({"day": day, "region": "east", "is_treated_region": False,
                     "is_post_rollout": True, "conversion_rate": 0.10 + 0.001 * day})
    return pd.DataFrame(rows)


def test_naive_estimate_ignores_control_region_trend():
    df = make_known_panel_data()
    result = compute_naive_before_after(df)
    # West alone changed by roughly +0.06 (the naive estimate includes
    # whatever "background trend" happened, since it ignores East entirely).
    assert abs(result["naive_estimate"] - 0.06) < 0.01


def test_did_estimate_removes_background_trend():
    df = make_known_panel_data()
    result = compute_difference_in_differences(df)
    # True DiD = West's change (0.06) minus East's change (0.02) = 0.04
    assert abs(result["did_estimate"] - 0.04) < 0.01


def test_did_estimate_is_smaller_than_naive_when_trend_exists():
    # This tests the CORE PURPOSE of DiD: when there's a shared background
    # trend, DiD's corrected estimate should be smaller than the naive one,
    # since DiD removes the part of the naive estimate caused by the trend.
    df = make_known_panel_data()
    naive = compute_naive_before_after(df)
    did = compute_difference_in_differences(df)
    assert did["did_estimate"] < naive["naive_estimate"]


def test_parallel_trends_detected_as_parallel_when_slopes_match():
    df = make_known_panel_data()
    # Both regions were built with the SAME +0.001/day pre-period slope,
    # so trends should correctly register as parallel.
    result = check_parallel_trends(df)
    assert result["trends_look_parallel"] == True


def test_parallel_trends_detected_as_not_parallel_when_slopes_differ():
    df = make_known_panel_data()
    # Manually break parallel trends: make West's pre-period slope much steeper.
    pre_west_mask = (df["region"] == "west") & (~df["is_post_rollout"])
    df.loc[pre_west_mask, "conversion_rate"] = 0.10 + 0.05 * df.loc[pre_west_mask, "day"]

    result = check_parallel_trends(df)
    assert result["trends_look_parallel"] == False


def test_regression_based_did_matches_simple_average_did():
    # Both methods should agree closely on the SAME balanced dataset --
    # this is an important cross-check between our two implementations.
    df = make_known_panel_data()
    simple_result = compute_difference_in_differences(df)
    regression_result = regression_based_did(df)

    assert abs(simple_result["did_estimate"] - regression_result["did_estimate"]) < 1e-9


def test_regression_based_did_confidence_interval_contains_estimate():
    df = make_known_panel_data()
    result = regression_based_did(df)
    # Basic sanity check: the point estimate should always fall within its own CI.
    assert result["ci_95_low"] <= result["did_estimate"] <= result["ci_95_high"]