import pandas as pd
from src.analysis.ab_test_stats import analyze_ab_test
from src.experiments.guardrails import check_sample_ratio_mismatch, check_minimum_sample_size


def make_known_results(n_treatment=1000, conv_treatment=150, n_control=1000, conv_control=100):
    """
    Builds a dataset with an EXACT, hand-picked number of conversions,
    so we can verify our statistics produce mathematically correct results.
    """
    treatment_rows = [{"group": "treatment", "converted": 1}] * conv_treatment + \
                     [{"group": "treatment", "converted": 0}] * (n_treatment - conv_treatment)
    control_rows = [{"group": "control", "converted": 1}] * conv_control + \
                   [{"group": "control", "converted": 0}] * (n_control - conv_control)
    return pd.DataFrame(treatment_rows + control_rows)


def test_conversion_rates_are_calculated_correctly():
    df = make_known_results(n_treatment=1000, conv_treatment=150, n_control=1000, conv_control=100)
    result = analyze_ab_test(df)

    # We know EXACTLY what these should be: 150/1000 and 100/1000.
    assert result["rate_treatment"] == 0.15
    assert result["rate_control"] == 0.10


def test_absolute_and_relative_lift_are_correct():
    df = make_known_results(n_treatment=1000, conv_treatment=150, n_control=1000, conv_control=100)
    result = analyze_ab_test(df)

    assert abs(result["absolute_lift"] - 0.05) < 1e-9
    assert abs(result["relative_lift"] - 0.5) < 1e-9  # 0.05 / 0.10 = 50% relative lift


def test_large_clear_difference_is_significant():
    df = make_known_results(n_treatment=1000, conv_treatment=150, n_control=1000, conv_control=100)
    result = analyze_ab_test(df)
    assert result["is_significant"] == True
    assert result["decision"] == "SHIP"


def test_identical_groups_are_not_significant():
    # Both groups convert at EXACTLY the same rate — there is truly no effect.
    df = make_known_results(n_treatment=1000, conv_treatment=100, n_control=1000, conv_control=100)
    result = analyze_ab_test(df)
    assert result["is_significant"] == False
    assert result["decision"] == "DO NOT SHIP"


def test_confidence_interval_contains_the_true_lift():
    df = make_known_results(n_treatment=1000, conv_treatment=150, n_control=1000, conv_control=100)
    result = analyze_ab_test(df)

    # The known true lift (0.05) should fall within our own computed CI.
    assert result["ci_95_low"] <= 0.05 <= result["ci_95_high"]


def test_srm_check_passes_on_balanced_split():
    df = pd.DataFrame({"group": ["treatment"] * 500 + ["control"] * 500})
    result = check_sample_ratio_mismatch(df)
    assert result["srm_detected"] == False


def test_srm_check_fails_on_badly_imbalanced_split():
    df = pd.DataFrame({"group": ["treatment"] * 700 + ["control"] * 300})
    result = check_sample_ratio_mismatch(df)
    assert result["srm_detected"] == True


def test_minimum_sample_size_check_fails_when_underpowered():
    result = check_minimum_sample_size(
        baseline_rate=0.08, minimum_detectable_effect=0.02, actual_n_per_group=100
    )
    assert result["has_enough_data"] is False


def test_minimum_sample_size_check_passes_when_well_powered():
    result = check_minimum_sample_size(
        baseline_rate=0.08, minimum_detectable_effect=0.02, actual_n_per_group=50000
    )
    assert result["has_enough_data"] is True