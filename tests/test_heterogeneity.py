import pandas as pd
import pytest

from src.analysis.heterogeneity import check_effect_heterogeneity


def make_cells(cells):
    """cells: list of (segment, group, n_users, n_conversions) -> one row per user."""
    rows = []
    for segment, group, n_users, n_conversions in cells:
        outcomes = [1] * n_conversions + [0] * (n_users - n_conversions)
        rows.extend({"segment": segment, "group": group, "converted": o} for o in outcomes)
    return pd.DataFrame(rows)


def test_identical_effects_give_a_large_p_value():
    # Both segments: control 10%, treatment 20%. The effect is exactly the same.
    df = make_cells([
        ("A", "control", 2000, 200), ("A", "treatment", 2000, 400),
        ("B", "control", 2000, 200), ("B", "treatment", 2000, 400),
    ])
    assert check_effect_heterogeneity(df, by="segment")["p_value"] > 0.99


def test_clearly_different_effects_give_a_tiny_p_value():
    # Segment A gains 10 points from treatment; segment B gains nothing.
    df = make_cells([
        ("A", "control", 2000, 200), ("A", "treatment", 2000, 400),
        ("B", "control", 2000, 200), ("B", "treatment", 2000, 200),
    ])
    assert check_effect_heterogeneity(df, by="segment")["p_value"] < 0.001


def test_degrees_of_freedom_is_segments_minus_one():
    df = make_cells([
        ("A", "control", 500, 50), ("A", "treatment", 500, 100),
        ("B", "control", 500, 50), ("B", "treatment", 500, 100),
        ("C", "control", 500, 50), ("C", "treatment", 500, 100),
    ])
    result = check_effect_heterogeneity(df, by="segment")
    assert result["n_segments"] == 3
    assert result["df"] == 2


def test_boolean_segment_column_works():
    df = make_cells([
        (True, "control", 500, 50), (True, "treatment", 500, 100),
        (False, "control", 500, 50), (False, "treatment", 500, 100),
    ])
    assert check_effect_heterogeneity(df, by="segment")["n_segments"] == 2


def test_input_dataframe_is_not_modified():
    df = make_cells([
        ("A", "control", 500, 50), ("A", "treatment", 500, 100),
        ("B", "control", 500, 50), ("B", "treatment", 500, 100),
    ])
    columns_before = list(df.columns)
    check_effect_heterogeneity(df, by="segment")
    assert list(df.columns) == columns_before


def test_single_segment_is_rejected():
    df = make_cells([("A", "control", 100, 10), ("A", "treatment", 100, 20)])
    with pytest.raises(ValueError):
        check_effect_heterogeneity(df, by="segment")
