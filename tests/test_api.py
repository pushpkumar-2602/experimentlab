import pytest
from fastapi.testclient import TestClient

from src.api.main import app

# Tags every test in this file as an integration test, because they
# read from the real PostgreSQL database.
pytestmark = pytest.mark.integration

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_experiments_returns_expected_fields():
    response = client.get("/experiments")
    assert response.status_code == 200

    experiments = response.json()
    assert isinstance(experiments, list)
    assert len(experiments) >= 1

    expected_fields = {
        "experiment_id", "name", "description",
        "start_date", "status", "primary_metric",
    }
    assert expected_fields.issubset(experiments[0].keys())


def test_analysis_returns_full_report_for_existing_experiment():
    # Look up a real experiment ID instead of hardcoding one, so this test
    # doesn't break if experiment IDs change when data is reloaded.
    experiment_id = client.get("/experiments").json()[0]["experiment_id"]

    response = client.get(f"/experiments/{experiment_id}/analysis")
    assert response.status_code == 200

    body = response.json()
    expected_keys = {"experiment_status", "final_decision", "guardrails", "analysis"}
    assert expected_keys.issubset(body.keys())


def test_unknown_experiment_returns_404():
    response = client.get("/experiments/999999/analysis")
    assert response.status_code == 404


def test_non_integer_experiment_id_is_rejected_with_422():
    # 422 = "Unprocessable Entity": FastAPI's standard response when the
    # request is well-formed but its values fail validation.
    response = client.get("/experiments/abc/analysis")
    assert response.status_code == 422


def test_invalid_experiment_never_gets_a_ship_decision():
    # The core safety promise of the platform: if the guardrails say an
    # experiment can't be trusted, the final decision must be INVESTIGATE,
    # no matter how good the statistics look.
    for experiment in client.get("/experiments").json():
        body = client.get(f"/experiments/{experiment['experiment_id']}/analysis").json()
        if body["experiment_status"] != "VALID":
            assert body["final_decision"] == "INVESTIGATE"


def test_p_values_are_never_exactly_zero():
    # Regression test for the floating-point bug we fixed (1 - cdf underflow).
    # If someone reintroduces that formula, this test fails.
    for experiment in client.get("/experiments").json():
        body = client.get(f"/experiments/{experiment['experiment_id']}/analysis").json()
        assert body["analysis"]["p_value"] > 0


def _first_experiment_id():
    return client.get("/experiments").json()[0]["experiment_id"]


def test_segments_returns_one_entry_per_segment():
    body = client.get(f"/experiments/{_first_experiment_id()}/segments?by=user_value_segment").json()
    assert body["n_segments"] == 3
    assert len(body["segments"]) == 3


def test_segments_bonferroni_threshold_is_alpha_divided_by_segment_count():
    body = client.get(f"/experiments/{_first_experiment_id()}/segments?by=user_value_segment").json()
    expected = 0.05 / body["n_segments"]
    assert abs(body["alpha_per_segment_after_correction"] - expected) < 1e-12


def test_segments_never_include_a_ship_decision():
    # Shipping to one slice of users is a human decision, not a p-value cutoff.
    body = client.get(f"/experiments/{_first_experiment_id()}/segments").json()
    for segment in body["segments"]:
        assert "decision" not in segment


def test_segments_rejects_arbitrary_column_names():
    response = client.get(f"/experiments/{_first_experiment_id()}/segments?by=name")
    assert response.status_code == 422


def test_correction_can_only_make_significance_stricter():
    # The corrected threshold is never larger than 0.05, so a segment that
    # passes the corrected test must also pass the uncorrected one.
    body = client.get(f"/experiments/{_first_experiment_id()}/segments?by=user_value_segment").json()
    for segment in body["segments"]:
        if "skipped" in segment:
            continue
        if segment["significant_after_correction"]:
            assert segment["is_significant"]
