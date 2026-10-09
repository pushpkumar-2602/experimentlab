from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from sqlalchemy import text

from src.analysis.ab_test_stats import analyze_ab_test
from src.database.db_connection import get_engine
from src.database.queries import get_experiment_dataframe
from src.experiments.guardrails import run_all_guardrails

app = FastAPI(
    title="ExperimentLab API",
    description="Causal experimentation platform: experiments, results, and guardrails.",
    version="0.2.0",
)

# Create the database engine once, when the app starts, and reuse it
# for every request instead of reconnecting each time.
engine = get_engine()


def to_native(value):
    """
    Recursively converts numpy types (np.bool_, np.int64, ...) into plain
    Python types, because FastAPI's JSON conversion can't handle all of them.
    """
    if isinstance(value, dict):
        return {key: to_native(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@app.get("/health")
def health_check():
    """Tells callers the API is running. Deployment platforms use this too."""
    return {"status": "ok"}


@app.get("/experiments")
def list_experiments():
    """Returns every experiment stored in the database."""
    query = text("""
        SELECT experiment_id, name, description, start_date, status, primary_metric
        FROM experiments
        ORDER BY experiment_id
    """)
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [dict(row) for row in rows]


@app.get("/experiments/{experiment_id}/analysis")
def analyze_experiment(experiment_id: int, minimum_detectable_effect: float = 0.02):
    """
    Runs guardrails and the A/B test for one experiment, and returns
    the full report: status, statistics, and a final decision.
    """
    df = get_experiment_dataframe(experiment_id)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for experiment {experiment_id}")

    control_rate = df[df["group"] == "control"]["converted"].mean()

    guardrails = run_all_guardrails(
        df,
        baseline_rate=control_rate,
        minimum_detectable_effect=minimum_detectable_effect,
    )
    analysis = analyze_ab_test(df)

    # Only trust the statistical verdict if the experiment itself is valid.
    if guardrails["experiment_status"] == "VALID":
        final_decision = analysis["decision"]
    else:
        final_decision = "INVESTIGATE"

    return to_native({
        "experiment_id": experiment_id,
        "experiment_status": guardrails["experiment_status"],
        "final_decision": final_decision,
        "guardrails": guardrails,
        "analysis": analysis,
    })


MIN_USERS_PER_GROUP = 30


@app.get("/experiments/{experiment_id}/segments")
def analyze_segments(
    experiment_id: int,
    by: Literal["is_new_user", "user_value_segment"] = "is_new_user",
):
    """
    Runs the A/B analysis separately for each segment (e.g. new vs returning
    users), with a Bonferroni correction because testing several segments
    increases the chance of a false positive.
    """
    df = get_experiment_dataframe(experiment_id)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for experiment {experiment_id}")

    n_segments = df[by].nunique()
    adjusted_alpha = 0.05 / n_segments

    segments = []
    for segment_value, subset in df.groupby(by):
        group_sizes = subset["group"].value_counts()
        too_small = (
            group_sizes.get("treatment", 0) < MIN_USERS_PER_GROUP
            or group_sizes.get("control", 0) < MIN_USERS_PER_GROUP
        )
        if too_small:
            segments.append({
                "segment": str(segment_value),
                "skipped": f"fewer than {MIN_USERS_PER_GROUP} users in a group",
            })
            continue

        result = analyze_ab_test(subset)
        result.pop("decision")  # shipping to one segment is a human decision
        result["segment"] = str(segment_value)
        result["significant_after_correction"] = result["p_value"] < adjusted_alpha
        segments.append(result)

    return to_native({
        "experiment_id": experiment_id,
        "sliced_by": by,
        "n_segments": n_segments,
        "alpha_per_segment_after_correction": adjusted_alpha,
        "segments": segments,
    })
