from fastapi import FastAPI
from sqlalchemy import text

from src.database.db_connection import get_engine

app = FastAPI(
    title="ExperimentLab API",
    description="Causal experimentation platform: experiments, results, and guardrails.",
    version="0.1.0",
)

# Create the database engine once, when the app starts, and reuse it
# for every request instead of reconnecting each time.
engine = get_engine()


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
