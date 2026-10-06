import pandas as pd
from sqlalchemy import text
from src.database.db_connection import get_engine


def load_experiment_to_db(csv_path: str, experiment_name: str, description: str,
                            start_date: str, primary_metric: str = "converted"):
    """
    Loads a pipeline's experiment_results.csv into the database as a new experiment,
    splitting it into the three normalized tables: experiments, user_assignments,
    and experiment_results.
    """
    engine = get_engine()
    df = pd.read_csv(csv_path)

    with engine.begin() as conn:
        # Step 1: insert the experiment record itself, and get back the auto-generated ID.
        result = conn.execute(
            text("""
                INSERT INTO experiments (name, description, start_date, primary_metric)
                VALUES (:name, :description, :start_date, :primary_metric)
                RETURNING experiment_id
            """),
            {
                "name": experiment_name,
                "description": description,
                "start_date": start_date,
                "primary_metric": primary_metric,
            },
        )
        experiment_id = result.scalar()
        print(f"Created experiment '{experiment_name}' with ID {experiment_id}")

        # Step 2: prepare and insert user assignments.
        assignments = df[["user_id", "group"]].copy()
        assignments["experiment_id"] = experiment_id
        assignments = assignments.rename(columns={"group": "group_name"})
        assignments.to_sql("user_assignments", conn, if_exists="append", index=False)
        print(f"Inserted {len(assignments):,} user assignments")

        # Step 3: prepare and insert results.
        results = df[["user_id", "converted", "revenue", "session_duration_minutes"]].copy()
        results["experiment_id"] = experiment_id
        results.to_sql("experiment_results", conn, if_exists="append", index=False)
        print(f"Inserted {len(results):,} experiment results")

    return experiment_id


if __name__ == "__main__":
    load_experiment_to_db(
        csv_path="data/experiment_results.csv",
        experiment_name="New Recommendation Algorithm",
        description="Testing a new recommendation algorithm's effect on conversion, revenue, and engagement.",
        start_date="2026-01-01",
        primary_metric="converted",
    )