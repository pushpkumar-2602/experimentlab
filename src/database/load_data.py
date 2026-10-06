import pandas as pd
from sqlalchemy import text
from src.database.db_connection import get_engine


def load_users(csv_path: str = "data/users.csv"):
    """
    Loads user profile data into the users table. This data doesn't belong
    to any single experiment -- it's loaded once and referenced by joins
    whenever we need segment/profile info for analysis.
    """
    engine = get_engine()
    df = pd.read_csv(csv_path)

    users = df[[
        "user_id", "is_new_user", "user_value_segment",
        "pre_experiment_purchases", "pre_experiment_avg_session_minutes",
    ]].copy()

    with engine.begin() as conn:
        # Clear existing users first, so re-running this doesn't create
        # duplicate-key errors (user_id is a PRIMARY KEY -- must be unique).
        conn.execute(text("DELETE FROM users"))
        users.to_sql("users", conn, if_exists="append", index=False)

    print(f"Loaded {len(users):,} users")


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

        assignments = df[["user_id", "group"]].copy()
        assignments["experiment_id"] = experiment_id
        assignments = assignments.rename(columns={"group": "group_name"})
        assignments.to_sql("user_assignments", conn, if_exists="append", index=False)
        print(f"Inserted {len(assignments):,} user assignments")

        results = df[["user_id", "converted", "revenue", "session_duration_minutes"]].copy()
        results["experiment_id"] = experiment_id
        results.to_sql("experiment_results", conn, if_exists="append", index=False)
        print(f"Inserted {len(results):,} experiment results")

    return experiment_id


def load_confounded_experiment_to_db(csv_path: str, experiment_name: str, description: str,
                                        start_date: str, primary_metric: str = "converted"):
    """
    Loads the confounded email campaign dataset, which has a different shape
    than our randomized experiments: 'received_email' instead of 'group',
    and no revenue/session_duration columns. We map it into the SAME schema
    by treating 'received_email' as the group indicator.
    """
    engine = get_engine()
    df = pd.read_csv(csv_path)

    with engine.begin() as conn:
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

        assignments = df[["user_id", "received_email"]].copy()
        assignments["experiment_id"] = experiment_id
        assignments["group_name"] = assignments["received_email"].map({1: "treatment", 0: "control"})
        assignments = assignments[["experiment_id", "user_id", "group_name"]]
        assignments.to_sql("user_assignments", conn, if_exists="append", index=False)
        print(f"Inserted {len(assignments):,} user assignments")

        results = df[["user_id", "converted"]].copy()
        results["experiment_id"] = experiment_id
        results["revenue"] = 0  # not tracked for this experiment
        results["session_duration_minutes"] = None
        results.to_sql("experiment_results", conn, if_exists="append", index=False)
        print(f"Inserted {len(results):,} experiment results")

    return experiment_id


if __name__ == "__main__":
    load_users()

    load_experiment_to_db(
        csv_path="data/experiment_results.csv",
        experiment_name="New Recommendation Algorithm",
        description="Testing a new recommendation algorithm's effect on conversion, revenue, and engagement.",
        start_date="2026-01-01",
        primary_metric="converted",
    )

    load_confounded_experiment_to_db(
        csv_path="data/confounded_results.csv",
        experiment_name="Personalized Email Campaign",
        description="Observational (non-randomized) email campaign targeting engaged users. Analyzed with propensity scores / IPW.",
        start_date="2026-02-01",
        primary_metric="converted",
    )
