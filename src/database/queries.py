import pandas as pd
from sqlalchemy import text

from src.database.db_connection import get_engine


def get_experiment_dataframe(experiment_id: int) -> pd.DataFrame:
    """
    Pulls one experiment's data out of PostgreSQL as a DataFrame shaped like
    the CSVs our analysis code already expects (a 'group' column, 'converted', etc.).
    """
    query = text("""
        SELECT
            ua.user_id,
            ua.group_name AS "group",
            er.converted,
            er.revenue::float AS revenue,
            er.session_duration_minutes::float AS session_duration_minutes,
            u.is_new_user,
            u.user_value_segment
        FROM user_assignments ua
        JOIN experiment_results er
            ON ua.experiment_id = er.experiment_id AND ua.user_id = er.user_id
        JOIN users u
            ON ua.user_id = u.user_id
        WHERE ua.experiment_id = :experiment_id
    """)

    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"experiment_id": experiment_id})

    return df
