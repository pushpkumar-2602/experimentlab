import os

from sqlalchemy import create_engine

# Used when no DATABASE_URL environment variable is set (i.e. on your Mac).
DEFAULT_DATABASE_URL = "postgresql+psycopg2://pushpkumar@localhost:5432/experimentlab"


def get_engine():
    """
    Creates a SQLAlchemy engine connected to PostgreSQL.

    The address comes from the DATABASE_URL environment variable when it is
    set (as it will be inside Docker); otherwise we use the local default.
    We explicitly specify "+psycopg2" to tell SQLAlchemy which driver to use.
    """
    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    return create_engine(database_url)
