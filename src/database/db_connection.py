import os

from sqlalchemy import create_engine

# Used when no DATABASE_URL environment variable is set (i.e. on your Mac).
DEFAULT_DATABASE_URL = "postgresql+psycopg2://pushpkumar@localhost:5432/experimentlab"


def normalize_database_url(url: str) -> str:
    """
    Hosted providers hand out addresses starting with 'postgres://' or
    'postgresql://'. We add the explicit '+psycopg2' driver so SQLAlchemy
    uses the driver we actually have installed.
    """
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg2://" + url[len(prefix):]
    return url


def get_engine():
    """
    Creates a SQLAlchemy engine connected to PostgreSQL.

    The address comes from the DATABASE_URL environment variable when it is
    set (Docker, hosting platforms); otherwise we use the local default.
    """
    database_url = normalize_database_url(os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    # pool_pre_ping tests each connection before use, which matters for hosted
    # databases that close idle connections.
    return create_engine(database_url, pool_pre_ping=True)
