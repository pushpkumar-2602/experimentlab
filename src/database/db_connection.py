from sqlalchemy import create_engine

def get_engine():
    """
    Creates a SQLAlchemy engine connected to our local PostgreSQL database.
    An 'engine' is SQLAlchemy's term for a reusable connection manager --
    we create one of these and reuse it, rather than opening a new raw
    connection every time we want to run a query.
    """
    # Format: postgresql+psycopg2://username:password@host:port/database_name
    # We explicitly specify "+psycopg2" to tell SQLAlchemy which driver to use,
    # since newer SQLAlchemy versions otherwise default to trying "psycopg" (v3),
    # which we don't have installed.
    connection_string = "postgresql+psycopg2://pushpkumar@localhost:5432/experimentlab"
    return create_engine(connection_string)
