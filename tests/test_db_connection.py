from src.database.db_connection import get_engine, normalize_database_url


def test_plain_postgresql_prefix_gets_psycopg2_driver():
    url = normalize_database_url("postgresql://user:pw@host/db")
    assert url == "postgresql+psycopg2://user:pw@host/db"


def test_short_postgres_prefix_is_also_converted():
    url = normalize_database_url("postgres://user:pw@host/db")
    assert url == "postgresql+psycopg2://user:pw@host/db"


def test_url_that_already_names_a_driver_is_unchanged():
    original = "postgresql+psycopg2://user:pw@db:5432/experimentlab"
    assert normalize_database_url(original) == original


def test_query_string_such_as_sslmode_is_preserved():
    url = normalize_database_url("postgresql://user:pw@host/db?sslmode=require")
    assert url.endswith("?sslmode=require")


def test_get_engine_reads_database_url_from_the_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@example.com/somedb")
    engine = get_engine()
    assert engine.url.host == "example.com"
    assert engine.url.drivername == "postgresql+psycopg2"
