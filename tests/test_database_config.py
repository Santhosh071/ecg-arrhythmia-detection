from src.layer6_storage.database import database_backend_label, resolve_database_url


def test_database_backend_label_detects_sqlite():
    assert database_backend_label("sqlite:///tmp/test.db") == "SQLite"


def test_resolve_database_url_returns_sqlite_for_path(tmp_path):
    url = resolve_database_url(str(tmp_path / "patient_history.db"), None)
    assert url.startswith("sqlite:///")
