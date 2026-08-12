"""
Unit tests for src/db.py using mocks — no real PostgreSQL required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import db

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_pool():
    """Ensure the module-level _pool is None before and after each test."""
    db._pool = None
    yield
    db._pool = None


def _make_pool(rows=None, fetchone_row=None):
    """Return a mock ThreadedConnectionPool that returns preset query results."""
    cursor = MagicMock()
    cursor.fetchall.return_value = rows or []
    cursor.fetchone.return_value = fetchone_row

    conn = MagicMock()
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.getconn.return_value = conn
    return pool, cursor


# ── is_available ──────────────────────────────────────────────────────────────

def test_is_available_false_without_pool():
    assert db.is_available() is False


def test_is_available_true_with_pool():
    db._pool = MagicMock()
    assert db.is_available() is True


# ── init_pool ─────────────────────────────────────────────────────────────────

def test_init_pool_no_url_leaves_pool_none(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db.init_pool()
    assert db._pool is None


def test_init_pool_sets_pool_on_success(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/test")
    mock_pool = MagicMock()
    with patch("psycopg2.pool.ThreadedConnectionPool", return_value=mock_pool):
        with patch.object(db, "_ensure_schema"):
            db.init_pool()
    assert db._pool is mock_pool


def test_init_pool_connection_error_leaves_pool_none(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/test")
    with patch("psycopg2.pool.ThreadedConnectionPool", side_effect=Exception("conn refused")):
        db.init_pool()
    assert db._pool is None


# ── get_patient ───────────────────────────────────────────────────────────────

def test_get_patient_returns_data_when_found():
    patient_data = {"static_info": {"RecordID": 1}, "timeline": []}
    pool, _ = _make_pool(fetchone_row=(patient_data,))
    db._pool = pool

    result = db.get_patient(1)
    assert result == patient_data


def test_get_patient_returns_none_when_not_found():
    pool, _ = _make_pool(fetchone_row=None)
    db._pool = pool

    result = db.get_patient(999999)
    assert result is None


# ── list_patients ─────────────────────────────────────────────────────────────

def test_list_patients_maps_rows_correctly():
    rows = [
        (132539, 0, 0.72, "65", "1", "Coronary Care Unit"),
        (132540, 1, 0.31, "45", "3", "Medical ICU"),
    ]
    pool, _ = _make_pool(rows=rows)
    db._pool = pool

    result = db.list_patients(limit=2)
    assert len(result) == 2
    assert result[0]["record_id"] == 132539
    assert result[0]["mortality_risk"] == 0.72
    assert result[0]["status"] == "High"
    assert result[1]["mortality_risk_percent"] == 31.0


def test_list_patients_status_thresholds():
    rows = [
        (1, 0, 0.85, "70", "1", "MICU"),  # Critical
        (2, 0, 0.60, "60", "1", "MICU"),  # High
        (3, 0, 0.25, "50", "1", "MICU"),  # Moderate
        (4, 0, 0.05, "40", "1", "MICU"),  # Low
    ]
    pool, _ = _make_pool(rows=rows)
    db._pool = pool

    result = db.list_patients(limit=4)
    statuses = [r["status"] for r in result]
    assert statuses == ["Critical", "High", "Moderate", "Low"]


def test_list_patients_handles_null_risk():
    rows = [(132539, 0, None, "65", "1", "MICU")]
    pool, _ = _make_pool(rows=rows)
    db._pool = pool

    result = db.list_patients(limit=1)
    assert result[0]["mortality_risk"] == 0.0


# ── upsert_patient ────────────────────────────────────────────────────────────

def test_upsert_patient_calls_execute_with_correct_args():
    pool, cursor = _make_pool()
    db._pool = pool

    data = {"static_info": {"RecordID": 42}, "timeline": []}
    db.upsert_patient(42, data, outcome=0, predicted_risk=0.15)

    cursor.execute.assert_called_once()
    call_args = cursor.execute.call_args[0]
    assert call_args[1][0] == 42                     # record_id
    assert json.loads(call_args[1][1]) == data        # data as JSON
    assert call_args[1][2] == 0                      # outcome
    assert call_args[1][3] == pytest.approx(0.15)    # predicted_risk


def test_upsert_patient_returns_connection_to_pool():
    pool, _ = _make_pool()
    db._pool = pool

    db.upsert_patient(1, {}, outcome=None, predicted_risk=None)

    pool.putconn.assert_called_once_with(pool.getconn.return_value)
