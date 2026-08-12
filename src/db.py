"""
PostgreSQL connection pool and CRUD helpers for patient data.

When DATABASE_URL is not set the module still imports cleanly; callers must
check `is_available()` before calling any query functions.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_pool = None  # psycopg2.pool.ThreadedConnectionPool


def is_available() -> bool:
    return _pool is not None


def init_pool(min_conn: int = 1, max_conn: int = 10) -> None:
    """Call once at application startup. No-ops if DATABASE_URL is unset."""
    global _pool
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.info("DATABASE_URL not set — running without PostgreSQL")
        return
    try:
        from psycopg2 import pool as pg_pool

        _pool = pg_pool.ThreadedConnectionPool(min_conn, max_conn, dsn=url)
        logger.info("PostgreSQL connection pool initialised (min=%d max=%d)", min_conn, max_conn)
        _ensure_schema()
    except Exception:
        logger.exception("Failed to connect to PostgreSQL — falling back to flat-file mode")
        _pool = None


def _ensure_schema() -> None:
    """Create the patients table if it doesn't exist."""
    sql = """
        CREATE TABLE IF NOT EXISTS patients (
            record_id      INTEGER PRIMARY KEY,
            data           JSONB        NOT NULL,
            outcome        SMALLINT,
            predicted_risk FLOAT,
            predicted_at   TIMESTAMPTZ  DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_patients_risk ON patients (predicted_risk DESC);
    """
    _execute(sql)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _execute(sql: str, params=None) -> None:
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
    finally:
        _pool.putconn(conn)


def _fetchall(sql: str, params=None) -> list[tuple]:
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        _pool.putconn(conn)


def _fetchone(sql: str, params=None) -> Optional[tuple]:
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()
    finally:
        _pool.putconn(conn)


# ── Public CRUD ───────────────────────────────────────────────────────────────

def upsert_patient(
    record_id: int,
    data: dict,
    outcome: Optional[int] = None,
    predicted_risk: Optional[float] = None,
) -> None:
    sql = """
        INSERT INTO patients (record_id, data, outcome, predicted_risk, predicted_at)
        VALUES (%s, %s, %s, %s, NOW())
        ON CONFLICT (record_id) DO UPDATE
            SET data           = EXCLUDED.data,
                outcome        = EXCLUDED.outcome,
                predicted_risk = EXCLUDED.predicted_risk,
                predicted_at   = NOW();
    """
    _execute(sql, (record_id, json.dumps(data), outcome, predicted_risk))


def get_patient(record_id: int) -> Optional[dict]:
    row = _fetchone(
        "SELECT data FROM patients WHERE record_id = %s",
        (record_id,),
    )
    return row[0] if row else None


def list_patients(limit: int = 500) -> list[dict]:
    """Return patients sorted by predicted_risk descending."""
    rows = _fetchall(
        """
        SELECT record_id, outcome, predicted_risk,
               data->>'Age' AS age, data->>'ICUType' AS icu_type,
               data->>'ICUTypeLabel' AS icu_type_label
        FROM   patients
        ORDER  BY predicted_risk DESC NULLS LAST
        LIMIT  %s
        """,
        (limit,),
    )
    results = []
    for record_id, outcome, risk, age, icu_type, icu_type_label in rows:
        risk = float(risk) if risk is not None else 0.0
        results.append({
            "record_id": record_id,
            "age": int(float(age)) if age else None,
            "icu_type": int(float(icu_type)) if icu_type else None,
            "icu_type_label": icu_type_label or "ICU",
            "mortality_risk": risk,
            "mortality_risk_percent": round(risk * 100, 1),
            "status": (
                "Critical" if risk >= 0.8
                else "High" if risk >= 0.5
                else "Moderate" if risk >= 0.2
                else "Low"
            ),
        })
    return results
