"""Append-only audit trail. Detail is JSON; never put raw PII (document, phone) in it."""
import json
import sqlite3

from src.bank import clock


def log(conn: sqlite3.Connection, event: str, *, customer_id: str | None = None,
        session_id: str | None = None, **detail) -> None:
    conn.execute(
        "insert into audit_log(ts, event, customer_id, session_id, detail) values (?,?,?,?,?)",
        (clock.now().isoformat(), event, customer_id, session_id, json.dumps(detail, default=str, sort_keys=True)),
    )
    conn.commit()
