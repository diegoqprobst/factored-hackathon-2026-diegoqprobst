import json

from src.bank import audit, config, db


def test_config_constants():
    assert config.SIM_TODAY.isoformat() == "2026-06-17"
    assert (config.SANDBOX_DAYS, config.REPROCESS_DAYS) == (120, 3)
    assert (config.SESSION_TTL_SECONDS, config.OTP_TTL_SECONDS, config.OTP_MAX_ATTEMPTS) == (900, 300, 3)


def test_create_schema_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "x.db")
    db.create_schema(conn)
    db.create_schema(conn)
    tables = {r["name"] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert {"customers", "products", "transactions", "fx_rates", "complaint_flags", "otp_challenges",
            "sandbox_outbox", "disputes", "card_blocks", "audit_log", "load_runs"} <= tables


def test_audit_log_writes_json_detail(bank, frozen):
    audit.log(bank, "unit_test", customer_id="CLI-A", session_id="s1", foo=1)
    row = bank.execute("select * from audit_log order by id desc limit 1").fetchone()
    assert (row["event"], row["customer_id"], row["session_id"]) == ("unit_test", "CLI-A", "s1")
    assert json.loads(row["detail"]) == {"foo": 1}
    assert row["ts"].startswith("2026-06-17T15:00:00")
