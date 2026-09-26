from datetime import date

from src.bank import db
from src.pipeline.sandbox import build_sandbox, refresh_transactions
from src.pipeline.silver import build_silver
from tests.helpers import txn_row, write_raw, write_txn_day


def _ids(conn):
    return [r[0] for r in conn.execute("select transaction_id from transactions order by 1")]


def test_full_build_loads_all_tables(tmp_path):
    raw, silver = tmp_path / "raw", tmp_path / "silver"
    write_raw(raw)
    build_silver(raw, silver, run_id="t")
    counts = build_sandbox(silver, tmp_path / "s.db", today=date(2026, 6, 16), days=10)
    assert counts == {"customers": 2, "products": 2, "fx_rates": 1, "complaint_flags": 1, "transactions": 4}
    conn = db.connect(tmp_path / "s.db")
    assert tuple(conn.execute("select * from complaint_flags").fetchone()) == ("CLI-B", 2, 1)
    assert tuple(conn.execute("select * from fx_rates").fetchone()) == ("2026-06-15", "COP", 0.00025)
    row = conn.execute("select transaction_ts_utc, local_date, is_fraud, amount_usd from transactions "
                       "where transaction_id='TRX-2'").fetchone()
    assert tuple(row) == ("2026-06-15T18:00:00", "2026-06-15", 0, None)
    assert tuple(conn.execute("select mode, rows_loaded from load_runs").fetchone()) == ("full", 4)


def test_late_arrival_fixture_incremental_refresh(tmp_path):
    """LATE-ARRIVAL FIXTURE (team-generated, synthetic): the source data is static, so update correctness
    is demonstrated by delivering a new day plus late rows for older days."""
    raw, silver, path = tmp_path / "raw", tmp_path / "silver", tmp_path / "s.db"
    write_raw(raw)
    build_silver(raw, silver, run_id="v1")
    build_sandbox(silver, path, today=date(2026, 6, 16), days=10)

    write_txn_day(raw, "2026-06-17", [txn_row("TRX-N1", "2026-06-17 18:00:00", "2026-06-17")])
    write_txn_day(raw, "2026-06-15", [  # re-delivered partition now contains a late row
        txn_row("TRX-2", "2026-06-15 18:00:00", "2026-06-15", usd=""),
        txn_row("TRX-L1", "2026-06-15 20:00:00", "2026-06-15"),
    ])
    write_txn_day(raw, "2026-06-01", [txn_row("TRX-OLD", "2026-06-01 18:00:00", "2026-06-01")])
    build_silver(raw, silver, run_id="v2")

    conn = db.connect(path)
    loaded = refresh_transactions(conn, silver, since=date(2026, 6, 17), until=date(2026, 6, 17))
    assert loaded == 6
    assert _ids(conn) == ["TRX-1", "TRX-2", "TRX-5", "TRX-8", "TRX-L1", "TRX-N1"]
    assert "TRX-OLD" not in _ids(conn)  # late beyond the 3-day window is NOT captured (documented limit)

    refresh_transactions(conn, silver, since=date(2026, 6, 17), until=date(2026, 6, 17))
    assert len(_ids(conn)) == 6  # idempotent: re-running does not duplicate
