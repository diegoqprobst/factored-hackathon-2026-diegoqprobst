import json

import duckdb

from src.pipeline.silver import build_silver
from tests.helpers import write_raw


def _silver(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "silver"
    write_raw(raw)
    return build_silver(raw, out, run_id="test"), out


def test_customers_contract_dedup_and_quarantine(tmp_path):
    report, out = _silver(tmp_path)
    s = report["tables"]["customers"]
    assert (s["rows_in"], s["quarantined"], s["duplicates_removed"], s["rows_out"]) == (5, 2, 1, 2)
    assert s["reject_reasons"] == {"document_type:bad_enum": 1, "document_number:missing": 1}
    status = duckdb.sql(f"select customer_status from '{out}/customers.parquet' where customer_id='CLI-A'").fetchone()
    assert status == ("Active",)  # newest last_updated wins


def test_transactions_contract_fk_and_timezone(tmp_path):
    report, out = _silver(tmp_path)
    s = report["tables"]["transactions"]
    assert (s["rows_in"], s["quarantined"], s["duplicates_removed"]) == (9, 2, 1)
    assert s["reject_reasons"] == {"amount:bad_type": 1, "transaction_status:bad_enum": 1}
    assert s["fk_reasons"] == {"fk:owner_mismatch": 1, "fk:product_missing": 1}
    assert s["rows_out"] == 4
    assert s["tz_offset_violations"] == 1
    ids = [r[0] for r in duckdb.sql(f"select transaction_id from '{out}/transactions.parquet' order by 1").fetchall()]
    assert ids == ["TRX-1", "TRX-2", "TRX-5", "TRX-8"]
    cols = duckdb.sql(f"describe select * from '{out}/transactions.parquet'").fetchall()
    types = {c[0]: c[1] for c in cols}
    assert types["amount"] == "DOUBLE" and types["local_date"] == "DATE" and types["is_fraud"] == "BOOLEAN"


def test_quarantine_files_keep_reason_and_source(tmp_path):
    _, out = _silver(tmp_path)
    q = duckdb.sql(f"select transaction_id, _reject_reason, filename from '{out}/_quarantine/transactions.parquet' order by 1").fetchall()
    assert [r[0] for r in q] == ["TRX-4", "TRX-7"]
    assert all(r[2].endswith(".csv") for r in q)
    fk = duckdb.sql(f"select transaction_id from '{out}/_quarantine/transactions_fk.parquet' order by 1").fetchall()
    assert fk == [("TRX-3",), ("TRX-6",)]


def test_quality_report_written(tmp_path):
    report, out = _silver(tmp_path)
    assert json.loads((out / "_quality" / "test.json").read_text()) == report
