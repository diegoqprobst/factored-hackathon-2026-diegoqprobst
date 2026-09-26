"""Bronze CSV -> silver Parquet. Every rejected row is kept in quarantine with its reason; every run
writes a quality report linked to the bronze manifest (lineage)."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from src.pipeline.contracts import CONTRACTS, Contract


def _source(c: Contract, raw_dir: Path) -> str:
    path = raw_dir / c.table / "**" / "*.csv" if c.partitioned else raw_dir / f"{c.table}.csv"
    return f"read_csv('{path}', all_varchar=true, header=true, filename=true, hive_partitioning=false)"


def _count(con, table: str) -> int:
    return con.execute(f"select count(*) from {table}").fetchone()[0]


def _reasons(con, table: str) -> dict[str, int]:
    rows = con.execute(
        f"select r, count(*) from (select unnest(string_split(_reject_reason, '; ')) r from {table} "
        f"where _reject_reason is not null) group by r order by 2 desc, 1").fetchall()
    return {r: n for r, n in rows}


def _write(con, table: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"copy {table} to '{path}' (format parquet)")


def build_table(con, c: Contract, raw_dir: Path) -> dict:
    t = c.table
    con.execute(f"create or replace temp table chk_{t} as select *, {c.reject_reason_sql()} as _reject_reason "
                f"from {_source(c, raw_dir)}")
    con.execute(f"create or replace temp table typed_{t} as select {c.select_sql()}, filename as _source_file "
                f"from chk_{t} where _reject_reason is null")
    pk = ", ".join(f'"{k}"' for k in c.pk)
    con.execute(f"create or replace temp table {t} as select * exclude (_rn) from ("
                f"select *, row_number() over (partition by {pk} order by {c.order_by}) as _rn from typed_{t}) "
                f"where _rn = 1")
    con.execute(f"create or replace temp table q_{t} as select * from chk_{t} where _reject_reason is not null")
    rows_in, typed, kept = _count(con, f"chk_{t}"), _count(con, f"typed_{t}"), _count(con, t)
    return {"rows_in": rows_in, "quarantined": rows_in - typed, "duplicates_removed": typed - kept,
            "rows_out": kept, "reject_reasons": _reasons(con, f"chk_{t}")}


def _transaction_integrity(con) -> dict:
    con.execute("""create or replace temp table txn_fk as
        select t.*, case when p.product_id is null then 'fk:product_missing'
                         when p.customer_id <> t.customer_id then 'fk:owner_mismatch' end as _reject_reason
        from transactions t left join products p on p.product_id = t.product_id""")
    con.execute("create or replace temp table q_transactions_fk as select * from txn_fk where _reject_reason is not null")
    # EDA Q9: transaction_date is UTC, process_date is the local (UTC-6) date partition.
    con.execute("""create or replace temp table transactions as
        select * exclude (_reject_reason), transaction_date as transaction_ts_utc, process_date as local_date
        from txn_fk where _reject_reason is null""")
    tz = con.execute("select count(*) from transactions "
                     "where cast(transaction_date - interval 6 hour as date) <> process_date").fetchone()[0]
    reasons = _reasons(con, "txn_fk")
    return {"fk_quarantined": sum(reasons.values()), "fk_reasons": reasons,
            "tz_offset_violations": tz, "rows_out": _count(con, "transactions")}


def build_silver(raw_dir: Path, out_dir: Path, run_id: str | None = None) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    con = duckdb.connect()
    manifest = raw_dir / "_manifest.csv"
    report = {"run_id": run_id,
              "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
              "tables": {}}
    for c in CONTRACTS:
        report["tables"][c.table] = build_table(con, c, raw_dir)
    report["tables"]["transactions"].update(_transaction_integrity(con))
    for c in CONTRACTS:
        _write(con, c.table, out_dir / f"{c.table}.parquet")
        _write(con, f"q_{c.table}", out_dir / "_quarantine" / f"{c.table}.parquet")
    _write(con, "q_transactions_fk", out_dir / "_quarantine" / "transactions_fk.parquet")
    (out_dir / "_quality").mkdir(parents=True, exist_ok=True)
    (out_dir / "_quality" / f"{run_id}.json").write_text(json.dumps(report, indent=2))
    return report
