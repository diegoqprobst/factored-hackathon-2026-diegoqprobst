"""Silver Parquet -> SQLite sandbox read by the mock bank services.
Dimensions are full-refreshed; transactions use partition overwrite over a reprocess window, which makes
the load idempotent and captures late arrivals up to `reprocess_days` old."""
import secrets
from datetime import date, timedelta
from pathlib import Path

import duckdb

from src.bank import clock, config, db

TXN_COLS = ("transaction_id, customer_id, product_id, transaction_ts_utc, local_date, transaction_type, amount, "
            "currency, amount_usd, channel, merchant_name, merchant_category, transaction_city, "
            "transaction_country, transaction_status, is_fraud")


def _pq(silver_dir: Path, table: str) -> str:
    return str(Path(silver_dir) / f"{table}.parquet")


def refresh_transactions(conn, silver_dir: Path, since: date, *, until: date = config.SIM_TODAY,
                         reprocess_days: int = config.REPROCESS_DAYS, mode: str = "incremental") -> int:
    start = since - timedelta(days=reprocess_days)
    rows = duckdb.connect().execute(f"""
        select transaction_id, customer_id, product_id, strftime(transaction_ts_utc, '%Y-%m-%dT%H:%M:%S'),
               strftime(local_date, '%Y-%m-%d'), transaction_type, amount, currency, amount_usd, channel,
               merchant_name, merchant_category, transaction_city, transaction_country, transaction_status,
               is_fraud::int
        from read_parquet('{_pq(silver_dir, "transactions")}')
        where local_date >= ? and local_date <= ?""", [start, until]).fetchall()
    with conn:
        conn.execute("delete from transactions where local_date >= ?", (start.isoformat(),))
        conn.executemany(f"insert or replace into transactions ({TXN_COLS}) values ({','.join('?' * 16)})", rows)
        conn.execute("insert into load_runs(run_id, started_at, mode, since, rows_loaded) values (?,?,?,?,?)",
                     (secrets.token_hex(6), clock.now().isoformat(), mode, start.isoformat(), len(rows)))
    return len(rows)


def build_sandbox(silver_dir: Path, sandbox_path: Path, *, today: date = config.SIM_TODAY,
                  days: int = config.SANDBOX_DAYS) -> dict:
    conn = db.connect(sandbox_path)
    db.create_schema(conn)
    duck = duckdb.connect()
    customers = duck.execute(
        "select customer_id, document_number, document_type, first_name, last_name, mobile_phone, country, "
        f"segment, customer_status from read_parquet('{_pq(silver_dir, 'customers')}')").fetchall()
    products = duck.execute(
        "select product_id, customer_id, product_type, product_number, currency, product_status "
        f"from read_parquet('{_pq(silver_dir, 'products')}')").fetchall()
    fx = duck.execute(
        "select strftime(date, '%Y-%m-%d'), source_currency, exchange_rate "
        f"from read_parquet('{_pq(silver_dir, 'daily_exchange_rates')}') where target_currency = 'USD'").fetchall()
    flags = duck.execute(
        "select customer_id, count(*), bool_or(is_repeat_complainer)::int "
        f"from read_parquet('{_pq(silver_dir, 'complaints')}') group by 1").fetchall()
    with conn:
        conn.executemany("insert or replace into customers values (?,?,?,?,?,?,?,?,?)", customers)
        conn.executemany("insert or replace into products values (?,?,?,?,?,?)", products)
        conn.executemany("insert or replace into fx_rates values (?,?,?)", fx)
        conn.executemany("insert or replace into complaint_flags values (?,?,?)", flags)
    n = refresh_transactions(conn, silver_dir, since=today - timedelta(days=days), until=today,
                             reprocess_days=0, mode="full")
    conn.close()
    return {"customers": len(customers), "products": len(products), "fx_rates": len(fx),
            "complaint_flags": len(flags), "transactions": n}
