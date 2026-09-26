"""SQLite sandbox: read-only banking tables (loaded by the pipeline) + writable service tables."""
import sqlite3
from pathlib import Path

SCHEMA = """
create table if not exists customers (
  customer_id text primary key, document_number text not null unique, document_type text not null,
  first_name text not null, last_name text not null, mobile_phone text, country text not null,
  segment text not null, customer_status text not null);
create table if not exists products (
  product_id text primary key, customer_id text not null, product_type text not null,
  product_number text not null, currency text not null, product_status text not null);
create index if not exists ix_products_customer on products(customer_id);
create table if not exists transactions (
  transaction_id text primary key, customer_id text not null, product_id text not null,
  transaction_ts_utc text not null, local_date text not null, transaction_type text not null,
  amount real not null, currency text not null, amount_usd real, channel text not null,
  merchant_name text, merchant_category text, transaction_city text, transaction_country text not null,
  transaction_status text not null, is_fraud integer not null);
create index if not exists ix_txn_customer_date on transactions(customer_id, local_date);
create table if not exists fx_rates (
  date text not null, currency text not null, rate_to_usd real not null, primary key (date, currency));
create table if not exists complaint_flags (
  customer_id text primary key, complaint_count integer not null, is_repeat_complainer integer not null);
create table if not exists otp_challenges (
  challenge_id text primary key, customer_id text, code_hash text not null, expires_at text not null,
  attempts integer not null default 0, consumed integer not null default 0);
create table if not exists sandbox_outbox (
  id integer primary key autoincrement, channel text not null, destination text not null,
  body text not null, created_at text not null);
create table if not exists disputes (
  dispute_id text primary key, idempotency_key text not null unique, session_id text not null,
  customer_id text not null, transaction_id text not null, dispute_type text not null, status text not null,
  policy_rule text not null, policy_version text not null, created_at text not null);
create table if not exists card_blocks (
  block_id text primary key, product_id text not null, customer_id text not null, session_id text not null,
  reason text not null, created_at text not null);
create table if not exists audit_log (
  id integer primary key autoincrement, ts text not null, event text not null, customer_id text,
  session_id text, detail text not null);
create table if not exists load_runs (
  run_id text primary key, started_at text not null, mode text not null, since text,
  rows_loaded integer not null);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
