# Plan 1 — Data pipeline, sandbox, and mock bank services — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the raw S3 mirror into contract-checked silver Parquet and a SQLite sandbox, and expose the mock bank services (OTP auth, scoped transactions, dispute policy, disputes, cards) that the agent will call — every guard enforced in code and covered by tests.

**Architecture:** Bronze (`data/raw`, already mirrored) → `src/pipeline/silver.py` applies per-table contracts in DuckDB (typed, deduplicated, quarantined with reasons, quality report) → `src/pipeline/sandbox.py` loads SQLite (`data/sandbox.db`) with an incremental, idempotent transaction refresh. `src/bank/*` are plain Python service modules over a `sqlite3.Connection`; each validates the session token and customer ownership itself. HTTP exposure comes in Plan 3.

**Tech Stack:** Python 3.12, uv, DuckDB, SQLite (stdlib `sqlite3`), PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-dispute-agent-design.md` (sections 1, 4, 5, 7). Evidence: `reports/eda_findings.md`.

## Global Constraints

- Python 3.12 via `uv`; run everything with `uv run ...`. Code, comments, docs in English.
- Simulated "today": `2026-06-17`. Sandbox transaction window: last **120 days**. Late-arrival reprocess window: **3 days**.
- Session TTL **15 min**. OTP TTL **5 min**, max **3** attempts, 6 digits, single use.
- Policy v1 (synthetic): P1 window **90 days** → ineligible · P2 `Declined` → ineligible · P3 `Reversed` → ineligible · P4 `amount_usd` > **500** (or not convertible) → requires_human · P5 ≥ **3** disputed transactions in session → requires_human · P6 repeat complainer → requires_human · P7 open dispute exists → ineligible (return existing id) · P8 → eligible. First match wins, in this order.
- Ownership: every read/write filters by the session's `customer_id`; foreign IDs raise `NotFound` (never reveal existence) and write an audit event.
- Data minimisation: the view given to the LLM contains only transaction id, local date, description (merchant or type), amount, currency, status, card last-4.
- A document number or email never authenticates on its own; only a verified OTP creates a session.
- Never commit `.env`, `data/`, or `docs/LATAM_Bank_Complete_Data_Dictionary*.pdf`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Customer without a mobile phone** (4,707 real customers) starts auth → must get `NoVerifiedChannel`, never a session (pinned in Task 4).
2. **`merchant_name` is null in ~77% of transactions** → search by merchant must not crash and the view must fall back to the transaction type (pinned in Task 5).
3. **Non-USD transaction with null `amount_usd` and no FX rate for that date** → policy must route to a human, not treat the amount as 0 (pinned in Tasks 5, 6, 7).
4. **Retrying `create_dispute` with the same idempotency key** (network retry) → returns the same case, no duplicate row (pinned in Task 7).
5. **Blocking a product that is not a card, is closed, or belongs to someone else** → `InvalidProduct` / `NotFound`, no state change (pinned in Task 8).

---

## File structure

| File | Responsibility |
|---|---|
| `src/bank/config.py` | Constants (dates, windows, TTLs, paths) and the session secret |
| `src/bank/clock.py` | `now()` — single time source, patched in tests |
| `src/bank/errors.py` | Typed service errors with stable `code`s |
| `src/bank/db.py` | SQLite connection + schema (read tables + writable tables) |
| `src/bank/audit.py` | Append-only audit events |
| `src/bank/auth.py` | OTP challenges, session tokens (HMAC), `verify_session` |
| `src/bank/transactions.py` | Scoped search/get, LLM-safe view, internal record, USD resolution |
| `src/bank/policy.py` + `policy/dispute_policy_v1.yaml` | Deterministic dispute policy |
| `src/bank/disputes.py` | Evaluate, create (idempotent, confirmed), get |
| `src/bank/cards.py` | Get and block cards (confirmed, idempotent, read-back) |
| `src/pipeline/contracts.py` | Per-table contracts → SQL for reject reasons and typed select |
| `src/pipeline/silver.py` | Bronze → silver Parquet + quarantine + quality report |
| `src/pipeline/sandbox.py` | Silver → SQLite, incremental transaction refresh |
| `src/pipeline/run.py` | CLI entrypoint |
| `tests/helpers.py` | Raw CSV fixture writer, `login`, `last_code` |
| `tests/conftest.py` | `frozen` clock and seeded `bank` fixtures |

---

### Task 1: Foundation — config, clock, errors, schema, audit, test harness

**Files:**
- Create: `src/bank/__init__.py`, `src/bank/config.py`, `src/bank/clock.py`, `src/bank/errors.py`, `src/bank/db.py`, `src/bank/audit.py`
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/helpers.py`, `tests/test_foundation.py`, `.env.example`
- Modify: `pyproject.toml` (deps + pytest config)

**Interfaces:**
- Produces: `config.SIM_TODAY: date`, `config.SANDBOX_DAYS=120`, `config.REPROCESS_DAYS=3`, `config.SESSION_TTL_SECONDS=900`, `config.OTP_TTL_SECONDS=300`, `config.OTP_MAX_ATTEMPTS=3`, `config.RAW_DIR`, `config.SILVER_DIR`, `config.SANDBOX_PATH`, `config.POLICY_PATH: Path`, `config.session_secret() -> bytes`; `clock.now() -> datetime (UTC)`; `db.connect(path) -> sqlite3.Connection` (row_factory=Row), `db.create_schema(conn) -> None`; `audit.log(conn, event: str, *, customer_id=None, session_id=None, **detail) -> None`; error classes listed below; test fixtures `frozen` (has `.advance(seconds)`) and `bank` (seeded connection).

- [ ] **Step 1: Add dependencies and pytest config**

```bash
uv add pyyaml
uv add --dev pytest
```

Append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Create `.env.example`:

```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-2
S3_BUCKET=factored-datathon-2026-s3-157725502942-us-east-2-an
BANK_SESSION_SECRET=change-me
```

- [ ] **Step 2: Write the failing test** — `tests/test_foundation.py`

```python
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
```

- [ ] **Step 3: Write `tests/conftest.py` and `tests/helpers.py` (fixtures used by all later tasks)**

`tests/__init__.py`: empty file.

`tests/conftest.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from src.bank import clock, db


class FrozenClock:
    def __init__(self, t: datetime):
        self.t = t

    def now(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)


@pytest.fixture
def frozen(monkeypatch):
    fc = FrozenClock(datetime(2026, 6, 17, 15, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(clock, "now", fc.now)
    return fc


CUSTOMERS = [
    ("CLI-A", "111", "CC", "Ana", "Gómez", "+57 300 000 1234", "Colombia", "Basic", "Active"),
    ("CLI-B", "222", "DNI", "Beto", "Paz", "+54 9 11 0000 2222", "Argentina", "Plus", "Active"),
    ("CLI-NOPHONE", "333", "CC", "Carla", "Ruiz", None, "Colombia", "Basic", "Active"),
    ("CLI-CLOSED", "444", "CC", "Dana", "Soto", "+57 300 000 4444", "Colombia", "Basic", "Closed"),
]
PRODUCTS = [
    ("PRD-A-CC", "CLI-A", "Tarjeta Crédito", "4000000000001111", "COP", "Active"),
    ("PRD-A-SAV", "CLI-A", "Cuenta Ahorro", "0011223344", "COP", "Active"),
    ("PRD-A-OLD", "CLI-A", "Tarjeta Débito", "4000000000003333", "COP", "Closed"),
    ("PRD-B-DC", "CLI-B", "Tarjeta Débito", "4000000000002222", "USD", "Active"),
]


def _t(tid, cid, pid, day, amount, cur, usd, merchant, status, fraud=0, ttype="Purchase"):
    return (tid, cid, pid, f"{day}T18:00:00", day, ttype, amount, cur, usd, "POS", merchant, None,
            "Bogotá", "Colombia", status, fraud)


TRANSACTIONS = [
    _t("TRX-A1", "CLI-A", "PRD-A-CC", "2026-06-16", 350.0, "COP", 0.09, "Oxxo", "Approved"),
    _t("TRX-A2", "CLI-A", "PRD-A-CC", "2026-06-10", 120000.0, "COP", None, None, "Approved"),
    _t("TRX-A3", "CLI-A", "PRD-A-CC", "2026-01-05", 50.0, "COP", 0.01, "Tienda", "Approved"),
    _t("TRX-A4", "CLI-A", "PRD-A-CC", "2026-06-15", 80.0, "COP", 0.02, "Rappi", "Declined"),
    _t("TRX-A5", "CLI-A", "PRD-A-CC", "2026-06-14", 90.0, "COP", 0.02, "Rappi", "Reversed"),
    _t("TRX-A6", "CLI-A", "PRD-A-CC", "2026-06-12", 2500000.0, "COP", 625.0, "Falabella", "Approved"),
    _t("TRX-A7", "CLI-A", "PRD-A-CC", "2026-06-11", 990.0, "ARS", None, "Kiosco", "Approved"),
    _t("TRX-A8", "CLI-A", "PRD-A-CC", "2026-06-16", 350.0, "COP", 0.09, "Oxxo", "Approved"),
    _t("TRX-A9", "CLI-A", "PRD-A-CC", "2026-06-13", 45.0, "COP", 0.01, "Uber", "Approved"),
    _t("TRX-B1", "CLI-B", "PRD-B-DC", "2026-06-16", 40.0, "USD", None, "Amazon", "Approved"),
]
FX = [("2026-06-10", "COP", 0.00025)]
FLAGS = [("CLI-B", 2, 1)]


@pytest.fixture
def bank(tmp_path):
    conn = db.connect(tmp_path / "bank.db")
    db.create_schema(conn)
    conn.executemany("insert into customers values (?,?,?,?,?,?,?,?,?)", CUSTOMERS)
    conn.executemany("insert into products values (?,?,?,?,?,?)", PRODUCTS)
    conn.executemany("insert into transactions values (" + ",".join("?" * 16) + ")", TRANSACTIONS)
    conn.executemany("insert into fx_rates values (?,?,?)", FX)
    conn.executemany("insert into complaint_flags values (?,?,?)", FLAGS)
    conn.commit()
    yield conn
    conn.close()
```

`tests/helpers.py` (the `login`/`last_code` helpers are used from Task 4 on; the raw writer from Task 2 on):

```python
import csv
import re
from pathlib import Path


def last_code(conn) -> str:
    body = conn.execute("select body from sandbox_outbox order by id desc limit 1").fetchone()["body"]
    return re.search(r"(\d{6})", body).group(1)


def login(conn, document_number: str) -> str:
    from src.bank import auth

    ch = auth.start_auth(conn, document_number)
    return auth.verify_otp(conn, ch.challenge_id, last_code(conn))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig mimics the BOM present in the real S3 files
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/test_foundation.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'src.bank'`.

- [ ] **Step 5: Implement the foundation modules**

`src/bank/__init__.py`: empty file.

`src/bank/config.py`:

```python
"""Project-wide constants for the mock bank and pipeline (values fixed by the design spec)."""
import os
from datetime import date
from pathlib import Path

SIM_TODAY = date(2026, 6, 17)  # dataset end; the sandbox "today"
SANDBOX_DAYS = 120
REPROCESS_DAYS = 3
SESSION_TTL_SECONDS = 15 * 60
OTP_TTL_SECONDS = 5 * 60
OTP_MAX_ATTEMPTS = 3

RAW_DIR = Path(os.environ.get("RAW_DIR", "data/raw"))
SILVER_DIR = Path(os.environ.get("SILVER_DIR", "data/silver"))
SANDBOX_PATH = Path(os.environ.get("SANDBOX_PATH", "data/sandbox.db"))
POLICY_PATH = Path(os.environ.get("POLICY_PATH", "policy/dispute_policy_v1.yaml"))


def session_secret() -> bytes:
    return os.environ.get("BANK_SESSION_SECRET", "dev-only-secret").encode()
```

`src/bank/clock.py`:

```python
"""Single time source so tests can freeze and advance time."""
from datetime import datetime, timezone


def now() -> datetime:
    return datetime.now(timezone.utc)
```

`src/bank/errors.py`:

```python
"""Typed errors raised by the mock bank services. `code` is stable and safe to show the orchestrator."""


class BankError(Exception):
    code = "bank_error"

    def __init__(self, message: str = ""):
        super().__init__(message or self.code)


class AuthError(BankError):
    code = "auth_failed"


class NoVerifiedChannel(AuthError):
    code = "no_verified_channel"


class AccountNotServiceable(AuthError):
    code = "account_not_serviceable"


class OtpInvalid(AuthError):
    code = "otp_invalid"


class OtpExpired(AuthError):
    code = "otp_expired"


class OtpLocked(AuthError):
    code = "otp_locked"


class SessionInvalid(AuthError):
    code = "session_invalid"


class SessionExpired(AuthError):
    code = "session_expired"


class NotFound(BankError):
    code = "not_found"


class InvalidProduct(BankError):
    code = "invalid_product"


class ConfirmationRequired(BankError):
    code = "confirmation_required"


class PolicyViolation(BankError):
    code = "policy_violation"

    def __init__(self, decision):
        super().__init__(decision.reason)
        self.decision = decision
```

`src/bank/db.py`:

```python
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
```

`src/bank/audit.py`:

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_foundation.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .env.example .gitignore src/__init__.py src/bank tests
git commit -m "feat(bank): foundation — config, clock, errors, sqlite schema, audit, test fixtures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Silver layer — contracts, quarantine, dedup, FK and timezone checks

**Files:**
- Create: `src/pipeline/contracts.py`, `src/pipeline/silver.py`, `tests/test_silver.py`
- Modify: `tests/helpers.py` (append `write_raw`)

**Interfaces:**
- Consumes: `tests.helpers.write_csv`.
- Produces: `contracts.CONTRACTS: tuple[Contract, ...]` (tables `customers`, `products`, `transactions`, `complaints`, `daily_exchange_rates`); `silver.build_silver(raw_dir: Path, out_dir: Path, run_id: str | None = None) -> dict` writing `out_dir/{table}.parquet`, `out_dir/_quarantine/{table}.parquet`, `out_dir/_quarantine/transactions_fk.parquet`, `out_dir/_quality/{run_id}.json`. Silver `transactions` has extra columns `transaction_ts_utc TIMESTAMP` and `local_date DATE`. Report shape: `{"run_id", "source_manifest_sha256", "tables": {table: {"rows_in", "quarantined", "duplicates_removed", "rows_out", "reject_reasons": {reason: n}}}}`; `transactions` also has `fk_quarantined`, `fk_reasons`, `tz_offset_violations`. `tests.helpers.write_raw(raw: Path) -> None`.

- [ ] **Step 1: Append the raw fixture writer to `tests/helpers.py`**

```python
TXN_HEADER = ["transaction_id", "transaction_date", "process_date", "product_id", "customer_id",
              "transaction_type", "transaction_category", "amount", "currency", "amount_usd", "channel",
              "merchant_name", "merchant_category", "transaction_country", "transaction_city",
              "transaction_status", "is_fraud"]


def txn_row(tid, ts, day, pid="PRD-A1", cid="CLI-A", amount="350.00", status="Approved", usd="0.09",
            merchant="Oxxo"):
    return [tid, ts, day, pid, cid, "Purchase", "Food", amount, "COP", usd, "POS", merchant, "Grocery",
            "Colombia", "Bogotá", status, "False"]


def write_txn_day(raw: Path, day: str, rows: list[list]) -> None:
    y, m, d = day.split("-")
    write_csv(raw / "transactions" / f"year={y}" / f"month={m}" / f"day={d}" / f"transactions_{y}{m}{d}.csv",
              TXN_HEADER, rows)


def write_raw(raw: Path) -> None:
    """Small bronze fixture with one planted defect of each kind the contracts must catch."""
    write_csv(raw / "customers.csv",
              ["customer_id", "document_number", "document_type", "first_name", "last_name", "mobile_phone",
               "country", "segment", "customer_status", "last_updated"],
              [["CLI-A", "111", "CC", "Ana", "Gómez", "+57 300 000 1234", "Colombia", "Basic", "Active", "2026-06-01 00:00:00"],
               ["CLI-A", "111", "CC", "Ana", "Gómez", "+57 300 000 1234", "Colombia", "Basic", "Inactive", "2026-01-01 00:00:00"],  # older duplicate
               ["CLI-B", "222", "DNI", "Beto", "Paz", "+54 9 11 0000 2222", "Argentina", "Plus", "Active", "2026-06-01 00:00:00"],
               ["CLI-BAD", "333", "XX", "Carla", "Ruiz", "", "Colombia", "Basic", "Active", "2026-06-01 00:00:00"],  # bad enum
               ["CLI-MISS", "", "CC", "Dana", "Soto", "", "Colombia", "Basic", "Active", "2026-06-01 00:00:00"]])  # missing doc
    write_csv(raw / "products.csv",
              ["product_id", "customer_id", "product_type", "product_number", "currency", "product_status", "last_updated"],
              [["PRD-A1", "CLI-A", "Tarjeta Crédito", "4000000000001111", "COP", "Active", "2026-06-01 00:00:00"],
               ["PRD-B1", "CLI-B", "Tarjeta Débito", "4000000000002222", "ARS", "Active", "2026-06-01 00:00:00"]])
    write_csv(raw / "daily_exchange_rates.csv",
              ["date", "source_currency", "target_currency", "exchange_rate"],
              [["2026-06-15", "COP", "USD", "0.00025"], ["2026-06-15", "USD", "COP", "4000"]])
    write_csv(raw / "complaints" / "year=2026" / "month=05" / "day=01" / "complaints_20260501.csv",
              ["complaint_id", "creation_date", "customer_id", "is_repeat_complainer"],
              [["CMP-1", "2026-05-01 10:00:00", "CLI-B", "True"], ["CMP-2", "2026-05-01 11:00:00", "CLI-B", "False"]])
    write_txn_day(raw, "2026-06-14", [txn_row("TRX-1", "2026-06-14 18:00:00", "2026-06-14")])
    write_txn_day(raw, "2026-06-15", [
        txn_row("TRX-2", "2026-06-15 18:00:00", "2026-06-15", usd=""),
        txn_row("TRX-3", "2026-06-15 18:00:00", "2026-06-15", cid="CLI-B"),       # product owned by CLI-A
        txn_row("TRX-4", "2026-06-15 18:00:00", "2026-06-15", amount="abc"),      # bad type
    ])
    write_txn_day(raw, "2026-06-16", [
        txn_row("TRX-5", "2026-06-16 18:00:00", "2026-06-16", merchant=""),
        txn_row("TRX-5", "2026-06-16 18:00:00", "2026-06-16", merchant=""),       # exact duplicate
        txn_row("TRX-6", "2026-06-16 18:00:00", "2026-06-16", pid="PRD-ZZ"),      # orphan product
        txn_row("TRX-7", "2026-06-16 18:00:00", "2026-06-16", status="Approvedd"),  # bad enum
        txn_row("TRX-8", "2026-06-16 03:00:00", "2026-06-16"),                    # breaks the UTC-6 rule
    ])
```

- [ ] **Step 2: Write the failing test** — `tests/test_silver.py`

```python
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
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_silver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline.silver'`.

- [ ] **Step 4: Implement `src/pipeline/contracts.py`**

```python
"""Data contracts for the tables the dispute workflow uses. Raw is read as VARCHAR; the contract
decides what is valid, casts it, and explains every rejection."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Col:
    name: str
    type: str = "VARCHAR"
    required: bool = False
    enum: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Contract:
    table: str
    pk: tuple[str, ...]
    columns: tuple[Col, ...]
    partitioned: bool
    order_by: str  # dedup tie-break: first row per PK in this order is kept

    def reject_reason_sql(self) -> str:
        parts = []
        for c in self.columns:
            q = f'"{c.name}"'
            present = f"nullif(trim({q}), '') is not null"
            if c.required:
                parts.append(f"case when not ({present}) then '{c.name}:missing' end")
            if c.type != "VARCHAR":
                parts.append(f"case when {present} and try_cast(trim({q}) as {c.type}) is null then '{c.name}:bad_type' end")
            if c.enum:
                vals = ", ".join("'" + v.replace("'", "''") + "'" for v in c.enum)
                parts.append(f"case when {present} and trim({q}) not in ({vals}) then '{c.name}:bad_enum' end")
        return "nullif(concat_ws('; ', " + ", ".join(parts) + "), '')"

    def select_sql(self) -> str:
        return ", ".join(f"try_cast(nullif(trim(\"{c.name}\"), '') as {c.type}) as \"{c.name}\"" for c in self.columns)


CURRENCIES = ("MXN", "COP", "ARS", "USD")
TS = "TIMESTAMP"

CONTRACTS = (
    Contract("customers", ("customer_id",), (
        Col("customer_id", required=True), Col("document_number", required=True),
        Col("document_type", required=True, enum=("DNI", "CURP", "CC", "CE", "Pasaporte", "Passport")),
        Col("first_name", required=True), Col("last_name", required=True), Col("mobile_phone"),
        Col("country", required=True, enum=("México", "Colombia", "Argentina")),
        Col("segment", required=True, enum=("Premium", "Plus", "Basic", "Student")),
        Col("customer_status", required=True, enum=("Active", "Inactive", "Suspended", "Closed")),
        Col("last_updated", TS, required=True),
    ), partitioned=False, order_by="last_updated desc"),
    Contract("products", ("product_id",), (
        Col("product_id", required=True), Col("customer_id", required=True),
        Col("product_type", required=True, enum=("Cuenta Ahorro", "Cuenta Corriente", "Tarjeta Crédito",
                                                  "Tarjeta Débito", "Préstamo Personal", "Préstamo Hipotecario",
                                                  "Inversión", "Seguro")),
        Col("product_number", required=True), Col("currency", required=True, enum=CURRENCIES),
        Col("product_status", required=True, enum=("Active", "Blocked", "Closed", "Suspended")),
        Col("last_updated", TS, required=True),
    ), partitioned=False, order_by="last_updated desc"),
    Contract("transactions", ("transaction_id",), (
        Col("transaction_id", required=True), Col("transaction_date", TS, required=True),
        Col("process_date", "DATE", required=True), Col("product_id", required=True), Col("customer_id", required=True),
        Col("transaction_type", required=True, enum=("Deposit", "Withdrawal", "Transfer", "Payment", "Purchase", "Adjustment")),
        Col("amount", "DOUBLE", required=True), Col("currency", required=True, enum=CURRENCIES),
        Col("amount_usd", "DOUBLE"),
        Col("channel", required=True, enum=("ATM", "Branch", "Web", "App", "POS", "Transfer")),
        Col("merchant_name"), Col("merchant_category"), Col("transaction_city"),
        Col("transaction_country", required=True),
        Col("transaction_status", required=True, enum=("Approved", "Declined", "Pending", "Reversed")),
        Col("is_fraud", "BOOLEAN", required=True),
    ), partitioned=True, order_by="process_date desc"),
    Contract("complaints", ("complaint_id",), (
        Col("complaint_id", required=True), Col("creation_date", TS, required=True),
        Col("customer_id", required=True), Col("is_repeat_complainer", "BOOLEAN", required=True),
    ), partitioned=True, order_by="creation_date desc"),
    Contract("daily_exchange_rates", ("date", "source_currency", "target_currency"), (
        Col("date", "DATE", required=True), Col("source_currency", required=True, enum=CURRENCIES),
        Col("target_currency", required=True, enum=CURRENCIES), Col("exchange_rate", "DOUBLE", required=True),
    ), partitioned=False, order_by="date"),
)
```

- [ ] **Step 5: Implement `src/pipeline/silver.py`**

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_silver.py -v`
Expected: 4 passed. If `tz_offset_violations` differs, check that `transaction_date - interval 6 hour` is applied to the TIMESTAMP column (not the VARCHAR).

- [ ] **Step 7: Commit**

```bash
git add src/pipeline/contracts.py src/pipeline/silver.py tests/test_silver.py tests/helpers.py
git commit -m "feat(pipeline): silver layer with contracts, quarantine, dedup, FK and TZ checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Sandbox loader with incremental, idempotent transaction refresh

**Files:**
- Create: `src/pipeline/sandbox.py`, `tests/test_sandbox.py`

**Interfaces:**
- Consumes: `silver.build_silver`, `db.connect`, `db.create_schema`, `clock.now`, `config.*`, `tests.helpers.write_raw/write_txn_day/txn_row`.
- Produces: `sandbox.build_sandbox(silver_dir: Path, sandbox_path: Path, *, today: date = config.SIM_TODAY, days: int = config.SANDBOX_DAYS) -> dict` (keys `customers`, `products`, `fx_rates`, `complaint_flags`, `transactions`); `sandbox.refresh_transactions(conn, silver_dir: Path, since: date, *, until: date = config.SIM_TODAY, reprocess_days: int = config.REPROCESS_DAYS, mode: str = "incremental") -> int` (rows loaded; overwrites every `local_date >= since - reprocess_days`).

- [ ] **Step 1: Write the failing test** — `tests/test_sandbox.py`

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_sandbox.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline.sandbox'`.

- [ ] **Step 3: Implement `src/pipeline/sandbox.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sandbox.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/sandbox.py tests/test_sandbox.py
git commit -m "feat(pipeline): sqlite sandbox with idempotent incremental transaction refresh

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Authentication — OTP challenges and signed sessions

**Files:**
- Create: `src/bank/auth.py`, `tests/test_auth.py`

**Interfaces:**
- Consumes: `db`, `audit.log`, `clock.now`, `config.*`, errors, fixtures `bank`, `frozen`, helpers `login`, `last_code`.
- Produces: `auth.Challenge(challenge_id: str, masked_destination: str, expires_at: datetime)`; `auth.Session(session_id: str, customer_id: str, expires_at: datetime)`; `auth.start_auth(conn, document_number: str) -> Challenge`; `auth.verify_otp(conn, challenge_id: str, code: str) -> str` (session token); `auth.verify_session(token: str) -> Session`; `auth.issue_token(session_id: str, customer_id: str, expires_at: datetime) -> str`. The OTP is "delivered" by inserting into `sandbox_outbox` with body `"Tu código de verificación es NNNNNN"`.

- [ ] **Step 1: Write the failing test** — `tests/test_auth.py`

```python
import base64
import json

import pytest

from src.bank import auth, config
from src.bank.errors import (AccountNotServiceable, NoVerifiedChannel, OtpExpired, OtpInvalid, OtpLocked,
                             SessionExpired, SessionInvalid)
from tests.helpers import last_code, login


def test_happy_path_creates_session(bank, frozen):
    s = auth.verify_session(login(bank, "111"))
    assert s.customer_id == "CLI-A"
    events = [r["event"] for r in bank.execute("select event from audit_log order by id")]
    assert events == ["otp_sent", "session_started"]


def test_challenge_masks_destination(bank, frozen):
    ch = auth.start_auth(bank, "111")
    assert ch.masked_destination == "***1234"


def test_wrong_codes_lock_the_challenge(bank, frozen):
    ch = auth.start_auth(bank, "111")
    good = last_code(bank)
    bad = "000000" if good != "000000" else "111111"
    with pytest.raises(OtpInvalid):
        auth.verify_otp(bank, ch.challenge_id, bad)
    with pytest.raises(OtpInvalid):
        auth.verify_otp(bank, ch.challenge_id, bad)
    with pytest.raises(OtpLocked):
        auth.verify_otp(bank, ch.challenge_id, bad)
    with pytest.raises(OtpLocked):
        auth.verify_otp(bank, ch.challenge_id, good)


def test_otp_expires(bank, frozen):
    ch = auth.start_auth(bank, "111")
    frozen.advance(config.OTP_TTL_SECONDS + 1)
    with pytest.raises(OtpExpired):
        auth.verify_otp(bank, ch.challenge_id, last_code(bank))


def test_otp_is_single_use(bank, frozen):
    ch = auth.start_auth(bank, "111")
    code = last_code(bank)
    auth.verify_otp(bank, ch.challenge_id, code)
    with pytest.raises(OtpInvalid):
        auth.verify_otp(bank, ch.challenge_id, code)


def test_unknown_document_gets_unverifiable_decoy(bank, frozen):
    ch = auth.start_auth(bank, "999999")
    assert ch.masked_destination.startswith("***") and len(ch.masked_destination) == 7
    with pytest.raises(OtpInvalid):
        auth.verify_otp(bank, ch.challenge_id, "123456")
    assert bank.execute("select count(*) from sandbox_outbox").fetchone()[0] == 0


def test_customer_without_phone_cannot_authenticate(bank, frozen):  # Review Focus 1
    with pytest.raises(NoVerifiedChannel):
        auth.start_auth(bank, "333")
    assert bank.execute("select count(*) from otp_challenges").fetchone()[0] == 0


def test_closed_customer_is_not_serviceable(bank, frozen):
    with pytest.raises(AccountNotServiceable):
        auth.start_auth(bank, "444")


def test_session_expires(bank, frozen):
    token = login(bank, "111")
    frozen.advance(config.SESSION_TTL_SECONDS)
    with pytest.raises(SessionExpired):
        auth.verify_session(token)


def test_tampered_or_garbage_token_rejected(bank, frozen):
    token = login(bank, "111")
    payload, sig = token.rsplit(".", 1)
    data = json.loads(base64.urlsafe_b64decode(payload))
    data["cid"] = "CLI-B"
    forged = base64.urlsafe_b64encode(json.dumps(data, sort_keys=True).encode()).decode() + "." + sig
    for bad in (forged, "garbage", ""):
        with pytest.raises(SessionInvalid):
            auth.verify_session(bad)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_auth.py -v`
Expected: FAIL with `ImportError: cannot import name 'auth'`.

- [ ] **Step 3: Implement `src/bank/auth.py`**

```python
"""Simulated OTP authentication and HMAC-signed session tokens.
A document number alone never authenticates: only a verified, unexpired, single-use OTP creates a session."""
import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.bank import audit, clock, config
from src.bank.errors import (AccountNotServiceable, NoVerifiedChannel, OtpExpired, OtpInvalid, OtpLocked,
                             SessionExpired, SessionInvalid)

SERVICEABLE_STATUSES = {"Active", "Inactive"}


@dataclass(frozen=True)
class Challenge:
    challenge_id: str
    masked_destination: str
    expires_at: datetime


@dataclass(frozen=True)
class Session:
    session_id: str
    customer_id: str
    expires_at: datetime


def _code_hash(challenge_id: str, code: str) -> str:
    return hashlib.sha256(f"{challenge_id}:{code}".encode()).hexdigest()


def _mask(phone: str) -> str:
    return "***" + "".join(ch for ch in phone if ch.isdigit())[-4:]


def start_auth(conn: sqlite3.Connection, document_number: str) -> Challenge:
    challenge_id = secrets.token_urlsafe(12)
    expires_at = clock.now() + timedelta(seconds=config.OTP_TTL_SECONDS)
    row = conn.execute("select customer_id, mobile_phone, customer_status from customers where document_number = ?",
                       (document_number.strip(),)).fetchone()
    if row is None:
        # Decoy: same response shape, can never verify -> no account enumeration.
        conn.execute("insert into otp_challenges(challenge_id, customer_id, code_hash, expires_at) values (?,?,?,?)",
                     (challenge_id, None, "!", expires_at.isoformat()))
        conn.commit()
        audit.log(conn, "auth_unknown_document")
        return Challenge(challenge_id, "***" + f"{secrets.randbelow(10**4):04d}", expires_at)
    customer_id = row["customer_id"]
    if row["customer_status"] not in SERVICEABLE_STATUSES:
        audit.log(conn, "auth_refused_status", customer_id=customer_id, status=row["customer_status"])
        raise AccountNotServiceable(row["customer_status"])
    if not row["mobile_phone"]:
        audit.log(conn, "auth_no_verified_channel", customer_id=customer_id)
        raise NoVerifiedChannel()
    code = f"{secrets.randbelow(10**6):06d}"
    conn.execute("insert into otp_challenges(challenge_id, customer_id, code_hash, expires_at) values (?,?,?,?)",
                 (challenge_id, customer_id, _code_hash(challenge_id, code), expires_at.isoformat()))
    conn.execute("insert into sandbox_outbox(channel, destination, body, created_at) values (?,?,?,?)",
                 ("sms", row["mobile_phone"], f"Tu código de verificación es {code}", clock.now().isoformat()))
    conn.commit()
    audit.log(conn, "otp_sent", customer_id=customer_id, challenge_id=challenge_id)
    return Challenge(challenge_id, _mask(row["mobile_phone"]), expires_at)


def verify_otp(conn: sqlite3.Connection, challenge_id: str, code: str) -> str:
    row = conn.execute("select * from otp_challenges where challenge_id = ?", (challenge_id,)).fetchone()
    if row is None or row["customer_id"] is None or row["consumed"]:
        audit.log(conn, "otp_rejected", reason="unknown_or_used")
        raise OtpInvalid()
    customer_id = row["customer_id"]
    if clock.now() > datetime.fromisoformat(row["expires_at"]):
        raise OtpExpired()
    if row["attempts"] >= config.OTP_MAX_ATTEMPTS:
        raise OtpLocked()
    if not hmac.compare_digest(_code_hash(challenge_id, code.strip()), row["code_hash"]):
        attempts = row["attempts"] + 1
        conn.execute("update otp_challenges set attempts = ? where challenge_id = ?", (attempts, challenge_id))
        conn.commit()
        audit.log(conn, "otp_rejected", customer_id=customer_id, reason="wrong_code", attempts=attempts)
        if attempts >= config.OTP_MAX_ATTEMPTS:
            raise OtpLocked()
        raise OtpInvalid()
    conn.execute("update otp_challenges set consumed = 1 where challenge_id = ?", (challenge_id,))
    conn.commit()
    session_id = secrets.token_urlsafe(12)
    expires_at = clock.now() + timedelta(seconds=config.SESSION_TTL_SECONDS)
    audit.log(conn, "session_started", customer_id=customer_id, session_id=session_id)
    return issue_token(session_id, customer_id, expires_at)


def _sign(payload: str) -> str:
    return hmac.new(config.session_secret(), payload.encode(), hashlib.sha256).hexdigest()


def issue_token(session_id: str, customer_id: str, expires_at: datetime) -> str:
    body = json.dumps({"sid": session_id, "cid": customer_id, "exp": expires_at.isoformat()}, sort_keys=True)
    payload = base64.urlsafe_b64encode(body.encode()).decode()
    return f"{payload}.{_sign(payload)}"


def verify_session(token: str) -> Session:
    try:
        payload, sig = token.rsplit(".", 1)
    except (ValueError, AttributeError):
        raise SessionInvalid()
    if not hmac.compare_digest(sig, _sign(payload)):
        raise SessionInvalid()
    data = json.loads(base64.urlsafe_b64decode(payload))
    expires_at = datetime.fromisoformat(data["exp"])
    if clock.now() >= expires_at:
        raise SessionExpired()
    return Session(data["sid"], data["cid"], expires_at)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_auth.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bank/auth.py tests/test_auth.py
git commit -m "feat(bank): simulated OTP auth with lockout, expiry, decoys and signed sessions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Transactions service — scoped search, LLM-safe view, USD resolution

**Files:**
- Create: `src/bank/transactions.py`, `tests/test_transactions.py`

**Interfaces:**
- Consumes: `auth.verify_session`, `audit.log`, `NotFound`, fixtures.
- Produces: `TransactionView(transaction_id: str, local_date: str, description: str, amount: float, currency: str, status: str, card_last4: str | None)`; `TransactionRecord(transaction_id: str, customer_id: str, product_id: str, local_date: date, transaction_type: str, amount: float, currency: str, amount_usd: float | None, status: str, is_fraud: bool, merchant_name: str | None)`; `search_transactions(conn, token, *, amount: float | None = None, date_from: date | None = None, date_to: date | None = None, merchant: str | None = None, limit: int = 10) -> list[TransactionView]` (max 25, newest first); `get_transaction(conn, token, transaction_id) -> TransactionView`; `get_transaction_record(conn, token, transaction_id) -> TransactionRecord`; `resolve_amount_usd(conn, record: TransactionRecord) -> float | None`.

- [ ] **Step 1: Write the failing test** — `tests/test_transactions.py`

```python
import json
from dataclasses import fields
from datetime import date

import pytest

from src.bank import config, transactions as tx
from src.bank.errors import NotFound, SessionExpired
from tests.helpers import login


def ids(views):
    return sorted(v.transaction_id for v in views)


def test_search_is_scoped_to_session_customer(bank, frozen):
    assert ids(tx.search_transactions(bank, login(bank, "222"))) == ["TRX-B1"]


def test_search_filters(bank, frozen):
    tok = login(bank, "111")
    assert ids(tx.search_transactions(bank, tok, amount=350)) == ["TRX-A1", "TRX-A8"]
    assert ids(tx.search_transactions(bank, tok, merchant="oxxo")) == ["TRX-A1", "TRX-A8"]
    assert ids(tx.search_transactions(bank, tok, merchant="RAPPI")) == ["TRX-A4", "TRX-A5"]
    assert ids(tx.search_transactions(bank, tok, date_from=date(2026, 6, 15))) == ["TRX-A1", "TRX-A4", "TRX-A8"]
    assert ids(tx.search_transactions(bank, tok, date_to=date(2026, 1, 31))) == ["TRX-A3"]


def test_search_limit_is_capped(bank, frozen):
    assert len(tx.search_transactions(bank, login(bank, "111"), limit=1000)) == 9  # all of CLI-A, cap is 25


def test_view_is_minimal(bank, frozen):
    v = tx.get_transaction(bank, login(bank, "111"), "TRX-A1")
    assert {f.name for f in fields(v)} == {"transaction_id", "local_date", "description", "amount", "currency",
                                          "status", "card_last4"}
    assert (v.description, v.card_last4, v.status) == ("Oxxo", "1111", "Approved")


def test_null_merchant_falls_back_to_type(bank, frozen):  # Review Focus 2
    tok = login(bank, "111")
    assert tx.get_transaction(bank, tok, "TRX-A2").description == "Purchase"
    assert "TRX-A2" not in ids(tx.search_transactions(bank, tok, merchant="oxxo"))


def test_foreign_transaction_is_not_found_and_audited(bank, frozen):
    tok = login(bank, "222")
    with pytest.raises(NotFound):
        tx.get_transaction(bank, tok, "TRX-A1")
    with pytest.raises(NotFound):
        tx.get_transaction(bank, tok, "TRX-DOES-NOT-EXIST")
    row = bank.execute("select customer_id, detail from audit_log where event='cross_customer_access_attempt'").fetchone()
    assert row["customer_id"] == "CLI-B" and json.loads(row["detail"]) == {"transaction_id": "TRX-A1"}


def test_expired_session_is_rejected(bank, frozen):
    tok = login(bank, "111")
    frozen.advance(config.SESSION_TTL_SECONDS)
    with pytest.raises(SessionExpired):
        tx.search_transactions(bank, tok)


def test_resolve_amount_usd(bank, frozen):  # Review Focus 3
    tok = login(bank, "111")
    rec = lambda t: tx.get_transaction_record(bank, tok, t)  # noqa: E731
    assert tx.resolve_amount_usd(bank, rec("TRX-A1")) == 0.09        # provided
    assert tx.resolve_amount_usd(bank, rec("TRX-A2")) == 30.0        # COP via fx_rates on 2026-06-10
    assert tx.resolve_amount_usd(bank, rec("TRX-A7")) is None        # ARS, no rate that day
    tokb = login(bank, "222")
    assert tx.resolve_amount_usd(bank, tx.get_transaction_record(bank, tokb, "TRX-B1")) == 40.0  # USD, null usd
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_transactions.py -v`
Expected: FAIL with `ImportError: cannot import name 'transactions'`.

- [ ] **Step 3: Implement `src/bank/transactions.py`**

```python
"""Transaction reads, always scoped to the authenticated customer.
`TransactionView` is the only shape that may reach the LLM (data minimisation);
`TransactionRecord` is internal, for policy evaluation."""
import sqlite3
from dataclasses import dataclass
from datetime import date

from src.bank import audit
from src.bank.auth import verify_session
from src.bank.errors import NotFound

MAX_RESULTS = 25
_SELECT = ("select t.*, p.product_number, p.product_type from transactions t "
           "join products p on p.product_id = t.product_id ")


@dataclass(frozen=True)
class TransactionView:
    transaction_id: str
    local_date: str
    description: str
    amount: float
    currency: str
    status: str
    card_last4: str | None


@dataclass(frozen=True)
class TransactionRecord:
    transaction_id: str
    customer_id: str
    product_id: str
    local_date: date
    transaction_type: str
    amount: float
    currency: str
    amount_usd: float | None
    status: str
    is_fraud: bool
    merchant_name: str | None


def _view(row) -> TransactionView:
    last4 = row["product_number"][-4:] if row["product_type"].startswith("Tarjeta") else None
    return TransactionView(row["transaction_id"], row["local_date"], row["merchant_name"] or row["transaction_type"],
                           row["amount"], row["currency"], row["transaction_status"], last4)


def _record(row) -> TransactionRecord:
    return TransactionRecord(row["transaction_id"], row["customer_id"], row["product_id"],
                             date.fromisoformat(row["local_date"]), row["transaction_type"], row["amount"],
                             row["currency"], row["amount_usd"], row["transaction_status"], bool(row["is_fraud"]),
                             row["merchant_name"])


def search_transactions(conn: sqlite3.Connection, token: str, *, amount: float | None = None,
                        date_from: date | None = None, date_to: date | None = None, merchant: str | None = None,
                        limit: int = 10) -> list[TransactionView]:
    s = verify_session(token)
    clauses, params = ["t.customer_id = ?"], [s.customer_id]
    if amount is not None:
        clauses.append("abs(t.amount - ?) <= max(0.01, abs(?) * 0.01)")
        params += [amount, amount]
    if date_from is not None:
        clauses.append("t.local_date >= ?")
        params.append(date_from.isoformat())
    if date_to is not None:
        clauses.append("t.local_date <= ?")
        params.append(date_to.isoformat())
    if merchant:
        clauses.append("lower(coalesce(t.merchant_name, t.transaction_type)) like ?")
        params.append(f"%{merchant.strip().lower()}%")
    params.append(max(1, min(limit, MAX_RESULTS)))
    rows = conn.execute(_SELECT + "where " + " and ".join(clauses) + " order by t.transaction_ts_utc desc limit ?",
                        params).fetchall()
    audit.log(conn, "txn_search", customer_id=s.customer_id, session_id=s.session_id, results=len(rows),
              filters={"amount": amount, "date_from": date_from, "date_to": date_to, "merchant": merchant})
    return [_view(r) for r in rows]


def _owned_row(conn: sqlite3.Connection, token: str, transaction_id: str):
    s = verify_session(token)
    row = conn.execute(_SELECT + "where t.transaction_id = ? and t.customer_id = ?",
                       (transaction_id, s.customer_id)).fetchone()
    if row is None:
        exists = conn.execute("select 1 from transactions where transaction_id = ?", (transaction_id,)).fetchone()
        if exists:
            audit.log(conn, "cross_customer_access_attempt", customer_id=s.customer_id, session_id=s.session_id,
                      transaction_id=transaction_id)
        raise NotFound()
    return row


def get_transaction(conn: sqlite3.Connection, token: str, transaction_id: str) -> TransactionView:
    return _view(_owned_row(conn, token, transaction_id))


def get_transaction_record(conn: sqlite3.Connection, token: str, transaction_id: str) -> TransactionRecord:
    return _record(_owned_row(conn, token, transaction_id))


def resolve_amount_usd(conn: sqlite3.Connection, record: TransactionRecord) -> float | None:
    if record.amount_usd is not None:
        return record.amount_usd
    if record.currency == "USD":
        return record.amount
    row = conn.execute("select rate_to_usd from fx_rates where currency = ? and date = ?",
                       (record.currency, record.local_date.isoformat())).fetchone()
    return round(record.amount * row["rate_to_usd"], 2) if row else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_transactions.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bank/transactions.py tests/test_transactions.py
git commit -m "feat(bank): customer-scoped transaction service with minimal LLM view and USD resolution

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Dispute policy v1 — deterministic, versioned

**Files:**
- Create: `policy/dispute_policy_v1.yaml`, `src/bank/policy.py`, `tests/test_policy.py`

**Interfaces:**
- Consumes: `transactions.TransactionRecord`, `config.POLICY_PATH`.
- Produces: `policy.DISPUTE_TYPES: frozenset[str]` = `{"unrecognized", "duplicate", "amount_mismatch", "undue_fee", "refund_not_received"}`; `Policy(version: str, window_days: int, auto_limit_usd: float, human_review_dispute_count: int)`; `load_policy(path: Path = config.POLICY_PATH) -> Policy`; `DisputeContext(transaction: TransactionRecord, amount_usd: float | None, disputed_in_session: int, is_repeat_complainer: bool, existing_dispute_id: str | None, today: date)`; `Decision(outcome: Literal["eligible", "ineligible", "requires_human"], rule_id: str, reason: str, policy_version: str, existing_dispute_id: str | None = None)`; `evaluate(ctx: DisputeContext, policy: Policy) -> Decision`.

- [ ] **Step 1: Write the failing test** — `tests/test_policy.py`

```python
from dataclasses import replace
from datetime import date

import pytest

from src.bank.policy import DisputeContext, evaluate, load_policy
from src.bank.transactions import TransactionRecord

POLICY = load_policy()
TODAY = date(2026, 6, 17)
REC = TransactionRecord("T1", "C1", "P1", date(2026, 6, 10), "Purchase", 100.0, "COP", 25.0, "Approved", False, "Oxxo")


def ctx(**kw):
    base = dict(transaction=REC, amount_usd=25.0, disputed_in_session=1, is_repeat_complainer=False,
                existing_dispute_id=None, today=TODAY)
    base.update(kw)
    return DisputeContext(**base)


def test_policy_file_values():
    assert (POLICY.version, POLICY.window_days, POLICY.auto_limit_usd, POLICY.human_review_dispute_count) == \
           ("dispute_policy_v1", 90, 500.0, 3)


@pytest.mark.parametrize("kw, outcome, rule", [
    ({}, "eligible", "P8"),
    ({"transaction": replace(REC, local_date=date(2026, 3, 18))}, "ineligible", "P1"),   # 91 days old
    ({"transaction": replace(REC, local_date=date(2026, 3, 19))}, "eligible", "P8"),     # exactly 90 days
    ({"transaction": replace(REC, status="Declined")}, "ineligible", "P2"),
    ({"transaction": replace(REC, status="Reversed")}, "ineligible", "P3"),
    ({"amount_usd": 500.01}, "requires_human", "P4"),
    ({"amount_usd": 500.0}, "eligible", "P8"),
    ({"amount_usd": None}, "requires_human", "P4"),                                        # Review Focus 3
    ({"disputed_in_session": 3}, "requires_human", "P5"),
    ({"is_repeat_complainer": True}, "requires_human", "P6"),
    ({"existing_dispute_id": "DSP-1"}, "ineligible", "P7"),
])
def test_rules(kw, outcome, rule):
    d = evaluate(ctx(**kw), POLICY)
    assert (d.outcome, d.rule_id, d.policy_version) == (outcome, rule, "dispute_policy_v1")


def test_first_match_wins():
    d = evaluate(ctx(transaction=replace(REC, status="Declined"), amount_usd=9999), POLICY)
    assert d.rule_id == "P2"


def test_existing_dispute_id_is_returned():
    assert evaluate(ctx(existing_dispute_id="DSP-9"), POLICY).existing_dispute_id == "DSP-9"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_policy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.bank.policy'`.

- [ ] **Step 3: Write `policy/dispute_policy_v1.yaml`**

```yaml
# SYNTHETIC POLICY — team assumptions for the hackathon prototype, not a real bank's rules.
version: dispute_policy_v1
window_days: 90                  # P1: older than this -> ineligible
auto_limit_usd: 500              # P4: above this (or not convertible to USD) -> human
human_review_dispute_count: 3    # P5: this many disputed transactions in one session -> human
```

- [ ] **Step 4: Implement `src/bank/policy.py`**

```python
"""Deterministic dispute policy. The LLM never evaluates these rules; it only communicates the Decision.
Rules are checked in order and the first non-eligible match wins."""
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import yaml

from src.bank import config
from src.bank.transactions import TransactionRecord

DISPUTE_TYPES = frozenset({"unrecognized", "duplicate", "amount_mismatch", "undue_fee", "refund_not_received"})
Outcome = Literal["eligible", "ineligible", "requires_human"]


@dataclass(frozen=True)
class Policy:
    version: str
    window_days: int
    auto_limit_usd: float
    human_review_dispute_count: int


@dataclass(frozen=True)
class DisputeContext:
    transaction: TransactionRecord
    amount_usd: float | None
    disputed_in_session: int
    is_repeat_complainer: bool
    existing_dispute_id: str | None
    today: date


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    rule_id: str
    reason: str
    policy_version: str
    existing_dispute_id: str | None = None


def load_policy(path: Path = config.POLICY_PATH) -> Policy:
    data = yaml.safe_load(Path(path).read_text())
    return Policy(str(data["version"]), int(data["window_days"]), float(data["auto_limit_usd"]),
                  int(data["human_review_dispute_count"]))


def evaluate(ctx: DisputeContext, policy: Policy) -> Decision:
    def d(outcome: Outcome, rule: str, reason: str, existing: str | None = None) -> Decision:
        return Decision(outcome, rule, reason, policy.version, existing)

    t = ctx.transaction
    if (ctx.today - t.local_date).days > policy.window_days:
        return d("ineligible", "P1", "outside dispute window")
    if t.status == "Declined":
        return d("ineligible", "P2", "no charge was made")
    if t.status == "Reversed":
        return d("ineligible", "P3", "already reversed")
    if ctx.amount_usd is None:
        return d("requires_human", "P4", "amount could not be converted to USD")
    if ctx.amount_usd > policy.auto_limit_usd:
        return d("requires_human", "P4", "amount above automatic limit")
    if ctx.disputed_in_session >= policy.human_review_dispute_count:
        return d("requires_human", "P5", "multiple disputed charges")
    if ctx.is_repeat_complainer:
        return d("requires_human", "P6", "repeat complainer review")
    if ctx.existing_dispute_id:
        return d("ineligible", "P7", "already disputed", ctx.existing_dispute_id)
    return d("eligible", "P8", "eligible for automatic filing")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_policy.py -v`
Expected: 14 passed.

- [ ] **Step 6: Commit**

```bash
git add policy/dispute_policy_v1.yaml src/bank/policy.py tests/test_policy.py
git commit -m "feat(bank): deterministic versioned dispute policy v1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Disputes service — evaluate, confirmed idempotent create, read-back

**Files:**
- Create: `src/bank/disputes.py`, `tests/test_disputes.py`

**Interfaces:**
- Consumes: `auth.verify_session`, `transactions.get_transaction_record`, `transactions.resolve_amount_usd`, `policy.*`, `audit.log`, errors, `config.SIM_TODAY`.
- Produces: `Dispute(dispute_id: str, transaction_id: str, dispute_type: str, status: str, created_at: str, policy_rule: str, policy_version: str)`; `evaluate_dispute(conn, token, transaction_id: str, dispute_type: str, *, declared_disputed_count: int = 0, policy: Policy | None = None) -> Decision`; `create_dispute(conn, token, transaction_id: str, dispute_type: str, *, idempotency_key: str, customer_confirmed: bool, declared_disputed_count: int = 0) -> Dispute` (raises `ConfirmationRequired`, `PolicyViolation`, `NotFound`, `ValueError` for unknown type); `get_dispute(conn, token, dispute_id: str) -> Dispute`. `disputed_in_session = max(<this session's other disputes in DB> + 1, declared_disputed_count)`.

- [ ] **Step 1: Write the failing test** — `tests/test_disputes.py`

```python
import pytest

from src.bank import disputes as ds
from src.bank.errors import ConfirmationRequired, NotFound, PolicyViolation
from tests.helpers import login


def create(conn, tok, txn, key, **kw):
    return ds.create_dispute(conn, tok, txn, kw.pop("dtype", "unrecognized"), idempotency_key=key,
                             customer_confirmed=kw.pop("confirmed", True), **kw)


def count(conn):
    return conn.execute("select count(*) from disputes").fetchone()[0]


def test_create_and_read_back(bank, frozen):
    tok = login(bank, "111")
    d = create(bank, tok, "TRX-A1", "k1")
    assert (d.transaction_id, d.status, d.policy_rule, d.policy_version) == ("TRX-A1", "open", "P8", "dispute_policy_v1")
    assert ds.get_dispute(bank, tok, d.dispute_id) == d
    assert bank.execute("select count(*) from audit_log where event='dispute_created'").fetchone()[0] == 1


def test_requires_explicit_confirmation(bank, frozen):
    with pytest.raises(ConfirmationRequired):
        create(bank, login(bank, "111"), "TRX-A1", "k1", confirmed=False)
    assert count(bank) == 0


def test_same_idempotency_key_returns_same_case(bank, frozen):  # Review Focus 4
    tok = login(bank, "111")
    first, again = create(bank, tok, "TRX-A1", "k1"), create(bank, tok, "TRX-A1", "k1")
    assert first == again and count(bank) == 1


def test_new_key_on_already_disputed_transaction_is_p7(bank, frozen):
    tok = login(bank, "111")
    first = create(bank, tok, "TRX-A1", "k1")
    with pytest.raises(PolicyViolation) as e:
        create(bank, tok, "TRX-A1", "k2")
    assert (e.value.decision.rule_id, e.value.decision.existing_dispute_id) == ("P7", first.dispute_id)


@pytest.mark.parametrize("txn, rule, outcome", [
    ("TRX-A3", "P1", "ineligible"), ("TRX-A4", "P2", "ineligible"), ("TRX-A5", "P3", "ineligible"),
    ("TRX-A6", "P4", "requires_human"), ("TRX-A7", "P4", "requires_human"),  # A7: ARS with no FX rate
])
def test_policy_is_enforced_by_the_service(bank, frozen, txn, rule, outcome):
    tok = login(bank, "111")
    with pytest.raises(PolicyViolation) as e:
        create(bank, tok, txn, "k-" + txn)
    assert (e.value.decision.rule_id, e.value.decision.outcome) == (rule, outcome)
    assert count(bank) == 0


def test_repeat_complainer_goes_to_human(bank, frozen):
    d = ds.evaluate_dispute(bank, login(bank, "222"), "TRX-B1", "unrecognized")
    assert (d.outcome, d.rule_id) == ("requires_human", "P6")


def test_third_dispute_in_session_goes_to_human(bank, frozen):
    tok = login(bank, "111")
    create(bank, tok, "TRX-A1", "k1")
    create(bank, tok, "TRX-A9", "k2")
    d = ds.evaluate_dispute(bank, tok, "TRX-A8", "duplicate")
    assert (d.outcome, d.rule_id) == ("requires_human", "P5")


def test_declared_count_from_orchestrator_is_respected(bank, frozen):
    d = ds.evaluate_dispute(bank, login(bank, "111"), "TRX-A1", "unrecognized", declared_disputed_count=5)
    assert d.rule_id == "P5"


def test_other_customer_cannot_see_or_replay(bank, frozen):
    d = create(bank, login(bank, "111"), "TRX-A1", "k1")
    tokb = login(bank, "222")
    with pytest.raises(NotFound):
        ds.get_dispute(bank, tokb, d.dispute_id)
    with pytest.raises(NotFound):
        create(bank, tokb, "TRX-B1", "k1")  # someone else's idempotency key


def test_unknown_dispute_type_rejected(bank, frozen):
    with pytest.raises(ValueError):
        ds.evaluate_dispute(bank, login(bank, "111"), "TRX-A1", "please_refund_me")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_disputes.py -v`
Expected: FAIL with `ImportError: cannot import name 'disputes'`.

- [ ] **Step 3: Implement `src/bank/disputes.py`**

```python
"""Dispute cases. The service re-evaluates policy itself on every create: it never trusts the caller's
claim that a dispute is eligible. Writes require explicit confirmation and are idempotent per key."""
import secrets
import sqlite3
from dataclasses import dataclass

from src.bank import audit, clock, config
from src.bank.auth import verify_session
from src.bank.errors import ConfirmationRequired, NotFound, PolicyViolation
from src.bank.policy import DISPUTE_TYPES, Decision, DisputeContext, Policy, evaluate, load_policy
from src.bank.transactions import get_transaction_record, resolve_amount_usd


@dataclass(frozen=True)
class Dispute:
    dispute_id: str
    transaction_id: str
    dispute_type: str
    status: str
    created_at: str
    policy_rule: str
    policy_version: str


def _dispute(row) -> Dispute:
    return Dispute(row["dispute_id"], row["transaction_id"], row["dispute_type"], row["status"], row["created_at"],
                   row["policy_rule"], row["policy_version"])


def evaluate_dispute(conn: sqlite3.Connection, token: str, transaction_id: str, dispute_type: str, *,
                     declared_disputed_count: int = 0, policy: Policy | None = None) -> Decision:
    if dispute_type not in DISPUTE_TYPES:
        raise ValueError(f"unknown dispute type: {dispute_type}")
    s = verify_session(token)
    record = get_transaction_record(conn, token, transaction_id)
    others = conn.execute("select count(*) from disputes where session_id = ? and transaction_id <> ?",
                          (s.session_id, transaction_id)).fetchone()[0]
    existing = conn.execute("select dispute_id from disputes where customer_id = ? and transaction_id = ? "
                            "and status = 'open'", (s.customer_id, transaction_id)).fetchone()
    flag = conn.execute("select is_repeat_complainer from complaint_flags where customer_id = ?",
                        (s.customer_id,)).fetchone()
    ctx = DisputeContext(transaction=record, amount_usd=resolve_amount_usd(conn, record),
                         disputed_in_session=max(others + 1, declared_disputed_count),
                         is_repeat_complainer=bool(flag and flag["is_repeat_complainer"]),
                         existing_dispute_id=existing["dispute_id"] if existing else None, today=config.SIM_TODAY)
    decision = evaluate(ctx, policy or load_policy())
    audit.log(conn, "policy_evaluated", customer_id=s.customer_id, session_id=s.session_id,
              transaction_id=transaction_id, dispute_type=dispute_type, outcome=decision.outcome,
              rule_id=decision.rule_id, policy_version=decision.policy_version)
    return decision


def get_dispute(conn: sqlite3.Connection, token: str, dispute_id: str) -> Dispute:
    s = verify_session(token)
    row = conn.execute("select * from disputes where dispute_id = ? and customer_id = ?",
                       (dispute_id, s.customer_id)).fetchone()
    if row is None:
        raise NotFound()
    return _dispute(row)


def create_dispute(conn: sqlite3.Connection, token: str, transaction_id: str, dispute_type: str, *,
                   idempotency_key: str, customer_confirmed: bool, declared_disputed_count: int = 0) -> Dispute:
    s = verify_session(token)
    prior = conn.execute("select * from disputes where idempotency_key = ?", (idempotency_key,)).fetchone()
    if prior is not None:
        if prior["customer_id"] != s.customer_id:
            raise NotFound()
        return _dispute(prior)
    if not customer_confirmed:
        raise ConfirmationRequired()
    decision = evaluate_dispute(conn, token, transaction_id, dispute_type,
                                declared_disputed_count=declared_disputed_count)
    if decision.outcome != "eligible":
        raise PolicyViolation(decision)
    dispute_id = "DSP-" + secrets.token_hex(5).upper()
    conn.execute("insert into disputes values (?,?,?,?,?,?,?,?,?,?)",
                 (dispute_id, idempotency_key, s.session_id, s.customer_id, transaction_id, dispute_type, "open",
                  decision.rule_id, decision.policy_version, clock.now().isoformat()))
    conn.commit()
    audit.log(conn, "dispute_created", customer_id=s.customer_id, session_id=s.session_id,
              dispute_id=dispute_id, transaction_id=transaction_id)
    return get_dispute(conn, token, dispute_id)  # read-back: what we return is what is stored
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_disputes.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bank/disputes.py tests/test_disputes.py
git commit -m "feat(bank): dispute service with in-service policy check, confirmation and idempotency

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Cards service — confirmed, idempotent block with read-back

**Files:**
- Create: `src/bank/cards.py`, `tests/test_cards.py`

**Interfaces:**
- Consumes: `auth.verify_session`, `audit.log`, `clock.now`, errors.
- Produces: `CARD_TYPES = frozenset({"Tarjeta Crédito", "Tarjeta Débito"})`; `CardStatus(product_id: str, product_type: str, last4: str, status: str)`; `list_cards(conn, token) -> list[CardStatus]`; `get_card(conn, token, product_id: str) -> CardStatus` (`NotFound` if not owned, `InvalidProduct` if not a card); `block_card(conn, token, product_id: str, *, customer_confirmed: bool, reason: str) -> CardStatus`.

- [ ] **Step 1: Write the failing test** — `tests/test_cards.py`

```python
import pytest

from src.bank import cards
from src.bank.errors import ConfirmationRequired, InvalidProduct, NotFound
from tests.helpers import login


def blocks(conn):
    return conn.execute("select count(*) from card_blocks").fetchone()[0]


def test_list_cards_only_own_cards(bank, frozen):
    got = cards.list_cards(bank, login(bank, "111"))
    assert sorted((c.product_id, c.last4, c.status) for c in got) == [
        ("PRD-A-CC", "1111", "Active"), ("PRD-A-OLD", "3333", "Closed")]


def test_block_with_confirmation_and_read_back(bank, frozen):
    tok = login(bank, "111")
    c = cards.block_card(bank, tok, "PRD-A-CC", customer_confirmed=True, reason="stolen")
    assert c.status == "Blocked" and cards.get_card(bank, tok, "PRD-A-CC").status == "Blocked"
    assert blocks(bank) == 1


def test_block_requires_confirmation(bank, frozen):
    with pytest.raises(ConfirmationRequired):
        cards.block_card(bank, login(bank, "111"), "PRD-A-CC", customer_confirmed=False, reason="stolen")
    assert blocks(bank) == 0


def test_block_is_idempotent(bank, frozen):
    tok = login(bank, "111")
    cards.block_card(bank, tok, "PRD-A-CC", customer_confirmed=True, reason="stolen")
    cards.block_card(bank, tok, "PRD-A-CC", customer_confirmed=True, reason="stolen")
    assert blocks(bank) == 1


@pytest.mark.parametrize("product, error", [
    ("PRD-A-SAV", InvalidProduct),   # savings account is not a card
    ("PRD-A-OLD", InvalidProduct),   # closed card
    ("PRD-B-DC", NotFound),          # someone else's card
    ("PRD-NOPE", NotFound),
])
def test_block_rejects_invalid_targets(bank, frozen, product, error):  # Review Focus 5
    with pytest.raises(error):
        cards.block_card(bank, login(bank, "111"), product, customer_confirmed=True, reason="stolen")
    assert blocks(bank) == 0
    assert bank.execute("select product_status from products where product_id='PRD-B-DC'").fetchone()[0] == "Active"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cards.py -v`
Expected: FAIL with `ImportError: cannot import name 'cards'`.

- [ ] **Step 3: Implement `src/bank/cards.py`**

```python
"""Card lookups and blocking, scoped to the authenticated customer. Blocking needs explicit confirmation,
is idempotent, and returns the state read back from storage."""
import secrets
import sqlite3
from dataclasses import dataclass

from src.bank import audit, clock
from src.bank.auth import verify_session
from src.bank.errors import ConfirmationRequired, InvalidProduct, NotFound

CARD_TYPES = frozenset({"Tarjeta Crédito", "Tarjeta Débito"})


@dataclass(frozen=True)
class CardStatus:
    product_id: str
    product_type: str
    last4: str
    status: str


def _card(row) -> CardStatus:
    return CardStatus(row["product_id"], row["product_type"], row["product_number"][-4:], row["product_status"])


def list_cards(conn: sqlite3.Connection, token: str) -> list[CardStatus]:
    s = verify_session(token)
    rows = conn.execute("select * from products where customer_id = ? and product_type in (?, ?) order by product_id",
                        (s.customer_id, *sorted(CARD_TYPES))).fetchall()
    return [_card(r) for r in rows]


def get_card(conn: sqlite3.Connection, token: str, product_id: str) -> CardStatus:
    s = verify_session(token)
    row = conn.execute("select * from products where product_id = ? and customer_id = ?",
                       (product_id, s.customer_id)).fetchone()
    if row is None:
        raise NotFound()
    if row["product_type"] not in CARD_TYPES:
        raise InvalidProduct("not a card")
    return _card(row)


def block_card(conn: sqlite3.Connection, token: str, product_id: str, *, customer_confirmed: bool,
               reason: str) -> CardStatus:
    s = verify_session(token)
    card = get_card(conn, token, product_id)
    if not customer_confirmed:
        raise ConfirmationRequired()
    if card.status == "Blocked":
        audit.log(conn, "card_block_noop", customer_id=s.customer_id, session_id=s.session_id, product_id=product_id)
        return card
    if card.status == "Closed":
        raise InvalidProduct("card is closed")
    with conn:
        conn.execute("update products set product_status = 'Blocked' where product_id = ? and customer_id = ?",
                     (product_id, s.customer_id))
        conn.execute("insert into card_blocks values (?,?,?,?,?,?)",
                     ("BLK-" + secrets.token_hex(5).upper(), product_id, s.customer_id, s.session_id, reason,
                      clock.now().isoformat()))
    audit.log(conn, "card_blocked", customer_id=s.customer_id, session_id=s.session_id, product_id=product_id,
              reason=reason)
    return get_card(conn, token, product_id)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cards.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bank/cards.py tests/test_cards.py
git commit -m "feat(bank): card service with confirmed idempotent blocking and read-back

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Pipeline CLI, Makefile, real-data run, README

**Files:**
- Create: `src/pipeline/run.py`, `Makefile`, `README.md`, `reports/data_quality_latest.json` (generated), `tests/test_run.py`

**Interfaces:**
- Consumes: `silver.build_silver`, `sandbox.build_sandbox`, `config.*`.
- Produces: `uv run python -m src.pipeline.run {silver|sandbox|all}`; `make setup | download | pipeline | test`.

- [ ] **Step 1: Write the failing test** — `tests/test_run.py`

```python
import json
import sys

from src.pipeline import run
from tests.helpers import write_raw


def test_run_all_end_to_end(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    write_raw(raw)
    monkeypatch.setattr(run.config, "RAW_DIR", raw)
    monkeypatch.setattr(run.config, "SILVER_DIR", tmp_path / "silver")
    monkeypatch.setattr(run.config, "SANDBOX_PATH", tmp_path / "s.db")
    monkeypatch.setattr(run, "QUALITY_REPORT", tmp_path / "quality.json")
    monkeypatch.setattr(sys, "argv", ["run", "all"])
    run.main()
    assert json.loads((tmp_path / "quality.json").read_text())["tables"]["customers"]["rows_out"] == 2
    assert (tmp_path / "s.db").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_run.py -v`
Expected: FAIL with `ImportError: cannot import name 'run'`.

- [ ] **Step 3: Implement `src/pipeline/run.py`**

```python
"""Pipeline entrypoint: uv run python -m src.pipeline.run {silver|sandbox|all}

The sandbox is a generated artifact: `sandbox` rebuilds it from scratch (disputes/blocks are reset)."""
import argparse
import json
from pathlib import Path

from src.bank import config
from src.pipeline.sandbox import build_sandbox
from src.pipeline.silver import build_silver

QUALITY_REPORT = Path("reports/data_quality_latest.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["silver", "sandbox", "all"])
    step = ap.parse_args().step
    if step in ("silver", "all"):
        report = build_silver(config.RAW_DIR, config.SILVER_DIR)
        QUALITY_REPORT.parent.mkdir(parents=True, exist_ok=True)
        QUALITY_REPORT.write_text(json.dumps(report, indent=2))
        summary = {t: {k: v for k, v in s.items() if not k.endswith("reasons")} for t, s in report["tables"].items()}
        print(json.dumps(summary, indent=2))
    if step in ("sandbox", "all"):
        Path(config.SANDBOX_PATH).unlink(missing_ok=True)
        print(build_sandbox(config.SILVER_DIR, config.SANDBOX_PATH))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_run.py -v`
Expected: 1 passed.

- [ ] **Step 5: Create `Makefile`**

```makefile
.PHONY: setup download pipeline test

setup:
	uv sync

download:
	uv run --env-file .env python -m src.pipeline.download --exclude digital_events

pipeline:
	uv run python -m src.pipeline.run all

test:
	uv run pytest -q
```

- [ ] **Step 6: Run the pipeline on the real data and sanity-check it**

Run: `make pipeline`
Expected (from the EDA): customers `rows_out` 150000; products 400000; transactions `rows_in` 4425008 with `quarantined` 0, `duplicates_removed` 0, `fk_quarantined` 0, and `tz_offset_violations` 0 (confirms Q9); the sandbox prints about 480k–500k transactions for the 120-day window. **If any number differs, do not "fix" the test data. Record the actual value and the reason in `reports/eda_findings.md` under a new "Pipeline v1 run" note.**

Then smoke-test a real customer from a Python shell:

```bash
uv run python -c "
from src.bank import db, auth, transactions as tx
from tests.helpers import last_code
c = db.connect('data/sandbox.db')
doc = c.execute(\"select document_number from customers where customer_status='Active' and mobile_phone is not null limit 1\").fetchone()[0]
ch = auth.start_auth(c, doc); tok = auth.verify_otp(c, ch.challenge_id, last_code(c))
print(ch.masked_destination, tx.search_transactions(c, tok, limit=3))
"
```

Expected: a masked phone and up to 3 `TransactionView`s belonging to that customer.

- [ ] **Step 7: Write `README.md` (setup and data sections only; later plans extend it)**

````markdown
# factored-hackathon-2026 — AI-first transaction-dispute agent

Customer-service system for LATAM Bank (synthetic) that takes in card/account transaction disputes in
Spanish and Portuguese, with a deterministic policy, permissions enforced in the service layer, and
structured human handoff. Design: `docs/superpowers/specs/2026-09-26-dispute-agent-design.md`.

## Setup

```bash
cp .env.example .env   # fill in the organizer-provided read-only S3 keys
make setup
make download          # S3 -> data/raw (+ lineage manifest)
make pipeline          # data/raw -> data/silver (contracts, quarantine, quality report) -> data/sandbox.db
make test
```

## Data

| Layer | Location | What it guarantees |
|---|---|---|
| Bronze | `data/raw/` + `_manifest.csv` | Exact S3 objects (ETag, size, timestamp) |
| Silver | `data/silver/*.parquet` | Typed, contract-checked, deduplicated; rejects in `_quarantine/` with reasons; run report in `_quality/` |
| Sandbox | `data/sandbox.db` | Last 120 days of transactions before simulated today (2026-06-17); incremental refresh with a 3-day reprocess window |

The dataset is synthetic (organizer-provided). Known limitations are listed in `reports/eda_findings.md`.
The dispute policy (`policy/dispute_policy_v1.yaml`) is a **synthetic, team-defined** policy.
````

- [ ] **Step 8: Run the full test suite**

Run: `make test`
Expected: all tests pass (≈64).

- [ ] **Step 9: Commit**

```bash
git add src/pipeline/run.py Makefile README.md reports/data_quality_latest.json tests/test_run.py \
        src/pipeline/download.py src/pipeline/views.py src/eda reports/eda_findings.md reports/eda_output.md \
        pyproject.toml uv.lock .python-version
git commit -m "feat(pipeline): CLI, Makefile, README; first full run on the real dataset

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Roadmap — following plans (each written after the previous one ships)

| Plan | Spec sections | Deliverable | Depends on |
|---|---|---|---|
| **2. Router** (day 3) | 6 | Labelled ES/PT utterance set (seed-grouped splits, hand-written test), keyword baseline vs TF-IDF+LR vs e5+LR, validation-tuned abstention threshold, versioned model artifact | none (independent of Plan 1) |
| **3. Orchestrator + LLM + baseline** (days 4–5) | 2, 3, 7 | FSM with per-stage tool allowlists, OpenRouter client with timeouts/retries/cost tracking, handoff builder, tracer, FastAPI endpoints, rules-only baseline | Plans 1, 2 |
| **4. UI + deploy + channels** (days 6, 8) | 9 | Web chat + live trace panel (CopilotKit or plain), container deploy, Slack then Telegram adapters | Plan 3 |
| **5. Evaluation** (days 7–9) | 8 | ~200 scripted held-out conversations, trace-based scorer, metrics report by language/segment, baseline vs hybrid, cost and latency | Plans 3 (and 2) |
