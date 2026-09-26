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
