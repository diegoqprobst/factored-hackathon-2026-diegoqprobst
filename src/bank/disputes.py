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
