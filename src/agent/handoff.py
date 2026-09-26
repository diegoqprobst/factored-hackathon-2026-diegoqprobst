"""Structured handoff to a human agent: the request, what was verified, what was done, and what is still open —
not a raw transcript."""
import json
import secrets

from src.agent.state import Conversation
from src.bank import clock, db


def _summary(conv: Conversation) -> str:
    parts = [f"intent={conv.intent or 'unknown'}"]
    if conv.dispute_type:
        parts.append(f"dispute_type={conv.dispute_type}")
    if conv.selected_view:
        v = conv.selected_view
        parts.append(f"charge={v['description']} {v['amount']:.2f} {v['currency']} on {v['local_date']}")
    if conv.disputes:
        parts.append(f"disputes_created={','.join(conv.disputes)}")
    return "; ".join(parts)


def _open_questions(conv: Conversation) -> list[str]:
    questions = []
    if not conv.token:
        questions.append("identity not verified")
    if not conv.transaction_id and conv.intent != "card_lost_stolen":
        questions.append("charge not identified")
    if conv.transaction_id and not conv.dispute_type:
        questions.append("dispute type not stated")
    if conv.charges_count and conv.charges_count > 1:
        questions.append(f"customer reports {conv.charges_count} disputed charges")
    for a in conv.actions:
        if a.get("verified") is False:
            questions.append(f"verify status of {a['action']} {a.get('id')} (write not confirmed by read-back)")
    return questions


def build_handoff(conv: Conversation, reason: str, *, trace_id: str) -> dict:
    return {
        "handoff_id": "HND-" + secrets.token_hex(5).upper(),
        "conversation_id": conv.id,
        "created_at": clock.now().isoformat(),
        "language": conv.language,
        "escalation_reason": reason,
        "customer_id": conv.customer_id if conv.token else None,
        "auth_level": "otp_verified" if conv.token else "unverified",
        "request_summary": _summary(conv),
        "transactions": [conv.selected_view] if conv.selected_view else list(conv.candidates),
        "dispute_type": conv.dispute_type,
        "policy_decision": conv.decision,
        "actions_taken": list(conv.actions),
        "open_questions": _open_questions(conv),
        "trace_id": trace_id,
    }


def save_handoff(conn, payload: dict) -> str:
    with db.LOCK:
        conn.execute("insert into handoffs values (?,?,?,?,?,?)",
                     (payload["handoff_id"], payload["conversation_id"], payload["customer_id"],
                      payload["escalation_reason"], json.dumps(payload, default=str), payload["created_at"]))
        conn.commit()
    return payload["handoff_id"]
