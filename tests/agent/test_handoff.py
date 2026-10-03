import json

from src.agent.handoff import build_handoff, save_handoff
from src.agent.state import Conversation, Stage

VIEW = {"transaction_id": "TRX-A6", "local_date": "2026-06-12", "description": "Falabella", "amount": 2500000.0,
        "currency": "COP", "status": "Approved", "card_last4": "1111"}


def test_verified_customer_handoff_payload(bank, frozen):
    c = Conversation("c1", stage=Stage.CLASSIFY, language="es", token="t", customer_id="CLI-A",
                     intent="dispute_unrecognized", dispute_type="unrecognized", transaction_id="TRX-A6",
                     selected_view=VIEW, decision={"outcome": "requires_human", "rule_id": "P4"},
                     actions=[{"action": "authenticate", "verified": True}])
    p = build_handoff(c, "policy_P4", trace_id="c1:5")
    assert p["handoff_id"].startswith("HND-") and p["auth_level"] == "otp_verified" and p["customer_id"] == "CLI-A"
    assert p["transactions"] == [VIEW] and p["policy_decision"]["rule_id"] == "P4"
    assert p["escalation_reason"] == "policy_P4" and p["trace_id"] == "c1:5"
    assert "Falabella" in p["request_summary"] and "unrecognized" in p["request_summary"]
    assert "transcript" not in json.dumps(p)
    hid = save_handoff(bank, p)
    row = bank.execute("select * from handoffs where handoff_id=?", (hid,)).fetchone()
    assert (row["conversation_id"], row["customer_id"], row["reason"]) == ("c1", "CLI-A", "policy_P4")
    assert json.loads(row["payload"]) == p


def test_unverified_handoff_lists_open_questions(bank, frozen):
    c = Conversation("c2", intent="human_request", language="pt", charges_count=3)
    p = build_handoff(c, "customer_requested_human", trace_id="c2:1")
    assert (p["auth_level"], p["customer_id"], p["transactions"]) == ("unverified", None, [])
    assert "identity not verified" in p["open_questions"] and "charge not identified" in p["open_questions"]
    assert "customer reports 3 disputed charges" in p["open_questions"]


def test_candidates_are_passed_when_no_charge_selected(bank, frozen):
    c = Conversation("c3", token="t", customer_id="CLI-A", candidates=[VIEW, {**VIEW, "transaction_id": "X"}])
    assert len(build_handoff(c, "could_not_identify_charge", trace_id="c3:4")["transactions"]) == 2
