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
