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
