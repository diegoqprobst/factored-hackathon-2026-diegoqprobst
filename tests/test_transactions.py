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
