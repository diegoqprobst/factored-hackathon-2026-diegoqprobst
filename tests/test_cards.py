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
