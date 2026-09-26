import re

import pytest

from src.agent.responder import TEMPLATES, render

TXN = {"transaction_id": "T1", "local_date": "2026-06-16", "description": "Oxxo", "amount": 350.0,
       "currency": "COP", "status": "Approved", "card_last4": "1111"}
SAMPLE = {"masked": "***1234", "options": [TXN, {**TXN, "description": "Uber"}], "txn": TXN,
          "dispute_type": "duplicate", "dispute_id": "DSP-1", "last4": "1111", "rule": "P7", "existing": "DSP-0",
          "reason": "policy_P4", "handoff_id": "HND-1", "topic": "oos_credit", "lead_key": "lead_card_blocked"}


def placeholders(s):
    return set(re.findall(r"{(\w+)}", s))


def test_languages_have_identical_keys_and_placeholders():
    assert set(TEMPLATES["es"]) == set(TEMPLATES["pt"])
    for key in TEMPLATES["es"]:
        assert placeholders(TEMPLATES["es"][key]) == placeholders(TEMPLATES["pt"][key]), key


@pytest.mark.parametrize("lang", ["es", "pt"])
def test_every_template_renders(lang):
    for key in TEMPLATES[lang]:
        text = render(key, lang, **SAMPLE)
        assert text and "{" not in text, key


def test_facts_are_rendered_verbatim():
    es = render("dispute_created", "es", dispute_id="DSP-9F", txn=TXN)
    assert "DSP-9F" in es and "Oxxo" in es and "350.00 COP" in es and "2026-06-16" in es and "1111" in es
    pt = render("choose", "pt", options=[TXN, {**TXN, "description": "Uber"}])
    assert "1. Oxxo" in pt and "2. Uber" in pt


def test_reason_labels_and_unknown_fallbacks():
    assert "asesor" in render("handoff", "es", reason="customer_requested_human", handoff_id="H")
    assert "H" in render("handoff", "es", reason="something_new", handoff_id="H")  # unknown reason -> generic
    assert render("greeting", "fr") == render("greeting", "es")
    assert "90" in render("ineligible", "es", rule="P1")
    assert "DSP-0" in render("ineligible", "pt", rule="P7", existing="DSP-0")


def test_card_options():
    text = render("choose_card", "es", options=[{"product_id": "P", "last4": "1111", "product_type": "Tarjeta Débito"}], kind="card")
    assert "1. Tarjeta Débito •1111" in text
