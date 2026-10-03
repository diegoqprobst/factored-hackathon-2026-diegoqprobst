from datetime import date

import pytest

from src.agent.nlu import (RuleExtractor, parse_amount, parse_charges_count, parse_confirm, parse_dates,
                           parse_document, parse_otp)
from src.agent.state import Stage

TODAY = date(2026, 6, 17)
X = RuleExtractor(today=TODAY)


@pytest.mark.parametrize("text, value", [
    ("un cargo de 350 en Oxxo", 350.0), ("me debitaron $45.000", 45000.0), ("fueron 1.250 pesos", 1250.0),
    ("R$ 89,90 na fatura", 89.90), ("como 45 mil", 45000.0), ("me cobraron 2 veces", None),
    ("hace 3 días", None), ("el 15/06", None), ("sin monto", None),
])
def test_parse_amount(text, value):
    assert parse_amount(text) == value


@pytest.mark.parametrize("text, expected", [
    ("fue ayer", (date(2026, 6, 16),) * 2), ("foi ontem", (date(2026, 6, 16),) * 2),
    ("anteayer en la tarde", (date(2026, 6, 15),) * 2), ("hoy", (TODAY, TODAY)),
    ("la semana pasada", (date(2026, 6, 3), date(2026, 6, 10))), ("el 15/06", (date(2026, 6, 15),) * 2),
    ("el 20/12", (date(2025, 12, 20),) * 2), ("sin fecha", (None, None)),
])
def test_parse_dates(text, expected):
    assert parse_dates(text, TODAY) == expected


@pytest.mark.parametrize("text, value", [
    ("sí", True), ("si, confirmo", True), ("dale", True), ("sim, pode", True), ("no", False), ("não", False),
    ("cancela", False), ("¿cuánto tarda?", None), ("claro que no", None), ("", None),  # Review Focus 1
])
def test_parse_confirm(text, value):
    assert parse_confirm(text) == value


def test_parse_document_and_otp():  # Review Focus 2
    assert parse_document("mi cédula es 1.234.567") == "1234567"
    assert parse_document("G8637940") == "G8637940"
    assert parse_document("no sé") is None
    assert parse_otp("el código es 123 456") == "123456"
    assert parse_otp("123456") == "123456"
    assert parse_otp("1234567") is None


def test_parse_charges_count():
    assert parse_charges_count("hay 3 compras que no hice") == 3
    assert parse_charges_count("tengo dos cargos raros") == 2
    assert parse_charges_count("várias cobranças") == 3
    assert parse_charges_count("un cargo") is None


def test_extract_by_stage():
    e = X.extract("No reconozco un cargo de 350 en Oxxo de ayer", Stage.INTAKE)
    assert (e.amount, e.merchant, e.date_from, e.dispute_type, e.source) == (350.0, "Oxxo", date(2026, 6, 16), "unrecognized", "rules")
    assert X.extract("mi documento es 111", Stage.AUTH_DOC).document_number == "111"
    assert X.extract("No reconozco 3 cargos de Uber", Stage.INTAKE).amount is None  # a count, not an amount
    assert X.extract("G8637940", Stage.AUTH_DOC).document_number == "G8637940"
    assert X.extract("123456", Stage.AUTH_OTP).otp_code == "123456"
    assert X.extract("123456", Stage.IDENTIFY).otp_code is None
    assert X.extract("el segundo", Stage.CHOOSE).choice == 2
    assert X.extract("3", Stage.CLASSIFY).dispute_type == "amount_mismatch"
    assert X.extract("sí", Stage.CONFIRM).confirm is True
    assert X.extract("quiero hablar con un asesor", Stage.CONFIRM).wants_human is True
    assert X.extract("devuélveme el dinero ya", Stage.CONFIRM).wants_refund_or_credit is True
    assert X.extract("no me han devuelto el reembolso", Stage.INTAKE).wants_refund_or_credit is False
