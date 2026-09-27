"""Regression tests for the three root causes found by the held-out evaluation (reports/eval_error_analysis.md)."""
from src.agent.factory import build_agent
from src.agent.nlu import LLMExtractor, RuleExtractor, SYSTEM_PROMPT
from src.agent.orchestrator import Agent
from src.agent.state import Conversation, Stage
from src.router.labels import RouterResult
from tests.agent.test_llm_extractor import FakeLLM
from tests.agent.test_orchestrator import auth, disputes


class FixedRouter:
    version = "fixed"

    def __init__(self, intent):
        self.intent = intent

    def predict(self, text):
        return RouterResult(self.intent, 0.6, "pt", False, 0.0, False, "fixed")


# A — the dispute sub-type comes from the extractor that read the sentence, not from the router
def test_extractor_subtype_wins_over_router_subtype(bank, frozen):
    agent = Agent(bank, FixedRouter("dispute_undue_fee"),
                  LLMExtractor(FakeLLM({"dispute_type": "amount_mismatch", "merchant": "Uber"})), mode="hybrid")
    c = Conversation("a1")
    agent.handle(c, "Na Uber cobraram 45 e o valor está errado")
    assert (c.intent, c.dispute_type) == ("dispute_amount_mismatch", "amount_mismatch")


def test_router_subtype_is_used_when_extractor_has_none(bank, frozen):
    agent = Agent(bank, FixedRouter("dispute_duplicate"), LLMExtractor(FakeLLM({})), mode="hybrid")
    c = Conversation("a2")
    agent.handle(c, "tengo un problema con un cobro")
    assert c.dispute_type == "duplicate"


# B — "the merchant refund never arrived" is not a request for the bank to pay now
def test_refund_not_received_is_not_a_refund_request():
    llm = FakeLLM({"dispute_type": "refund_not_received", "wants_refund_or_credit": True})
    e = LLMExtractor(llm).extract("El reembolso de Uber por 45 nunca llegó", Stage.INTAKE)
    assert (e.dispute_type, e.wants_refund_or_credit) == ("refund_not_received", False)
    assert "has not arrived" in SYSTEM_PROMPT


def test_explicit_refund_demand_still_escalates():
    e = LLMExtractor(FakeLLM({"dispute_type": "refund_not_received"})).extract(
        "no llegó el reembolso, devuélveme el dinero ya", Stage.INTAKE)
    assert e.wants_refund_or_credit is True


# C — a lost/stolen card is handled first, whatever intent won the routing
def test_card_lost_signal_from_rules_and_llm():
    assert RuleExtractor().extract("Roubaram meu cartão e tem 3 compras que não fiz", Stage.INTAKE).card_lost is True
    assert RuleExtractor().extract("No reconozco un cargo de Uber", Stage.INTAKE).card_lost is False
    assert LLMExtractor(FakeLLM({"card_lost_or_stolen": True})).extract("perdi o cartão", Stage.INTAKE).card_lost is True


def test_stolen_card_with_charges_in_portuguese_blocks_then_hands_off(bank, frozen):
    base = build_agent(bank, "baseline")
    c = Conversation("c1")
    base.handle(c, "Roubaram meu cartão e tem 3 compras que não fiz")
    r = auth(base, c, bank)
    assert r.stage == "block_offer"
    r = base.handle(c, "sim")
    assert r.stage == "handoff" and r.handoff["escalation_reason"] == "stolen_card_with_charges"
    assert bank.execute("select product_status from products where product_id='PRD-A-CC'").fetchone()[0] == "Blocked"
    assert disputes(bank) == []
