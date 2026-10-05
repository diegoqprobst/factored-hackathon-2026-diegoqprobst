"""Regression tests for the Plan 3 final-review findings (Critical 1, Important 2-11)."""
import http.client

import pytest
from fastapi.testclient import TestClient

from src.agent.api import create_app
from src.agent.factory import build_agent
from src.agent.llm import OpenRouterLLM
from src.agent.nlu import LLMExtractor, build_extraction_messages, parse_confirm
from src.agent.orchestrator import Agent
from src.agent.state import Conversation, Stage
from src.bank import db
from src.router.labels import RouterResult
from tests.agent.test_llm_extractor import FakeLLM
from tests.agent.test_orchestrator import auth, disputes
from tests.helpers import last_code


@pytest.fixture
def base(bank, frozen):
    return build_agent(bank, "baseline")


def card_status(bank):
    return bank.execute("select product_status from products where product_id='PRD-A-CC'").fetchone()[0]


# Critical 1 — questions and conditionals are never a "yes"
@pytest.mark.parametrize("text", [
    "¿qué pasa si la bloqueo? ¿puedo seguir usando la app?", "¿cuánto tarda si la abro?",
    "Pode me explicar o que acontece depois?", "¿qué pasa si confirmo?", "¿Es correcto el monto?",
    "Pode me explicar o que é isso?", "si me cobran otra vez qué hago", "claro que no", "no, ya la bloqueé por la app",
])
def test_questions_and_conditionals_are_not_confirmations(text):
    assert parse_confirm(text) is None


@pytest.mark.parametrize("text, value", [
    ("sí", True), ("si", True), ("Sí, confirmo", True), ("dale", True), ("sim, pode", True), ("ok, gracias", True),
    ("isso mesmo", True), ("no", False), ("não", False), ("no, gracias", False), ("mejor no", False),
])
def test_plain_answers_still_work(text, value):
    assert parse_confirm(text) is value


def test_question_at_block_offer_does_not_block(base, bank):
    c = Conversation("c1")
    base.handle(c, "Me robaron la tarjeta")
    auth(base, c, bank)
    r = base.handle(c, "¿qué pasa si la bloqueo? ¿puedo seguir usando la app?")
    assert r.stage == "block_offer" and card_status(bank) == "Active"


def test_question_at_confirm_does_not_file(base, bank):
    c = Conversation("c2")
    base.handle(c, "No reconozco un cargo de Uber")
    auth(base, c, bank)
    assert base.handle(c, "¿cuánto tarda si la abro?").stage == "confirm" and disputes(bank) == []


def test_hybrid_write_needs_rules_and_llm_to_agree():
    assert LLMExtractor(FakeLLM({"confirm": None})).extract("sí", Stage.CONFIRM).confirm is None
    assert LLMExtractor(FakeLLM({"confirm": True})).extract("sí", Stage.CONFIRM).confirm is True
    assert LLMExtractor(FakeLLM({"confirm": True})).extract("¿cuánto tarda?", Stage.CONFIRM).confirm is None


# Important 2 — declared charges are never silently dropped
def test_extra_charges_declared_at_confirm_reach_policy(base, bank):
    c = Conversation("m1")
    base.handle(c, "No reconozco un cargo de Uber")
    auth(base, c, bank)
    assert base.handle(c, "sí, y además hay 3 cargos más que tampoco reconozco").stage == "confirm"
    r = base.handle(c, "sí")
    assert r.stage == "handoff" and r.handoff["escalation_reason"] == "policy_P5" and disputes(bank) == []


def test_ineligible_first_charge_with_many_declared_hands_off(base, bank):
    c = Conversation("m2")
    base.handle(c, "No reconozco 3 cargos en Rappi")
    auth(base, c, bank)
    r = base.handle(c, "1")
    assert r.stage == "handoff" and r.handoff["escalation_reason"] == "policy_P5"


def test_stolen_card_with_charges_hands_off_even_if_block_declined(base, bank):
    c = Conversation("m3")
    base.handle(c, "Me robaron la tarjeta y hay 3 compras que no hice")
    auth(base, c, bank)
    r = base.handle(c, "no")
    assert r.stage == "handoff" and r.handoff["escalation_reason"] == "stolen_card_with_charges"


def test_new_charge_at_block_offer_is_not_read_as_no(base, bank):
    c = Conversation("m4")
    base.handle(c, "No reconozco un cargo de Uber")
    auth(base, c, bank)
    base.handle(c, "sí")
    r = base.handle(c, "No reconozco otro cargo en Oxxo también")
    assert r.stage == "block_offer" and card_status(bank) == "Active" and c.charges_count == 2
    assert "un cargo a la vez" in r.reply


# Important 3 — a different customer after session expiry starts clean
def test_different_customer_after_expiry_does_not_inherit_case(base, bank, frozen):
    c = Conversation("x1")
    base.handle(c, "No reconozco un cargo de Uber")
    auth(base, c, bank)
    frozen.advance(15 * 60 + 1)
    base.handle(c, "sí")
    r = auth(base, c, bank, doc="222")
    assert r.stage == "identify" and r.handoff is None
    assert (c.customer_id, c.transaction_id, c.selected_view, c.disputes) == ("CLI-B", None, None, [])


# Important 4 — a committed write is reported even if its read-back fails
def test_unverified_write_is_in_the_handoff(bank, frozen):
    agent = build_agent(bank, "baseline", faults={"get_dispute": 99})
    c = Conversation("w1")
    agent.handle(c, "No reconozco un cargo de Uber")
    auth(agent, c, bank)
    r = agent.handle(c, "sí")
    assert r.stage == "handoff" and len(disputes(bank)) == 1
    unverified = [a for a in r.handoff["actions_taken"] if a["action"] == "create_dispute"]
    assert unverified == [{"action": "create_dispute", "id": disputes(bank)[0]["dispute_id"], "verified": False}]
    assert any("create_dispute" in q for q in r.handoff["open_questions"])


# Important 5 — amounts are not documents; a document can be corrected at the OTP step
def test_amount_is_not_taken_as_document(base, bank):
    c = Conversation("d1")
    base.handle(c, "No reconozco un cargo de Uber")
    r = base.handle(c, "fue un cargo de 450 pesos")
    assert r.stage == "auth_doc" and bank.execute("select count(*) from otp_challenges").fetchone()[0] == 0


def test_document_can_be_corrected_at_otp_step(base, bank):
    c = Conversation("d2")
    base.handle(c, "No reconozco un cargo de Uber")
    base.handle(c, "999999")
    first = c.challenge_id
    r = base.handle(c, "perdón, mi documento es 111")
    assert r.stage == "auth_otp" and c.challenge_id != first
    assert bank.execute("select customer_id from otp_challenges where challenge_id=?", (c.challenge_id,)).fetchone()[0] == "CLI-A"


# Important 6 — an approved block survives a session expiry
def test_block_offer_resumes_after_expiry(base, bank, frozen):
    c = Conversation("b1")
    base.handle(c, "No reconozco un cargo de Uber")
    auth(base, c, bank)
    base.handle(c, "sí")
    frozen.advance(15 * 60 + 1)
    assert base.handle(c, "sí").stage == "auth_doc"
    r = auth(base, c, bank)
    assert r.stage == "block_offer" and "1111" in r.reply
    assert base.handle(c, "sí").stage == "done" and card_status(bank) == "Blocked"


# Important 7 — short answers at confirmation-like stages never switch language
class PtRouter:
    version = "pt"

    def predict(self, text):
        return RouterResult("oos_other", 0.9, "pt", False, 0.0, False, "pt")


def test_language_does_not_flip_outside_free_text_stages(bank, frozen):
    agent = Agent(bank, PtRouter(), LLMExtractor(FakeLLM({})), mode="hybrid")
    c = Conversation("l1", stage=Stage.AUTH_OTP, challenge_id="x")
    agent.handle(c, "vale")
    assert c.language == "es"


# Important 8 — the demo SMS is bound to the challenge, not the phone
def test_demo_sms_is_bound_to_challenge_when_phones_are_shared(bank, frozen):
    bank.execute("insert into customers values ('CLI-SH','555','CC','Sara','Hernández','+57 300 000 1234','Colombia','Basic','Active')")
    bank.commit()
    client = TestClient(create_app(conn=bank, agent=build_agent(bank, "baseline"), demo_mode=True))
    ids = []
    for doc in ("111", "555"):
        cid = client.post("/v1/conversations").json()["conversation_id"]
        client.post("/v1/chat", json={"conversation_id": cid, "message": "No reconozco un cargo de Uber"})
        client.post("/v1/chat", json={"conversation_id": cid, "message": doc})
        ids.append(cid)
    code = client.get(f"/v1/demo/sms/{ids[0]}").json()["sms"].split()[-1]
    out = client.post("/v1/chat", json={"conversation_id": ids[0], "message": code}).json()
    assert out["stage"] != "auth_otp"


def test_outbox_column_is_migrated_on_old_databases(tmp_path):
    conn = db.connect(tmp_path / "old.db")
    conn.execute("create table sandbox_outbox (id integer primary key autoincrement, channel text not null, "
                 "destination text not null, body text not null, created_at text not null)")
    db.create_schema(conn)
    assert "challenge_id" in {r[1] for r in conn.execute("pragma table_info(sandbox_outbox)")}


# Important 9 — PII is redacted before the LLM sees the message
def test_pii_is_redacted_for_the_llm():
    from datetime import date
    msg = "soy Ana, cédula 1.234.567, cel 300 000 1234, ana.g@mail.com, me cobraron 2.500.000 COP en Falabella"
    content = build_extraction_messages(msg, Stage.INTAKE, None, date(2026, 6, 17))[1]["content"]
    assert "1.234.567" not in content and "300 000 1234" not in content and "ana.g@mail.com" not in content
    assert "2.500.000 COP" in content and "Falabella" in content


# Important 10 — any LLM failure falls back to rules; one retry by default
def test_unexpected_llm_exception_falls_back():
    e = LLMExtractor(FakeLLM(error=http.client.IncompleteRead(b""))).extract("No reconozco un cargo de Uber", Stage.INTAKE)
    assert (e.source, e.merchant) == ("llm_fallback", "Uber")


def test_llm_defaults_follow_spec_one_retry():
    llm = OpenRouterLLM("m", api_key="k", transport=lambda b, t: {})
    assert (llm.max_retries, llm.timeout) == (1, 15.0)


# Important 11 — the low-confidence counter resets once the customer is understood
def test_low_confidence_counter_resets(base, bank):
    c = Conversation("lc")
    assert base.handle(c, "mmm qwerty").stage == "intake"
    base.handle(c, "No reconozco un cargo")
    r = auth(base, c, bank)
    assert r.stage == "identify"
    assert base.handle(c, "ehh zzz").stage == "identify"


class _NoMeansDisputeRouter:
    """Mimics the deployed TF-IDF router, which scores a bare "no" as dispute_unrecognized (0.81): its
    training openings are full of "no reconozco…". A plain yes/no answer must still win over the route."""
    version = "stub"

    def __init__(self):
        from src.router.keyword import KeywordRouter
        self.inner = KeywordRouter()

    def predict(self, text):
        r = self.inner.predict(text)
        if parse_confirm(text) is False:
            return RouterResult("dispute_unrecognized", 0.81, r.language, False, 0.0, False, "stub")
        return r


@pytest.mark.parametrize("answer", ["no", "No.", "não"])
def test_plain_no_at_block_offer_declines_even_if_router_sees_a_dispute(bank, frozen, answer):
    agent = build_agent(bank, "baseline", router=_NoMeansDisputeRouter())
    c = Conversation("n1")
    agent.handle(c, "No reconozco un cargo de Uber")
    auth(agent, c, bank)
    agent.handle(c, "sí")
    r = agent.handle(c, answer)
    assert r.stage == "done" and card_status(bank) == "Active" and "un cargo a la vez" not in r.reply


def test_plain_no_at_confirm_cancels_even_if_router_sees_a_dispute(bank, frozen):
    agent = build_agent(bank, "baseline", router=_NoMeansDisputeRouter())
    c = Conversation("n2")
    agent.handle(c, "No reconozco un cargo de Uber")
    auth(agent, c, bank)
    r = agent.handle(c, "no")
    assert r.stage == "done" and disputes(bank) == []


def test_model_only_no_does_not_swallow_a_new_charge_at_confirm(bank, frozen):  # Final review Important 4
    from src.agent.nlu import LLMExtractor
    agent = Agent(bank, _NoMeansDisputeRouter(), LLMExtractor(FakeLLM({"confirm": False})), mode="hybrid")
    c = Conversation("n3")
    agent.handle(c, "No reconozco un cargo de Uber")
    auth(agent, c, bank)
    r = agent.handle(c, "No, el que no reconozco es otro de 80 en Oxxo")
    assert r.stage == "confirm" and "un cargo a la vez" in r.reply and disputes(bank) == []


@pytest.mark.parametrize("text", [  # seen in the live demo: an explicit action verb is a clear yes
    "sim, bloqueia", "Sim, bloqueie", "sí, bloquéala", "sí bloquéala por favor", "bloquéala", "dale, ábrela",
    "sim, pode abrir", "sí, ábrela",
])
def test_yes_with_the_action_verb_is_a_yes(text):
    assert parse_confirm(text) is True


@pytest.mark.parametrize("text", ["no la bloquees", "não bloqueia", "¿la bloqueo?", "bloquéala si vuelve a pasar"])
def test_action_verb_does_not_turn_a_no_or_a_condition_into_a_yes(text):
    assert parse_confirm(text) is not True


@pytest.mark.parametrize("text", [  # seen live: an amount pasted at the OTP step burned attempts
    "No reconozco un cargo de 451998.09 en Laboratorio Central", "fueron 123456,50 pesos", "1.451998",
])
def test_decimal_amount_is_never_an_otp(text):
    from src.agent.nlu import parse_otp
    assert parse_otp(text) is None


@pytest.mark.parametrize("text, code", [("123456", "123456"), ("mi código es 123 456", "123456"), ("123456.", "123456")])
def test_plain_otp_still_parses(text, code):
    from src.agent.nlu import parse_otp
    assert parse_otp(text) == code


class _AbstainingRouter:
    """The router has no confident opinion (the confirmation run's one unsafe case: confidence 0.33)."""
    version = "stub"

    def predict(self, text):
        return RouterResult("dispute_unrecognized", 0.33, "pt", False, 0.0, True, "stub")


def test_rules_and_llm_disagree_on_the_type_when_the_router_abstains(bank, frozen):  # residual unsafe, closed
    agent = Agent(bank, _AbstainingRouter(), LLMExtractor(FakeLLM({"dispute_type": "unrecognized"})), mode="hybrid")
    c = Conversation("u1")
    agent.handle(c, "Essa cobrança da Uber é uma cobrança indevida")
    r = auth(agent, c, bank)
    assert r.stage == "classify" and disputes(bank) == []  # the customer is asked, nothing is filed on a guess


def test_router_abstains_but_rules_and_llm_agree_keeps_the_type(bank, frozen):
    agent = Agent(bank, _AbstainingRouter(), LLMExtractor(FakeLLM({"dispute_type": "unrecognized"})), mode="hybrid")
    c = Conversation("u2")
    agent.handle(c, "No reconozco un cargo de Uber")
    r = auth(agent, c, bank)
    assert r.stage == "confirm" and "não reconhecida" in r.reply


def test_router_abstains_rules_silent_llm_type_stands(bank, frozen):  # "Usaram meu cartão… não fui eu" must not ask
    agent = Agent(bank, _AbstainingRouter(), LLMExtractor(FakeLLM({"dispute_type": "unrecognized"})), mode="hybrid")
    c = Conversation("u3")
    agent.handle(c, "Usaram meu cartão na Uber, não fui eu")
    r = auth(agent, c, bank)
    assert r.stage == "confirm" and "não reconhecida" in r.reply
