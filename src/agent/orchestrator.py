"""Dispute-intake orchestrator: a finite-state machine that owns the conversation state, runs bank tools only
through the stage allowlist, gates every write behind explicit confirmation, reads every write back before
claiming it, and hands off to a human with a structured payload. The router and the extractor only *understand*
the message; they never act."""
import time
from dataclasses import asdict, dataclass

from src.agent import responder
from src.agent.handoff import build_handoff, save_handoff
from src.agent.state import Conversation, Stage
from src.agent.tools import ToolNotAllowed, Tools, ToolUnavailable
from src.agent.trace import Tracer, persist
from src.bank.auth import verify_session
from src.bank.errors import (AccountNotServiceable, BankError, NoVerifiedChannel, OtpExpired, OtpInvalid, OtpLocked,
                             PolicyViolation, SessionExpired, SessionInvalid)
from src.router.labels import DISPUTE_TYPE_BY_INTENT

MAX_MESSAGE_CHARS = 2000
DISPUTE_INTENTS = set(DISPUTE_TYPE_BY_INTENT)
INTENT_BY_DISPUTE_TYPE = {v: k for k, v in DISPUTE_TYPE_BY_INTENT.items()}


@dataclass(frozen=True)
class TurnResult:
    conversation_id: str
    reply: str
    stage: str
    language: str
    trace_id: str
    events: list
    handoff: dict | None
    latency_ms: float
    cost_usd: float


def _has_words(text: str) -> bool:
    return sum(ch.isalpha() for ch in text) >= 3


class Agent:
    def __init__(self, conn, router, extractor, *, mode: str, faults: dict | None = None, max_low_confidence: int = 2,
                 max_injections: int = 2, max_not_found: int = 2, max_choose_retries: int = 2):
        self.conn, self.router, self.extractor, self.mode, self.faults = conn, router, extractor, mode, faults
        self.max_low_confidence, self.max_injections = max_low_confidence, max_injections
        self.max_not_found, self.max_choose_retries = max_not_found, max_choose_retries

    # ---- entry point -------------------------------------------------------------------------------------------
    def handle(self, conv: Conversation, message: str) -> TurnResult:
        start = time.perf_counter()
        conv.turns += 1
        tracer = Tracer(conv.id, conv.turns)
        tools = Tools(self.conn, tracer, faults=self.faults)
        text = (message or "")[:MAX_MESSAGE_CHARS]
        try:
            key, facts = self._turn(conv, text, tracer, tools)
        except (SessionExpired, SessionInvalid):
            key, facts = self._session_lost(conv, tracer)
        except ToolUnavailable:
            key, facts = self._handoff(conv, tracer, "action_failed")
        except ToolNotAllowed:
            key, facts = self._handoff(conv, tracer, "internal_guard")
        except Exception as exc:  # safe fallback: never leave the customer without an answer or act blindly
            tracer.record("error", error=type(exc).__name__)
            key, facts = self._handoff(conv, tracer, "internal_error")
        payload = facts.pop("payload", None)  # returned, never stored on the shared agent (concurrent requests)
        reply = responder.render(key, conv.language, **facts)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        tracer.record("reply", key=key, stage=conv.stage.value)
        persist(self.conn, tracer, conv.id, conv.turns, conv.stage.value, latency_ms)
        return TurnResult(conv.id, reply, conv.stage.value, conv.language, tracer.trace_id, tracer.events,
                          payload, latency_ms, tracer.cost_usd)

    def _turn(self, conv, text, tracer, tools):
        if conv.stage is Stage.HANDOFF:
            return "already_handed_off", {"handoff_id": conv.handoff_id}
        route = self.router.predict(text)
        tracer.record("router", intent=route.intent, confidence=round(route.confidence, 3), language=route.language,
                      injection=route.injection, abstain=route.abstain, model=route.model_version)
        if _has_words(text) and (conv.stage in (Stage.INTAKE, Stage.DONE) or not route.abstain):
            conv.language = route.language
        ext = self.extractor.extract(text, conv.stage, self._context(conv), tracer=tracer)
        tracer.record("extraction", source=ext.source, fields=ext.present_fields())
        if route.injection:
            conv.injection_count += 1
            tracer.record("injection_detected", count=conv.injection_count)
            if conv.injection_count > self.max_injections:
                return self._handoff(conv, tracer, "repeated_manipulation")
        if (route.intent == "human_request" and not route.abstain) or ext.wants_human:
            return self._handoff(conv, tracer, "customer_requested_human")
        if ext.wants_refund_or_credit:
            return self._handoff(conv, tracer, "refund_or_credit_requested")
        handler = {Stage.INTAKE: self._intake, Stage.AUTH_DOC: self._auth_doc, Stage.AUTH_OTP: self._auth_otp,
                   Stage.IDENTIFY: self._identify, Stage.CHOOSE: self._choose, Stage.CLASSIFY: self._classify,
                   Stage.CONFIRM: self._confirm, Stage.BLOCK_OFFER: self._block_offer, Stage.DONE: self._done}
        return handler[conv.stage](conv, route, ext, tracer, tools)

    def _context(self, conv):
        if conv.stage is Stage.CHOOSE and conv.choose_kind == "transaction":
            return {"options": [responder._txn(o, conv.language) for o in conv.candidates]}
        return None

    # ---- stages ------------------------------------------------------------------------------------------------
    def _intake(self, conv, route, ext, tracer, tools):
        intent = None if route.abstain else route.intent
        if intent is None and ext.dispute_type:
            intent = INTENT_BY_DISPUTE_TYPE[ext.dispute_type]  # the LLM understood what the router could not
        if intent in DISPUTE_INTENTS or intent == "card_lost_stolen":
            conv.intent = intent
            conv.dispute_type = DISPUTE_TYPE_BY_INTENT.get(intent)
            conv.remember(ext)
            conv.stage = Stage.AUTH_DOC
            return "ask_document", {}
        if route.injection:
            return "injection_refused", {}
        if intent is None:
            return self._unclear(conv, tracer)
        if intent == "greeting_smalltalk":
            return "greeting", {}
        return "out_of_scope", {"topic": intent}

    def _unclear(self, conv, tracer):
        conv.low_conf_count += 1
        if conv.low_conf_count >= self.max_low_confidence:
            return self._handoff(conv, tracer, "not_understood")
        return "clarify", {}

    def _auth_doc(self, conv, route, ext, tracer, tools):
        if not ext.document_number:
            return "ask_document", {}
        try:
            challenge = tools.call(conv.stage, "start_auth", document_number=ext.document_number)
        except (NoVerifiedChannel, AccountNotServiceable) as exc:
            return self._handoff(conv, tracer, f"auth_{exc.code}")
        except OtpLocked:
            return self._handoff(conv, tracer, "auth_rate_limited")
        conv.challenge_id = challenge.challenge_id
        conv.stage = Stage.AUTH_OTP
        return "otp_sent", {"masked": challenge.masked_destination}

    def _auth_otp(self, conv, route, ext, tracer, tools):
        if not ext.otp_code:
            return "ask_otp", {}
        try:
            token = tools.call(conv.stage, "verify_otp", challenge_id=conv.challenge_id, code=ext.otp_code)
        except OtpInvalid:
            return "otp_invalid", {}
        except OtpExpired:
            conv.stage = Stage.AUTH_DOC
            return "otp_expired", {}
        except OtpLocked:
            return self._handoff(conv, tracer, "identity_not_verified")
        conv.token, conv.customer_id = token, verify_session(token).customer_id
        conv.actions.append({"action": "authenticate", "verified": True})
        tracer.record("authenticated")
        return self._after_auth(conv, tracer, tools)

    def _after_auth(self, conv, tracer, tools):
        if conv.intent == "card_lost_stolen":
            return self._start_block_flow(conv, tracer, tools)
        if conv.transaction_id and conv.dispute_type:
            return self._evaluate(conv, tracer, tools)  # resuming after a session expiry
        conv.stage = Stage.IDENTIFY
        if conv.slots:
            return self._search(conv, tracer, tools)
        return "ask_charge", {}

    def _identify(self, conv, route, ext, tracer, tools):
        if route.intent == "card_lost_stolen" and not route.abstain:
            conv.intent = "card_lost_stolen"
            return self._start_block_flow(conv, tracer, tools)
        if route.intent in DISPUTE_INTENTS and not route.abstain:
            conv.dispute_type = DISPUTE_TYPE_BY_INTENT[route.intent]
        elif ext.dispute_type:
            conv.dispute_type = ext.dispute_type
        conv.remember(ext)
        if not conv.slots:
            return self._unclear(conv, tracer) if route.abstain and not ext.present_fields() else ("ask_charge", {})
        return self._search(conv, tracer, tools)

    def _search(self, conv, tracer, tools):
        s = conv.slots
        results = tools.call(conv.stage, "search_transactions", token=conv.token, amount=s.get("amount"),
                             date_from=s.get("date_from"), date_to=s.get("date_to"), merchant=s.get("merchant"),
                             limit=5)
        if not results:
            conv.not_found_count += 1
            conv.slots = {}
            if conv.not_found_count >= self.max_not_found:
                return self._handoff(conv, tracer, "charge_not_found")
            return "not_found", {}
        if len(results) == 1:
            conv.transaction_id, conv.selected_view = results[0].transaction_id, asdict(results[0])
            return self._classify_or_evaluate(conv, tracer, tools)
        conv.candidates, conv.choose_kind, conv.stage = [asdict(v) for v in results], "transaction", Stage.CHOOSE
        return "choose", {"options": conv.candidates}

    def _choose(self, conv, route, ext, tracer, tools):
        n = len(conv.candidates)
        idx = ext.choice
        if idx is None and ext.amount is not None and conv.choose_kind == "transaction":
            hits = [i for i, c in enumerate(conv.candidates, 1) if abs(c["amount"] - ext.amount) <= max(0.01, ext.amount * 0.01)]
            idx = hits[0] if len(hits) == 1 else None
        if idx == -1:
            idx = n
        if not idx or not 1 <= idx <= n:
            conv.choose_retries += 1
            if conv.choose_retries >= self.max_choose_retries:
                return self._handoff(conv, tracer, "could_not_identify_charge")
            key = "choose_card" if conv.choose_kind == "card" else "choose"
            return key, {"options": conv.candidates, "kind": conv.choose_kind}
        chosen = conv.candidates[idx - 1]
        if conv.choose_kind == "card":
            conv.product_id, conv.stage = chosen["product_id"], Stage.BLOCK_OFFER
            return "offer_block", {"last4": chosen["last4"]}
        view = tools.call(conv.stage, "get_transaction", token=conv.token, transaction_id=chosen["transaction_id"])
        conv.transaction_id, conv.selected_view = view.transaction_id, asdict(view)
        return self._classify_or_evaluate(conv, tracer, tools)

    def _classify_or_evaluate(self, conv, tracer, tools):
        if conv.dispute_type:
            return self._evaluate(conv, tracer, tools)
        conv.stage = Stage.CLASSIFY
        return "ask_type", {"txn": conv.selected_view}

    def _classify(self, conv, route, ext, tracer, tools):
        dispute_type = ext.dispute_type or (DISPUTE_TYPE_BY_INTENT.get(route.intent) if not route.abstain else None)
        if not dispute_type:
            conv.low_conf_count += 1
            if conv.low_conf_count >= self.max_low_confidence:
                return self._handoff(conv, tracer, "not_understood")
            return "ask_type", {"txn": conv.selected_view}
        conv.dispute_type = dispute_type
        return self._evaluate(conv, tracer, tools)

    def _declared_count(self, conv) -> int:
        return max(len(conv.disputes) + 1, conv.charges_count or 0)

    def _evaluate(self, conv, tracer, tools):
        conv.stage = Stage.CLASSIFY
        decision = tools.call(conv.stage, "evaluate_dispute", token=conv.token, transaction_id=conv.transaction_id,
                              dispute_type=conv.dispute_type, declared_disputed_count=self._declared_count(conv))
        return self._apply_decision(conv, tracer, decision)

    def _apply_decision(self, conv, tracer, decision):
        conv.decision = asdict(decision)
        tracer.record("policy", outcome=decision.outcome, rule_id=decision.rule_id, version=decision.policy_version)
        if decision.outcome == "eligible":
            conv.stage = Stage.CONFIRM
            return "confirm_dispute", {"txn": conv.selected_view, "dispute_type": conv.dispute_type}
        if decision.outcome == "requires_human":
            return self._handoff(conv, tracer, f"policy_{decision.rule_id}")
        conv.stage = Stage.DONE
        return "ineligible", {"rule": decision.rule_id, "existing": decision.existing_dispute_id or ""}

    def _confirm(self, conv, route, ext, tracer, tools):
        if ext.confirm is None:
            return "confirm_dispute", {"txn": conv.selected_view, "dispute_type": conv.dispute_type}
        if ext.confirm is False:
            conv.stage = Stage.DONE
            return "cancelled", {}
        session = verify_session(conv.token)
        try:
            created = tools.call(conv.stage, "create_dispute", token=conv.token, transaction_id=conv.transaction_id,
                                 dispute_type=conv.dispute_type, idempotency_key=f"{session.session_id}:{conv.transaction_id}",
                                 customer_confirmed=True, declared_disputed_count=self._declared_count(conv))
        except PolicyViolation as exc:
            return self._apply_decision(conv, tracer, exc.decision)
        stored = tools.call(conv.stage, "get_dispute", token=conv.token, dispute_id=created.dispute_id)
        if (stored.transaction_id, stored.status) != (conv.transaction_id, "open"):
            conv.actions.append({"action": "create_dispute", "id": created.dispute_id, "verified": False})
            return self._handoff(conv, tracer, "action_not_verified")
        conv.disputes.append(stored.dispute_id)
        conv.actions.append({"action": "create_dispute", "id": stored.dispute_id, "verified": True})
        tracer.record("action_verified", action="create_dispute", id=stored.dispute_id)
        last4 = (conv.selected_view or {}).get("card_last4")
        card = next((c for c in tools.call(conv.stage, "list_cards", token=conv.token)
                     if c.last4 == last4 and c.status == "Active"), None) if last4 else None
        if conv.dispute_type == "unrecognized" and card:
            conv.product_id, conv.stage = card.product_id, Stage.BLOCK_OFFER
            return "dispute_created_offer_block", {"dispute_id": stored.dispute_id, "txn": conv.selected_view,
                                                   "last4": card.last4}
        conv.stage = Stage.DONE
        return "dispute_created", {"dispute_id": stored.dispute_id, "txn": conv.selected_view}

    def _start_block_flow(self, conv, tracer, tools):
        conv.stage = Stage.BLOCK_OFFER
        active = [c for c in tools.call(conv.stage, "list_cards", token=conv.token) if c.status == "Active"]
        if not active:
            return self._handoff(conv, tracer, "no_active_card")
        if len(active) == 1:
            conv.product_id = active[0].product_id
            return "offer_block", {"last4": active[0].last4}
        conv.candidates = [{"product_id": c.product_id, "last4": c.last4, "product_type": c.product_type} for c in active]
        conv.choose_kind, conv.stage = "card", Stage.CHOOSE
        return "choose_card", {"options": conv.candidates, "kind": "card"}

    def _block_offer(self, conv, route, ext, tracer, tools):
        card = tools.call(conv.stage, "get_card", token=conv.token, product_id=conv.product_id)
        if ext.confirm is None:
            return "offer_block", {"last4": card.last4}
        if ext.confirm is False:
            conv.stage = Stage.DONE
            return "block_declined", {}
        tools.call(conv.stage, "block_card", token=conv.token, product_id=conv.product_id, customer_confirmed=True,
                   reason=conv.intent or "customer_request")
        stored = tools.call(conv.stage, "get_card", token=conv.token, product_id=conv.product_id)
        if stored.status != "Blocked":
            conv.actions.append({"action": "block_card", "id": conv.product_id, "verified": False})
            return self._handoff(conv, tracer, "action_not_verified")
        conv.actions.append({"action": "block_card", "id": conv.product_id, "verified": True})
        tracer.record("action_verified", action="block_card", id=conv.product_id)
        if conv.intent == "card_lost_stolen" and conv.charges_count:
            return self._handoff(conv, tracer, "stolen_card_with_charges", lead_key="lead_card_blocked",
                                 last4=stored.last4)
        conv.stage = Stage.DONE
        return "card_blocked", {"last4": stored.last4}

    def _done(self, conv, route, ext, tracer, tools):
        intent = None if route.abstain else route.intent
        if intent is None and ext.dispute_type:
            intent = INTENT_BY_DISPUTE_TYPE[ext.dispute_type]
        if intent in DISPUTE_INTENTS or intent == "card_lost_stolen":
            conv.reset_case()
            conv.intent, conv.dispute_type = intent, DISPUTE_TYPE_BY_INTENT.get(intent)
            conv.remember(ext)
            if not conv.token:
                conv.stage = Stage.AUTH_DOC
                return "ask_document", {}
            return self._after_auth(conv, tracer, tools)
        if route.injection:
            return "injection_refused", {}
        if intent == "greeting_smalltalk":
            return "goodbye", {}
        if intent is None:
            return "anything_else", {}
        return "out_of_scope", {"topic": intent}

    # ---- exits -------------------------------------------------------------------------------------------------
    def _session_lost(self, conv, tracer):
        conv.token = conv.customer_id = conv.challenge_id = None
        conv.stage = Stage.AUTH_DOC
        tracer.record("session_expired")
        return "session_expired", {}

    def _handoff(self, conv, tracer, reason, **lead):
        payload = build_handoff(conv, reason, trace_id=tracer.trace_id)
        conv.handoff_id = save_handoff(self.conn, payload)
        conv.stage = Stage.HANDOFF
        tracer.record("handoff", reason=reason, handoff_id=conv.handoff_id)
        return "handoff", {"reason": reason, "handoff_id": conv.handoff_id, "payload": payload, **lead}
