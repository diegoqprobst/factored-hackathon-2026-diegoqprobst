"""The only path from the orchestrator to the bank. A per-stage allowlist means that even an orchestrator bug
cannot, say, create a dispute before the customer identified a charge; the bank services still enforce session,
ownership, policy and confirmation on their own."""
import sqlite3
import time

from src.agent.state import Stage
from src.bank import auth, cards, disputes, transactions
from src.bank.errors import BankError

STAGE_TOOLS = {
    Stage.AUTH_DOC: frozenset({"start_auth"}),
    Stage.AUTH_OTP: frozenset({"verify_otp"}),
    Stage.IDENTIFY: frozenset({"search_transactions"}),
    Stage.CHOOSE: frozenset({"get_transaction"}),
    Stage.CLASSIFY: frozenset({"evaluate_dispute"}),
    Stage.CONFIRM: frozenset({"evaluate_dispute", "create_dispute", "get_dispute", "list_cards"}),
    Stage.BLOCK_OFFER: frozenset({"list_cards", "block_card", "get_card"}),
}
REGISTRY = {
    "start_auth": auth.start_auth, "verify_otp": auth.verify_otp,
    "search_transactions": transactions.search_transactions, "get_transaction": transactions.get_transaction,
    "evaluate_dispute": disputes.evaluate_dispute, "create_dispute": disputes.create_dispute,
    "get_dispute": disputes.get_dispute, "list_cards": cards.list_cards, "get_card": cards.get_card,
    "block_card": cards.block_card,
}
SAFE_ARGS = {"transaction_id", "product_id", "dispute_id", "dispute_type", "amount", "merchant", "date_from",
             "date_to", "limit"}


class ToolNotAllowed(Exception):
    pass


class ToolUnavailable(Exception):
    pass


class Tools:
    def __init__(self, conn, tracer, *, faults: dict[str, int] | None = None, retries: int = 2, sleep=time.sleep):
        self.conn, self.tracer, self.faults, self.retries, self.sleep = conn, tracer, faults, retries, sleep

    def call(self, stage: Stage, name: str, **kwargs):
        args = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in kwargs.items()
                if k in SAFE_ARGS and v is not None}
        if name not in STAGE_TOOLS.get(stage, frozenset()):
            self.tracer.record("tool_denied", name=name, stage=stage.value)
            raise ToolNotAllowed(f"{name} is not allowed at stage {stage.value}")
        start, attempts = time.perf_counter(), 0
        while True:
            attempts += 1
            try:
                if self.faults and self.faults.get(name, 0) > 0:
                    self.faults[name] -= 1
                    raise sqlite3.OperationalError("injected fault")
                result = REGISTRY[name](self.conn, **kwargs)
            except BankError as exc:
                self._trace(name, stage, attempts, start, args, error=exc.code)
                raise
            except sqlite3.OperationalError:
                if attempts > self.retries:
                    self._trace(name, stage, attempts, start, args, error="unavailable")
                    raise ToolUnavailable(name)
                self.sleep(0.1 * 2 ** (attempts - 1))
                continue
            self._trace(name, stage, attempts, start, args)
            return result

    def _trace(self, name, stage, attempts, start, args, error=None):
        event = {"name": name, "stage": stage.value, "ok": error is None, "attempts": attempts,
                 "latency_ms": round((time.perf_counter() - start) * 1000, 2), "args": args}
        if error:
            event["error"] = error
        self.tracer.record("tool", **event)
