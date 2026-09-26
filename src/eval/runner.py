"""Plays evaluation cases against an agent. The scripted customer is reactive: it answers what the agent's current
stage asks for, so baseline and hybrid face the same customer even when they take different paths."""
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta

from src.agent.state import Conversation
from src.bank import clock, config
from src.eval.cases import Case

CAPS = {"document": 3, "otp": 3, "details": 2, "choice": 2, "type": 2, "confirm": 3, "block": 2, "clarification": 1}
WRITE_TOOLS = {"create_dispute", "block_card"}


@dataclass
class CaseRun:
    case_id: str
    system: str
    conversation_id: str
    turns: list[dict] = field(default_factory=list)
    final_stage: str = "intake"
    handoff: dict | None = None
    policy_rules: list[str] = field(default_factory=list)
    write_attempts: list[str] = field(default_factory=list)
    llm_calls: int = 0
    cost_usd: float = 0.0
    error: str | None = None


def next_message(case: Case, stage: str, counters: dict, otp_lookup) -> tuple[str, str] | None:
    def once(kind, text):
        if text is None or counters.get(kind, 0) >= CAPS.get(kind, 1):
            return None
        counters[kind] = counters.get(kind, 0) + 1
        return kind, text

    if case.inject and case.inject["stage"] == stage and not counters.get("inject"):
        counters["inject"] = 1
        return "inject", case.inject["message"]
    if stage == "intake":
        return once("clarification", case.clarification)
    if stage == "auth_doc":
        return once("document", case.document)
    if stage == "auth_otp":
        real = otp_lookup()
        code = ("000000" if real != "000000" else "111111") if case.otp_mode == "wrong" else real
        return once("otp", code)
    if stage == "identify":
        return once("details", case.details)
    if stage == "choose":
        return once("choice", case.choice)
    if stage == "classify":
        return once("type", case.type_text)
    if stage == "confirm":
        return once("confirm", case.confirm)
    if stage == "block_offer":
        return once("block", case.block)
    return None


class OffsetClock:
    """Adds an offset to the bank clock so session-expiry cases can jump forward. Install only while no other case
    runs; restore afterwards."""

    def __init__(self):
        self._original = None
        self.offset = timedelta(0)

    def install(self):
        self._original = clock.now
        original = self._original
        clock.now = lambda: original() + self.offset

    def advance(self, seconds: float):
        self.offset += timedelta(seconds=seconds)

    def restore(self):
        clock.now = self._original


class LockedRouter:
    def __init__(self, router):
        self._router, self._lock, self.version = router, threading.Lock(), getattr(router, "version", "unknown")

    def predict(self, text):
        with self._lock:
            return self._router.predict(text)


def _otp_lookup(conn, conv):
    def lookup():
        if not conv.challenge_id:
            return None
        row = conn.execute("select body from sandbox_outbox where challenge_id = ? order by id desc limit 1",
                           (conv.challenge_id,)).fetchone()
        return row["body"].split()[-1] if row else None
    return lookup


def run_case(agent, conn, case: Case, system: str, clock_ctl: OffsetClock | None = None, max_turns: int = 14) -> CaseRun:
    conv = Conversation(f"{system}-{case.id}")
    run = CaseRun(case.id, system, conv.id)
    counters: dict = {}
    kind, message = "opening", case.opening
    try:
        while message is not None and len(run.turns) < max_turns:
            result = agent.handle(conv, message)
            reply_key = next((e["key"] for e in reversed(result.events) if e["kind"] == "reply"), None)
            run.turns.append({"kind": kind, "reply": result.reply, "reply_key": reply_key, "stage": result.stage,
                              "latency_ms": result.latency_ms, "cost_usd": result.cost_usd})
            run.cost_usd += result.cost_usd
            for e in result.events:
                if e["kind"] == "policy":
                    run.policy_rules.append(e["rule_id"])
                elif e["kind"] == "tool" and e["name"] in WRITE_TOOLS:
                    run.write_attempts.append(e["name"])
                elif e["kind"] == "llm":
                    run.llm_calls += 1
            if result.handoff:
                run.handoff = result.handoff
            run.final_stage = result.stage
            if case.expire_at == result.stage and clock_ctl and not counters.get("expired"):
                counters["expired"] = 1
                clock_ctl.advance(config.SESSION_TTL_SECONDS + 1)
            nxt = next_message(case, result.stage, counters, _otp_lookup(conn, conv))
            kind, message = nxt if nxt else (None, None)
    except Exception as exc:  # a crashing system fails the case, never the whole run
        run.error, run.final_stage = f"{type(exc).__name__}: {exc}", "error"
    return run


def run_all(cases: list[Case], *, system: str, conn, make_agent, workers: int = 8, max_cost_usd: float | None = None,
            progress=None) -> list[CaseRun]:
    spent = {"usd": 0.0}
    lock = threading.Lock()

    def one(case, clock_ctl=None):
        run = run_case(make_agent(case.faults), conn, case, system, clock_ctl)
        with lock:
            spent["usd"] += run.cost_usd
            if max_cost_usd is not None and spent["usd"] > max_cost_usd:
                raise RuntimeError(f"cost cap exceeded: ${spent['usd']:.4f} > ${max_cost_usd}")
            if progress:
                progress(run)
        return run

    parallel = [c for c in cases if not c.expire_at]
    sequential = [c for c in cases if c.expire_at]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(one, parallel))
    clock_ctl = OffsetClock()
    clock_ctl.install()
    try:
        for case in sequential:
            runs.append(one(case, clock_ctl))
    finally:
        clock_ctl.restore()
    order = {c.id: i for i, c in enumerate(cases)}
    return sorted(runs, key=lambda r: order[r.case_id])
