"""Per-turn audit trail. Events carry decisions, tool outcomes, policy rules and LLM usage — never raw PII."""
import json
import time

from src.bank import clock, db


class Tracer:
    def __init__(self, conversation_id: str, turn: int):
        self.trace_id = f"{conversation_id}:{turn}"
        self.events: list[dict] = []
        self._t0 = time.perf_counter()

    def record(self, kind: str, **data) -> None:
        self.events.append({"t_ms": round((time.perf_counter() - self._t0) * 1000, 2), "kind": kind, **data})

    @property
    def cost_usd(self) -> float:
        return round(sum(e.get("cost_usd", 0.0) for e in self.events if e["kind"] == "llm"), 8)


def persist(conn, tracer: Tracer, conversation_id: str, turn: int, stage: str, latency_ms: float) -> None:
    with db.LOCK:
        conn.execute("insert or replace into agent_traces values (?,?,?,?,?,?,?,?)",
                     (tracer.trace_id, conversation_id, turn, stage, json.dumps(tracer.events, default=str),
                      latency_ms, tracer.cost_usd, clock.now().isoformat()))
        conn.commit()
