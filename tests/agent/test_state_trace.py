import json
from dataclasses import dataclass

from src.agent.state import Conversation, Stage
from src.agent.trace import Tracer, persist


@dataclass
class Ext:
    amount: float | None = None
    date_from: object = None
    date_to: object = None
    merchant: str | None = None
    charges_count: int | None = None


def test_remember_merges_slots_and_keeps_max_charges():
    c = Conversation("c1")
    c.remember(Ext(amount=350.0, merchant="Oxxo", charges_count=3))
    c.remember(Ext(merchant="Uber", charges_count=1))
    assert c.slots == {"amount": 350.0, "merchant": "Uber"} and c.charges_count == 3


def test_reset_case_keeps_auth_and_history():
    c = Conversation("c1", stage=Stage.DONE, token="t", customer_id="CLI-A", transaction_id="T1",
                     dispute_type="duplicate", slots={"amount": 1.0}, disputes=["D1"])
    c.reset_case()
    assert (c.token, c.customer_id, c.disputes) == ("t", "CLI-A", ["D1"])
    assert (c.transaction_id, c.dispute_type, c.slots) == (None, None, {})


def test_tracer_records_events_and_cost(bank):
    t = Tracer("c1", 2)
    t.record("router", intent="x")
    t.record("llm", cost_usd=0.0001, prompt_tokens=10)
    t.record("llm", cost_usd=0.0002, prompt_tokens=10)
    assert t.trace_id == "c1:2" and t.cost_usd == 0.0003 and [e["kind"] for e in t.events] == ["router", "llm", "llm"]
    persist(bank, t, "c1", 2, "intake", 12.5)
    row = bank.execute("select * from agent_traces").fetchone()
    assert (row["trace_id"], row["turn"], row["stage"], row["latency_ms"], row["cost_usd"]) == ("c1:2", 2, "intake", 12.5, 0.0003)
    assert json.loads(row["events"])[0]["kind"] == "router"


def test_new_tables_exist(bank):
    names = {r["name"] for r in bank.execute("select name from sqlite_master where type='table'")}
    assert {"handoffs", "agent_traces"} <= names
