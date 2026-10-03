import json

import pytest

from src.agent.state import Stage
from src.agent.tools import STAGE_TOOLS, Tools, ToolNotAllowed, ToolUnavailable
from src.agent.trace import Tracer
from src.bank.errors import NotFound
from tests.helpers import login


def test_allowlist_matches_the_plan():
    assert STAGE_TOOLS[Stage.IDENTIFY] == {"search_transactions"}
    assert STAGE_TOOLS[Stage.CONFIRM] == {"evaluate_dispute", "create_dispute", "get_dispute", "list_cards"}
    assert Stage.INTAKE not in STAGE_TOOLS and Stage.DONE not in STAGE_TOOLS


def test_denied_tool_is_blocked_and_traced(bank, frozen):
    t = Tracer("c", 1)
    with pytest.raises(ToolNotAllowed):
        Tools(bank, t).call(Stage.IDENTIFY, "create_dispute", token="x", transaction_id="T")
    assert t.events[-1]["kind"] == "tool_denied" and bank.execute("select count(*) from disputes").fetchone()[0] == 0


def test_successful_call_is_traced_without_pii(bank, frozen):
    t = Tracer("c", 1)
    ch = Tools(bank, t).call(Stage.AUTH_DOC, "start_auth", document_number="111")
    assert ch.masked_destination == "***1234"
    event = t.events[-1]
    assert (event["kind"], event["name"], event["ok"], event["attempts"], event["args"]) == ("tool", "start_auth", True, 1, {})
    assert "111" not in json.dumps(t.events)


def test_bank_errors_propagate_with_code(bank, frozen):
    t, tok = Tracer("c", 1), login(bank, "111")
    with pytest.raises(NotFound):
        Tools(bank, t).call(Stage.CHOOSE, "get_transaction", token=tok, transaction_id="TRX-B1")
    assert (t.events[-1]["ok"], t.events[-1]["error"], t.events[-1]["args"]) == (False, "not_found", {"transaction_id": "TRX-B1"})


def test_transient_fault_is_retried(bank, frozen):
    t, tok, slept = Tracer("c", 1), login(bank, "111"), []
    faults = {"search_transactions": 1}
    out = Tools(bank, t, faults=faults, sleep=slept.append).call(Stage.IDENTIFY, "search_transactions", token=tok, merchant="uber")
    assert [v.transaction_id for v in out] == ["TRX-A9"] and t.events[-1]["attempts"] == 2 and faults == {"search_transactions": 0}
    assert slept == [0.1]


def test_persistent_fault_becomes_unavailable(bank, frozen):
    t, tok = Tracer("c", 1), login(bank, "111")
    with pytest.raises(ToolUnavailable):
        Tools(bank, t, faults={"search_transactions": 99}, sleep=lambda s: None).call(
            Stage.IDENTIFY, "search_transactions", token=tok)
    assert (t.events[-1]["ok"], t.events[-1]["error"], t.events[-1]["attempts"]) == (False, "unavailable", 3)
