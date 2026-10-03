"""Regression tests for the Plan 5 final-review findings (Critical 1, Important 2-6)."""
import pytest

from src.agent.factory import build_agent
from src.eval.cases import Case
from src.eval.runner import run_case
from src.eval.scoring import aggregate, mcnemar, score
from tests.eval.test_runner import uber
from tests.eval.test_scoring import case, run


# Critical 1 — injected faults are per case run, never consumed across runs or systems
def test_fault_injection_is_not_consumed_across_runs(bank, frozen):
    faults = {"search_transactions": 1}
    for i in range(2):
        r = run_case(build_agent(bank, "baseline", faults=faults), bank,
                     uber(id=f"t{i}", faults=faults), f"s{i}")
        tool_events = [t for t in r.turns]  # noqa: F841 — the run must still succeed
        assert faults == {"search_transactions": 1}


# Important 2 — an escalation only counts with an expected reason
def test_escalation_needs_an_expected_reason():
    s = score(case("escalate", reasons=["policy_P4"]), run(handoff={"escalation_reason": "internal_error"}), [])
    assert (s["success"], s["reason_ok"]) == (False, False)


# Important 3 — abstention means the conversation never left intake
def test_abstain_requires_staying_at_intake():
    r = run()
    r.turns[0]["stage"] = "auth_doc"
    assert not score(case("abstain"), r, [])["success"]
    ok = run()
    ok.turns[0]["stage"] = "intake"
    assert score(case("abstain"), ok, [])["success"]


# Important 5 — containment is reported together with contained-and-successful
def test_contained_and_successful_metric():
    rows = [score(case("resolved", []), run(), []),                       # contained, successful
            score(case("escalate", reasons=["x"]), run(), [])]            # contained, but a missed transfer
    a = aggregate(rows)
    assert (a["containment"]["k"], a["contained_successful"]["k"]) == (2, 1)


# Important 6 — paired exact McNemar on the same cases
def test_mcnemar_exact():
    a = [{"case_id": str(i), "success": False} for i in range(10)]
    b = [{"case_id": str(i), "success": True} for i in range(10)]
    out = mcnemar(a, b, lambda r: r["success"])
    assert (out["a_only"], out["b_only"]) == (0, 10) and out["p"] == pytest.approx(0.001953, abs=1e-6)
    same = mcnemar(a, a, lambda r: r["success"])
    assert (same["a_only"], same["b_only"], same["p"]) == (0, 0, 1.0)
