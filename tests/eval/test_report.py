from src.eval.report import render_report
from src.eval.scoring import aggregate, score
from tests.eval.test_scoring import case, run


def test_report_renders_both_systems_and_undefined_cost():
    rows = [score(case("resolved", []), run(), [])]
    summary = {"meta": {"cases": 1, "cases_sha256": "abc", "git_sha": "def", "router": "e5_v1", "llm_model": "m",
                        "prompt_sha256": "p", "policy_version": "v1", "date": "2026-09-26", "leakage_hits": 0},
               "systems": {"baseline": aggregate(rows), "hybrid": aggregate(rows)},
               "repeats": {"cases": 1, "agreement_rate": 1.0, "sar_by_rep": [1.0], "sar_min": 1.0, "sar_max": 1.0}}
    text = render_report(summary)
    for heading in ("Safe automated resolution", "Escalation quality", "Unsafe outcomes", "Latency and cost",
                    "By language", "By customer segment", "By category", "Repeated runs", "Limitations"):
        assert heading in text
    assert "| baseline |" in text and "| hybrid |" in text and "1/1" in text


def test_run_refuses_to_score_a_misconfigured_environment(monkeypatch):
    import pytest

    from src.eval import run as eval_run
    monkeypatch.delenv("BANK_SESSION_SECRET", raising=False)
    with pytest.raises(SystemExit, match="BANK_SESSION_SECRET"):
        eval_run.main(["run", "--systems", "baseline"])
