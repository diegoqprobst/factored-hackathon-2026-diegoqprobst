import pytest

from src.agent.budget import BudgetedLLM
from src.agent.llm import LLMError, LLMResponse
from src.agent.nlu import LLMExtractor
from src.agent.state import Stage


class Paid:
    model = "m"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, **kw):
        self.calls += 1
        return LLMResponse('{"merchant": "Uber"}', "m", 10, 5, 0.3, 5.0)


def test_budget_blocks_after_cap_and_resets_next_day():
    now = {"t": 86400 * 100 + 10}
    inner = Paid()
    llm = BudgetedLLM(inner, daily_usd=0.5, clock=lambda: now["t"])
    llm.complete([])
    llm.complete([])  # spent 0.6 >= 0.5 after this call
    with pytest.raises(LLMError, match="budget"):
        llm.complete([])
    assert inner.calls == 2 and llm.spent_today == pytest.approx(0.6)
    now["t"] += 86400
    llm.complete([])
    assert inner.calls == 3 and llm.spent_today == pytest.approx(0.3)


def test_extractor_keeps_working_when_budget_is_exhausted():  # Review Focus 1
    llm = BudgetedLLM(Paid(), daily_usd=0.0)
    e = LLMExtractor(llm).extract("No reconozco un cargo de Uber", Stage.INTAKE)
    assert (e.source, e.merchant) == ("llm_fallback", "Uber")
