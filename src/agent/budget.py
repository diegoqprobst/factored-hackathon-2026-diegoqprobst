"""Daily spend cap for the LLM behind a public demo. Over the cap, calls fail fast with LLMError and the extractor
falls back to rules, so the agent keeps answering (deterministically) instead of spending or erroring."""
import threading
import time

from src.agent.llm import LLMError, LLMResponse


class BudgetedLLM:
    def __init__(self, llm, daily_usd: float, clock=time.time):
        self._llm, self.daily_usd, self._clock = llm, daily_usd, clock
        self.model = getattr(llm, "model", "unknown")
        self._day, self._spent, self._lock = None, 0.0, threading.Lock()

    @property
    def spent_today(self) -> float:
        with self._lock:
            return self._spent if self._day == int(self._clock() // 86400) else 0.0

    def complete(self, messages, **kw) -> LLMResponse:
        day = int(self._clock() // 86400)
        with self._lock:
            if self._day != day:
                self._day, self._spent = day, 0.0
            if self._spent >= self.daily_usd:
                raise LLMError("daily LLM budget exhausted")
        resp = self._llm.complete(messages, **kw)
        with self._lock:
            self._spent += resp.cost_usd
        return resp
