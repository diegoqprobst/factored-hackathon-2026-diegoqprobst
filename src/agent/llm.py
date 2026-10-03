"""OpenRouter chat client: bounded retries on transient failures, hard timeout, exact cost from `usage.cost`."""
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_MODEL = "google/gemma-4-31b-it"
URL = "https://openrouter.ai/api/v1/chat/completions"
TRANSIENT_HTTP = {408, 409, 429, 500, 502, 503, 504}


class LLMError(Exception):
    pass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float


def _urllib_transport(api_key: str):
    def send(body: dict, timeout: float) -> dict:
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    return send


class OpenRouterLLM:
    def __init__(self, model: str = DEFAULT_MODEL, *, api_key: str | None = None, timeout: float = 15.0,
                 max_retries: int = 1, transport=None, sleep=time.sleep):
        self.model, self.timeout, self.max_retries, self._sleep = model, timeout, max_retries, sleep
        self._transport = transport or _urllib_transport(api_key or os.environ["OPENROUTER_API_KEY"])

    def complete(self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 250) -> LLMResponse:
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                "usage": {"include": True}}
        start, last = time.perf_counter(), None
        for attempt in range(self.max_retries + 1):
            try:
                data = self._transport(body, self.timeout)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in TRANSIENT_HTTP:
                    raise LLMError(f"HTTP {exc.code}") from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
            if attempt < self.max_retries:
                self._sleep(0.5 * 2 ** attempt)
        else:
            raise LLMError(f"unavailable after {self.max_retries + 1} attempts: {last!r}")
        try:
            usage = data.get("usage") or {}
            return LLMResponse(text=data["choices"][0]["message"]["content"], model=self.model,
                               prompt_tokens=int(usage.get("prompt_tokens", 0)),
                               completion_tokens=int(usage.get("completion_tokens", 0)),
                               cost_usd=float(usage.get("cost", 0.0)),
                               latency_ms=round((time.perf_counter() - start) * 1000, 1))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("malformed response") from exc


def parse_json_object(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text or "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("not a JSON object")
    return obj
