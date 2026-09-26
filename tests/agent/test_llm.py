import io
import urllib.error

import pytest

from src.agent.llm import LLMError, OpenRouterLLM, parse_json_object

OK = {"choices": [{"message": {"content": '{"a": 1}'}}],
      "usage": {"prompt_tokens": 200, "completion_tokens": 20, "cost": 0.00006}}


def http_error(code):
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(b"{}"))


def test_complete_parses_text_usage_and_cost():
    seen = {}

    def transport(body, timeout):
        seen.update(body=body, timeout=timeout)
        return OK
    r = OpenRouterLLM("m", api_key="k", transport=transport).complete([{"role": "user", "content": "hi"}])
    assert (r.text, r.model, r.prompt_tokens, r.completion_tokens, r.cost_usd) == ('{"a": 1}', "m", 200, 20, 0.00006)
    assert seen["body"]["usage"] == {"include": True} and seen["body"]["temperature"] == 0.0 and seen["timeout"] == 20.0


def test_retries_transient_errors_with_backoff():
    calls, slept = [], []

    def transport(body, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise http_error(429) if len(calls) == 1 else TimeoutError()
        return OK
    r = OpenRouterLLM("m", api_key="k", transport=transport, sleep=slept.append).complete([])
    assert r.text == '{"a": 1}' and len(calls) == 3 and slept == [0.5, 1.0]


def test_gives_up_after_bounded_retries():
    def transport(body, timeout):
        raise urllib.error.URLError("down")
    with pytest.raises(LLMError):
        OpenRouterLLM("m", api_key="k", transport=transport, max_retries=2, sleep=lambda s: None).complete([])


def test_client_errors_are_not_retried():
    calls = []

    def transport(body, timeout):
        calls.append(1)
        raise http_error(400)
    with pytest.raises(LLMError):
        OpenRouterLLM("m", api_key="k", transport=transport, sleep=lambda s: None).complete([])
    assert len(calls) == 1


def test_malformed_response_is_llm_error():
    with pytest.raises(LLMError):
        OpenRouterLLM("m", api_key="k", transport=lambda b, t: {"choices": []}).complete([])


def test_parse_json_object():
    assert parse_json_object('```json\n{"x": [1, 2]}\n```') == {"x": [1, 2]}
    with pytest.raises(ValueError):
        parse_json_object("no json")
    with pytest.raises(ValueError):
        parse_json_object("[1, 2]")
