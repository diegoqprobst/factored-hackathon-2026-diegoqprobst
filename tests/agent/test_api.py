import pytest
from fastapi.testclient import TestClient

from src.agent.api import create_app
from src.agent.factory import build_agent


@pytest.fixture
def client(bank, frozen):
    return TestClient(create_app(conn=bank, agent=build_agent(bank, "baseline"), demo_mode=True))


def chat(client, cid, message):
    r = client.post("/v1/chat", json={"conversation_id": cid, "message": message})
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "mode": "baseline", "router": "keyword_v1"}


def test_full_dispute_over_http_with_demo_sms(client, bank):
    cid = client.post("/v1/conversations").json()["conversation_id"]
    assert chat(client, cid, "No reconozco un cargo de Uber")["stage"] == "auth_doc"
    assert chat(client, cid, "111")["stage"] == "auth_otp"
    sms = client.get(f"/v1/demo/sms/{cid}").json()["sms"]
    code = sms.split()[-1]
    out = chat(client, cid, code)
    assert out["stage"] == "confirm" and out["events"] and out["trace_id"] == f"{cid}:3"
    out = chat(client, cid, "sí")
    assert out["stage"] == "block_offer" and "DSP-" in out["reply"]
    trace = client.get(f"/v1/conversations/{cid}/trace").json()
    assert [t["turn"] for t in trace] == [1, 2, 3, 4]


def test_new_conversation_when_id_omitted(client):
    out = client.post("/v1/chat", json={"message": "hola"}).json()
    assert out["conversation_id"] and out["stage"] == "intake"


def test_unknown_conversation_is_404(client):
    assert client.post("/v1/chat", json={"conversation_id": "nope", "message": "hola"}).status_code == 404


@pytest.mark.parametrize("message", ["", "x" * 2001])
def test_message_length_is_validated(client, message):
    assert client.post("/v1/chat", json={"message": message}).status_code == 422


def test_demo_sms_is_disabled_outside_demo_mode(bank, frozen):
    c = TestClient(create_app(conn=bank, agent=build_agent(bank, "baseline"), demo_mode=False))
    cid = c.post("/v1/conversations").json()["conversation_id"]
    assert c.get(f"/v1/demo/sms/{cid}").status_code == 404


def test_metrics(client):
    cid = client.post("/v1/conversations").json()["conversation_id"]
    chat(client, cid, "Quiero hablar con un asesor")
    m = client.get("/v1/metrics").json()
    assert (m["turns"], m["conversations"], m["handoffs"], m["escalation_rate"]) == (1, 1, 1, 1.0)
    assert m["latency_ms_p95"] >= m["latency_ms_p50"] >= 0 and m["cost_usd_total"] == 0.0


def test_conversation_store_evicts_expired_and_caps_size():  # Final review Important 1
    import time as _time
    from src.agent.api import ConversationStore
    s = ConversationStore(ttl_seconds=0.05, max_items=3)
    old = s.create()
    s.lock(old.id)
    _time.sleep(0.06)
    ids = [s.create().id for _ in range(5)]
    assert len(s._items) <= 3 and len(s._locks) <= 3
    assert old.id not in s._items and old.id not in s._locks
    assert s.get(ids[-1]) is not None  # the newest survive


def test_metrics_latency_is_rounded(bank, frozen):  # the live footer showed 3228.1699999999996 ms
    from src.agent.api import metrics_summary
    bank.execute("insert into agent_traces values ('t1', 'c', 1, 'done', '[]', 3228.1699999999996, 0.1234567891, 'x')")
    m = metrics_summary(bank)
    assert m["latency_ms_p50"] == 3228.2 and m["latency_ms_p95"] == 3228.2


def test_demo_customers_answers_503_fast_while_the_pool_builds(bank, frozen, monkeypatch):
    import threading
    import time as _time

    from fastapi.testclient import TestClient

    import src.agent.api as api
    from src.agent.factory import build_agent
    release = threading.Event()
    monkeypatch.setattr(api, "demo_pool", lambda conn: (release.wait(5), {})[1])
    c = TestClient(api.create_app(conn=bank, agent=build_agent(bank, "baseline"), demo_mode=True))
    start = _time.monotonic()
    r = c.get("/v1/demo/customers")
    assert r.status_code == 503 and _time.monotonic() - start < 3
    release.set()
