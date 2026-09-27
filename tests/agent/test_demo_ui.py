from fastapi.testclient import TestClient

from src.agent.api import create_app
from src.agent.demo import demo_scenarios
from src.agent.factory import build_agent


def test_demo_scenarios_from_fixture(bank, frozen):
    s = {x["scenario"]: x for x in demo_scenarios(bank)}
    assert s["normal"]["document"] == "111" and "Uber" in s["normal"]["message_es"]
    assert s["ambiguous"]["document"] == "111" and "Oxxo" in s["ambiguous"]["message_pt"]
    assert s["stolen_card"]["document"] in ("111", "222") and s["no_phone"]["document"] == "333"
    assert "large_amount" not in s  # the fixture has no large movement without a merchant


def test_config_customers_and_static_routes(bank, frozen):
    c = TestClient(create_app(conn=bank, agent=build_agent(bank, "baseline"), demo_mode=True))
    assert c.get("/v1/config").json() == {"mode": "baseline", "router": "keyword_v1", "llm_model": None, "demo": True}
    assert any(x["scenario"] == "normal" for x in c.get("/v1/demo/customers").json())
    html = c.get("/")
    assert html.status_code == 200 and "text/html" in html.headers["content-type"] and 'id="chat"' in html.text


def test_demo_customers_hidden_outside_demo(bank, frozen):
    c = TestClient(create_app(conn=bank, agent=build_agent(bank, "baseline"), demo_mode=False))
    assert c.get("/v1/demo/customers").status_code == 404 and c.get("/v1/config").json()["demo"] is False
