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
    assert c.head("/").status_code == 200


def test_demo_customers_hidden_outside_demo(bank, frozen):
    c = TestClient(create_app(conn=bank, agent=build_agent(bank, "baseline"), demo_mode=False))
    assert c.get("/v1/demo/customers").status_code == 404 and c.get("/v1/config").json()["demo"] is False


def _names(conn):
    return {x["scenario"]: x for x in demo_scenarios(conn)}


def _add_second_normal_candidate(conn):
    conn.execute("insert into transactions values (" + ",".join("?" * 16) + ")",
                 ("TRX-A10", "CLI-A", "PRD-A-CC", "2026-06-15T18:00:00", "2026-06-15", "Purchase", 20.0, "COP",
                  0.01, "POS", "Cabify", None, "Bogotá", "Colombia", "Approved", 0))
    conn.commit()


def _open_dispute(conn, cid, tid):
    conn.execute("insert into disputes values (?,?,?,?,?,?,?,?,?,?)",
                 (f"DSP-{tid}", f"k-{tid}", "s", cid, tid, "unrecognized", "open", "P1", "v1", "2026-06-17T14:00:00"))
    conn.commit()


def test_demo_rotates_past_a_disputed_transaction(bank, frozen):
    _add_second_normal_candidate(bank)
    first = _names(bank)["normal"]
    assert "Cabify" in first["message_es"]  # candidates are ordered by transaction id: TRX-A10 < TRX-A9
    _open_dispute(bank, "CLI-A", "TRX-A10")
    assert "Uber" in _names(bank)["normal"]["message_es"]
    _open_dispute(bank, "CLI-A", "TRX-A9")
    assert "normal" not in _names(bank)  # a used-up scenario is hidden rather than shown broken


def test_demo_hides_customer_with_recent_otp_or_blocked_card(bank, frozen):
    bank.execute("insert into otp_challenges(challenge_id, customer_id, code_hash, expires_at, created_at) "
                 "values ('c1', 'CLI-A', 'h', '2026-06-17T15:05:00+00:00', '2026-06-17T14:59:00+00:00')")
    bank.commit()
    s = _names(bank)
    assert not {"normal", "ambiguous", "stolen_card"} & set(s)  # another judge is mid-flow on CLI-A
    assert s["no_phone"]["document"] == "333"
    bank.execute("delete from otp_challenges")
    bank.execute("insert into card_blocks values ('B1', 'PRD-A-CC', 'CLI-A', 's', 'stolen', '2026-06-17T14:00:00')")
    bank.commit()
    assert "stolen_card" not in _names(bank)


def test_demo_customers_endpoint_reflects_new_disputes(bank, frozen):
    _add_second_normal_candidate(bank)
    c = TestClient(create_app(conn=bank, agent=build_agent(bank, "baseline"), demo_mode=True))
    before = {x["scenario"]: x for x in c.get("/v1/demo/customers").json()}["normal"]
    _open_dispute(bank, "CLI-A", "TRX-A10")
    after = {x["scenario"]: x for x in c.get("/v1/demo/customers").json()}["normal"]
    assert "Cabify" in before["message_es"] and "Uber" in after["message_es"]
    assert set(after) == {"scenario", "document", "message_es", "message_pt"}  # no internal ids leak


def test_page_has_the_guided_flow_elements(bank, frozen):
    c = TestClient(create_app(conn=bank, agent=build_agent(bank, "baseline"), demo_mode=True))
    html = c.get("/").text
    for element in ('id="steps"', 'id="next"', 'id="active"', 'id="sms"', 'id="welcome"'):
        assert element in html
    js = c.get("/static/app.js").text
    assert "startScenario" in js and "renderNext" in js


def test_ui_assets_are_cache_busted(bank, frozen):  # seen live: a returning browser kept the old app.js
    import re
    c = TestClient(create_app(conn=bank, agent=build_agent(bank, "baseline"), demo_mode=True))
    r = c.get("/")
    assert "no-cache" in r.headers.get("cache-control", "")
    js = re.search(r'/static/app\.js\?v=([0-9a-f]{8,})"', r.text)
    css = re.search(r'/static/styles\.css\?v=([0-9a-f]{8,})"', r.text)
    assert js and css
    assert c.head("/").status_code == 200


def test_demo_pool_failure_does_not_escape_its_thread(bank, frozen, monkeypatch):
    # In CI the fixture closed the connection while the pool thread was scanning it (segfault). The close now takes
    # db.LOCK; a pool build that fails anyway (closed or broken database) is logged and the page shows no scenarios.
    import sqlite3
    import threading

    from fastapi.testclient import TestClient

    import src.agent.api as api
    from src.agent.factory import build_agent
    escaped = []
    monkeypatch.setattr(threading, "excepthook", lambda args: escaped.append(args.exc_value))

    def broken(conn):
        raise sqlite3.ProgrammingError("Cannot operate on a closed database.")
    monkeypatch.setattr(api, "demo_pool", broken)
    c = TestClient(api.create_app(conn=bank, agent=build_agent(bank, "baseline"), demo_mode=True))
    assert c.get("/v1/demo/customers").json() == [] and escaped == []
