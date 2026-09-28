import hashlib
import hmac
import json
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agent.api import ConversationStore
from src.agent.factory import build_agent
from src.channels.bridge import ChannelBridge
from src.channels.slack import slack_router, verify_slack_signature
from src.channels.telegram import telegram_router


def app_with(router):
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_bridge_keeps_one_conversation_per_chat_and_shows_demo_sms(bank, frozen):
    b = ChannelBridge(build_agent(bank, "baseline"), ConversationStore(), bank, demo=True)
    assert "documento" in b.handle("tg", "42", "No reconozco un cargo de Uber")[0]
    out = b.handle("tg", "42", "111")
    assert len(out) == 2 and "SMS simulado" in out[1] and out[1].split()[-1].isdigit()
    from src.agent.responder import render
    assert b.handle("tg", "42", "/nueva") == [render("greeting", "es")]
    assert "documento" in b.handle("tg", "42", "No reconozco un cargo de Uber")[0]  # a fresh conversation


def test_telegram_webhook(bank, frozen):
    sent = []
    b = ChannelBridge(build_agent(bank, "baseline"), ConversationStore(), bank, demo=False)
    c = app_with(telegram_router(b, bot_token="t", secret="s3", send=lambda chat, text: sent.append((chat, text))))
    upd = {"update_id": 1, "message": {"chat": {"id": 42}, "text": "Quiero hablar con un asesor"}}
    assert c.post("/v1/channels/telegram", json=upd).status_code == 401
    assert c.post("/v1/channels/telegram", json=upd, headers={"X-Telegram-Bot-Api-Secret-Token": "s3"}).status_code == 200
    assert sent and sent[0][0] == 42 and "asesor" in sent[0][1]
    sent.clear()
    sticker = {"update_id": 2, "message": {"chat": {"id": 42}, "sticker": {}}}  # Review Focus 3
    assert c.post("/v1/channels/telegram", json=sticker, headers={"X-Telegram-Bot-Api-Secret-Token": "s3"}).status_code == 200
    assert sent == []


def sign(secret, ts, body):
    return "v0=" + hmac.new(secret.encode(), f"v0:{ts}:{body.decode()}".encode(), hashlib.sha256).hexdigest()


def test_slack_signature():
    body, now = b'{"a":1}', 1_700_000_000
    good = sign("sec", now, body)
    assert verify_slack_signature("sec", str(now), body, good, now)
    assert not verify_slack_signature("sec", str(now), body, "v0=bad", now)
    assert not verify_slack_signature("sec", str(now - 301), body, sign("sec", now - 301, body), now)  # Review Focus 2


def test_slack_events(bank, frozen):
    sent, now = [], int(time.time())
    b = ChannelBridge(build_agent(bank, "baseline"), ConversationStore(), bank, demo=False)
    c = app_with(slack_router(b, bot_token="x", signing_secret="sec", send=lambda ch, t: sent.append((ch, t)), clock=lambda: now))

    def post(payload, extra=None):
        body = json.dumps(payload).encode()
        h = {"X-Slack-Request-Timestamp": str(now), "X-Slack-Signature": sign("sec", now, body), "content-type": "application/json"}
        h.update(extra or {})
        return c.post("/v1/channels/slack/events", content=body, headers=h)

    assert post({"type": "url_verification", "challenge": "abc"}).json() == {"challenge": "abc"}
    ev = {"type": "event_callback", "event_id": "E1", "event": {"type": "message", "channel": "D1", "user": "U1", "text": "Quiero hablar con un asesor"}}
    assert post(ev).status_code == 200 and sent and sent[0][0] == "D1"
    assert post(ev).status_code == 200 and len(sent) == 1  # duplicate event_id ignored
    assert post({**ev, "event_id": "E2"}, {"X-Slack-Retry-Num": "1"}).status_code == 200 and len(sent) == 1
    bot = {"type": "event_callback", "event_id": "E3", "event": {"type": "message", "channel": "D1", "bot_id": "B", "text": "loop"}}
    assert post(bot).status_code == 200 and len(sent) == 1
    bad = c.post("/v1/channels/slack/events", content=b"{}", headers={"X-Slack-Request-Timestamp": str(now), "X-Slack-Signature": "v0=x"})
    assert bad.status_code == 401


class _CountingBridge:
    def __init__(self, delay=0.0):
        self.calls, self.delay = [], delay

    def handle(self, channel, chat_id, text):
        self.calls.append(text)
        time.sleep(self.delay)
        return ["ok"]


def test_telegram_redelivery_after_failed_send_does_not_run_the_turn_twice():  # Final review Important 2
    bridge, H = _CountingBridge(), {"X-Telegram-Bot-Api-Secret-Token": "s3"}

    def failing_send(chat, text):
        raise OSError("telegram 429")
    c = app_with(telegram_router(bridge, bot_token="t", secret="s3", send=failing_send))
    upd = {"update_id": 7, "message": {"chat": {"id": 42}, "text": "sí"}}
    assert c.post("/v1/channels/telegram", json=upd, headers=H).status_code == 200  # no 500 -> no redelivery
    assert c.post("/v1/channels/telegram", json=upd, headers=H).status_code == 200  # a redelivery anyway
    assert bridge.calls == ["sí"]


def test_channel_turn_does_not_block_the_server():  # Final review Important 3
    import socket
    import threading
    import urllib.request

    import uvicorn
    bridge = _CountingBridge(delay=1.5)
    app = FastAPI()
    app.include_router(telegram_router(bridge, bot_token="t", secret="s3", send=lambda chat, text: None))
    app.get("/health")(lambda: {"status": "ok"})
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.02)
    body = json.dumps({"update_id": 8, "message": {"chat": {"id": 42}, "text": "hola"}}).encode()
    req = urllib.request.Request(base + "/v1/channels/telegram", data=body, method="POST", headers={
        "Content-Type": "application/json", "X-Telegram-Bot-Api-Secret-Token": "s3"})
    t = threading.Thread(target=lambda: urllib.request.urlopen(req, timeout=5).read())
    t.start()
    time.sleep(0.3)
    start = time.monotonic()
    assert urllib.request.urlopen(base + "/health", timeout=5).status == 200
    elapsed = time.monotonic() - start
    t.join()
    server.should_exit = True
    assert elapsed < 0.5 and bridge.calls == ["hola"]
