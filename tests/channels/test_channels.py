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
