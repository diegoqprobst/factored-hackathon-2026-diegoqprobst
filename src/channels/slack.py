"""Slack Events API adapter (signed requests). Subscribe the app to message.im and app_mention events."""
import hashlib
import hmac
import json
import re
import threading
import time
import urllib.request

from fastapi import APIRouter, HTTPException, Request


def verify_slack_signature(signing_secret: str, timestamp: str, body: bytes, signature: str, now: float) -> bool:
    try:
        if abs(now - int(timestamp)) > 300:
            return False
    except (TypeError, ValueError):
        return False
    expected = "v0=" + hmac.new(signing_secret.encode(), f"v0:{timestamp}:".encode() + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")


def _sender(bot_token: str):
    def send(channel, text):
        body = json.dumps({"channel": channel, "text": text}).encode()
        req = urllib.request.Request("https://slack.com/api/chat.postMessage", data=body, headers={
            "Content-Type": "application/json; charset=utf-8", "Authorization": f"Bearer {bot_token}"})
        urllib.request.urlopen(req, timeout=10).read()
    return send


def slack_router(bridge, *, bot_token: str, signing_secret: str, send=None, clock=time.time) -> APIRouter:
    send = send or _sender(bot_token)
    router = APIRouter()
    seen: set[str] = set()
    lock = threading.Lock()

    @router.post("/v1/channels/slack/events")
    async def events(request: Request):
        body = await request.body()
        if not verify_slack_signature(signing_secret, request.headers.get("X-Slack-Request-Timestamp", ""), body,
                                      request.headers.get("X-Slack-Signature", ""), clock()):
            raise HTTPException(401, "bad signature")
        payload = json.loads(body or b"{}")
        if payload.get("type") == "url_verification":
            return {"challenge": payload.get("challenge")}
        if request.headers.get("X-Slack-Retry-Num"):
            return {"ok": True}  # the first delivery is (or was) being processed
        event, event_id = payload.get("event") or {}, payload.get("event_id")
        with lock:
            if event_id in seen:
                return {"ok": True}
            seen.add(event_id)
        if event.get("bot_id") or event.get("subtype") or event.get("type") not in ("message", "app_mention"):
            return {"ok": True}
        text = re.sub(r"<@[A-Z0-9]+>", "", event.get("text") or "").strip()
        channel = event.get("channel")
        if text and channel:
            for reply in bridge.handle("slack", f"{channel}:{event.get('user')}", text):
                send(channel, reply)
        return {"ok": True}

    return router
