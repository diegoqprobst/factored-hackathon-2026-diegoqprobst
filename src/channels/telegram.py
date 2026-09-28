"""Telegram webhook adapter. Register with setWebhook(url, secret_token=TELEGRAM_WEBHOOK_SECRET)."""
import hmac
import json
import logging
import threading
import urllib.request
from collections import OrderedDict

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

log = logging.getLogger(__name__)
SEEN_MAX = 2000


def _sender(bot_token: str):
    def send(chat_id, text):
        body = json.dumps({"chat_id": chat_id, "text": text}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{bot_token}/sendMessage", data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    return send


def telegram_router(bridge, *, bot_token: str, secret: str, send=None) -> APIRouter:
    send = send or _sender(bot_token)
    router = APIRouter()
    seen: OrderedDict = OrderedDict()
    lock = threading.Lock()

    def first_delivery(update_id) -> bool:
        # Telegram re-delivers an update until it gets a 2xx: replaying "sí" would act on a question the
        # customer never saw (e.g. block the card), so each update_id runs at most once.
        if update_id is None:
            return True
        with lock:
            if update_id in seen:
                return False
            seen[update_id] = True
            while len(seen) > SEEN_MAX:
                seen.popitem(last=False)
            return True

    def process(chat, text):
        for reply in bridge.handle("telegram", str(chat), text):
            try:
                send(chat, reply)
            except Exception as exc:  # the turn already happened; a 500 would only trigger a replay
                log.warning("telegram send failed: %s", type(exc).__name__)

    @router.post("/v1/channels/telegram")
    async def webhook(request: Request):
        if not hmac.compare_digest(request.headers.get("X-Telegram-Bot-Api-Secret-Token", ""), secret):
            raise HTTPException(401, "bad secret")
        update = await request.json()
        message = update.get("message") or {}
        text, chat = message.get("text"), (message.get("chat") or {}).get("id")
        if not text or chat is None:
            return {"ok": True}  # stickers, photos, edits: nothing to answer
        if first_delivery(update.get("update_id")):
            await run_in_threadpool(process, chat, text)  # LLM + sqlite + HTTP are blocking: keep the loop free
        return {"ok": True}

    return router
