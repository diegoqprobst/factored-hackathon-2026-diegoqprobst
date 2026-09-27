"""Telegram webhook adapter. Register with setWebhook(url, secret_token=TELEGRAM_WEBHOOK_SECRET)."""
import hmac
import json
import urllib.request

from fastapi import APIRouter, HTTPException, Request


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

    @router.post("/v1/channels/telegram")
    async def webhook(request: Request):
        if not hmac.compare_digest(request.headers.get("X-Telegram-Bot-Api-Secret-Token", ""), secret):
            raise HTTPException(401, "bad secret")
        update = await request.json()
        message = update.get("message") or {}
        text, chat = message.get("text"), (message.get("chat") or {}).get("id")
        if not text or chat is None:
            return {"ok": True}  # stickers, photos, edits: nothing to answer
        for reply in bridge.handle("telegram", str(chat), text):
            send(chat, reply)
        return {"ok": True}

    return router
