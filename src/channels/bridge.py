"""Channel-agnostic bridge: one conversation per (channel, chat id); the same agent and guarantees as the web API."""
import threading

from src.agent.responder import render
from src.bank import db

RESET = {"/start", "/reset", "/nueva", "/nova"}


class ChannelBridge:
    def __init__(self, agent, store, conn, demo: bool):
        self.agent, self.store, self.conn, self.demo = agent, store, conn, demo
        self._map: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def _conversation(self, channel: str, chat_id: str, reset: bool):
        with self._lock:
            cid = None if reset else self._map.get((channel, chat_id))
            conv = self.store.get(cid) if cid else None
            if conv is None:
                conv = self.store.create()
                self._map[(channel, chat_id)] = conv.id
            return conv

    def handle(self, channel: str, chat_id: str, text: str) -> list[str]:
        text = (text or "").strip()
        if text.lower() in RESET:
            self._conversation(channel, chat_id, reset=True)
            return [render("greeting", "es")]
        conv = self._conversation(channel, chat_id, reset=False)
        with self.store.lock(conv.id):
            result = self.agent.handle(conv, text)
        replies = [result.reply]
        if self.demo and result.stage == "auth_otp" and conv.challenge_id:
            with db.LOCK:
                row = self.conn.execute("select body from sandbox_outbox where challenge_id = ? order by id desc limit 1",
                                        (conv.challenge_id,)).fetchone()
            if row:
                replies.append(f"📱 SMS simulado (solo demo): {row['body']}")
        return replies
