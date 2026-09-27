"""HTTP API for the dispute agent. Run: uv run --group embeddings --env-file .env uvicorn --factory src.agent.api:create_app
Single process, in-memory conversation store with TTL (a declared capacity limit)."""
import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent.demo import demo_scenarios
from src.agent.factory import build_agent
from src.agent.state import Conversation
from src.bank import config, db

STATIC_DIR = Path(__file__).parent / "static"


class ChatIn(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)


class ConversationStore:
    def __init__(self, ttl_seconds: int = 1800):
        self.ttl, self._items, self._locks, self._guard = ttl_seconds, {}, {}, threading.Lock()

    def create(self) -> Conversation:
        conv = Conversation(uuid.uuid4().hex)
        with self._guard:
            self._items[conv.id] = (conv, time.monotonic())
            self._locks[conv.id] = threading.Lock()
        return conv

    def get(self, cid: str) -> Conversation | None:
        with self._guard:
            item = self._items.get(cid)
            if item is None or time.monotonic() - item[1] > self.ttl:
                self._items.pop(cid, None)
                return None
            self._items[cid] = (item[0], time.monotonic())
            return item[0]

    def lock(self, cid: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(cid, threading.Lock())


def metrics_summary(conn) -> dict:
    with db.LOCK:
        rows = conn.execute("select latency_ms, cost_usd from agent_traces").fetchall()
        conversations = conn.execute("select count(distinct conversation_id) from agent_traces").fetchone()[0]
        handoffs = conn.execute("select count(distinct conversation_id) from handoffs").fetchone()[0]
    latencies = [r["latency_ms"] for r in rows]
    total = round(sum(r["cost_usd"] for r in rows), 6)
    return {"turns": len(rows), "conversations": conversations, "handoffs": handoffs,
            "escalation_rate": handoffs / conversations if conversations else None,
            "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else None,
            "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
            "cost_usd_total": total, "cost_usd_per_conversation": total / conversations if conversations else None}


def create_app(conn=None, agent=None, demo_mode: bool | None = None) -> FastAPI:
    if conn is None:
        conn = db.connect(config.SANDBOX_PATH)
        db.create_schema(conn)
    mode = os.environ.get("AGENT_MODE", "hybrid")
    agent = agent or build_agent(conn, mode, llm_budget_usd=float(os.environ.get("AGENT_LLM_DAILY_BUDGET_USD", "0.5"))
                                 if mode == "hybrid" else None)
    demo = demo_mode if demo_mode is not None else os.environ.get("DEMO_MODE") == "1"
    store = ConversationStore()
    app = FastAPI(title="LATAM Bank dispute agent", version="0.3.0")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    cache: dict = {}

    @app.api_route("/", methods=["GET", "HEAD"])  # HEAD for uptime monitors and preview readiness probes
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/v1/config")
    def app_config():
        llm = getattr(agent.extractor, "llm", None)
        return {"mode": agent.mode, "router": getattr(agent.router, "version", "unknown"),
                "llm_model": getattr(llm, "model", None), "demo": demo}

    @app.get("/v1/demo/customers")
    def demo_customers():
        if not demo:
            raise HTTPException(404, "not available")
        if "scenarios" not in cache:
            cache["scenarios"] = demo_scenarios(conn)
        return cache["scenarios"]

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": agent.mode, "router": getattr(agent.router, "version", "unknown")}

    @app.post("/v1/conversations")
    def new_conversation():
        return {"conversation_id": store.create().id}

    @app.post("/v1/chat")
    def chat(body: ChatIn):
        conv = store.create() if body.conversation_id is None else store.get(body.conversation_id)
        if conv is None:
            raise HTTPException(404, "unknown or expired conversation")
        with store.lock(conv.id):
            return asdict(agent.handle(conv, body.message))

    @app.get("/v1/conversations/{cid}/trace")
    def trace(cid: str):
        with db.LOCK:
            rows = conn.execute("select * from agent_traces where conversation_id = ? order by turn", (cid,)).fetchall()
        return [{**dict(r), "events": json.loads(r["events"])} for r in rows]

    @app.get("/v1/demo/sms/{cid}")
    def demo_sms(cid: str):
        conv = store.get(cid) if demo else None
        if conv is None or not conv.challenge_id:
            raise HTTPException(404, "not available")
        with db.LOCK:
            row = conn.execute("select body from sandbox_outbox where challenge_id = ? order by id desc limit 1",
                               (conv.challenge_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "not available")
        return {"sms": row["body"], "note": "SIMULATED SMS - demo mode only"}

    @app.get("/v1/metrics")
    def metrics():
        return metrics_summary(conn)

    if os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("SLACK_BOT_TOKEN"):
        from src.channels.bridge import ChannelBridge
        bridge = ChannelBridge(agent, store, conn, demo)
        if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_WEBHOOK_SECRET"):
            from src.channels.telegram import telegram_router
            app.include_router(telegram_router(bridge, bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
                                               secret=os.environ["TELEGRAM_WEBHOOK_SECRET"]))
        if os.environ.get("SLACK_BOT_TOKEN") and os.environ.get("SLACK_SIGNING_SECRET"):
            from src.channels.slack import slack_router
            app.include_router(slack_router(bridge, bot_token=os.environ["SLACK_BOT_TOKEN"],
                                            signing_secret=os.environ["SLACK_SIGNING_SECRET"]))

    return app
