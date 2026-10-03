# Plan 4 — Web UI with live trace, public deployment (Hugging Face Spaces), Slack/Telegram adapters — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the agent in front of judges. Build a zero-build web chat with a live trace panel, demo scenarios and a simulated SMS inbox. Protect the public deployment with an LLM spend cap. Deploy on Hugging Face Spaces (Docker, free tier), pulling the sandbox database from a private HF dataset repo. Add Slack and Telegram webhook adapters that go live once tokens exist.

**Architecture:**
- **UI.** Static files in `src/agent/static/` (HTML + vanilla JS + CSS, no framework, no build step), served by the same FastAPI app that serves the API, so one container holds everything.
- **Demo endpoints.** `src/agent/demo.py` finds one real customer per demo scenario via SQL.
- **Spend cap.** `src/agent/budget.py` wraps the LLM with a daily spend cap. When the cap is reached, extraction falls back to rules and the conversation continues.
- **Channels.** `src/channels/` holds a channel-agnostic bridge (chat id → conversation) plus Telegram (webhook with secret header) and Slack (Events API with HMAC signature) adapters, mounted as FastAPI routes only when their tokens are configured.
- **Deployment.** `deploy/hf_space/` holds the Dockerfile, entrypoint and Space README, and `scripts/deploy_hf.py` creates the private dataset repo, uploads `sandbox.db`, creates the Space, sets secrets and uploads the app.

**Tech Stack:** FastAPI StaticFiles, vanilla JS (fetch), CSS custom properties (light/dark), `huggingface_hub` (already installed with sentence-transformers), Docker (built remotely by HF), stdlib `hmac`/`urllib` for channels.

**Spec:** `docs/superpowers/specs/2026-09-26-dispute-agent-design.md` §2 (channels), §7 (security, observability), §9 (tooling and deployment; CopilotKit is replaced by the documented plain-chat fallback).

## Global Constraints

- **Deployment target (decided 2026-09-27).** Hugging Face Spaces, Docker SDK, `app_port: 7860`. Cloud Run was blocked because every GCP billing account is closed.
- **Data exposure.** `sandbox.db` lives only in a **private** HF dataset repo and is downloaded at container start with a Space secret. It is never inside the public Space repo or the public GitHub repo.
- **Public app safety.**
  - The app runs with `DEMO_MODE=1` (simulated SMS visible, declared in the UI as a demo).
  - LLM spend is capped by `AGENT_LLM_DAILY_BUDGET_USD` (default **0.50**). Over the cap, the hybrid uses rules extraction, flagged `budget_fallback` in the trace.
  - Messages are capped at 2000 chars (existing).
- **UI rules.**
  - No external scripts or fonts.
  - Works at 375 px width.
  - Light/dark via `prefers-color-scheme`.
  - Every trace event shown comes from `/v1/chat` `events`, never from client-side guesses.
- **Channels.**
  - **Telegram** is active only if `TELEGRAM_BOT_TOKEN` and `TELEGRAM_WEBHOOK_SECRET` are set. It verifies the `X-Telegram-Bot-Api-Secret-Token` header.
  - **Slack** is active only if `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET` are set. It verifies the `v0` HMAC signature and rejects timestamps more than 300 s off. It acknowledges retries (`X-Slack-Retry-Num`) without reprocessing and deduplicates `event_id`.
  - In demo mode, both channels append the simulated SMS after an `auth_otp` reply, because a channel user cannot see the web inbox.
- **Secrets.** Only from the environment and `.env`, never committed. Space secrets hold `BANK_SESSION_SECRET`, `OPENROUTER_API_KEY` and `HF_DATA_TOKEN`.
- **Commits.** End with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Public abuse of the LLM key.** Repeated conversations eventually hit the daily cap. After that, turns still get answers (rules fallback) and never error (pinned in Task 1).
2. **A Slack request with a stale timestamp or a wrong signature.** Answered 401 with no agent call, even if the body is valid (pinned in Task 4).
3. **A Telegram update without text** (sticker, photo, edited message). Ignored with 200 and no agent call (pinned in Task 4).
4. **The container starts before the dataset download finishes, or with a bad token.** The entrypoint fails loudly (non-zero exit) instead of serving an empty sandbox (pinned in Task 5, via the script's `set -e` and a size check).
5. **Two browser tabs on the same conversation id.** Already serialised by the per-conversation lock (Plan 3). The UI never reuses a conversation after a handoff without the user choosing "new conversation" (pinned in Task 3's manual check).

---

### Task 1: LLM daily budget guard

**Files:** Create `src/agent/budget.py`, `tests/agent/test_budget.py`. Modify `src/agent/factory.py`.

**Interfaces:**
- `BudgetedLLM(llm, daily_usd: float, clock=time.time)` exposes `.model`, `.spent_today`, and `.complete(messages, **kw) -> LLMResponse`.
  - Once today's spend is ≥ `daily_usd`, `.complete` raises `LLMError("daily LLM budget exhausted")` before calling the wrapped LLM.
  - "Today" is `int(clock() // 86400)` (UTC day).
- `build_agent(..., llm_budget_usd: float | None = None)` wraps the LLM when a budget is given. `create_app` reads `AGENT_LLM_DAILY_BUDGET_USD` (default `0.5`) for hybrid mode.
- `LLMExtractor` already falls back to rules on any exception and records `llm_error` with `error="LLMError"`.

- [ ] **Step 1: Failing tests** — `tests/agent/test_budget.py`

```python
import pytest

from src.agent.budget import BudgetedLLM
from src.agent.llm import LLMError, LLMResponse
from src.agent.nlu import LLMExtractor
from src.agent.state import Stage


class Paid:
    model = "m"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, **kw):
        self.calls += 1
        return LLMResponse('{"merchant": "Uber"}', "m", 10, 5, 0.3, 5.0)


def test_budget_blocks_after_cap_and_resets_next_day():
    now = {"t": 86400 * 100 + 10}
    inner = Paid()
    llm = BudgetedLLM(inner, daily_usd=0.5, clock=lambda: now["t"])
    llm.complete([])
    llm.complete([])  # spent 0.6 >= 0.5 after this call
    with pytest.raises(LLMError, match="budget"):
        llm.complete([])
    assert inner.calls == 2 and llm.spent_today == pytest.approx(0.6)
    now["t"] += 86400
    llm.complete([])
    assert inner.calls == 3 and llm.spent_today == pytest.approx(0.3)


def test_extractor_keeps_working_when_budget_is_exhausted():  # Review Focus 1
    llm = BudgetedLLM(Paid(), daily_usd=0.0)
    e = LLMExtractor(llm).extract("No reconozco un cargo de Uber", Stage.INTAKE)
    assert (e.source, e.merchant) == ("llm_fallback", "Uber")
```

- [ ] **Step 2: Run** `uv run pytest tests/agent/test_budget.py -q`. Expected: FAIL with `ModuleNotFoundError: src.agent.budget`.

- [ ] **Step 3: Implement** `src/agent/budget.py`

```python
"""Daily spend cap for the LLM behind a public demo. Over the cap, calls fail fast with LLMError and the extractor
falls back to rules, so the agent keeps answering (deterministically) instead of spending or erroring."""
import threading
import time

from src.agent.llm import LLMError, LLMResponse


class BudgetedLLM:
    def __init__(self, llm, daily_usd: float, clock=time.time):
        self._llm, self.daily_usd, self._clock = llm, daily_usd, clock
        self.model = getattr(llm, "model", "unknown")
        self._day, self._spent, self._lock = None, 0.0, threading.Lock()

    @property
    def spent_today(self) -> float:
        with self._lock:
            return self._spent if self._day == int(self._clock() // 86400) else 0.0

    def complete(self, messages, **kw) -> LLMResponse:
        day = int(self._clock() // 86400)
        with self._lock:
            if self._day != day:
                self._day, self._spent = day, 0.0
            if self._spent >= self.daily_usd:
                raise LLMError("daily LLM budget exhausted")
        resp = self._llm.complete(messages, **kw)
        with self._lock:
            self._spent += resp.cost_usd
        return resp
```

In `src/agent/factory.py`, add the `llm_budget_usd: float | None = None` keyword to `build_agent`. After building `llm`, add `if llm_budget_usd is not None: llm = BudgetedLLM(llm, llm_budget_usd)`. In `src/agent/api.py` `create_app`, replace `build_agent(conn, os.environ.get("AGENT_MODE", "hybrid"))` with:

```python
    mode = os.environ.get("AGENT_MODE", "hybrid")
    agent = agent or build_agent(conn, mode, llm_budget_usd=float(os.environ.get("AGENT_LLM_DAILY_BUDGET_USD", "0.5"))
                                 if mode == "hybrid" else None)
```

- [ ] **Step 4: Run** `uv run pytest tests/agent -q`. Expected: all pass.
- [ ] **Step 5: Commit** — `feat(agent): daily LLM spend cap with rules fallback for the public demo`

---

### Task 2: Demo scenarios, config endpoint and static UI routes

**Files:** Create `src/agent/demo.py`, `src/agent/static/index.html` (placeholder in this task), `tests/agent/test_demo_ui.py`. Modify `src/agent/api.py`.

**Interfaces:**
- `demo_scenarios(conn) -> list[dict]`. Each entry has keys `scenario` (`normal|ambiguous|large_amount|stolen_card|no_phone`), `document`, `message_es` and `message_pt`. A scenario with no matching data is omitted. Selection is deterministic (lowest `customer_id`).
- `GET /v1/config` returns `{"mode", "router", "llm_model", "demo"}`.
- `GET /v1/demo/customers` returns 404 unless demo mode is on; otherwise the list is computed once and cached.
- `GET /` returns `static/index.html`, and `/static/*` serves the other assets.

- [ ] **Step 1: Failing tests** — `tests/agent/test_demo_ui.py`

```python
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
```

- [ ] **Step 2: Run** — Expected: FAIL (`src.agent.demo` missing).

- [ ] **Step 3: Implement** `src/agent/demo.py`

```python
"""One real sandbox customer per demo scenario, so judges can try every path without knowing the data."""
from src.bank import db

ACTIVE = ("u.customer_status = 'Active' and u.mobile_phone is not null and u.customer_id not in "
          "(select customer_id from complaint_flags where is_repeat_complainer = 1)")
CARD = "p.product_type like 'Tarjeta%' and p.product_status = 'Active'"
USD = "coalesce(t.amount_usd, case when t.currency = 'USD' then t.amount end)"


def _amount(a: float, lang: str) -> str:
    s = f"{a:.2f}".removesuffix(".00")
    return s.replace(".", ",") if lang == "pt" else s


def demo_scenarios(conn) -> list[dict]:
    q = lambda sql: conn.execute(sql).fetchone()  # noqa: E731
    out = []
    with db.LOCK:
        r = q(f"""select u.document_number d, t.merchant_name m, t.amount a from transactions t
                  join products p on p.product_id = t.product_id join customers u on u.customer_id = t.customer_id
                  where {ACTIVE} and {CARD} and t.transaction_status = 'Approved' and t.merchant_name is not null
                    and t.local_date >= '2026-04-01' and {USD} < 300 and (select count(*) from transactions t2 where t2.customer_id = t.customer_id
                                         and t2.merchant_name = t.merchant_name) = 1
                  order by u.customer_id, t.transaction_id limit 1""")
        if r:
            out.append({"scenario": "normal", "document": r["d"],
                        "message_es": f"No reconozco un cargo de {_amount(r['a'], 'es')} en {r['m']}",
                        "message_pt": f"Não reconheço uma cobrança de {_amount(r['a'], 'pt')} na {r['m']}"})
        r = q(f"""select u.document_number d, t.merchant_name m from transactions t
                  join products p on p.product_id = t.product_id join customers u on u.customer_id = t.customer_id
                  where {ACTIVE} and {CARD} and t.transaction_status = 'Approved' and t.merchant_name is not null
                    and t.local_date >= '2026-04-01' and {USD} < 300
                  group by u.customer_id, t.merchant_name having count(*) between 2 and 3
                  order by u.customer_id limit 1""")
        if r:
            out.append({"scenario": "ambiguous", "document": r["d"],
                        "message_es": f"No reconozco un cargo en {r['m']}",
                        "message_pt": f"Não reconheço uma cobrança na {r['m']}"})
        r = q(f"""select u.document_number d, t.amount a from transactions t
                  join products p on p.product_id = t.product_id join customers u on u.customer_id = t.customer_id
                  where {ACTIVE} and {CARD} and t.transaction_status = 'Approved' and t.merchant_name is null
                    and t.local_date >= '2026-04-01' and {USD} > 600 and t.transaction_type = 'Withdrawal'
                  order by u.customer_id limit 1""")
        if r:
            out.append({"scenario": "large_amount", "document": r["d"],
                        "message_es": f"No reconozco un retiro de {_amount(r['a'], 'es')} en mi tarjeta",
                        "message_pt": f"Não reconheço um saque de {_amount(r['a'], 'pt')} no meu cartão"})
        r = q(f"""select u.document_number d from customers u join products p on p.customer_id = u.customer_id
                  where {ACTIVE} and {CARD} group by u.customer_id having count(*) = 1
                  order by u.customer_id limit 1""")
        if r:
            out.append({"scenario": "stolen_card", "document": r["d"],
                        "message_es": "Me robaron la tarjeta y hay 3 compras que no hice",
                        "message_pt": "Roubaram meu cartão e tem 3 compras que não fiz"})
        r = q("""select document_number d from customers where customer_status = 'Active' and mobile_phone is null
                 order by customer_id limit 1""")
        if r:
            out.append({"scenario": "no_phone", "document": r["d"],
                        "message_es": "No reconozco un cargo de mi tarjeta",
                        "message_pt": "Não reconheço uma cobrança do meu cartão"})
    return out
```

In `src/agent/api.py`:
- Add `from pathlib import Path`, `from fastapi.responses import FileResponse`, `from fastapi.staticfiles import StaticFiles` and `from src.agent.demo import demo_scenarios`.
- Set `STATIC_DIR = Path(__file__).parent / "static"`.
- Inside `create_app`, after the `app = FastAPI(...)` line, add:

```python
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    cache: dict = {}

    @app.get("/")
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
```

Create a placeholder `src/agent/static/index.html` with `<!doctype html><title>Dispute agent</title><div id="chat"></div>`. Task 3 replaces it.

- [ ] **Step 4: Run** `uv run pytest tests/agent -q`. Expected: all pass.
- [ ] **Step 5: Commit** — `feat(agent): demo scenarios, config endpoint, static UI routes`

---

### Task 3: Web UI — chat, live trace, demo scenarios, simulated SMS, handoff view

**Files:** Replace `src/agent/static/index.html`. Create `src/agent/static/app.js` and `src/agent/static/styles.css`. Create `.claude/launch.json` (local preview).

**Behaviour:**
- **Header:** product name, language tip ("Escribe en español o português"), and badges for mode, router and LLM from `/v1/config`.
- **Chat column:**
  - Message bubbles; the input is disabled while a request is in flight.
  - A "Nueva conversación" button.
  - When `handoff` is present, a card shows the reason and a collapsible "Lo que recibe el asesor" block with the pretty-printed payload.
  - After a handoff the input shows "Caso transferido: inicia una nueva conversación" but still allows typing (the agent replies with the reference).
- **Side column, top: demo scenarios.** Only in demo mode. One card per scenario, with a label, the document, and three buttons: "Mensaje ES", "Mensaje PT" and "Enviar documento".
- **Side column, SMS simulado.** After any reply whose `stage` is `auth_otp`, fetch `/v1/demo/sms/{id}` and show the code with an "Enviar código" button. It is marked "Simulado — solo demo".
- **Side column, trace.** One block per turn, newest first, showing:
  - turn number, stage, latency and cost;
  - one line per event, formatted from its kind: `router` (intent, conf, lang, ⚠ injection, abstain), `extraction` (source + fields), `tool` (✓/✗ name, args, ms, error), `policy` (rule, outcome, version), `llm` (tokens, $), `handoff`, `action_verified`, `session_expired` and `reply`.
- **Footer:** metrics from `/v1/metrics` (turns, conversations, escalation rate, p50/p95, cost), refreshed after each turn.
- **Errors:** a network or 4xx/5xx error shows an inline error bubble and never loses typed text.

- [ ] **Step 1: Write the three static files**

`src/agent/static/index.html`:

```html
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LATAM Bank · Disputas</title>
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
  <header class="top">
    <div>
      <h1>LATAM Bank · Asistente de disputas</h1>
      <p class="sub">Escribe en español o português · datos sintéticos · demo</p>
    </div>
    <div id="badges" class="badges"></div>
  </header>
  <main class="layout">
    <section class="chat-col">
      <div id="chat" class="chat" aria-live="polite"></div>
      <form id="composer" class="composer">
        <input id="input" autocomplete="off" maxlength="2000" placeholder="Ej: No reconozco un cargo de 350 en Oxxo">
        <button id="send" type="submit">Enviar</button>
      </form>
      <button id="new" class="link" type="button">Nueva conversación</button>
    </section>
    <aside class="side-col">
      <section id="demo" class="panel" hidden>
        <h2>Clientes de demo</h2>
        <div id="scenarios"></div>
      </section>
      <section id="sms" class="panel" hidden>
        <h2>SMS simulado <span class="tag">solo demo</span></h2>
        <p id="sms-body" class="mono"></p>
        <button id="sms-send" type="button">Enviar código</button>
      </section>
      <section class="panel">
        <h2>Traza en vivo</h2>
        <p class="hint">Lo que el sistema decidió en cada turno: router, extracción, herramientas, política, LLM.</p>
        <div id="trace" class="trace"></div>
      </section>
    </aside>
  </main>
  <footer id="metrics" class="metrics"></footer>
  <script src="/static/app.js"></script>
</body>
</html>
```

`src/agent/static/styles.css`:

```css
:root { --bg:#f7f7f5; --panel:#fff; --text:#1d1d1f; --muted:#6b6b70; --line:#e3e3e0; --accent:#0b6e4f;
        --me:#e7f4ee; --agent:#fff; --warn:#b54708; --bad:#b42318; --ok:#067647; --mono:ui-monospace,SFMono-Regular,Menlo,monospace; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#121214; --panel:#1b1b1f; --text:#ececef; --muted:#9a9aa2; --line:#2c2c31; --accent:#3ccf91;
          --me:#173327; --agent:#1f1f24; --warn:#fdb022; --bad:#f97066; --ok:#47cd89; }
}
* { box-sizing: border-box; }
body { margin:0; font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }
.top { display:flex; justify-content:space-between; align-items:flex-end; gap:12px; padding:16px; border-bottom:1px solid var(--line); flex-wrap:wrap; }
h1 { font-size:18px; margin:0; } h2 { font-size:14px; margin:0 0 8px; } .sub, .hint { color:var(--muted); margin:2px 0 0; font-size:13px; }
.badges { display:flex; gap:6px; flex-wrap:wrap; } .badge, .tag { font-size:12px; padding:2px 8px; border:1px solid var(--line); border-radius:999px; color:var(--muted); }
.layout { display:grid; grid-template-columns: minmax(0,1.3fr) minmax(0,1fr); gap:16px; padding:16px; }
@media (max-width: 860px) { .layout { grid-template-columns: 1fr; } }
.chat-col, .side-col { display:flex; flex-direction:column; gap:12px; min-width:0; }
.chat { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; min-height:52vh; max-height:66vh; overflow-y:auto; display:flex; flex-direction:column; gap:8px; }
.msg { max-width:85%; padding:8px 12px; border-radius:12px; white-space:pre-wrap; overflow-wrap:anywhere; border:1px solid var(--line); }
.msg.me { align-self:flex-end; background:var(--me); } .msg.agent { align-self:flex-start; background:var(--agent); }
.msg.error { align-self:center; color:var(--bad); border-color:var(--bad); font-size:13px; }
.handoff { align-self:stretch; border:1px solid var(--warn); border-radius:12px; padding:8px 12px; font-size:13px; }
.handoff pre { max-height:260px; overflow:auto; font:12px/1.4 var(--mono); }
.composer { display:flex; gap:8px; } .composer input { flex:1; min-width:0; padding:10px 12px; border-radius:10px; border:1px solid var(--line); background:var(--panel); color:var(--text); font:inherit; }
button { font:inherit; padding:8px 12px; border-radius:10px; border:1px solid var(--accent); background:var(--accent); color:#fff; cursor:pointer; }
button:disabled { opacity:.5; cursor:default; } button.link { background:none; color:var(--accent); border:none; align-self:flex-start; padding:0; }
button.ghost { background:none; color:var(--accent); }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; min-width:0; }
.scenario { border-top:1px solid var(--line); padding:8px 0; display:flex; flex-direction:column; gap:6px; } .scenario:first-child { border-top:none; }
.scenario .row { display:flex; gap:6px; flex-wrap:wrap; } .scenario button { font-size:12px; padding:4px 8px; }
.mono { font-family:var(--mono); font-size:13px; overflow-wrap:anywhere; }
.trace { display:flex; flex-direction:column; gap:10px; max-height:52vh; overflow-y:auto; }
.turn { border:1px solid var(--line); border-radius:10px; padding:8px; font:12px/1.5 var(--mono); }
.turn .head { font-weight:600; margin-bottom:4px; } .ev { overflow-wrap:anywhere; } .ok { color:var(--ok); } .bad { color:var(--bad); } .warn { color:var(--warn); }
.metrics { padding:8px 16px 20px; color:var(--muted); font-size:12px; }
```

`src/agent/static/app.js`:

```javascript
"use strict";
const $ = (id) => document.getElementById(id);
const state = { cid: null, turn: 0, busy: false, handedOff: false };

function bubble(text, who) {
  const el = document.createElement("div");
  el.className = `msg ${who}`;
  el.textContent = text;
  $("chat").appendChild(el);
  $("chat").scrollTop = $("chat").scrollHeight;
  return el;
}

function handoffCard(payload) {
  const card = document.createElement("div");
  card.className = "handoff";
  const title = document.createElement("strong");
  title.textContent = `Transferido a un asesor · motivo: ${payload.escalation_reason} · ${payload.handoff_id}`;
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Lo que recibe el asesor (payload estructurado, sin transcripción)";
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(payload, null, 2);
  details.append(summary, pre);
  card.append(title, details);
  $("chat").appendChild(card);
  $("chat").scrollTop = $("chat").scrollHeight;
}

function fmtEvent(e) {
  const k = e.kind;
  if (k === "router") return [`router → ${e.intent} (${e.confidence}) lang=${e.language}${e.abstain ? " · abstain" : ""}${e.injection ? " · ⚠ injection" : ""}`, e.injection ? "warn" : ""];
  if (k === "extraction") return [`extraction[${e.source}] ${(e.fields || []).join(", ") || "—"}`, e.source === "llm_fallback" ? "warn" : ""];
  if (k === "tool") return [`${e.ok ? "✓" : "✗"} ${e.name} ${JSON.stringify(e.args)} ${e.latency_ms}ms${e.attempts > 1 ? ` ×${e.attempts}` : ""}${e.error ? " · " + e.error : ""}`, e.ok ? "ok" : "bad"];
  if (k === "tool_denied") return [`⛔ denied ${e.name} at ${e.stage}`, "bad"];
  if (k === "policy") return [`policy ${e.rule_id} → ${e.outcome} (${e.version})`, e.outcome === "eligible" ? "ok" : "warn"];
  if (k === "llm") return [`llm ${e.model} ${e.prompt_tokens}+${e.completion_tokens} tok · $${e.cost_usd} · ${e.latency_ms}ms`, ""];
  if (k === "llm_error") return [`llm error → rules fallback (${e.error})`, "warn"];
  if (k === "handoff") return [`handoff → ${e.reason} (${e.handoff_id})`, "warn"];
  if (k === "action_verified") return [`read-back verified: ${e.action} ${e.id}`, "ok"];
  if (k === "reply") return [`reply template: ${e.key}`, ""];
  return [`${k} ${JSON.stringify(Object.fromEntries(Object.entries(e).filter(([x]) => !["kind", "t_ms"].includes(x))))}`, ""];
}

function renderTurn(r) {
  const block = document.createElement("div");
  block.className = "turn";
  const head = document.createElement("div");
  head.className = "head";
  head.textContent = `turn ${state.turn} · stage ${r.stage} · ${Math.round(r.latency_ms)} ms · $${r.cost_usd}`;
  block.appendChild(head);
  for (const e of r.events) {
    const [text, cls] = fmtEvent(e);
    const line = document.createElement("div");
    line.className = `ev ${cls}`;
    line.textContent = text;
    block.appendChild(line);
  }
  $("trace").prepend(block);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

async function refreshMetrics() {
  try {
    const m = await api("/v1/metrics");
    const pct = m.escalation_rate == null ? "—" : `${Math.round(m.escalation_rate * 100)}%`;
    $("metrics").textContent = `Servicio: ${m.turns} turnos · ${m.conversations} conversaciones · escalamiento ${pct} · latencia p50/p95 ${m.latency_ms_p50 ?? "—"}/${m.latency_ms_p95 ?? "—"} ms · costo LLM $${m.cost_usd_total}`;
  } catch (_) { /* metrics are optional */ }
}

async function showSms() {
  try {
    const s = await api(`/v1/demo/sms/${state.cid}`);
    const code = s.sms.split(" ").pop();
    $("sms-body").textContent = s.sms;
    $("sms").hidden = false;
    $("sms-send").onclick = () => send(code);
  } catch (_) { $("sms").hidden = true; }
}

async function send(text) {
  text = (text ?? $("input").value).trim();
  if (!text || state.busy) return;
  state.busy = true;
  $("send").disabled = true;
  bubble(text, "me");
  if (text === $("input").value.trim()) $("input").value = "";
  try {
    const r = await api("/v1/chat", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ conversation_id: state.cid, message: text }) });
    state.cid = r.conversation_id;
    state.turn += 1;
    bubble(r.reply, "agent");
    renderTurn(r);
    if (r.handoff) { handoffCard(r.handoff); state.handedOff = true; $("input").placeholder = "Caso transferido: inicia una nueva conversación"; }
    if (r.stage === "auth_otp") await showSms(); else $("sms").hidden = true;
    refreshMetrics();
  } catch (err) {
    bubble(`No se pudo enviar (${err.message}). Tu texto sigue en el campo.`, "error");
    $("input").value = text;
  } finally {
    state.busy = false;
    $("send").disabled = false;
    $("input").focus();
  }
}

function resetConversation() {
  Object.assign(state, { cid: null, turn: 0, handedOff: false });
  $("chat").replaceChildren();
  $("trace").replaceChildren();
  $("sms").hidden = true;
  $("input").placeholder = "Ej: No reconozco un cargo de 350 en Oxxo";
  $("input").focus();
}

const LABELS = { normal: "Cargo no reconocido (1 coincidencia)", ambiguous: "Cargo ambiguo (varias coincidencias)",
  large_amount: "Monto alto → revisión humana (P4)", stolen_card: "Tarjeta robada con cargos → bloqueo + fraude",
  no_phone: "Sin celular registrado → no se puede verificar" };

async function loadDemo() {
  const cfg = await api("/v1/config");
  for (const [k, v] of [["modo", cfg.mode], ["router", cfg.router], ["LLM", cfg.llm_model || "ninguno"]]) {
    const b = document.createElement("span");
    b.className = "badge";
    b.textContent = `${k}: ${v}`;
    $("badges").appendChild(b);
  }
  if (!cfg.demo) return;
  const list = await api("/v1/demo/customers");
  $("demo").hidden = false;
  for (const s of list) {
    const card = document.createElement("div");
    card.className = "scenario";
    const title = document.createElement("div");
    title.textContent = LABELS[s.scenario] || s.scenario;
    const doc = document.createElement("div");
    doc.className = "mono";
    doc.textContent = `documento: ${s.document}`;
    const row = document.createElement("div");
    row.className = "row";
    for (const [label, text] of [["Mensaje ES", s.message_es], ["Mensaje PT", s.message_pt], ["Enviar documento", s.document]]) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "ghost";
      b.textContent = label;
      b.onclick = () => send(text);
      row.appendChild(b);
    }
    card.append(title, doc, row);
    $("scenarios").appendChild(card);
  }
}

$("composer").addEventListener("submit", (ev) => { ev.preventDefault(); send(); });
$("new").addEventListener("click", resetConversation);
loadDemo().catch(() => {});
refreshMetrics();
```

`.claude/launch.json`:

```json
{
  "version": "0.0.1",
  "configurations": [
    { "name": "agent-baseline-demo", "runtimeExecutable": "make", "runtimeArgs": ["serve-demo-baseline"], "port": 8000 }
  ]
}
```

Add the Makefile target (and add it to `.PHONY`):

```makefile
serve-demo-baseline:
	DEMO_MODE=1 AGENT_MODE=baseline uv run --env-file .env uvicorn --factory src.agent.api:create_app --port 8000
```

- [ ] **Step 2: Run the unit tests** — `uv run pytest tests/agent/test_demo_ui.py -q`. Expected: 3 passed (`id="chat"` is present).
- [ ] **Step 3: Manual browser check (baseline, no LLM cost)** in the in-app browser, started with `preview_start` `agent-baseline-demo`:
  - Run the "normal" scenario end to end: message → document → SMS button → sí → no.
  - Run "stolen card": the handoff card and payload appear.
  - Check the trace blocks.
  - Check mobile width (resize to 375 px).
  - Check the dark scheme.
  - Fix any layout or JS issue found. Take a screenshot for the ledger.
- [ ] **Step 4: Commit** — `feat(ui): web chat with live trace, demo scenarios, simulated SMS and handoff view`

---

### Task 4: Slack and Telegram adapters

**Files:** Create `src/channels/__init__.py`, `src/channels/bridge.py`, `src/channels/telegram.py`, `src/channels/slack.py`, `tests/channels/__init__.py`, `tests/channels/test_channels.py`. Modify `src/agent/api.py` (mount the routers when configured).

**Interfaces:**
- `ChannelBridge(agent, store, conn, demo: bool)`:
  - `.handle(channel: str, chat_id: str, text: str) -> list[str]` returns the replies to send (reply, plus the simulated SMS in demo mode).
  - `/start`, `/reset` and `/nueva` start a new conversation.
- `telegram_router(bridge, *, bot_token, secret, send=None) -> APIRouter` exposes `POST /v1/channels/telegram`. `send(chat_id, text)` defaults to a `sendMessage` urllib call.
- `slack_router(bridge, *, bot_token, signing_secret, send=None, clock=time.time) -> APIRouter` exposes `POST /v1/channels/slack/events`. `send(channel, text)` defaults to `chat.postMessage`.
- `verify_slack_signature(signing_secret, timestamp, body: bytes, signature, now) -> bool`.

- [ ] **Step 1: Failing tests** — `tests/channels/test_channels.py`

```python
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
```

- [ ] **Step 2: Run** — Expected: FAIL (`src.channels` missing).

- [ ] **Step 3: Implement**

`src/channels/bridge.py`:

```python
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
```

`src/channels/telegram.py`:

```python
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
```

`src/channels/slack.py`:

```python
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
```

In `src/agent/api.py` `create_app`, before `return app`:

```python
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
```

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** — `feat(channels): Telegram webhook and Slack events adapters over a shared bridge`

---

### Task 5: Hugging Face Space packaging and deploy script

**Files:** Create `deploy/hf_space/Dockerfile`, `deploy/hf_space/entrypoint.sh`, `deploy/hf_space/README.md`, `scripts/deploy_hf.py`, `tests/test_deploy_files.py`. Modify `Makefile` (`deploy`), `.env.example`.

**Interfaces:**
- `scripts/deploy_hf.py --user <hf_user> [--space dispute-agent] [--dataset latam-bank-sandbox]` does the following:
  1. Creates the **private** dataset repo and uploads `data/sandbox.db` (skipped if the remote sha256 already matches).
  2. Creates the **public** Docker Space.
  3. Sets the Space secrets `BANK_SESSION_SECRET`, `OPENROUTER_API_KEY` and `HF_DATA_TOKEN` (= `HF_TOKEN`), and the variables `AGENT_MODE=hybrid`, `DEMO_MODE=1`, `AGENT_LLM_DAILY_BUDGET_USD=0.5` and `HF_DATASET=<user>/<dataset>`.
  4. Uploads the Space files: `deploy/hf_space/*` at the root, plus `src/`, `policy/`, `models/`, `pyproject.toml` and `uv.lock`.
  5. Prints the Space URL.
- The container:
  - Runs as uid 1000 and listens on 7860.
  - At start it downloads `sandbox.db` into `$HOME/data`, checks the size is > 50 MB, then execs uvicorn.
  - Pre-downloads the pinned e5 model at build time.

- [ ] **Step 1: Failing test** — `tests/test_deploy_files.py`

```python
from pathlib import Path

D = Path("deploy/hf_space")


def test_space_files_are_consistent():
    readme = (D / "README.md").read_text()
    assert "sdk: docker" in readme and "app_port: 7860" in readme
    docker = (D / "Dockerfile").read_text()
    assert "download.pytorch.org/whl/cpu" in docker and "useradd -m -u 1000" in docker and "EXPOSE 7860" in docker
    assert "COPY data" not in docker and "COPY --chown=user data" not in docker  # data never baked into the image
    entry = (D / "entrypoint.sh").read_text()
    assert entry.startswith("#!/bin/sh") and "set -e" in entry and "HF_DATA_TOKEN" in entry and "52428800" in entry
```

- [ ] **Step 2: Run** — Expected: FAIL (the files do not exist).

- [ ] **Step 3: Write the files**

`deploy/hf_space/README.md`:

```markdown
---
title: LATAM Bank Dispute Agent
emoji: 🏦
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

AI-first transaction-dispute agent (Factored AI & Data Hackathon 2026). Synthetic data, demo mode: the SMS/OTP is
simulated and shown in the page. Source: see the GitHub repository linked in the submission.
```

`deploy/hf_space/Dockerfile`:

```dockerfile
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860 HOME=/home/user HF_HOME=/home/user/.hf \
    SANDBOX_PATH=/home/user/data/sandbox.db AGENT_MODE=hybrid DEMO_MODE=1
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
RUN pip install uv
COPY --chown=user pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --group embeddings --no-hashes --no-emit-project -o req.txt \
 && TORCH=$(sed -nE 's/^torch==([^ ;]+).*/\1/p' req.txt | head -1) \
 && pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH}" \
 && grep -vE '^(torch==|nvidia-|triton)' req.txt > req-nogpu.txt \
 && pip install -r req-nogpu.txt
COPY --chown=user src ./src
COPY --chown=user policy ./policy
COPY --chown=user models ./models
COPY --chown=user entrypoint.sh ./entrypoint.sh
USER user
RUN python -c "from sentence_transformers import SentenceTransformer; from src.router.classifier import E5_REVISION; SentenceTransformer('intfloat/multilingual-e5-small', revision=E5_REVISION)"
EXPOSE 7860
CMD ["sh", "./entrypoint.sh"]
```

`deploy/hf_space/entrypoint.sh`:

```sh
#!/bin/sh
# Download the sandbox database from the PRIVATE dataset repo, refuse to start without it, then serve.
set -e
mkdir -p "$(dirname "$SANDBOX_PATH")"
if [ ! -s "$SANDBOX_PATH" ]; then
  python -c "
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ['HF_DATASET'], 'sandbox.db', repo_type='dataset', token=os.environ['HF_DATA_TOKEN'])
shutil.copy(p, os.environ['SANDBOX_PATH'])"
fi
SIZE=$(wc -c < "$SANDBOX_PATH")
if [ "$SIZE" -lt 52428800 ]; then echo "sandbox.db too small ($SIZE bytes): refusing to start" >&2; exit 1; fi
exec uvicorn --factory src.agent.api:create_app --host 0.0.0.0 --port "${PORT:-7860}"
```

`scripts/deploy_hf.py`:

```python
"""Deploy to Hugging Face: private dataset (sandbox.db) + public Docker Space (app).
Usage: uv run --group embeddings --env-file .env python scripts/deploy_hf.py --user <hf_user>"""
import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--space", default="dispute-agent")
    ap.add_argument("--dataset", default="latam-bank-sandbox")
    a = ap.parse_args()
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    dataset, space = f"{a.user}/{a.dataset}", f"{a.user}/{a.space}"

    api.create_repo(dataset, repo_type="dataset", private=True, exist_ok=True)
    db_path = ROOT / "data" / "sandbox.db"
    local = sha256(db_path)
    remote = next((f.lfs.sha256 for f in api.list_repo_tree(dataset, repo_type="dataset")
                   if getattr(f, "path", "") == "sandbox.db" and getattr(f, "lfs", None)), None)
    if remote != local:
        api.upload_file(path_or_fileobj=str(db_path), path_in_repo="sandbox.db", repo_id=dataset, repo_type="dataset",
                        commit_message="sandbox.db (synthetic, organizer dataset derivative; private)")
    print("dataset:", dataset, "(private)", "uploaded" if remote != local else "unchanged")

    api.create_repo(space, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    for key, value in {"BANK_SESSION_SECRET": os.environ["BANK_SESSION_SECRET"],
                       "OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"], "HF_DATA_TOKEN": token}.items():
        api.add_space_secret(space, key, value)
    for key, value in {"AGENT_MODE": "hybrid", "DEMO_MODE": "1", "AGENT_LLM_DAILY_BUDGET_USD": "0.5",
                       "HF_DATASET": dataset}.items():
        api.add_space_variable(space, key, value)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for f in (ROOT / "deploy" / "hf_space").iterdir():
            shutil.copy(f, stage / f.name)
        for d in ("src", "policy", "models"):
            shutil.copytree(ROOT / d, stage / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for f in ("pyproject.toml", "uv.lock"):
            shutil.copy(ROOT / f, stage / f)
        api.upload_folder(folder_path=str(stage), repo_id=space, repo_type="space", commit_message="deploy app")
    print("space: https://huggingface.co/spaces/" + space)
    print("app:   https://" + space.replace("/", "-").replace("_", "-").lower() + ".hf.space")


if __name__ == "__main__":
    main()
```

Append to `.env.example`: `HF_TOKEN=` (write token, used only by the deploy script) and `AGENT_LLM_DAILY_BUDGET_USD=0.5`. Add the Makefile target `deploy` (usage: `make deploy HF_USER=<user>`):

```makefile
deploy:
	uv run --group embeddings --env-file .env python scripts/deploy_hf.py --user $(HF_USER)
```

- [ ] **Step 4: Run** `uv run pytest tests/test_deploy_files.py -q`. Expected: 1 passed.
- [ ] **Step 5: Commit** — `feat(deploy): Hugging Face Space packaging with private dataset download and deploy script`

---

### Task 6: Deploy and verify the public URL

- [ ] **Step 1:** Confirm that `.env` has `HF_TOKEN` (write) and get the HF username from Diego. Run `make deploy HF_USER=<user>`.
- [ ] **Step 2:** Wait for the Space build (the page shows "Building" → "Running"; about 5–10 min for torch and e5). Poll `https://<user>-dispute-agent.hf.space/health` until it returns 200 with `mode: hybrid`, `router: e5_v1`.
- [ ] **Step 3: Remote smoke test** in the in-app browser:
  - normal scenario ES → dispute created;
  - PT unsupported;
  - stolen card → block + handoff;
  - trace panel populated with the `llm` cost lines.
  - Also confirm `/v1/demo/customers` returns scenarios, and that the dataset repo is private (open its URL logged-out: 404).
- [ ] **Step 4:** Append a "Live demo" section to `README.md` with the URL, the demo-mode caveats (simulated SMS, synthetic data, $0.5/day LLM cap, possible cold start after sleep), how to reset a conversation, and how to connect Slack/Telegram later (env vars + webhook URLs). Commit `docs: live demo`.
