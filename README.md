# factored-hackathon-2026 — AI-first transaction-dispute agent

Customer-service system for LATAM Bank (synthetic) that takes in card/account transaction disputes in
Spanish and Portuguese, with a deterministic policy, permissions enforced in the service layer, and
structured human handoff. Design: `docs/superpowers/specs/2026-09-26-dispute-agent-design.md`.

## Setup

```bash
cp .env.example .env   # fill in the S3 keys and a random BANK_SESSION_SECRET (required; placeholders are rejected)
make setup
make download          # S3 -> data/raw (+ lineage manifest)
make pipeline          # data/raw -> data/silver (contracts, quarantine, quality report) -> data/sandbox.db
make test
```

## Data

| Layer | Location | What it guarantees |
|---|---|---|
| Bronze | `data/raw/` + `_manifest.csv` | Exact S3 objects (ETag, size, timestamp) |
| Silver | `data/silver/*.parquet` | Typed, contract-checked, deduplicated; rejects in `_quarantine/` with reasons; run report in `_quality/` |
| Sandbox | `data/sandbox.db` | Last 120 days of transactions before simulated today (2026-06-17); incremental refresh with a 3-day reprocess window |

The dataset is synthetic (organizer-provided). Known limitations are listed in `reports/eda_findings.md`.
The dispute policy (`policy/dispute_policy_v1.yaml`) is a **synthetic, team-defined** policy.

## Running the agent

```bash
make serve            # hybrid: learned router + LLM extraction (needs OPENROUTER_API_KEY and the embeddings group)
make serve-baseline   # rules-only baseline: keyword router + regex extraction, no LLM
curl -s localhost:8000/v1/chat -H 'content-type: application/json' -d '{"message": "No reconozco un cargo de Uber"}'
```

The agent is a finite-state machine: authenticate → identify the charge → classify → policy → confirm → act → verify.
- **Understanding only.** The router and the LLM understand messages; they never act and never write the replies the customer sees. Replies are templates filled with facts read back from the bank services.
- **Guarded tools.** Each stage can call only the bank tools it needs.
- **Checked writes.** Every write needs an explicit "sí"/"sim" and is read back before the agent claims it happened.
- **Structured handoffs.** Anything outside policy goes to a human with a structured payload (`handoffs` table): the request, verified facts, actions taken and open questions.
- **Traces.** Every turn writes a trace (`agent_traces`: router decision, tool calls, policy rule and version, LLM tokens and cost) available at `/v1/conversations/{id}/trace`. Aggregate latency, cost and escalation are at `/v1/metrics`.
- **Demo mode.** `DEMO_MODE=1` exposes the simulated SMS (OTP) at `/v1/demo/sms/{id}`; it exists only for the demo.
