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
make serve-demo       # hybrid + DEMO_MODE=1 (exposes the simulated SMS/OTP; demo only, never in production)
curl -s localhost:8000/v1/chat -H 'content-type: application/json' -d '{"message": "No reconozco un cargo de Uber"}'
```

The agent is a finite-state machine: authenticate → identify the charge → classify → policy → confirm → act → verify.
- **Understanding only.** The router and the LLM understand messages; they never act and never write the replies the customer sees. Replies are templates filled with facts read back from the bank services.
- **Guarded tools.** Each stage can call only the bank tools it needs.
- **Checked writes.** Every write needs an explicit "sí"/"sim" and is read back before the agent claims it happened.
- **Structured handoffs.** Anything outside policy goes to a human with a structured payload (`handoffs` table): the request, verified facts, actions taken and open questions.
- **Traces.** Every turn writes a trace (`agent_traces`: router decision, tool calls, policy rule and version, LLM tokens and cost) available at `/v1/conversations/{id}/trace`. Aggregate latency, cost and escalation are at `/v1/metrics`.
- **Demo mode.** `DEMO_MODE=1` (`make serve-demo`) exposes the simulated SMS (OTP) for the conversation's own challenge at `/v1/demo/sms/{id}`; with it on, anyone who knows a document number can authenticate, so it exists only for the demo.
- **PII to the LLM.** Credentials never reach the LLM; e-mails and document- or phone-like numbers are redacted from every other message before it is sent.

## Evaluation (held-out, offline)

230 sealed scripted conversations in ES/PT, built from real sandbox records (normal, ambiguous, unsupported, human-required, adversarial). Full report: `reports/eval_report.md`; error analysis with root causes and proposed fixes: `reports/eval_error_analysis.md`.

| | baseline (rules) | hybrid (router + LLM) |
|---|---|---|
| Safe automated resolution | 86.5% (90/104, 95% CI 78.7%–91.8%) | 88.5% (92/104, 95% CI 80.9%–93.3%) |
| Containment | 72.6% (167/230, 95% CI 66.5%–78.0%) | 63.9% (147/230, 95% CI 57.5%–69.8%) |
| Escalated when required | 81.1% (60/74, 95% CI 70.7%–88.4%) | 94.6% (70/74, 95% CI 86.9%–97.9%) |
| Unnecessary transfers | 2.2% (3/138, 95% CI 0.7%–6.2%) | 4.3% (6/138, 95% CI 2.0%–9.2%) |
| Unsafe outcomes | 0.0% (0/230, 95% CI 0.0%–1.6%) | 2.6% (6/230, 95% CI 1.2%–5.6%) |
| Turn latency p50 / p95 | 2.4 / 13.0 ms | 1404.6 / 6762.0 ms |
| Cost (230 cases) / per successful resolution | $0 | $0.029145 / $0.000317 |

Hybrid repeated runs (63 cases × 3): 100% per-case outcome agreement. The 6 hybrid unsafe outcomes are disputes filed with a wrong reason that the scripted customer confirmed without reading (root cause A in the error analysis). Offline simulation on synthetic data — not a production measurement.
