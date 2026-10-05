# factored-hackathon-2026 — AI-first transaction-dispute agent

[![tests](https://github.com/diegoqprobst/factored-hackathon-2026-diegoqprobst/actions/workflows/tests.yml/badge.svg)](https://github.com/diegoqprobst/factored-hackathon-2026-diegoqprobst/actions/workflows/tests.yml) [![keep-warm](https://github.com/diegoqprobst/factored-hackathon-2026-diegoqprobst/actions/workflows/keep-warm.yml/badge.svg)](https://github.com/diegoqprobst/factored-hackathon-2026-diegoqprobst/actions/workflows/keep-warm.yml)

Customer-service system for LATAM Bank (synthetic) that takes in card/account transaction disputes in
Spanish and Portuguese, with a deterministic policy, permissions enforced in the service layer, and
structured human handoff. Design: `docs/superpowers/specs/2026-09-26-dispute-agent-design.md`.

## Live demo

**https://latam-dispute-agent.onrender.com** — hybrid mode (TF-IDF router + `google/gemma-4-31b-it` extraction), demo mode on.

- Guided demo: the side panel lists real sandbox customers, one per path (normal dispute, ambiguous charge,
  large amount → human, stolen card, no registered phone). "▶ Probar (ES/PT)" starts a fresh conversation; a
  progress bar follows the agent's real stage, and "next step" buttons under the input offer the document, the OTP
  from the simulated SMS (demo mode only), a wrong code, the matching charges, and yes/no. Free text always works.
- The live trace shows, per turn, the route, the extracted fields, the tool calls, the policy rule and the LLM cost.
- Each scenario rotates through 12 customers and skips one once it is used (open dispute, blocked card, or an OTP
  requested in the current window), so several people can try it at once.
- Free Render instance: it sleeps after 15 min idle and takes about a minute to wake. The sandbox database is
  downloaded fresh from a private Hugging Face dataset on every start, so demo writes reset. The LLM spend is
  capped per day (`AGENT_LLM_DAILY_BUDGET_USD`); past the cap the agent falls back to rules.
- Deployed-config evaluation (same sealed confirmation cases): [`reports/eval_confirm_tfidf_final/eval_report.md`](reports/eval_confirm_tfidf_final/eval_report.md).

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


## Cost per resolution and ROI

[`reports/roi_analysis.md`](reports/roi_analysis.md) (`uv run python -m src.eda.roi`): dispute-intake volume, handling
time and satisfaction from the dataset, automation rate and LLM cost from the evaluation. Per contact the agent costs
about 5% of a human-handled one (assumed $10/agent-hour); at the evaluated containment it frees ~1,100 agent hours a
year for this (small, synthetic) bank. Agent cost per hour and infrastructure cost are stated assumptions with scenarios.

## Evaluation (held-out, offline)

Scripted conversations in ES/PT built from real sandbox records, covering normal, ambiguous, unsupported, human-required and adversarial cases (injection, session expiry, tool faults, cross-customer). Both systems ran the identical sealed cases. Headline numbers come from a **confirmation set of 230 fresh records** (seed 2027), sealed before scoring, with the agent frozen:

| | baseline (rules) | hybrid (router + LLM) | paired exact McNemar |
|---|---|---|---|
| Safe automated resolution | 86.5% (90/104) | **99.0% (103/104)** | p = 0.001 |
| Escalated when required | 86.5% (64/74) | **98.7% (73/74)** | p = 0.012 |
| Unnecessary transfers | 2.2% (3/138) | 0.0% (0/138) | — |
| Unsafe outcomes | 0/230 | 1/230 (0.4%, 95% CI 0.1–2.4%) | p = 1.0 |
| Turn latency p50 | 3 ms | 2.6 s | — |
| Cost | $0 | $0.029 per 230 cases ($0.00028 per resolution) | — |

**How we got there.** The first evaluation found that the hybrid, while better at escalation and Portuguese, filed 6 disputes with the wrong reason (2.6% unsafe). We traced three root causes, tried a fix that made things worse (kept in the record), and landed on a design where the agent **asks the customer** when its router and its LLM disagree instead of guessing. The confirmation run on fresh records shows 99% safe resolution with 1 residual unsafe case, a documented risk with a proposed fix that was deliberately not applied to keep the estimate clean. The full history (v1→v4), root causes and limitations are in `reports/eval_error_analysis.md`. Reports: `reports/eval_confirm/eval_report.md` (headline) and `reports/eval_v1…v3/` (history).

This is an offline simulation on synthetic data with a scripted customer and shared message templates, not a production measurement.
