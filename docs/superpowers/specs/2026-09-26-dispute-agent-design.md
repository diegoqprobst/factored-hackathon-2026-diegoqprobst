# Design — AI-first transaction-dispute intake agent

- **Date:** 2026-09-26 · **Status:** approved in brainstorming, pending written-spec review
- **Context:** Factored AI & Data Hackathon 2026 (solo entry, 10 days, submission ~2026-10-05)
- **Goal:** win the hackathon — every component must earn points against the published rubric.
- **Evidence base:** `reports/eda_findings.md` (findings referenced as Q1–Q12).

## 1. Problem and scope

**Workflow:** intake of card/account transaction disputes ("cargo no reconocido", "cobro indebido").
Chosen because complaint contacts have the worst first-contact resolution (FCR 0.44 vs 0.92 for
transactional contacts, 63% need follow-up), charge disputes are ~40% of formal complaints, and
transactions are the only fully consistent source in the dataset (Q7).

**In scope**
- Authenticate the customer (simulated OTP), find the disputed transaction(s), classify the dispute,
  check eligibility against a deterministic policy, create a dispute case, block a card on suspected
  fraud, and hand off to a human with structured context.
- Spanish (MX/CO/AR variants) and Portuguese.
- Three required paths: normal automated resolution; ambiguous/unsupported (clarify or abstain);
  human-required (structured handoff).

**Out of scope (explicit)**
- Refunds, provisional credit, compensation, any movement of money → always human.
- Other workflows (balances, credit, onboarding) → polite abstention with redirection.
- Credit/bidding ideas (neocrédito) → "future work" slide only; the dataset's delinquency is
  independent of credit score (≈10% DPD>30 in every score band), so it cannot support a risk model.

**Action permissions**

| Action | Who | Guard |
|---|---|---|
| `start_auth`, `verify_otp` | Agent | — |
| `search_transactions`, `get_transaction` | Agent (read) | Valid session; rows filtered by session `customer_id` |
| Classify dispute type | Agent | — |
| `evaluate_policy` | Orchestrator only | Deterministic, versioned rules |
| `create_dispute` | Agent | Explicit customer confirmation + policy `eligible` + idempotency key + read-back verification |
| `block_card` | Agent | Separate explicit confirmation + card owned by session customer + read-back verification |
| Refund / provisional credit / compensation | Never the agent | Always `create_handoff` |

## 2. Architecture

```
Channels: Web chat + live trace panel (primary) · Slack · Telegram   (thin adapters: message + session_id)
    │
Agent core (FastAPI)
  1. Learned router        → intent · language · escalate · injection (+ confidence)
  2. Orchestrator (FSM)    → owns state, per-stage tool allowlist, confirmation gates, post-action verification
  3. Stage LLM (OpenRouter)→ converses and extracts slots; sees only the tools its stage enables
  4. Handoff builder       → structured JSON for the human agent
  5. Tracer                → per-turn audit record
    │
Mock bank services (each enforces its own permissions)
  Identity/OTP · Transactions · Disputes · Cards · Policy (versioned rules)
    │
Data pipeline: S3 CSV (bronze) → contracts + quality (silver, Parquet) → SQLite sandbox
Evaluation harness: held-out scripted conversations ES/PT → baseline vs hybrid → metrics report
```

**Design principles**
1. **Security never depends on the prompt.** Services validate session and ownership on every call (Q6
   shows the dataset itself cross-links products between customers).
2. **The LLM never decides policy.** It only communicates the policy service's decision.
3. **Nothing is claimed without verification.** After a write, the orchestrator reads state back; only a
   confirmed read lets the agent tell the customer it happened.
4. **The orchestrator owns state, not the LLM.** Additional disputes mentioned mid-conversation are queued
   by the orchestrator.

**Baseline vs proposed.** Baseline = same orchestrator, services, and policy, but keyword-rule router and
template responses/regex slot extraction (no LLM). Proposed (hybrid) = learned router + stage LLM.
Everything else is shared so the comparison isolates the learned/LLM components.

## 3. Conversation flow (orchestrator stages)

| # | Stage | Behaviour | Tools enabled for the LLM |
|---|---|---|---|
| 0 | Intake | Router classifies intent/language/injection. Non-dispute intents → abstain + redirect. | none |
| 1 | Auth | Document number → simulated OTP to registered phone → session (TTL 15 min). | `start_auth`, `verify_otp` |
| 2 | Identify | LLM extracts amount/date/merchant. 0 matches → ask for more; >1 → present candidates; 1 → confirm. | `search_transactions` |
| 3 | Classify | Type ∈ {unrecognized/fraud, duplicate, amount_mismatch, undue_fee, refund_not_received}. | none |
| 4 | Policy | Orchestrator calls `evaluate_policy` → `eligible` / `ineligible(reason)` / `requires_human(reason)`. | none |
| 5 | Confirm | Summary + explicit yes. Fraud → separate offer to block card. | none |
| 6 | Execute + verify | `create_dispute` (idempotent) → `get_dispute`; `block_card` → `get_card`. | `create_dispute`, `block_card`, `get_*` |
| 7 | Close / handoff | Verified case number + next steps, or structured handoff. | `create_handoff` |

**Escalation triggers (any stage):** customer asks for a human; policy `requires_human`; request for
refund/provisional credit/compensation; tool failure after retries; router confidence below threshold
twice in a row; repeated injection attempts; read-back verification fails.

**Handoff payload (JSON):** `request_summary`, `language`, `customer_id` (verified), `auth_level`,
`transactions` (verified facts), `dispute_type`, `policy_decision` + rule ids + version,
`actions_taken` (with verification status), `open_questions`, `escalation_reason`, `trace_id`.
No raw transcript dump.

**Demo scripts**
- Normal: "No reconozco un cargo de 350 pesos en Oxxo de ayer" → 1 match → confirm → case created and verified.
- Ambiguous: "Me cobraron algo raro" → 3 candidates → customer picks one.
- Unsupported: "Quero um empréstimo" → abstain + redirect (Portuguese).
- Human: "Me robaron la tarjeta, hay 5 compras que no hice" → confirmed card block → handoff.

## 4. Dispute policy v1 (synthetic, versioned, documented as such)

Policy lives in `policy/dispute_policy_v1.yaml`; every decision records rule ids and version.
"Today" is simulated as 2026-06-17 (dataset end).

| Rule | Condition | Outcome |
|---|---|---|
| P1 window | `transaction_date` older than 90 days | ineligible("outside dispute window") |
| P2 status | status `Declined` | ineligible("no charge was made") |
| P3 status | status `Reversed` | ineligible("already reversed") |
| P4 amount | `amount_usd` > 500 (if `amount_usd` is null, convert with `daily_exchange_rates` for the transaction date; if no rate exists → requires_human) | requires_human("amount above automatic limit") |
| P5 volume | ≥ 3 disputed transactions in the session | requires_human("multiple disputed charges") |
| P6 repeat | `is_repeat_complainer` in complaints history | requires_human("repeat complainer review") |
| P7 duplicate | open dispute already exists for the transaction | ineligible("already disputed", returns existing case id) |
| P8 default | otherwise | eligible |

Rules are evaluated in order; the first `ineligible` or `requires_human` match wins. Thresholds are team
assumptions, flagged as such in README and slides.

## 5. Data pipeline

- **Bronze:** S3 mirror in `data/raw` + `_manifest.csv` (key, size, ETag, last_modified). Done.
- **Silver (Parquet):** one contract per table (types, enums, not-null, PK uniqueness, FK integrity).
  Timestamps normalised to UTC with explicit local date (Q9). Deduplication on natural keys. Rejected rows
  go to a quarantine table with a reason. Each run writes a quality report.
- **Sandbox (SQLite):** customers, card/account products, transactions of the last 120 days before
  2026-06-17, complaints history (repeat-complainer flag), plus writable `disputes`, `card_blocks`,
  `handoffs`, `otp_sessions`, `audit_log`.
- **Update policy:** incremental load by `process_date` partition, reprocess window of 3 days for late
  arrivals, upsert by PK. Update correctness demonstrated with a clearly labelled test fixture.
- **Lineage:** source ETag → `run_id` → output tables, recorded per run.
- **Excluded:** `digital_events`, `campaign_sends`, `marketing_campaigns`, `satisfaction_surveys` (not
  needed by the workflow; documented).

## 6. Learned component — router

- **Labels:** team-generated utterance set (transcripts carry no intent signal, Q3). Intents: 5 dispute
  types, `card_lost_stolen`, `human_request`, several out-of-scope classes, `greeting_noise`. Language:
  es / pt. Flag: injection.
- **Generation:** hand-written seeds → paraphrases with a free/cheap OpenRouter model → manual review
  of a random sample to estimate label quality. The held-out test set includes hand-written utterances
  (not generated) in both languages.
- **Leakage prevention:** split grouped by seed; paraphrases of one seed never cross train/test.
- **Models compared:** (1) keyword rules (baseline), (2) char TF-IDF + logistic regression,
  (3) small multilingual embeddings (e5-small class) + logistic regression. Selection by macro-F1 and
  calibration. Abstention threshold chosen on validation only.
- **Tracking:** each model version stores metrics and dataset hash.

## 7. Security, failure handling, observability

- **Session:** signed token (customer_id, expiry). Expired → 401 → orchestrator returns to Auth keeping
  dispute context.
- **Ownership:** every query filters by session customer. Foreign IDs return "not found" (no existence
  leak) and write an audit event.
- **Identity:** document number or email alone never authenticates (email is shared across customers, Q8).
- **Prompt injection:** router detection; data fields (merchant names etc.) passed to the LLM as delimited
  data; stage-scoped tools; confirmation on all writes.
- **Data minimisation:** LLM sees merchant, amount, date, last-4 only. Logs are masked.
- **Retries:** max 2 with backoff, idempotent operations only; then safe fallback + handoff.
- **LLM failure:** timeout or schema-invalid output → one retry → deterministic template or handoff.
- **Trace per turn:** input, router decision + confidence, stage, tool calls (result, latency), policy
  rule + version, tokens, cost. Feeds the live trace panel and the audit log. Service metrics endpoint:
  p50/p95 latency, escalation rate, cost per case.
- **Declared limitations:** single process/no queue, simulated OTP, no real Portuguese data, synthetic
  generator artefacts (Q12), missing production work (real IdP, DB encryption, retention policy).

## 8. Evaluation

- **Held-out set:** ~200 scripted conversations, ES + PT, across normal / ambiguous / unsupported /
  human-required / adversarial (injection, expired session, cross-customer access, forced tool failure,
  wrong data, ES/PT code-switching).
- **Reference labels per case:** expected outcome (resolve / clarify / abstain / escalate), expected
  actions, forbidden actions.
- **Simulated customer:** deterministic scripts (reproducible, zero cost). Scoring is trace-based. Any LLM
  judge for wording quality is validated against a hand-scored sample, with its rubric documented.
- **Metrics:** safe automated resolution (over all in-scope cases) + automation-attempt share; containment;
  escalation quality (missed and unnecessary transfers); unsafe outcomes with counts and denominators;
  p50/p95 latency; cost per attempted case and per successful resolution ("not defined" if none). Broken
  down by language and customer segment. 3 repeated runs on a subset for variability. Model and prompt
  versions recorded.
- **Budget:** OpenRouter balance ≈ US$5.66. Free models for development and paraphrasing; response
  caching; cost estimate printed before any full run.

## 9. Tooling and deployment

- Python 3.12 + uv; DuckDB for pipeline; FastAPI; SQLite sandbox; OpenRouter (OpenAI-compatible API).
- UI: CopilotKit (AG-UI events for the live trace panel) if it integrates cleanly with the Python
  orchestrator within half a day; otherwise a plain chat page served by FastAPI.
- Optional: mozilla.ai guardrail as a second injection detector for comparison.
- Not used: Exa (unverified external source, injection risk), Trigger.dev (no background jobs needed).
- Backend container on Render/Railway/Fly free tier; frontend on Vercel if Next.js.
- `make setup && make pipeline && make eval` reproduces everything. Secrets only via `.env` (never
  committed); the credential-bearing data-dictionary PDF is git-ignored.

## 10. Plan and cut order

| Day | Deliverable |
|---|---|
| 1–2 | Silver pipeline, sandbox, bank services, policy — with tests |
| 3 | Router dataset + compared models |
| 4–5 | Orchestrator, LLM stages, baseline |
| 6 | Web UI with trace panel, deployed |
| 7 | Evaluation harness + first full run |
| 8 | Fixes from evaluation; Slack and Telegram adapters |
| 9 | Report, slides, video |
| 10 | Buffer and submission |

**Cut order if late:** Telegram → Slack → CopilotKit (fallback to plain chat). Evaluation is never cut.

## 11. Submission checklist

Public repo `factored-hackathon-2026-[team]`, deployed URL, 4–6 slides, short video pitch, sent to
hackathon.admin@factored.ai.
