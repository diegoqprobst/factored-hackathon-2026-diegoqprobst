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
