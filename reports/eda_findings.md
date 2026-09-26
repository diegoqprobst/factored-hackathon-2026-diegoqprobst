# EDA findings — LATAM Bank dataset (v1.0.0)

Every number here is reproducible: `uv run python -m src.eda.run_eda` regenerates
[`eda_output.md`](eda_output.md) with the exact SQL behind each figure.
Source: S3 `data/` prefix, mirrored 2026-09-25 (lineage in `data/raw/_manifest.csv`).
`digital_events` (3.7 GB) not yet profiled.

## 1. Data quality — what is actually usable

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| Q1 | Fact tables hold **~86%** of the documented rows (interactions 686k/800k, transactions 4.43M/5M, complaints 67k/80k, surveys 213k/250k). Dimensions match exactly. PKs are unique; no content-level duplicates found on natural keys. | Row-count check | Report the gap; the advertised ~2% duplicates are not present in this delivery — our dedup step must still exist (contract), but will be a no-op on v1.0.0. |
| Q2 | `contact_reason` is **identical** to `reason_category` (6 values). No granular contact reason exists in structured data. | Demand-by-reason check | Workflow sizing must combine interactions with complaint subcategories. |
| Q3 | Transcripts are **2 templates** (2 opening lines, 546 distinct texts over 171k rows), all with **unfilled placeholders** (`{monto} {moneda}`), 1 intent value (`consulta_general`), 1 language (`es`). The opening line is **independent of `main_topics`** (each template spreads ~35/22/17/15/8/3% over the 6 topics). | Transcript diversity + opening-line vs topic | Transcript text carries **no signal** for intent. Cannot train/evaluate an intent classifier on it. A labelled utterance set must be team-generated (and declared as such). |
| Q4 | Complaint `description` is a single template per category ("Queja relacionada con fees"). | Sample | No free text to learn from in complaints either. |
| Q5 | `complaints.origin_interaction_id` is **100% null** → complaints cannot be joined to the contact that created them. | Linkage check | Contact→complaint funnel must be estimated at customer/time level, not by key. |
| Q6 | **0 of 44,570** complaints with an `affected_product_id` reference a product owned by the complaining customer. | Linkage check | Referential integrity is broken across customers. Our tool layer must enforce `product.customer_id == session.customer_id` — this dataset proves the check is necessary, not theoretical. |
| Q7 | Transactions → products ownership is **100% consistent** (0 orphans, 0 owner mismatches). | Ownership check | Transactions are the trustworthy ground truth to anchor responses. |
| Q8 | `email` is shared across customers (91k distinct for 150k customers). | Identity check | Email (and document number alone) cannot authenticate. Need a test session/OTP service, as the brief requires. |
| Q9 | 25% of transactions have `transaction_date::date != process_date`, with timestamps ranging 06:01 → 05:59 next day: a **constant 6h offset** (UTC timestamps vs local-date partitions), not late arrival. | Date check | Normalise timezone in the contract; don't mistake it for late data. |
| Q10 | Requests and Suggestions have `compensation_granted` (449 + 237) while never having a `claimed_amount`. | case_type check | Flag as policy inconsistency; don't learn compensation rules from these rows. |
| Q11 | Nulls: `duration_seconds` 14%, accent 30%, wait time 30% (non-phone channels have no wait). | Null check | Above the advertised ~5%; nullable fields handled explicitly. |
| Q12 | Distributions are near-uniform/flat where real data would not be: escalation ≈10% in every reason, complaint categories ~equal, fraud ≈0.1% in every transaction type, `fraud_score` vs `is_fraud` corr = 0.12. | Several checks | Differences between segments are mostly generator noise — don't over-claim insights; report effect sizes with CIs. |

## 2. Demand — where the workflow should be

Contact volume (686k interactions, 2023-06 → 2026-06):

| Reason | Share | FCR | Follow-up | Median duration |
|---|---|---|---|---|
| Transaccional | 35.0% | 0.915 | 22% | 205 s |
| Producto | 22.0% | 0.896 | 24% | 263 s |
| **Queja** | 17.1% | **0.436** | **63%** | **431 s** |
| Técnico | 15.0% | 0.699 | 41% | 360 s |
| Comercial | 8.0% | 0.652 | 45% | 540 s |
| Retención | 3.0% | 0.602 | 49% | 478 s |

Complaints (67k): **"Cargo no reconocido" (Transactions) + "Cobro indebido" (Fees) ≈ 40%**,
~20% SLA breach, median 15–16 days to resolution, average claimed amount ≈ 2,500 (local currency).

## 3. Recommendation: **transaction-dispute intake** ("cargo no reconocido / cobro indebido")

- **Pain is where FCR is worst.** Queja contacts resolve on first contact 44% of the time vs 92% for
  transactional ones, take 2× longer, and 63% need follow-up. Disputes are ~40% of formal complaints.
- **It has the one reliable ground truth in the dataset.** Transactions are consistent with products
  and customers (Q7), with status (Approved/Declined/Pending/Reversed), merchant, channel, amount,
  and a fraud label — so the agent can *verify* the charge the customer is talking about.
- **It naturally exercises all three required paths:**
  normal → identify the charge, confirm it, file a dispute case (verified by reading it back);
  ambiguous → "a charge I don't recognise" matching several transactions → clarify;
  human → high amount / fraud signals / repeat complainer / card must be blocked → structured handoff.
- **It stresses exactly what's graded:** customer-record isolation (Q6 shows why), authentication (Q8),
  action permissions (filing a case ≠ refunding money — the agent never moves funds).
- **Learned component with real labels:** a dispute-triage model (auto-resolvable vs needs human, or
  transaction-matching ranker) trained on transaction features, evaluated against a rules baseline on a
  **time-based** held-out split. Intent/language routing uses a team-labelled ES/PT utterance set (Q3).

Alternatives considered: *account/payment inquiries* (highest volume, but high FCR already → little to
gain, weak escalation story); *credit eligibility* (strong policy angle, but needs a synthetic policy
service and has no outcome labels in the data); *card support* (overlaps disputes, less evidence).

## 4. Declared limitations (so far)
- All data is synthetic Spanish; **no Portuguese** anywhere → PT behaviour is evaluated only on a
  team-generated, labelled test set.
- No usable conversational text (Q3/Q4) → intent evaluation data is team-generated.
- Generator artefacts (Q12) limit how much "insight" the analytics can honestly claim.

## 5. Pipeline v1 run (2026-09-26)
`make pipeline` on the full mirror: 0 contract rejections, 0 duplicates, 0 FK violations in every table
(customers 150,000 · products 400,000 · transactions 4,425,008 · complaints 67,095 · FX 13,164).
The sandbox holds 497,202 transactions (120 days to 2026-06-17), 3,291 USD FX rates, 54,145 complaint flags.
`tz_offset_violations` = **51**: every one is timestamped exactly `06:00:00` on the day after its partition,
so partitions are `(06:00, 06:00]` UTC — an inclusive-end boundary convention, not late or wrong data.
The UTC-6 reading of Q9 holds; the metric is kept as a monitor (a jump above ~51 would signal real drift).
