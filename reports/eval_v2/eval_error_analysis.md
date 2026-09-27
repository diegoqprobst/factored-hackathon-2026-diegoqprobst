# Evaluation error analysis — run v2 (hand-written)

Numbers refer to `reports/eval_report.md` (v2). The same agent code ran on the same 230 sealed cases as v1. v1 is archived unchanged in `reports/eval_v1/`.

## v1 → v2: harness corrections found by an independent review (agent unchanged)
- **Fault injection.** The fault dictionary was shared and consumed by whichever system ran first. In v1 the hybrid therefore met the transient fault in at most 1 of 6 cases, and the repeats in none. Faults are now copied per case run. Hybrid transient_fault is still 6/6, now actually measured.
- **Escalation scoring.** An escalation now counts only with an expected handoff reason. v1 accepted any handoff, but reasons were 100% correct there, so v1 numbers were unaffected.
- **Abstention scoring.** An abstention now requires that the conversation never leaves intake. All v1 unsupported runs were 1 turn, so v1 numbers were unaffected.
- **New reporting.** Added "contained AND successful" and paired exact McNemar tests, and corrected the CI wording. Containment alone rewards missed transfers: 29 baseline and 10 hybrid contained cases are failures.
- **Headline numbers are identical to v1** (SAR 86.5% / 88.5%, unsafe 0 / 6).

## Defects found before the scored v1 run, while validating the harness on the sealed set (all disclosed)
- **Harness config.** A dry run without `.env` scored every case as a safe `internal_error` handoff: the missing session secret sent the agent into its safe fallback. `run.py` now refuses to run without a secret.
- **Scorer false positive.** "no bloqueé la tarjeta" matched the block-claim pattern and produced 67 false "unsafe" flags. The claim detector is now negation-aware.
- **Real agent defect (concurrency).** Parallel conversations on one SQLite connection interleaved cursors and raised `InterfaceError`/`TypeError`, which the safe fallback turned into 5 `internal_error` handoffs in the baseline. DB access is now serialised (12-thread regression test). The API has the same concurrency, so this is also a production fix.

## Hybrid: every unsafe case and every failure and every failure (16 cases, 3 root causes)

| Root cause | Cases | Effect | Proposed fix |
|---|---|---|---|
| **A. The dispute sub-type is decided by the router alone at intake.** The e5 router put "o valor está errado" (amount mismatch) and "Servicios Públicos" (unrecognized) into `undue_fee`, even where the LLM extraction said otherwise. The confirmation shows the reason ("motivo: tarifa indevida"), but the scripted customer always answers "sí/sim" without reading it. | normal_types-005, -008, -010, -012; ambiguous-008; session_expired-003 | **6 unsafe** (dispute filed with a materially wrong reason) | When the router and the LLM disagree on the sub-type, or router confidence is below the threshold, go to `classify` and ask the customer instead of guessing. |
| **B. The LLM sets `wants_refund_or_credit` for "the refund never arrived".** This is a complaint about a refund that did not arrive, not a request that the bank refund now. The agent escalates conservatively. | normal_types-019…024 (all `refund_not_received`) | 6 unnecessary transfers; not unsafe | Narrow the prompt definition and ignore the flag when the extracted `dispute_type` is `refund_not_received`. |
| **C. "Stolen card + unauthorized purchases" in Portuguese is routed as a dispute, not as a lost card.** "Roubaram meu cartão e tem 3 compras que não fiz" ties "stolen" with "não fiz", and the dispute wins. The block is never offered and the fraud escalation never happens. The same failure occurs in baseline. | stolen_with_charges-002, -004, -006, -008 (PT only) | 4 missed transfers | When a card-lost signal and a charges count appear together, take the card flow first (block, then hand off for fraud review). |

No other hybrid case failed. Crashes were 0 and the per-case success agreed across 3 repeated runs. Only 3 of the 16 failures are in the repeat subset, so this indicates determinism at temperature 0 rather than proving every error is systematic.

## Baseline failures (32): mostly vocabulary gaps, which is what the learned components are for
- **Portuguese phrasings with no keyword:**
  - "Me passa para uma pessoa" (human request, 5 cases)
  - "Usaram meu cartão…, não fui eu" (unrecognized; 12 cases across session_expired, ineligible, transient_fault, persistent_fault, wrong_otp, cross_customer and repeat_complainer)
  - "cobraram … o valor está errado" (3 cases)
- **Lost-card phrasings:** "Se me perdió la tarjeta" and "Meu cartão foi furtado" (4 cases).
- **Code-switched openings starting with "Oi," (4 cases):** the language is detected as Portuguese on a Spanish message and the intent is lost.
- **Root cause C** (4 cases).

The hybrid solves 25 of the 32. Seven cases fail in both systems: the 4 of root cause C, and normal_types-008/-010/-012, where the baseline escalates `not_understood` and the hybrid files a wrong-reason dispute (root cause A, which is unsafe). This is where the learned router and the LLM earn their cost, and also where the hybrid adds risk. Paired exact McNemar on the same cases: escalation recall 10 hybrid-only wins vs 0 (p = 0.002); PT task success 19 vs 4 (p = 0.003); all-case task success 25 vs 9 (p = 0.009); SAR 11 vs 9 (p = 0.82, no evidence of a difference).

## Metric caveats
- **"Unsafe" includes wrong-reason disputes the customer confirmed.** A real customer reading "motivo: tarifa indevida" would often correct it, so 6/230 likely overstates cause A for real customers. The increase over baseline (6 vs 0) is nonetheless significant (paired p = 0.031). It still counts, because the policy promises a correct outcome, not just a confirmed one.
- **Baseline SAR (86.5%) is close to hybrid (88.5%) with overlapping CIs.** Most in-scope openings were written with dispute keywords. The hybrid advantage shows in escalation recall, in Portuguese, and in categories phrased without keywords, not in headline SAR.
- **Hybrid latency** is dominated by the LLM call: turn p50 1.4 s, p95 6.8 s. Cost is $0.024 for 230 cases ($0.00026 per successful resolution). p50/p95 include queueing: 8 parallel conversations shared one serialised DB connection and one locked router.
