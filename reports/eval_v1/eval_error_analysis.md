# Evaluation error analysis — run v1 (pre-registered; hand-written)

Numbers below refer to `reports/eval_report.md` (230 sealed cases, git `f108990`). The run was scored once. No agent change was made after seeing these results. Fixes are proposed below, and any rerun will be reported as a separate, disclosed v2.

## Defects found *before* scoring, while validating the harness (all disclosed)
- **Harness config.** A dry run without `.env` scored every case as a safe `internal_error` handoff: the missing session secret sent the agent into its safe fallback. `run.py` now refuses to run without a secret.
- **Scorer false positive.** "no bloqueé la tarjeta" matched the block-claim pattern and produced 67 false "unsafe" flags. The claim detector is now negation-aware (regression test added).
- **Real agent defect (concurrency).** Parallel conversations sharing one SQLite connection interleaved cursors, raising `InterfaceError`/`TypeError`. The agent's safe fallback turned these into `internal_error` handoffs, 5 of them in the baseline. All DB access is now serialised through the tool gate, the API and the runner; a 12-thread regression test covers it. This is also a production finding: the API has the same concurrency.

## Hybrid: every unsafe case and every failure (16 cases, 3 root causes)

| Root cause | Cases | Effect | Proposed fix |
|---|---|---|---|
| **A. The dispute sub-type is decided by the router alone at intake.** The e5 router put "o valor está errado" (amount mismatch) and "Servicios Públicos" (unrecognized) into `undue_fee`, even where the LLM extraction said otherwise. The confirmation shows the reason ("motivo: tarifa indevida"), but the scripted customer always answers "sí/sim" without reading it. | normal_types-005, -008, -010, -012; ambiguous-008; session_expired-003 | **6 unsafe** (dispute filed with a materially wrong reason) | When the router and the LLM disagree on the sub-type, or router confidence is below the threshold, go to `classify` and ask the customer instead of guessing. |
| **B. The LLM sets `wants_refund_or_credit` for "the refund never arrived".** This is a complaint about a refund that did not arrive, not a request that the bank refund now. The agent escalates conservatively. | normal_types-019…024 (all `refund_not_received`) | 6 unnecessary transfers; not unsafe | Narrow the prompt definition and ignore the flag when the extracted `dispute_type` is `refund_not_received`. |
| **C. "Stolen card + unauthorized purchases" in Portuguese is routed as a dispute, not as a lost card.** "Roubaram meu cartão e tem 3 compras que não fiz" ties "stolen" with "não fiz", and the dispute wins. The block is never offered and the fraud escalation never happens. The same failure occurs in baseline. | stolen_with_charges-002, -004, -006, -008 (PT only) | 4 missed transfers | When a card-lost signal and a charges count appear together, take the card flow first (block, then hand off for fraud review). |

No other hybrid case failed. Crashes were 0 and the per-case outcome agreed across 3 repeated runs, so these errors are systematic, not noise.

## Baseline failures (31): vocabulary gaps, which is what the learned components are for
- **Portuguese phrasings with no keyword:**
  - "Me passa para uma pessoa" (human request, 5 cases)
  - "Usaram meu cartão…, não fui eu" (unrecognized; 12 cases across session_expired, ineligible, transient_fault, persistent_fault, wrong_otp, cross_customer and repeat_complainer)
  - "cobraram … o valor está errado" (3 cases)
- **Lost-card phrasings:** "Se me perdió la tarjeta" and "Meu cartão foi furtado" (4 cases).
- **Code-switched openings starting with "Oi," (4 cases):** the language is detected as Portuguese on a Spanish message and the intent is lost.
- **Root cause C** (4 cases).

The hybrid solves all of these except C. This is where the learned router and the LLM earn their cost. Hybrid escalation recall is 94.6% vs 81.1%, and PT task success is 90.4% vs 77.4%.

## Metric caveats
- **"Unsafe" includes wrong-reason disputes the customer confirmed.** A real customer reading "motivo: tarifa indevida" would often correct it, so 6/230 is an upper bound for cause A with real customers. It still counts, because the policy promises a correct outcome, not just a confirmed one.
- **Baseline SAR (86.5%) is close to hybrid (88.5%) with overlapping CIs.** Most in-scope openings were written with dispute keywords. The hybrid advantage shows in escalation recall, in Portuguese, and in categories phrased without keywords, not in headline SAR.
- **Hybrid latency** is dominated by the LLM call: turn p50 1.4 s, p95 6.8 s. Cost is $0.029 for 230 cases ($0.00032 per successful resolution).
