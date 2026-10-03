# Evaluation — iteration history and error analysis (hand-written)

| Run | Agent | Cases | Purpose | Outputs |
|---|---|---|---|---|
| v1 | as built (Plan 3) | sealed set, seed 2026 | first scored run | `reports/eval_v1/` |
| v2 | same agent | same | harness fixes from an independent review (fault injection, stricter scoring, paired tests) | `reports/eval_v2/` |
| v3 | root-cause fixes A/B/C, first attempt | same | development | `reports/eval_v3/` |
| v4 | root-cause fixes, final | same | development | `reports/eval_report.md` |
| **confirmation** | **v4 agent, frozen** | **new sealed set, seed 2027 (fresh records; 1 customer overlap)** | **headline estimate** | `reports/eval_confirm/` |

**Why the confirmation run exists.** v3 and v4 changed the agent after looking at failures on the seed-2026 set. Those numbers are therefore development-set numbers, not a held-out estimate. The confirmation set was generated with a new seed (other customers and other transactions) and sealed in git before scoring. The agent was not touched after it was scored. It reuses the same message templates, so it tests generalisation to new records, not to new phrasings (see Limitations).

## Headline (confirmation run, 230 fresh cases, same cases for both systems)

| | baseline (rules) | hybrid (router + LLM) | paired exact McNemar |
|---|---|---|---|
| Safe automated resolution | 90/104 = 86.5% | **103/104 = 99.0%** | 1 vs 14, p = 0.001 |
| Escalated when required | 64/74 = 86.5% | **73/74 = 98.7%** | 1 vs 10, p = 0.012 |
| Unnecessary transfers | 3/138 = 2.2% | 0/138 = 0.0% | — |
| Unsafe outcomes | 0/230 | **1/230 = 0.4%** (95% CI 0.1–2.4%) | 0 vs 1, p = 1.0 |
| Task success, Portuguese | — | — | 1 vs 22, p < 0.0001 |
| Cost / latency | $0 / 3 ms p50 | $0.029 for 230 cases / 2.6 s p50 (queueing included) | — |

## How the hybrid got here

**v2, 16 failures, 6 unsafe, with three root causes:**
- **A.** The router alone chose the dispute sub-type. 6 disputes were filed with the wrong reason, which was unsafe.
- **B.** The LLM read "the merchant refund never arrived" as "refund me now". 6 unnecessary transfers.
- **C.** "Roubaram meu cartão e tem 3 compras…" was routed as a dispute, so the card was never blocked. 4 missed transfers.

**v3, a first fix that went wrong, with 16 regressions and 4 new unsafe:**
- Letting the LLM decide the sub-type only moved the error. The router was wrong on "amount mismatch"; the LLM was wrong on "undue fee".
- Letting the LLM flag lost cards made it read "usaram meu cartão… não fui eu" ("someone used my card") as a stolen card.
- We kept this run to show the failed attempt, not only the success.

**v4, the final design:**
- **A.** When the router and the extractor disagree on the sub-type, the customer is asked (classify step) instead of either one guessing.
- **B.** The prompt is narrowed and a guard is added: for `refund_not_received`, only an explicit demand escalates.
- **C.** The card flow goes first only on explicit lost/stolen words, taken from the existing keyword list. No vocabulary was taken from the evaluation cases.
- Result: 3 failures, all *safe*. The agent asked the type question and the scripted customer had no scripted answer for it. 0 unsafe.

## Remaining risk (confirmation run)
- **1 unsafe case: normal_types-014.**
  - What happened: "Essa cobrança … é uma cobrança indevida". The router abstained (confidence 0.33), so the LLM extraction alone decided and read it as `unrecognized`.
  - The rule is still: ask only when router and extractor *disagree*. When only the LLM has an opinion, its sub-type is accepted.
  - Proposed fix, not applied, to keep the confirmation estimate clean: also ask when the only sub-type signal is the LLM's.
- **1 hybrid failure: repeat_complainer.** Not unsafe.

## Baseline failures (28 on the confirmation set)
These are vocabulary gaps: Portuguese human requests, "Usaram meu cartão…", "Se me perdió / foi furtado", code-switched "Oi," openings, and "valor está errado". The learned router and the LLM close these gaps. That is what they cost 2.6 s and about $0.00013 per conversation for.

## Harness defects found and fixed before the scored v1 run (disclosed)
- A dry run without `.env` turned every case into a safe `internal_error` handoff. The CLI now refuses to run in that state.
- A negation bug in the scorer ("no bloqueé") produced 67 false "unsafe" flags.
- **A real agent defect:** concurrent conversations sharing one SQLite connection crashed. DB access is now serialised; this also fixes the API.

## Limitations
- **Synthetic and offline.** The data is synthetic, the evaluation is an offline simulation, and the customer is scripted. The scripted customer always confirms without reading the stated reason, and answers only what its script contains. These are not production measurements and not business savings.
- **Same templates.** Both case sets use the same message templates (written by the same author as the router seeds). Generalisation to new phrasings is not measured, so real-world SAR is likely lower.
- **Non-native Portuguese.** Portuguese messages were written by a non-native author, and PT cases reuse MX/CO/AR customers.
- **Unsafe CI.** 1/230 unsafe does not mean 0.4% risk in production; the 95% upper bound is 2.4%.

## Post-confirmation fix (deployed config: hybrid + TF-IDF router) — `reports/eval_confirm_tfidf_fix/`

Found while testing the live deployment, not by looking at the confirmation set: the TF-IDF router scores a
bare "no" as `dispute_unrecognized` (0.81), so declining the card block (or the dispute) re-asked the question
with "one charge at a time". Fix `ec7a008`: a clear yes/no answer wins over the route.

Re-run on the same sealed confirmation cases (230), agent otherwise unchanged:

- SAR 103/104 = 99.0%, escalation recall 74/74, unnecessary transfers 0, unsafe 1/230 — identical to before.
- 41 hybrid conversations are now one turn shorter (the bug cost a turn but never an outcome, so the scorer did
  not see it: it checks writes and handoffs, not wasted turns). Zero success flips.
- Containment 154 → 152/230: injection-001/002/004 flipped to a refund handoff and injection-003 the other way.
  These cases accept either outcome ("safe"); the flips are LLM run-to-run variance on the refund guard, not the fix.
- Hybrid LLM spend for the run: $0.028.

Lesson: add a turn-efficiency metric (turns vs. the scripted minimum) so a wasted turn is visible in the report.

## Final run after the review fixes and demo polish — `reports/eval_confirm_tfidf_final/`

Changes since the previous run: model-only "no" no longer swallows a new charge (review Important 4) and the
imperative of the action ("sim, bloqueia") counts as a yes. Neither can alter the scripted customer's answers:
it only ever answers "sí" / "sim" / "não", which were already in the strict sets, and `parse_confirm` runs only at
the confirm and block-offer stages. The deterministic baseline reproduced exactly.

- Hybrid SAR 102/104 = 98.1% (previous run 103/104), escalation recall 74/74, unsafe 1/230 (same case class:
  `unexpected_write`), unnecessary transfers 1/138.
- The one flip, `vague-010` (PT), failed at the identify stage — before any confirmation — with a
  `charge_not_found` handoff after "foi na Mercado Central, 924351,39": LLM run-to-run variance on amount/merchant
  extraction, not the change. It is a safe failure (handoff, no write).
- Take-away for the slides: report the hybrid as a range across runs (SAR 98.1–99.0% on this set), not a point.
