# Router error analysis (hand-written; complements the generated `router_eval.md`)

Chosen router: e5 (`models/router_v1`), threshold 0.356 tuned on validation. Test set: 159 sealed, hand-written utterances.

**17 intent errors in 159. 14 of them are abstentions**, so the orchestrator asks a clarifying question instead of acting.
Overall, 38 of 159 messages abstain (coverage 0.761) and 118 of the 121 accepted answers are right (selective accuracy 0.975).

The 3 errors that are *not* abstained:

| id | message | gold → predicted | consequence in Plan 3 |
|---|---|---|---|
| t-es-059 | "tienen CDT a 6 meses? qué tasa pagan" | oos_other → oos_balance_movements | Both are out of scope, so the answer is the same polite redirection. |
| t-pt-015 | "no posto era 150 e veio 210 na fatura" | dispute_amount_mismatch → dispute_duplicate | Wrong sub-type. The dispute flow asks the customer to confirm the charge and the type before filing, and policy re-checks. No unsafe action. |
| t-es-080 (`multi_intent`) | "no reconozco un cargo y además quiero hablar con alguien" | dispute_unrecognized → human_request | Defensible. The orchestrator escalates with the dispute recorded as an open question. |

Patterns among the abstained errors:
- **Vague or colloquial disputes** ("algo raro", "algo esquisito", "una compra en Netflix y yo no tengo Netflix") get low confidence. The agent clarifies, which is the intended path.
- **Amount mismatch vs duplicate** get confused when both amounts are in the message.
- **Credit vs balance** ("cuánto me prestan", "quanto vocês me emprestam") get confused because both mention quantities.
- **Injection rows:** the intent is often uncertain (abstained), but `injection=True` was detected on all 18 (injection F1 1.00, n=18, no CI). Tool scoping and confirmation gates (Plan 1) remain the real defence. The router is a signal, not the control.
- **Code-switching:** 7 of 7 languages correct. The one intent miss (t-es-078, "estorno" inside a Spanish sentence) abstained.

**Calibration:** ECE is 0.385. The e5 heads are *under*-confident: most correct answers sit at 0.3–0.6 confidence across 11 classes. The pre-registered rule selected on macro-F1 only, not on calibration as spec §6 asks. That deviation is declared in `router_eval.md` and kept, not fixed after seeing test results. Confidences are used only as scores against a validation-tuned threshold.

**Deployment:** e5 needs the `embeddings` dependency group and a one-time ~470 MB model download at a pinned revision (`614241f…`). `load_router` warms it up at startup and fails loudly if the dependency is missing. After warm-up p50 is 9.4 ms per message. TF-IDF (test macro-F1 0.886, ~1 ms, no download) is the documented fallback if the deployment target cannot carry the model. Using it would be a deployment decision, not a re-selection on test.
