# Video pitch — script and demo run-sheet (target 3:00, hard max 3:30)

The brief asks for a short video that **demonstrates the working solution** and **explains core architectural
decisions**. Record the screen and your voice (QuickTime: File → New Screen Recording, microphone on; or Loom).

## Before recording (5 minutes)

1. **Wake the app 2 minutes early.** Open https://latam-dispute-agent.onrender.com. The free instance sleeps; the
   first load shows Render's "waking up" screen for about a minute, then the demo list takes up to another minute
   ("Cargando clientes de demo…"). Reload until the five scenarios appear.
2. Browser at 1920×1080, zoom 110–125%, one tab only, notifications off (Do Not Disturb).
3. Open the deck in Present mode in a second window so you can switch with ⌘-Tab.
4. Do one dry run of the demo below. Demo customers rotate after use, so document numbers change between takes:
   always use the scenario buttons, never type document numbers from memory.

## Script

| Time | Screen | Say |
|---|---|---|
| 0:00–0:15 | Slide 1 (cover) | "Hi, I'm Diego. This is a dispute agent for LATAM Bank that knows when *not* to act. It handles card disputes end to end in Spanish and Portuguese, and hands off to a person when policy says so." |
| 0:15–0:35 | Slide 2 (why) | "We chose disputes from the data. Complaint contacts are fixed on first contact only 44% of the time, against 92% for transactional ones, and unrecognized charges and undue fees are about 40% of formal complaints. Disputes also have a trustworthy ground truth: every transaction ties cleanly to its product and customer, so the agent can verify the exact charge." |
| 0:35–0:45 | App, empty chat | "Here it is live. On the right, demo customers taken from the real sandbox, and below, a live trace of every decision." |
| 0:45–1:05 | Click **Cargo ambiguo → Mensaje PT**, then **Enviar documento**, then **Enviar código** | "A customer writes in Portuguese about a charge at Boutique Moda. The router detects Portuguese and a dispute, and before looking at any data the agent verifies identity: document, then a one-time code, simulated here as an SMS." |
| 1:05–1:20 | Agent lists two matching charges; type `o de 165` | "There are two charges at that merchant, so instead of guessing, it asks. I answer in my own words and the model maps that to the right option." |
| 1:20–1:35 | Type `o que acontece se eu confirmar?` | "Before any write it asks for an explicit yes. A question is not a yes: it asks again and writes nothing." |
| 1:35–1:50 | Type `sim`, then `sim, bloqueia`; scroll the trace | "Now it files the dispute and offers to block the card. In the trace you can see each tool call, the read-back that verifies the write, the policy rule, and the cost of the turn: fractions of a cent." |
| 1:50–2:05 | **Nueva conversación** → **Monto alto → Mensaje ES** → document → code | "And when it should not act: a large withdrawal. Policy says a person must review it, so it hands off with a reference, and the advisor receives everything already verified." |
| 2:05–2:30 | Slide 3 (architecture) | "The core decision: the model understands, code decides. A learned router and an LLM read the message; the LLM never sees the document or the code. A state machine, a versioned policy and customer-scoped bank tools decide and act. When the router and the model disagree, the agent asks the customer instead of guessing." |
| 2:30–2:45 | Slide 4 (router), then slide 5 (results) | "The router was chosen by a rule fixed before the sealed test set was opened, and it beats keywords by more than 20 points. End to end, on 230 held-out conversations on fresh records, the hybrid safely resolves 98% of disputes against 86.5% for rules, makes every required handoff, and has one unsafe outcome, which we report." |
| 2:45–3:00 | Slide 6 (production) | "It runs today with tracing, retries, a spending cap and a safe fallback, and we list what is missing to make it real. Build something that works, prove it works, and know when not to act. Thank you." |

## If something goes wrong while recording

- **Demo list empty:** the instance just woke up; wait for "Cargando…" to finish or reload.
- **A reply takes 5–9 s:** normal for the LLM on the free tier; keep talking over it or cut the pause in editing.
- **"sim, bloqueia" asks again:** the deploy predates the fix; answer `sim`.
- **A scenario says the charge already has a dispute:** click *Nueva conversación*; the list refreshes with an unused customer.

## Numbers used (all from the repo)

- 44% vs 92% first-contact resolution, ≈40% of complaints: `reports/eda_findings.md` §2.
- Router test macro-F1 0.666 / 0.886 / 0.892: `reports/router_eval.md`.
- 98.1% vs 86.5% SAR (p = 0.004), 74/74 handoffs, 1/230 unsafe, $0.0003 per resolution:
  `reports/eval_confirm_tfidf_final/eval_report.md`; run-to-run range in `reports/eval_error_analysis.md`.
