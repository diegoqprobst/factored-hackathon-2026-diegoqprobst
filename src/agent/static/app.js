"use strict";
const $ = (id) => document.getElementById(id);
const state = { cid: null, turn: 0, busy: false, handedOff: false, stage: null, lang: "es", scenario: null,
  outcomes: new Set(), reached: new Set() };
const WELCOME = $("welcome").cloneNode(true);

// ---- guided flow ---------------------------------------------------------------------------------------------
const STEP_OF = { intake: "intake", auth_doc: "auth_doc", auth_otp: "auth_otp", identify: "identify", choose: "identify",
  classify: "identify", confirm: "confirm", block_offer: "confirm", done: "done", handoff: "done" };
const STEPS = ["intake", "auth_doc", "auth_otp", "identify", "confirm", "done"];
// a successful tool call proves its step happened even when the stage jumped past it (one match, or a handoff)
const TOOL_STEP = { start_auth: "auth_doc", verify_otp: "auth_otp", search_transactions: "identify",
  get_transaction: "identify", evaluate_dispute: "identify", create_dispute: "confirm", block_card: "confirm" };

const T = {
  es: { doc: (d) => `Enviar documento ${d}`, code: (c) => `Enviar código ${c}`, wrong: "Probar código incorrecto",
    yesDispute: "Sí, abrir disputa", yes: "sí", no: "No", noSend: "no", yesBlock: "Sí, bloquear", yesBlockSend: "sí, bloquéala",
    noThanks: "No, gracias", again: "Repetir la descripción del cargo", other: "▶ Probar otro escenario",
    noSms: "Este documento no existe en el banco: no llega ningún SMS. El agente responde igual a propósito, para no revelar qué documentos son clientes." },
  pt: { doc: (d) => `Enviar documento ${d}`, code: (c) => `Enviar código ${c}`, wrong: "Testar código errado",
    yesDispute: "Sim, abrir contestação", yes: "sim", no: "Não", noSend: "não", yesBlock: "Sim, bloquear", yesBlockSend: "sim, bloqueia",
    noThanks: "Não, obrigado", again: "Repetir a descrição da cobrança", other: "▶ Testar outro cenário",
    noSms: "Este documento não existe no banco: nenhum SMS chega. O agente responde igual de propósito, para não revelar quais documentos são clientes." },
};

function renderSteps() {
  const current = STEP_OF[state.stage] || null;
  if (current) state.reached.add(current);
  const idx = current ? STEPS.indexOf(current) : -1;
  for (const li of $("steps").children) {
    const step = li.dataset.step, i = STEPS.indexOf(step);
    // a step the conversation jumped over (e.g. policy handed off before "Confirmar") is shown as skipped, not done
    li.className = current === "done" && i === idx ? "done"
      : i < idx ? (state.reached.has(step) || step === "intake" ? "done" : "skipped")
      : i === idx ? "current" : "";
  }
  const last = $("steps").lastElementChild;
  last.textContent = current !== "done" ? "Resultado"
    : state.handedOff ? (state.outcomes.has("block_card") ? "Tarjeta bloqueada + asesor" : "Pasado a un asesor")
    : state.outcomes.has("block_card") && state.outcomes.has("create_dispute") ? "Disputa abierta + tarjeta bloqueada"
    : state.outcomes.has("create_dispute") ? "Disputa abierta"
    : state.outcomes.has("block_card") ? "Tarjeta bloqueada" : "Terminado";
}

function chip(label, onclick, kind = "") {
  const b = document.createElement("button");
  b.type = "button";
  b.className = `chip ${kind}`;
  b.textContent = label;
  b.onclick = onclick;
  $("next").appendChild(b);
}

function numberedOptions(reply) {
  return reply.split("\n").map((l) => l.match(/^\s*(\d+)\.\s+(.*)$/)).filter(Boolean)
    .map((m) => ({ n: m[1], text: m[2].length > 48 ? m[2].slice(0, 46) + "…" : m[2] }));
}

async function renderNext(r) {
  $("next").replaceChildren();
  const t = T[state.lang] || T.es;
  const sc = state.scenario;
  const stage = r ? r.stage : state.stage;
  if (stage === "auth_doc" && sc) chip(t.doc(sc.document), () => send(sc.document), "primary");
  if (stage === "auth_otp") {
    const code = await fetchSms();
    if (code) {
      chip(t.code(code), () => send(code), "primary");
      const wrong = String((Number(code) + 1) % 1e6).padStart(6, "0");
      chip(t.wrong, () => send(wrong));
    } else {
      const note = document.createElement("p");
      note.className = "note";
      note.textContent = t.noSms;
      $("next").appendChild(note);
    }
  }
  if ((stage === "choose" || stage === "classify") && r) {
    for (const o of numberedOptions(r.reply)) chip(`${o.n} · ${o.text}`, () => send(o.n));
  }
  if (stage === "identify" && sc) chip(t.again, () => send(state.lang === "pt" ? sc.message_pt : sc.message_es));
  if (stage === "confirm") { chip(t.yesDispute, () => send(t.yes), "primary"); chip(t.no, () => send(t.noSend)); }
  if (stage === "block_offer") { chip(t.yesBlock, () => send(t.yesBlockSend), "primary"); chip(t.noThanks, () => send(t.noSend)); }
  if ((stage === "done" || stage === "handoff") && !$("demo").hidden) {
    chip(t.other, () => { newConversation(); $("demo").scrollIntoView({ behavior: "smooth" }); }, "primary");
  }
}

// ---- chat and trace ------------------------------------------------------------------------------------------
function bubble(text, who) {
  $("welcome")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${who}`;
  el.textContent = text;
  $("chat").appendChild(el);
  $("chat").scrollTop = $("chat").scrollHeight;
  return el;
}

function handoffCard(payload) {
  const card = document.createElement("div");
  card.className = "handoff";
  const title = document.createElement("strong");
  title.textContent = `Transferido a un asesor · motivo: ${payload.escalation_reason} · ${payload.handoff_id}`;
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Lo que recibe el asesor (payload estructurado, sin transcripción)";
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(payload, null, 2);
  details.append(summary, pre);
  card.append(title, details);
  $("chat").appendChild(card);
  $("chat").scrollTop = $("chat").scrollHeight;
}

function fmtEvent(e) {
  const k = e.kind;
  if (k === "router") return [`router → ${e.intent} (${e.confidence}) lang=${e.language}${e.abstain ? " · abstain" : ""}${e.injection ? " · ⚠ injection" : ""}`, e.injection ? "warn" : ""];
  if (k === "extraction") return [`extraction[${e.source}] ${(e.fields || []).join(", ") || "—"}`, e.source === "llm_fallback" ? "warn" : ""];
  if (k === "tool") return [`${e.ok ? "✓" : "✗"} ${e.name} ${JSON.stringify(e.args)} ${e.latency_ms}ms${e.attempts > 1 ? ` ×${e.attempts}` : ""}${e.error ? " · " + e.error : ""}`, e.ok ? "ok" : "bad"];
  if (k === "tool_denied") return [`⛔ denied ${e.name} at ${e.stage}`, "bad"];
  if (k === "policy") return [`policy ${e.rule_id} → ${e.outcome} (${e.version})`, e.outcome === "eligible" ? "ok" : "warn"];
  if (k === "llm") return [`llm ${e.model} ${e.prompt_tokens}+${e.completion_tokens} tok · $${e.cost_usd} · ${e.latency_ms}ms`, ""];
  if (k === "llm_error") return [`llm error → rules fallback (${e.error})`, "warn"];
  if (k === "handoff") return [`handoff → ${e.reason} (${e.handoff_id})`, "warn"];
  if (k === "action_verified") return [`read-back verified: ${e.action} ${e.id}`, "ok"];
  if (k === "reply") return [`reply template: ${e.key}`, ""];
  return [`${k} ${JSON.stringify(Object.fromEntries(Object.entries(e).filter(([x]) => !["kind", "t_ms"].includes(x))))}`, ""];
}

function renderTurn(r) {
  const block = document.createElement("div");
  block.className = "turn";
  const head = document.createElement("div");
  head.className = "head";
  head.textContent = `turn ${state.turn} · stage ${r.stage} · ${Math.round(r.latency_ms)} ms · $${r.cost_usd}`;
  block.appendChild(head);
  for (const e of r.events) {
    const [text, cls] = fmtEvent(e);
    const line = document.createElement("div");
    line.className = `ev ${cls}`;
    line.textContent = text;
    block.appendChild(line);
    if (e.kind === "action_verified") state.outcomes.add(e.action);
    if (e.kind === "tool" && e.ok && e.name === "create_dispute") state.outcomes.add("create_dispute");
    if (e.kind === "tool" && e.ok && TOOL_STEP[e.name]) state.reached.add(TOOL_STEP[e.name]);
  }
  $("trace").prepend(block);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

async function refreshMetrics() {
  try {
    const m = await api("/v1/metrics");
    const pct = m.escalation_rate == null ? "—" : `${Math.round(m.escalation_rate * 100)}%`;
    $("metrics").textContent = `Servicio: ${m.turns} turnos · ${m.conversations} conversaciones · escalamiento ${pct} · latencia p50/p95 ${m.latency_ms_p50 ?? "—"}/${m.latency_ms_p95 ?? "—"} ms · costo LLM $${m.cost_usd_total}`;
  } catch (_) { /* metrics are optional */ }
}

async function fetchSms() {
  try {
    const s = await api(`/v1/demo/sms/${state.cid}`);
    $("sms-body").textContent = s.sms;
    $("sms").hidden = false;
    return s.sms.split(" ").pop();
  } catch (_) {
    $("sms").hidden = true;
    return null;
  }
}

function setBusy(busy) {
  state.busy = busy;
  $("send").disabled = busy;
  for (const b of $("next").querySelectorAll("button")) b.disabled = busy;
  $("chat").classList.toggle("typing", busy);
}

async function send(text) {
  text = (text ?? $("input").value).trim();
  if (!text || state.busy) return;
  setBusy(true);
  bubble(text, "me");
  if (text === $("input").value.trim()) $("input").value = "";
  try {
    const r = await api("/v1/chat", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ conversation_id: state.cid, message: text }) });
    state.cid = r.conversation_id;
    state.turn += 1;
    state.stage = r.stage;
    if (r.language) state.lang = r.language;
    bubble(r.reply, "agent");
    renderTurn(r);
    if (r.handoff) { handoffCard(r.handoff); state.handedOff = true; $("input").placeholder = "Caso transferido: inicia una nueva conversación"; }
    if (r.stage !== "auth_otp") $("sms").hidden = true;
    renderSteps();
    await renderNext(r);
    refreshMetrics();
  } catch (err) {
    bubble(`No se pudo enviar (${err.message}). Tu texto sigue en el campo.`, "error");
    $("input").value = text;
  } finally {
    setBusy(false);
    $("input").focus();
  }
}

function resetConversation() {
  Object.assign(state, { cid: null, turn: 0, handedOff: false, stage: null, scenario: null, outcomes: new Set(),
    reached: new Set() });
  $("chat").replaceChildren(WELCOME.cloneNode(true));
  $("trace").replaceChildren();
  $("next").replaceChildren();
  $("sms").hidden = true;
  $("active").hidden = true;
  for (const c of $("scenarios").children) c.classList.remove("selected", "dim");
  $("input").placeholder = "Ej: No reconozco un cargo de 350 en Oxxo";
  renderSteps();
  $("input").focus();
}

function newConversation() {
  resetConversation();
  if (!$("demo").hidden) loadScenarios().catch(() => {});
}

// ---- demo scenarios ------------------------------------------------------------------------------------------
const SCENARIOS = {
  normal: ["Cargo no reconocido", "Un solo movimiento coincide: el agente abre la disputa y ofrece bloquear la tarjeta."],
  ambiguous: ["Cargo ambiguo", "Varios cargos del mismo comercio: pregunta cuál es en vez de adivinar."],
  large_amount: ["Monto alto → asesor", "Más de 500 USD: el agente no actúa y pasa el caso a una persona (regla P4)."],
  stolen_card: ["Tarjeta robada con compras", "Bloquea la tarjeta y deriva las compras a fraude con un asesor."],
  no_phone: ["Sin celular registrado", "No puede verificar la identidad por SMS, así que no revela ningún dato."],
};

function startScenario(s, lang, card) {
  resetConversation();
  state.scenario = s;
  state.lang = lang;
  for (const c of $("scenarios").children) c.classList.toggle("dim", c !== card);
  card.classList.add("selected");
  $("active-label").textContent = `Escenario: ${SCENARIOS[s.scenario]?.[0] || s.scenario} · documento ${s.document} · ${lang.toUpperCase()}`;
  $("active").hidden = false;
  send(lang === "pt" ? s.message_pt : s.message_es);
}

async function loadDemo() {
  const cfg = await api("/v1/config");
  for (const [k, v] of [["modo", cfg.mode], ["router", cfg.router], ["LLM", cfg.llm_model || "ninguno"]]) {
    const b = document.createElement("span");
    b.className = "badge";
    b.textContent = `${k}: ${v}`;
    $("badges").appendChild(b);
  }
  if (!cfg.demo) return;
  $("demo").hidden = false;
  await loadScenarios();
}

// Re-fetched on every new conversation: the server skips customers another visitor has already used.
async function loadScenarios() {
  if (!$("scenarios").children.length) $("scenarios").textContent = "Cargando clientes de demo…";
  let list;
  try {
    list = await api("/v1/demo/customers");
  } catch {
    $("scenarios").textContent = "Cargando clientes de demo… (el servidor acaba de despertar, tarda ~1 min)";
    setTimeout(loadScenarios, 5000);
    return;
  }
  $("scenarios").replaceChildren();
  if (!list.length) $("scenarios").textContent = "No hay clientes de demo disponibles ahora; usa «Nueva conversación» más tarde.";
  for (const s of list) {
    const [title, what] = SCENARIOS[s.scenario] || [s.scenario, ""];
    const card = document.createElement("div");
    card.className = "scenario";
    const h = document.createElement("div");
    h.className = "title";
    h.textContent = title;
    const p = document.createElement("div");
    p.className = "what";
    p.textContent = what;
    const doc = document.createElement("div");
    doc.className = "mono";
    doc.textContent = `documento ${s.document}`;
    const row = document.createElement("div");
    row.className = "row";
    for (const lang of ["es", "pt"]) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = `▶ Probar (${lang.toUpperCase()})`;
      b.onclick = () => { if (!state.busy) startScenario(s, lang, card); };
      row.appendChild(b);
    }
    card.append(h, p, doc, row);
    $("scenarios").appendChild(card);
  }
}

$("composer").addEventListener("submit", (ev) => { ev.preventDefault(); send(); });
$("new").addEventListener("click", newConversation);
$("active-exit").addEventListener("click", newConversation);
renderSteps();
loadDemo().catch(() => {});
refreshMetrics();
