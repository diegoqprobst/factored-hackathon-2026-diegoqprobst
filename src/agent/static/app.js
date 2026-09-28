"use strict";
const $ = (id) => document.getElementById(id);
const state = { cid: null, turn: 0, busy: false, handedOff: false };

function bubble(text, who) {
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

async function showSms() {
  try {
    const s = await api(`/v1/demo/sms/${state.cid}`);
    const code = s.sms.split(" ").pop();
    $("sms-body").textContent = s.sms;
    $("sms").hidden = false;
    $("sms-send").onclick = () => send(code);
  } catch (_) { $("sms").hidden = true; }
}

async function send(text) {
  text = (text ?? $("input").value).trim();
  if (!text || state.busy) return;
  state.busy = true;
  $("send").disabled = true;
  bubble(text, "me");
  if (text === $("input").value.trim()) $("input").value = "";
  try {
    const r = await api("/v1/chat", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ conversation_id: state.cid, message: text }) });
    state.cid = r.conversation_id;
    state.turn += 1;
    bubble(r.reply, "agent");
    renderTurn(r);
    if (r.handoff) { handoffCard(r.handoff); state.handedOff = true; $("input").placeholder = "Caso transferido: inicia una nueva conversación"; }
    if (r.stage === "auth_otp") await showSms(); else $("sms").hidden = true;
    refreshMetrics();
  } catch (err) {
    bubble(`No se pudo enviar (${err.message}). Tu texto sigue en el campo.`, "error");
    $("input").value = text;
  } finally {
    state.busy = false;
    $("send").disabled = false;
    $("input").focus();
  }
}

function resetConversation() {
  Object.assign(state, { cid: null, turn: 0, handedOff: false });
  $("chat").replaceChildren();
  $("trace").replaceChildren();
  $("sms").hidden = true;
  $("input").placeholder = "Ej: No reconozco un cargo de 350 en Oxxo";
  $("input").focus();
}

const LABELS = { normal: "Cargo no reconocido (1 coincidencia)", ambiguous: "Cargo ambiguo (varias coincidencias)",
  large_amount: "Monto alto → revisión humana (P4)", stolen_card: "Tarjeta robada con cargos → bloqueo + fraude",
  no_phone: "Sin celular registrado → no se puede verificar" };

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
    const card = document.createElement("div");
    card.className = "scenario";
    const title = document.createElement("div");
    title.textContent = LABELS[s.scenario] || s.scenario;
    const doc = document.createElement("div");
    doc.className = "mono";
    doc.textContent = `documento: ${s.document}`;
    const row = document.createElement("div");
    row.className = "row";
    for (const [label, text] of [["Mensaje ES", s.message_es], ["Mensaje PT", s.message_pt], ["Enviar documento", s.document]]) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "ghost";
      b.textContent = label;
      b.onclick = () => send(text);
      row.appendChild(b);
    }
    card.append(title, doc, row);
    $("scenarios").appendChild(card);
  }
}

$("composer").addEventListener("submit", (ev) => { ev.preventDefault(); send(); });
$("new").addEventListener("click", () => { resetConversation(); if (!$("demo").hidden) loadScenarios().catch(() => {}); });
loadDemo().catch(() => {});
refreshMetrics();
