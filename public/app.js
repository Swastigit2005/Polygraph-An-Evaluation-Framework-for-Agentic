"use strict";

const $ = (s) => document.querySelector(s);
const STEP_MS = 750;
const API = "/api/ask";
const KIND = {
  normal: { tag: "💬 Normal question" },
  vague: { tag: "🌀 Vague · may need rephrasing" },
  trick: { tag: "🪤 Trick · not in the docs" },
};

let recorded = [];
let busy = false;

// ---------------------------------------------------------------- helpers
function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const norm = (s) => String(s || "").toLowerCase().replace(/\s+/g, " ").trim();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pct = (x) => `${Math.round(x * 100)}%`;
const pageOf = (title) => String(title || "").split(" > ")[0];

// ---------------------------------------------------------------- examples + warm-up
async function loadExamples() {
  try {
    const data = await (await fetch("data/examples.json")).json();
    recorded = data.recorded || [];
    const seen = new Set();
    for (const ex of data.examples) {
      if (seen.has(ex.kind)) continue; // one chip per kind keeps it simple
      seen.add(ex.kind);
      const b = el("button", `chip ${ex.kind}`, `<span class="tag">${KIND[ex.kind].tag}</span><span class="txt">${esc(ex.question)}</span>`);
      b.type = "button";
      b.onclick = () => { $("#q").value = ex.question; ask(ex.question); };
      $("#chips").append(b);
    }
  } catch (e) { /* chips are optional */ }
}
fetch(API).catch(() => {}); // wake the serverless function (loads models) while the visitor reads

// ---------------------------------------------------------------- polygraph strip
const SHAPES = {
  retrieve: { cls: "#334155", f: (t) => 6 * Math.sin(t * Math.PI * 4) },
  pass: { cls: "#059669", f: (t) => (t > .35 && t < .65 ? -16 * Math.sin((t - .35) / .3 * Math.PI) : 0) },
  fail: { cls: "#e11d48", f: (t) => { const a = [0, -46, 40, -38, 30, -22, 12, 0]; const i = t * (a.length - 1); const k = Math.floor(i); return a[k] + (a[Math.min(k + 1, a.length - 1)] - a[k]) * (i - k); } },
  rewrite: { cls: "#7c3aed", f: (t) => 18 * Math.sin(t * Math.PI * 6) * Math.sin(t * Math.PI) },
  generate: { cls: "#059669", f: (t) => 3 * Math.sin(t * Math.PI * 8) * (1 - t) },
  refuse: { cls: "#e11d48", f: () => 0 },
};
function shapeOf(step) {
  if (step.node === "grade_context") return step.data.relevant ? "pass" : "fail";
  if (step.node === "rewrite_query") return "rewrite";
  if (step.node === "generate" && step.data.refused) return "refuse";
  return SHAPES[step.node] ? step.node : "retrieve";
}
function drawStrip(steps) {
  const svg = $("#strip");
  svg.querySelectorAll(".seg").forEach((n) => n.remove());
  const w = 1000 / steps.length;
  steps.forEach((st, i) => {
    const s = SHAPES[shapeOf(st)];
    let d = "";
    for (let x = 0; x <= w; x += 4) d += `${x ? "L" : "M"}${(i * w + x).toFixed(1)},${(60 + s.f(x / w)).toFixed(1)}`;
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", d); p.setAttribute("class", "seg"); p.setAttribute("clip-path", "url(#reveal)");
    p.style.stroke = s.cls;
    svg.insertBefore(p, $("#pen"));
  });
  return w;
}
function resetStrip(idle) {
  $("#strip").querySelectorAll(".seg").forEach((n) => n.remove());
  $("#reveal-rect").setAttribute("width", 0);
  $("#pen").setAttribute("cx", 0); $("#pen").setAttribute("cy", 60);
  $("#strip-idle").setAttribute("d", idle ? "M0,60 L1000,60" : "");
}

// ---------------------------------------------------------------- stations + log
function station(name, cls, badge) {
  const li = $(`[data-st="${name}"]`);
  li.className = cls || "";
  li.querySelector(".badge")?.remove();
  if (badge) li.append(el("span", "badge", esc(badge)));
}
function log(cls, icon, html) {
  const li = el("li", cls, `<span class="ico">${icon}</span><span>${html}</span>`);
  $("#log").append(li);
  return li;
}

function narrate(st, ctx) {
  const d = st.data;
  if (st.node === "retrieve") {
    const pages = [...new Set(d.retrieved_ids.map((id) => pageOf(ctx.titles[id])))].slice(0, 3);
    const head = ctx.searches++ === 0
      ? `Searched the docs for <em>“${esc(d.query)}”</em> and kept the ${d.retrieved_ids.length} best snippets`
      : `Searched again with the new wording`;
    log("search", "🔎", `${head}<span class="pages">from: ${pages.map(esc).join(" · ")}</span>`);
    station("search", "pass", `${ctx.searches}×`);
    station("check", "active");
  } else if (st.node === "grade_context") {
    ctx.grades++;
    if (d.relevant) {
      log("pass", "✅", `Fact-checker: <em>the evidence answers the question</em><span class="why">“${esc(d.reason)}”</span>`);
      station("check", "pass", "✓ passed");
      station("answer", "active");
    } else {
      ctx.fails++;
      log("fail", "❌", `Fact-checker: <em>not good enough</em><span class="why">“${esc(d.reason)}”</span>`);
      station("check", "fail", `✗ rejected${ctx.fails > 1 ? ` ×${ctx.fails}` : ""}`);
    }
  } else if (st.node === "rewrite_query") {
    log("rewrite", "✏️", `Rephrased the search as <em>“${esc(d.to)}”</em>`);
    station("rephrase", "used", `${d.attempt}×`);
    station("search", "active");
  } else if (st.node === "generate") {
    if (d.refused) {
      log("refuse", "🛑", `The writer saw the evidence didn't contain the answer, so it <em>refused instead of guessing</em>`);
      station("answer", "fail", "refused");
    } else {
      const n = ctx.payload.sources.length;
      log("done", "💬", `Wrote an answer using <em>${n} source${n === 1 ? "" : "s"}</em> from the docs`);
      station("answer", "pass", "answered");
    }
  } else if (st.node === "refuse") {
    log("refuse", "🛑", `Still no solid evidence after ${ctx.payload.stats.rewrites} rephrasings, so it <em>refused instead of guessing</em>`);
    station("answer", "fail", "refused");
  }
}

// ---------------------------------------------------------------- result card
function renderAnswer(p) {
  const v = $("#verdict");
  if (p.refused) {
    v.className = "verdict no"; v.textContent = "✗ Not in the docs · refused to guess";
    const lastCheck = [...p.steps].reverse().find((s) => s.node === "grade_context");
    $("#answer").innerHTML =
      `<p class="refusal-explain">The agent couldn't find this in the Docker documentation, so it said so instead of making something up.
       <b>For a question outside the docs, that's the right answer.</b> A typical chatbot would have produced a confident-sounding guess.</p>` +
      (lastCheck ? `<p class="refusal-explain">Last fact-check: “${esc(lastCheck.data.reason)}”</p>` : "");
  } else {
    v.className = "verdict ok"; v.textContent = "✓ Answer backed by the docs";
    const html = DOMPurify.sanitize(marked.parse(p.answer));
    $("#answer").innerHTML = html.replace(/\[(\d{1,2})\]/g, (m, n) =>
      p.sources.some((s) => String(s.n) === n) ? `<a class="cite" href="#src-${n}">${n}</a>` : m);
  }
  const src = $("#sources");
  src.innerHTML = "";
  if (p.sources.length) {
    src.append(el("h4", "", "Sources"));
    for (const s of p.sources) {
      const a = el("a", "src", `<span class="n">${s.n}</span><span><span class="t">${esc(s.title)}</span><div class="x">${esc(s.excerpt)}</div></span>`);
      a.id = `src-${s.n}`; a.href = s.url; a.target = "_blank"; a.rel = "noopener";
      src.append(a);
    }
  }
  const st = p.stats;
  $("#meta").innerHTML = [
    `⏱ ${st.seconds.toFixed(1)} s`, `🤖 ${st.llm_calls} AI call${st.llm_calls === 1 ? "" : "s"}`,
    `🔢 ${st.tokens.toLocaleString()} tokens`, `✏️ ${st.rewrites} rephrasing${st.rewrites === 1 ? "" : "s"}`,
    p.models?.generator ? `writer: ${esc(p.models.generator)}` : "", p.models?.grader ? `fact-checker: ${esc(p.models.grader)}` : "",
  ].filter(Boolean).map((t) => `<span>${t}</span>`).join("");
  $("#raw").textContent = JSON.stringify(p.steps, null, 2);
  $("#result").hidden = false;
}

// ---------------------------------------------------------------- run
async function play(p, notice) {
  resetStrip(false);
  const w = drawStrip(p.steps);
  const ctx = { payload: p, titles: p.titles || {}, searches: 0, grades: 0, fails: 0 };
  $("#log").innerHTML = "";
  ["search", "check", "rephrase", "answer"].forEach((s) => station(s, ""));
  station("search", "active");
  for (let i = 0; i < p.steps.length; i++) {
    narrate(p.steps[i], ctx);
    $("#reveal-rect").setAttribute("width", (i + 1) * w);
    $("#pen").setAttribute("cx", (i + 1) * w);
    await sleep(STEP_MS);
  }
  if (!p.stats.rewrites) station("rephrase", "skip", "not needed");
  $("#notice").hidden = !notice;
  if (notice) $("#notice").innerHTML = notice;
  renderAnswer(p);
}

async function callApi(question) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 70000);
  try {
    const res = await fetch(API, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }), signal: ctrl.signal,
    });
    let body = {};
    try { body = await res.json(); } catch (e) { body = { error: "The server returned an unexpected response.", busy: true }; }
    return res.ok ? { ok: true, body } : { ok: false, body, busy: body.busy || res.status >= 500 };
  } catch (e) {
    return { ok: false, body: { error: "Couldn't reach the server." }, busy: true };
  } finally {
    clearTimeout(timer);
  }
}

async function ask(question) {
  question = question.trim();
  if (!question || busy) return;
  busy = true;
  $("#go").disabled = true;
  $("#run").hidden = false;
  $("#result").hidden = true;
  $("#run-q").textContent = question;
  $("#log").innerHTML = "";
  ["search", "check", "rephrase", "answer"].forEach((s) => station(s, ""));
  station("search", "active");
  resetStrip(true);
  $("#run").scrollIntoView({ behavior: "smooth", block: "start" });

  const waiting = log("wait", "⏳", "The agent is working on it…");
  const slow = setTimeout(() => {
    waiting.lastChild.innerHTML = "Still working… the first question can take a few extra seconds while the server wakes up.";
  }, 6000);

  const r = await callApi(question);
  clearTimeout(slow);
  waiting.remove();

  try {
    if (r.ok) {
      await play(r.body, "");
    } else {
      const rec = r.busy && recorded.find((x) => norm(x.question) === norm(question));
      if (rec) {
        await play(rec, `<b>Heads-up:</b> the live AI is busy right now (free-tier quota), so this is a <b>recorded run</b> of the same agent on this question from ${esc(rec.recorded_at)}.`);
      } else {
        resetStrip(false);
        station("search", "");
        log("refuse", "⚠️", `<span class="error">${esc(r.body.error || "Something went wrong.")}</span>${r.busy ? '<span class="why">Tip: the example questions above always work, even when the live AI is busy.</span>' : ""}`);
      }
    }
  } finally {
    busy = false;
    $("#go").disabled = false;
  }
}

$("#ask").addEventListener("submit", (e) => { e.preventDefault(); ask($("#q").value); });

// ---------------------------------------------------------------- results section
async function loadStats() {
  let s;
  try { s = await (await fetch("data/stats.json")).json(); } catch (e) { return; }
  const b = s.configs.baseline, f = s.configs.full;
  if (!b || !f) return;
  const metrics = [
    { name: "Correct answers", hint: `on the ${f.answerable} questions the docs can answer`, key: "end_to_end_accuracy",
      delta: `+${Math.round((f.end_to_end_accuracy - b.end_to_end_accuracy) * 100)} pts` },
    { name: "Trick questions refused", hint: `on the ${f.unanswerable} questions the docs can't answer`, key: "correct_refusal_rate",
      delta: `+${Math.round((f.correct_refusal_rate - b.correct_refusal_rate) * 100)} pts` },
    { name: "Good questions wrongly refused", hint: "lower is better", key: "false_refusal_rate",
      delta: f.false_refusal_rate > 0 ? `${Math.round(b.false_refusal_rate / f.false_refusal_rate)}× fewer` : "none" },
  ];
  for (const m of metrics) {
    const card = el("div", "metric", `<div class="metric-head"><div><h3>${m.name}</h3><span class="hint">${m.hint}</span></div><span class="delta">${m.delta}</span></div>`);
    for (const [cls, label, val] of [["base", "Plain AI search", b[m.key]], ["ours", "Polygraph", f[m.key]]]) {
      const row = el("div", `row ${cls}`, `<span class="lbl">${label}</span><span class="track"><span class="fill" data-w="${val * 100}"></span></span><span class="val">${pct(val)}</span>`);
      card.append(row);
    }
    $("#bars").append(card);
  }
  $("#fine").innerHTML =
    `${f.items} test questions (${f.answerable} answerable, ${f.unanswerable} deliberately not in the docs), drafted by an LLM and reviewed by an AI, not a human. ` +
    `One run per setup, so small differences are noise. Numbers come straight from the evaluation result files ` +
    `(<code>${esc(b.file)}</code>, <code>${esc(f.file)}</code>); ` +
    `<a href="https://github.com/Swastigit2005/Polygraph-An-Evaluation-Framework-for-Agentic#evaluation-results" target="_blank" rel="noopener">full metrics on GitHub</a>.`;
  const io = new IntersectionObserver((entries) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    document.querySelectorAll(".fill").forEach((n) => { n.style.width = `${n.dataset.w}%`; });
    io.disconnect();
  }, { threshold: 0.2 });
  io.observe($("#bars"));
}

loadExamples();
loadStats();
