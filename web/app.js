"use strict";
// Every string from the server or the model is escaped before it becomes
// markup, so model output and retrieved text can never inject HTML.

const $ = s => document.querySelector(s);
const MODES = ["chat", "coding", "agent"];
const MODE_LABEL = { chat: "assistant", coding: "coder", agent: "agent" };
const S = {
  view: "chat", mode: "chat", cid: null, last: { chat: null, coding: null, agent: null },
  busy: null, projectId: null, files: [], health: null, categories: [],
};

// ------------------------------------------------------------ helpers

function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "html") e.innerHTML = v;
    else if (k === "cls") e.className = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : String(kid));
  return e;
}

async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opt);
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON error */ }
  if (!res.ok) throw new Error((data && data.error) || res.status + " " + res.statusText);
  return data;
}

function toast(msg, bad) {
  const t = $("#toast");
  t.textContent = msg; t.className = bad ? "err" : ""; t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, bad ? 7000 : 3500);
}

async function guard(fn, okMsg) {
  try { const r = await fn(); if (okMsg) toast(okMsg); return r; }
  catch (e) { toast(e.message, true); return null; }
}

function when(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000), now = new Date();
  const same = d.toDateString() === now.toDateString();
  return same ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
              : d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " +
                d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function copyText(text, btn) {
  const done = () => { if (btn) { const o = btn.textContent; btn.textContent = "Copied"; setTimeout(() => btn.textContent = o, 1200); } };
  if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text, done));
  else fallbackCopy(text, done);
}
function fallbackCopy(text, done) {
  const ta = h("textarea"); ta.value = text; document.body.append(ta); ta.select();
  try { document.execCommand("copy"); done(); } finally { ta.remove(); }
}

// ----------------------------------------------------------- markdown

const KW = new Set(("abstract and as assert async await break case catch class const continue def default del " +
  "delete do elif else enum except export extends false final finally fn for from func function global go if " +
  "impl implements import in inherited instanceof interface is lambda let match mod mut new nil none not null " +
  "or override package pass private protected pub public raise range return select self sharing static struct " +
  "super switch this throw throws trigger true try type typeof use var virtual void where while with without " +
  "yield None True False insert update delete upsert from order by limit group").split(" "));

function highlight(code, lang) {
  lang = (lang || "").toLowerCase();
  if (!lang || lang === "text" || lang === "txt") return esc(code);
  if (lang === "diff" || lang === "patch") return diffHtml(code, true);
  const hash = /^(py|python|sh|bash|zsh|shell|console|rb|ruby|ya?ml|toml|r|perl|make|makefile|dockerfile|ini|conf)$/.test(lang);
  const dash = /^(sql|soql|lua|haskell)$/.test(lang);
  const comments = ["\\/\\*[\\s\\S]*?\\*\\/", hash ? "#[^\\n]*" : "\\/\\/[^\\n]*"];
  if (dash) comments.push("--[^\\n]*");
  const re = new RegExp(
    "(" + comments.join("|") + ")" +
    "|(\"\"\"[\\s\\S]*?\"\"\"|'''[\\s\\S]*?'''|\"(?:\\\\.|[^\"\\\\\\n])*\"|'(?:\\\\.|[^'\\\\\\n])*'|`(?:\\\\.|[^`\\\\])*`)" +
    "|(\\b\\d[\\d_]*(?:\\.\\d+)?\\b)" +
    "|([A-Za-z_][A-Za-z0-9_]*)(?=(\\s*\\()?)", "g");
  let out = "", last = 0, m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    const t = esc(m[0]);
    if (m[1]) out += '<span class="tk-c">' + t + "</span>";
    else if (m[2]) out += '<span class="tk-s">' + t + "</span>";
    else if (m[3]) out += '<span class="tk-n">' + t + "</span>";
    else if (KW.has(m[4]) || KW.has(m[4].toLowerCase()) && /^(sql|soql|apex|cls)$/.test(lang)) out += '<span class="tk-k">' + t + "</span>";
    else if (m[5] !== undefined) out += '<span class="tk-f">' + t + "</span>";
    else out += t;
    last = re.lastIndex;
  }
  return out + esc(code.slice(last));
}

function inline(s) {
  // s is already escaped
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return "\u0000" + (codes.length - 1) + "\u0000"; });
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
       .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,:;!?]|$)/g, "$1<em>$2</em>")
       .replace(/(^|[\s(])_([^_\n]+)_(?=[\s).,:;!?]|$)/g, "$1<em>$2</em>")
       .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => "<code>" + codes[+i] + "</code>");
}

function blocks(text) {
  const lines = esc(text).split("\n");
  let out = "", para = [], list = null;
  const flushPara = () => { if (para.length) { out += "<p>" + inline(para.join("<br>")) + "</p>"; para = []; } };
  const flushList = () => { if (list) { out += "<" + list.tag + ">" + list.items.map(i => "<li>" + inline(i) + "</li>").join("") + "</" + list.tag + ">"; list = null; } };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    let m;
    if (!line.trim()) { flushPara(); flushList(); continue; }
    if ((m = line.match(/^(#{1,4})\s+(.*)$/))) { flushPara(); flushList(); out += "<h" + (m[1].length + 1) + ">" + inline(m[2]) + "</h" + (m[1].length + 1) + ">"; continue; }
    if ((m = line.match(/^\s*([-*+]|\d+[.)])\s+(.*)$/))) {
      flushPara();
      const tag = /\d/.test(m[1]) ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push(m[2]); continue;
    }
    if ((m = line.match(/^&gt;\s?(.*)$/))) { flushPara(); flushList(); out += "<blockquote>" + inline(m[1]) + "</blockquote>"; continue; }
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      flushPara(); flushList();
      const cells = l => l.trim().replace(/^\||\|$/g, "").split("|").map(c => inline(c.trim()));
      let t = "<table><tr>" + cells(line).map(c => "<th>" + c + "</th>").join("") + "</tr>";
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) { t += "<tr>" + cells(lines[i]).map(c => "<td>" + c + "</td>").join("") + "</tr>"; i++; }
      i--; out += t + "</table>"; continue;
    }
    if (list && /^\s{2,}\S/.test(line)) { list.items[list.items.length - 1] += " " + line.trim(); continue; }
    flushList(); para.push(line);
  }
  flushPara(); flushList();
  return out;
}

function markdown(src) {
  const parts = [];
  const re = /```([\w+#.-]*)[^\n]*\n([\s\S]*?)(```|$)/g;
  let last = 0, m;
  while ((m = re.exec(src))) {
    parts.push(blocks(src.slice(last, m.index)));
    const lang = m[1] || "", code = m[2].replace(/\n$/, "");
    parts.push('<div class="codeblock"><span class="lang">' + esc(lang) + '</span><button class="copy small">Copy</button><pre><code>' +
               highlight(code, lang) + "</code></pre></div>");
    last = re.lastIndex;
    if (!m[3]) break;
  }
  parts.push(blocks(src.slice(last)));
  return parts.join("");
}

function wireCopy(root) {
  for (const b of root.querySelectorAll(".codeblock .copy")) {
    if (b.dataset.wired) continue;
    b.dataset.wired = "1";
    b.onclick = () => copyText(b.parentElement.querySelector("code").innerText, b);
  }
}

function diffHtml(diff, inner) {
  const rows = String(diff || "").split("\n").map(l => {
    const c = l.startsWith("+++") || l.startsWith("---") ? "f" : l.startsWith("@@") ? "h" : l.startsWith("+") ? "a" : l.startsWith("-") ? "d" : "";
    return '<div class="' + c + '">' + (esc(l) || " ") + "</div>";
  }).join("");
  return inner ? rows : '<div class="diff">' + rows + "</div>";
}

// ---------------------------------------------------------- navigation

function show(view) {
  if (S.busy && view !== S.view && MODES.includes(S.view)) { toast("Stop the current answer first."); return; }
  S.view = view;
  for (const b of document.querySelectorAll("nav button")) b.classList.toggle("on", b.dataset.view === view);
  const convo = MODES.includes(view);
  $("#v-convo").hidden = !convo;
  for (const v of ["history", "memory", "projects", "settings", "health"]) $("#v-" + v).hidden = view !== v;
  if (convo) setMode(view);
  else ({ history: loadHistory, memory: loadMemory, projects: loadProjects, settings: loadSettings, health: loadHealth })[view]();
}

function setMode(mode) {
  if (S.mode !== mode) { S.last[S.mode] = S.cid; S.mode = mode; S.cid = S.last[mode]; }
  $("#coderBar").hidden = mode !== "coding";
  $("#side").hidden = mode === "chat";
  $("#sideCoding").hidden = mode !== "coding";
  $("#sideAgent").hidden = mode !== "agent";
  $("#input").placeholder = { chat: "Message the assistant", coding: "Ask about the code, or request a change",
                              agent: "Ask, plan, or say e.g. 'remind me tomorrow at 9 to send the report'" }[mode];
  loadConvs();
  if (S.cid) openConv(S.cid, true); else clearLog();
  if (mode === "coding") { loadProjectsSelect(); refreshCodingSide(); }
  if (mode === "agent") refreshAgentSide();
}

for (const b of document.querySelectorAll("nav button")) b.onclick = () => show(b.dataset.view);

// ------------------------------------------------------- conversations

function clearLog() {
  $("#log").innerHTML = "";
  const intro = { chat: "Ask anything. Things you tell me about your work are remembered across conversations, and you can review them under Memory.",
                  coding: "Pick a project, then ask. I search the indexed repository for the relevant code; proposed edits appear as diffs and change nothing until you apply them.",
                  agent: "I can keep tasks and reminders, recall earlier conversations and your preferences, and run approved tools. Anything that changes something waits for your approval." }[S.mode];
  $("#log").append(h("div", { cls: "turn" }, h("div", { cls: "status-line" }, intro)));
  $("#ctxInfo").textContent = "";
}

async function loadConvs() {
  const box = $("#convs");
  const q = $("#convSearch").value.trim();
  try {
    if (q) {
      const r = await api("/api/conversations?q=" + encodeURIComponent(q));
      box.innerHTML = "";
      if (!r.results.length) box.append(h("div", { cls: "dim" }, "No matches."));
      for (const c of r.results) {
        const row = h("div", { cls: "conv", onclick: () => openConv(c.id) },
          h("span", { cls: "t" }, h("span", {}, c.title), h("span", { cls: "snip", html: snippet(c.snippet) })));
        box.append(row);
      }
      return;
    }
    let url = "/api/conversations?type=" + S.mode;
    const r = await api(url);
    box.innerHTML = "";
    if (!r.conversations.length) box.append(h("div", { cls: "dim" }, "No conversations yet."));
    for (const c of r.conversations) {
      const row = h("div", { cls: "conv" + (c.id === S.cid ? " on" : ""), title: c.title, onclick: () => openConv(c.id) },
        h("span", { cls: "t" }, c.title),
        h("button", { cls: "x", title: "Delete", onclick: ev => { ev.stopPropagation(); deleteConv(c.id); } }, "\u00d7"));
      box.append(row);
    }
  } catch (e) { box.innerHTML = ""; box.append(h("div", { cls: "err" }, e.message)); }
}

function snippet(s) {
  // the server marks matches with [ ]; everything else is escaped
  return esc(s || "").replace(/\[/g, "<mark>").replace(/\]/g, "</mark>");
}

$("#convSearch").oninput = () => { clearTimeout(loadConvs.t); loadConvs.t = setTimeout(loadConvs, 250); };
$("#newConv").onclick = () => { if (S.busy) return; S.cid = null; clearLog(); loadConvs(); $("#input").focus(); };

async function deleteConv(id) {
  if (!confirm("Delete this conversation and its messages? Memories learned from it stay (delete them under Memory).")) return;
  await guard(() => api("/api/conversations/delete", { id }), "Deleted.");
  if (id === S.cid) { S.cid = null; clearLog(); }
  loadConvs();
}

async function openConv(id, quiet) {
  if (S.busy) return;
  let r;
  try { r = await api("/api/conversation?id=" + id); }
  catch (e) { if (!quiet) toast(e.message, true); S.cid = null; clearLog(); return; }
  const type = MODES.includes(r.conversation.conversation_type) ? r.conversation.conversation_type : "chat";
  if (type !== S.mode || S.view !== type) {
    S.last[type] = id;
    if (type === S.mode) S.cid = id;
    show(type);   // re-enters openConv once the view and mode match
    return;
  }
  S.cid = id;
  if (S.mode === "coding" && r.conversation.project_id) { S.projectId = r.conversation.project_id; $("#projectSel").value = String(S.projectId); }
  $("#log").innerHTML = "";
  if (r.summary) $("#log").append(h("div", { cls: "turn" }, h("div", { cls: "status-line" }, "Earlier turns are summarized for the model; the full text is kept here.")));
  const msgs = r.messages;
  msgs.forEach((m, i) => {
    if (m.role === "user") {
      addUser(m.content);
      if (m.metadata && m.metadata.unanswered && !(msgs[i + 1] && msgs[i + 1].role === "assistant")) {
        const t = addAssistant();
        t.setError({ message: m.metadata.error || "No answer was produced.", retryable: true });
        t.finish();
      }
    } else if (m.role === "assistant") {
      const t = addAssistant();
      t.setText(m.content);
      const md = m.metadata || {};
      if (md.sources) t.setSources(md.sources);
      if (md.stats) t.setStats(md.stats);
      if (md.stopped) t.note("stopped by you", false);
      if (md.error) t.note("ended early: " + md.error, false);
      for (const p of r.proposals.filter(p => p.message_id === m.id)) t.addProposal(p);
      t.finish();
    }
  });
  markRegenerate();
  scrollDown(true);
  loadConvs();
  if (S.mode === "coding") refreshCodingSide();
  if (S.mode === "agent") refreshAgentSide();
}

function scrollDown(force) {
  const log = $("#log");
  if (force || log.scrollHeight - log.scrollTop - log.clientHeight < 140) log.scrollTop = log.scrollHeight;
}

function addUser(text) {
  const body = h("div", { cls: "body" });
  body.textContent = text;
  const copy = h("button", { onclick: () => copyText(text, copy) }, "Copy");
  $("#log").append(h("div", { cls: "turn user" }, h("div", { cls: "who" }, "you", h("span", { cls: "acts" }, copy)), body));
  scrollDown(true);
}

function addAssistant() {
  let text = "", raf = 0;
  const body = h("div", { cls: "body" });
  const notes = h("div");
  const extra = h("div");
  const status = h("div", { cls: "status-line" });
  const copy = h("button", { onclick: () => copyText(text, copy) }, "Copy");
  const regen = h("button", { cls: "regen", hidden: true, onclick: () => regenerate() }, "Regenerate");
  const wrap = h("div", { cls: "turn bot" },
    h("div", { cls: "who" }, MODE_LABEL[S.mode], h("span", { cls: "acts" }, copy, regen)), notes, body, status, extra);
  $("#log").append(wrap);
  const paint = () => { raf = 0; body.innerHTML = markdown(text); wireCopy(body); scrollDown(); };
  return {
    wrap,
    append(piece) { text += piece; status.textContent = ""; if (!raf) raf = requestAnimationFrame(paint); },
    setText(t) { text = t; paint(); },
    get text() { return text; },
    busy(on) { body.classList.toggle("blink", on); },
    status(t) { status.textContent = t || ""; },
    note(t, ok, node) { notes.append(h("div", { cls: "notice" + (ok ? " ok" : "") }, t, node || null)); scrollDown(); },
    node(n) { extra.append(n); scrollDown(); },
    setSources(src) {
      if (!src || !src.length) return;
      const label = { memory: "memory", project_memory: "project", conversation: "earlier chat", code: "code", note: "note", file: "file" };
      extra.append(h("div", { cls: "sources" }, h("b", {}, "Used: "),
        src.slice(0, 14).map(s => h("span", { title: s.label }, (label[s.kind] || s.kind) + " · " + String(s.label).slice(0, 60)))));
    },
    setStats(st) {
      if (!st) return;
      const bits = [];
      if (st.tokens_per_second) bits.push(st.tokens_per_second.toFixed(1) + " tok/s");
      if (st.ttft_ms != null) bits.push((st.ttft_ms / 1000).toFixed(2) + "s to first token");
      if (st.total_ms != null) bits.push((st.total_ms / 1000).toFixed(1) + "s total");
      if (st.prompt_tokens) bits.push(st.prompt_tokens + " prompt tokens");
      if (st.completion_tokens) bits.push(st.completion_tokens + " generated");
      if (st.queue_wait_ms) bits.push("queued " + (st.queue_wait_ms / 1000).toFixed(1) + "s");
      if (st.model) bits.push(st.model);
      extra.append(h("div", { cls: "stats" }, bits.join(" · ")));
    },
    setError(err) {
      const box = h("div", { cls: "errbox" }, h("div", { cls: "err" }, err.message || "Something went wrong."));
      if (err.hint) box.append(h("div", { cls: "hint" }, err.hint));
      const row = h("div", { cls: "row", style: "margin:8px 0 0" });
      if (err.retryable !== false) row.append(h("button", { cls: "small", onclick: () => regenerate() }, "Retry"));
      if (err.kind === "unavailable" || err.kind === "loading") {
        row.append(h("button", { cls: "small", onclick: () => checkHealth(true) }, "Check connection"));
        if (S.health && S.health.colibri && S.health.colibri.managed) row.append(h("button", { cls: "small", onclick: () => colibriAction("start") }, "Start colibri"));
        row.append(h("button", { cls: "small", onclick: () => show("settings") }, "Model settings"));
      }
      box.append(row);
      extra.append(box); scrollDown();
    },
    addProposal(p) { extra.append(proposalCard(p, true)); scrollDown(); },
    finish() { if (raf) { cancelAnimationFrame(raf); paint(); } body.classList.remove("blink"); status.textContent = ""; },
  };
}

function markRegenerate() {
  const bots = [...document.querySelectorAll("#log .turn.bot")];
  bots.forEach((b, i) => { const r = b.querySelector(".regen"); if (r) r.hidden = i !== bots.length - 1; });
}

// ---------------------------------------------------------------- turns

const input = $("#input");
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
document.addEventListener("keydown", e => { if (e.key === "Escape" && S.busy) S.busy.abort(); });
input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 220) + "px"; });
$("#send").onclick = () => S.busy ? S.busy.abort() : send();

function send() {
  const text = input.value.trim();
  if (!text || S.busy) return;
  input.value = ""; input.style.height = "auto";
  if ($("#log .status-line") && !S.cid) $("#log").innerHTML = "";
  addUser(text);
  const body = { mode: S.mode, message: text, conversation_id: S.cid };
  if (S.mode === "coding") {
    body.project_id = S.projectId;
    body.files = S.files.slice();
    const code = $("#codeInput").value.trim();
    if (code) { body.code = code; $("#codeInput").value = ""; }
  }
  runTurn(body);
}

function regenerate() {
  if (S.busy || !S.cid) return;
  const bots = [...document.querySelectorAll("#log .turn.bot")];
  const lastUser = [...document.querySelectorAll("#log .turn.user")].pop();
  // drop every answer after the last question
  for (const b of bots) if (lastUser && (lastUser.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING)) b.remove();
  runTurn({ mode: S.mode, conversation_id: S.cid, regenerate: true, project_id: S.projectId,
            files: S.mode === "coding" ? S.files.slice() : [] });
}

async function runTurn(body) {
  const ctrl = new AbortController();
  S.busy = ctrl;
  setStreaming(true);
  const t = addAssistant();
  t.busy(true);
  let sawDone = false, failed = false;
  try {
    const res = await fetch("/api/turn", { method: "POST", headers: { "Content-Type": "application/json" },
                                           body: JSON.stringify(body), signal: ctrl.signal });
    if (!res.ok) throw new Error(await res.text());
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const ev = JSON.parse(line.slice(5));
        if (ev.type === "done") sawDone = true;
        if (ev.type === "error") failed = true;
        handleEvent(ev, t);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") t.note("stopped by you; the partial answer is saved", false);
    else { failed = true; t.setError({ message: String(e.message || e), kind: "unavailable", retryable: true }); }
  }
  t.finish();
  if (!sawDone && !failed && !ctrl.signal.aborted) t.note("the connection closed before the answer finished", false);
  S.busy = null;
  setStreaming(false);
  markRegenerate();
  loadConvs();
  if (S.mode === "coding") refreshCodingSide();
  if (S.mode === "agent") refreshAgentSide();
  input.focus();
}

function handleEvent(ev, t) {
  switch (ev.type) {
    case "conversation":
      S.cid = ev.id; S.last[S.mode] = ev.id;
      if (ev.created) loadConvs();
      break;
    case "intent":
      if (ev.kind && ev.kind !== "conversational") t.status("Understood as: " + ev.kind);
      break;
    case "status":
      t.status(ev.text);
      break;
    case "action":
      t.note((ev.status === "succeeded" ? "\u2713 " : "\u2717 ") + ev.tool + ": " + ev.summary, ev.status === "succeeded");
      break;
    case "approval":
      t.node(approvalCard(ev));
      break;
    case "context": {
      const b = ev.budget || {};
      $("#ctxInfo").textContent = "prompt ~" + ev.prompt_tokens + " tokens of " + (b.available || "?") +
        (ev.history_total ? " · history " + ev.history_kept + "/" + ev.history_total : "") + (ev.summary_used ? " · summary used" : "");
      t.setSources(ev.sources);
      break;
    }
    case "memory":
      for (const m of ev.saved || []) t.note("Remembered (" + (m.scope === "project" ? "project " : "") + (m.category || "") + "): " + m.content, true);
      for (const m of ev.removed || []) t.note("Forgot: " + m.content, true);
      for (const r of ev.refused || []) t.note(r, false);
      break;
    case "token":
      t.append(ev.text);
      break;
    case "error":
      t.setError(ev);
      if (ev.kind === "unavailable" || ev.kind === "loading") checkHealth(true);
      break;
    case "stats":
      t.setStats(ev);
      break;
    case "proposal":
      t.addProposal(ev);
      break;
    case "proposal_error":
      t.note("Could not turn the suggested edit into a diff: " + ev.message, false);
      break;
  }
}

function setStreaming(on) {
  $("#send").textContent = on ? "Stop" : "Send";
  $("#stStream").innerHTML = on ? "<b>streaming</b>" : "idle";
}

// --------------------------------------------------- approvals & runs

function approvalCard(ev) {
  const id = ev.run_id;
  const args = Object.assign({}, ev.arguments || {});
  delete args._project_id;
  const card = h("div", { cls: "approval" },
    h("b", {}, "Approval needed: " + ev.tool),
    h("span", { cls: "mono" }, args.command ? args.command : JSON.stringify(args, null, 1)));
  if (ev.warning) card.append(h("div", { cls: "warn" }, "\u26a0 " + ev.warning));
  const out = h("div");
  const row = h("div", { cls: "row" },
    h("button", { cls: "small primary", onclick: async () => {
      row.remove();
      const r = await guard(() => api("/api/tools/approve", { run_id: id }));
      if (r) out.append(runResult(r));
      refreshSides();
    } }, "Approve & run"),
    h("button", { cls: "small", onclick: async () => {
      row.remove();
      if (await guard(() => api("/api/tools/reject", { run_id: id }))) out.append(h("div", { cls: "dim" }, "Rejected; nothing ran."));
      refreshSides();
    } }, "Reject"));
  card.append(row, out);
  return card;
}

function runResult(r) {
  const res = r.result;
  const box = h("div", { cls: "run" }, h("span", { cls: "badge " + r.status }, r.status), " ", r.tool || "");
  if (r.error) box.append(h("div", { cls: "err" }, r.error));
  if (res && typeof res === "object" && "exit_code" in res) {
    box.append(h("div", { cls: "meta" }, "exit " + res.exit_code + (res.seconds != null ? " · " + res.seconds + "s" : "")));
    const text = (res.stdout || "") + (res.stderr ? "\n[stderr]\n" + res.stderr : "");
    if (text.trim()) box.append(h("pre", {}, text));
  } else if (res != null) {
    box.append(h("pre", {}, typeof res === "string" ? res : JSON.stringify(res, null, 1)));
  }
  return box;
}

function runRow(r) {
  if (r.status === "pending_approval") return approvalCard({ run_id: r.id, tool: r.tool, arguments: r.arguments, warning: r.warning });
  const box = runResult({ ...r, tool: r.tool });
  const args = Object.assign({}, r.arguments || {}); delete args._project_id;
  box.insertBefore(h("div", { cls: "meta" }, when(r.created_at) + " · " + (r.origin || "") + " · " +
    (args.command || JSON.stringify(args).slice(0, 90))), box.children[1] || null);
  return box;
}

function refreshSides() {
  if (S.mode === "coding") refreshCodingSide();
  if (S.mode === "agent") refreshAgentSide();
  checkHealth();
}

// ------------------------------------------------------- coding mode

async function loadProjectsSelect() {
  const r = await guard(() => api("/api/projects"));
  if (!r) return;
  const sel = $("#projectSel");
  sel.innerHTML = "";
  sel.append(h("option", { value: "" }, r.projects.length ? "— choose —" : "none yet (create under Projects)"));
  for (const p of r.projects) sel.append(h("option", { value: p.id }, p.name + (p.repository_path ? "" : " (no folder)")));
  if (!S.projectId && r.projects.length === 1) S.projectId = r.projects[0].id;
  sel.value = S.projectId ? String(S.projectId) : "";
}
$("#projectSel").onchange = e => { S.projectId = +e.target.value || null; S.files = []; renderChips(); refreshCodingSide(); };

$("#fileFilter").oninput = () => {
  clearTimeout(loadConvs.f);
  loadConvs.f = setTimeout(async () => {
    const q = $("#fileFilter").value.trim(), box = $("#fileHits");
    if (!q || !S.projectId) { box.hidden = true; return; }
    const r = await guard(() => api("/api/project/files?id=" + S.projectId + "&q=" + encodeURIComponent(q)));
    box.innerHTML = "";
    for (const f of (r ? r.files : []).slice(0, 40)) {
      box.append(h("div", { onclick: () => { if (!S.files.includes(f.path) && S.files.length < 5) S.files.push(f.path); renderChips(); box.hidden = true; $("#fileFilter").value = ""; } }, f.path));
    }
    box.hidden = !box.children.length;
  }, 200);
};
document.addEventListener("click", e => { if (!e.target.closest("#fileHits, #fileFilter")) $("#fileHits").hidden = true; });

function renderChips() {
  const box = $("#chosenFiles");
  box.innerHTML = "";
  for (const f of S.files) box.append(h("span", { cls: "chip" }, f, h("button", { onclick: () => { S.files = S.files.filter(x => x !== f); renderChips(); } }, "\u00d7")));
}
$("#toggleCode").onclick = () => { const c = $("#codeInput"); c.hidden = !c.hidden; if (!c.hidden) c.focus(); };

function proposalCard(p, inline) {
  const card = h("div", { cls: "prop" });
  const draw = (p) => {
    card.innerHTML = "";
    card.append(h("div", {}, h("span", { cls: "badge " + p.status }, p.status), " ", h("b", {}, "Change #" + p.id), " ",
      h("span", { cls: "meta" }, (p.files || []).join(", "))));
    if (p.summary && !inline) card.append(h("div", { cls: "meta" }, p.summary));
    const problems = p.problems || (p.result && p.result.problems);
    if (problems && problems.length) card.append(h("div", { cls: "warn" }, problems.join("; ")));
    if (p.result && p.result.error) card.append(h("div", { cls: "err" }, p.result.error));
    const diff = h("div", { html: diffHtml(p.diff) });
    if (p.status !== "pending" && !inline) diff.hidden = true;
    card.append(diff);
    const row = h("div", { cls: "row", style: "margin:6px 0 0" });
    if (p.status !== "pending" && !inline) row.append(h("button", { cls: "small", onclick: () => { diff.hidden = !diff.hidden; } }, "Diff"));
    if (p.status === "pending") {
      row.append(h("button", { cls: "small primary", onclick: () => act("apply") }, "Apply to files"),
                 h("button", { cls: "small", onclick: () => act("reject") }, "Reject"));
    }
    if (p.status === "applied") row.append(h("button", { cls: "small", onclick: () => act("revert") }, "Revert"));
    card.append(row);
  };
  const act = async (what) => {
    if (what === "apply" && !confirm("Write these changes to " + (p.files || []).join(", ") + "?")) return;
    const r = await guard(() => api("/api/proposals/" + what, { id: p.id }), { apply: "Applied.", reject: "Rejected.", revert: "Reverted." }[what]);
    if (r) {
      const fresh = await guard(() => api("/api/proposal?id=" + p.id));
      if (fresh) { p = fresh.proposal; draw(p); }
      refreshCodingSide();
    }
  };
  draw(p);
  return card;
}

async function refreshCodingSide() {
  const props = $("#proposals"), runs = $("#runs");
  if (!S.projectId) { props.innerHTML = '<div class="dim">Choose a project to see its proposed changes.</div>'; }
  else {
    const r = await guard(() => api("/api/proposals?project_id=" + S.projectId));
    props.innerHTML = "";
    if (r && !r.proposals.length) props.append(h("div", { cls: "dim" }, "None yet. Ask for a change and the diff appears here."));
    for (const p of (r ? r.proposals : []).slice(0, 15)) props.append(proposalCard(p, false));
  }
  const rr = await guard(() => api("/api/tools/runs?limit=25"));
  runs.innerHTML = "";
  const list = (rr ? rr.runs : []).filter(x => ["run_command", "read_file", "search_code"].includes(x.tool));
  if (!list.length) runs.append(h("div", { cls: "dim" }, "Nothing has run."));
  for (const x of list) runs.append(runRow(x));
}

$("#cmdRun").onclick = async () => {
  const command = $("#cmdInput").value.trim();
  if (!command) return;
  if (!S.projectId) { toast("Choose a project first.", true); return; }
  const r = await guard(() => api("/api/tools/run", { tool: "run_command", arguments: { command }, project_id: S.projectId, conversation_id: S.cid }));
  if (r) { $("#cmdInput").value = ""; if (r.status === "failed") toast(r.error, true); refreshCodingSide(); }
};

// --------------------------------------------------------- agent mode

async function refreshAgentSide() {
  const [pending, tasks, runs] = await Promise.all([
    guard(() => api("/api/tools/runs?status=pending_approval")),
    guard(() => api("/api/tasks")),
    guard(() => api("/api/tools/runs?limit=15")),
  ]);
  const ap = $("#approvals");
  ap.innerHTML = "";
  if (pending && !pending.runs.length) ap.append(h("div", { cls: "dim" }, "Nothing waiting."));
  for (const r of pending ? pending.runs : []) ap.append(runRow(r));
  const tb = $("#tasks");
  tb.innerHTML = "";
  const list = tasks ? tasks.tasks : [];
  if (!list.length) tb.append(h("div", { cls: "dim" }, "No tasks."));
  for (const t of list.filter(t => t.status === "open").concat(list.filter(t => t.status !== "open").slice(0, 8))) {
    const box = h("input", { type: "checkbox", checked: t.status === "done", onchange: async () => {
      await guard(() => api("/api/tasks/update", { id: t.id, status: box.checked ? "done" : "open" })); refreshAgentSide(); } });
    tb.append(h("div", { cls: "task" + (t.status === "done" ? " done" : "") },
      h("label", { cls: "check" }, box, h("span", {}, "#" + t.id + " " + t.title)),
      t.due_label ? h("div", { cls: "meta dim" }, "due " + t.due_label) : null,
      h("button", { cls: "link small", onclick: async () => { if (confirm("Delete this task?")) { await guard(() => api("/api/tasks/delete", { id: t.id })); refreshAgentSide(); } } }, "delete")));
  }
  const ar = $("#agentRuns");
  ar.innerHTML = "";
  for (const r of (runs ? runs.runs : []).filter(r => r.status !== "pending_approval").slice(0, 10)) ar.append(runRow(r));
}

$("#taskAdd").onclick = async () => {
  const title = $("#taskTitle").value.trim();
  if (!title) return;
  const r = await guard(() => api("/api/tasks/create", { title, due: $("#taskDue").value.trim() || null }));
  if (r) { $("#taskTitle").value = ""; $("#taskDue").value = ""; toast("Added: " + r.task.title + (r.task.due_label ? " (" + r.task.due_label + ")" : "")); refreshAgentSide(); }
};

async function pollReminders() {
  try {
    const r = await api("/api/reminders/due");
    for (const t of r.due) toast("\u23f0 Reminder: " + t.title);
    if (r.due.length && S.mode === "agent") refreshAgentSide();
  } catch (e) { /* offline server; the status bar reports it */ }
}

// ------------------------------------------------------------ history

async function loadHistory() {
  const q = $("#histSearch").value.trim(), box = $("#histList");
  box.innerHTML = "";
  try {
    if (q) {
      const r = await api("/api/conversations?q=" + encodeURIComponent(q));
      if (!r.results.length) box.append(h("div", { cls: "dim" }, "No matches."));
      for (const c of r.results) box.append(histRow(c, c.snippet));
      return;
    }
    const type = $("#histType").value, tag = $("#histTag").value.trim();
    const r = await api("/api/conversations?limit=300" + (type ? "&type=" + type : "") + (tag ? "&tag=" + encodeURIComponent(tag) : ""));
    if (!r.conversations.length) box.append(h("div", { cls: "dim" }, "No conversations."));
    for (const c of r.conversations) box.append(histRow(c));
  } catch (e) { box.append(h("div", { cls: "err" }, e.message)); }
}

function histRow(c, snip) {
  const row = h("div", { cls: "hist" },
    h("div", {}, h("b", {}, c.title), " ", h("span", { cls: "badge" }, c.conversation_type || ""),
      " ", (c.tags || []).map(t => h("span", { cls: "badge" }, "#" + t))),
    h("div", { cls: "meta" }, when(c.updated_at) + (c.n_messages != null ? " · " + c.n_messages + " messages" : "") + (c.category ? " · " + c.category : "")),
    snip ? h("div", { cls: "snip", html: snippet(snip) }) : null,
    h("div", { cls: "row", style: "margin:6px 0 0" },
      h("button", { cls: "small", onclick: () => openConv(c.id) }, "Open"),
      h("button", { cls: "small", onclick: async () => {
        const title = prompt("Rename conversation", c.title);
        if (title && title.trim()) { await guard(() => api("/api/conversations/update", { id: c.id, title: title.trim() }), "Renamed."); loadHistory(); loadConvs(); }
      } }, "Rename"),
      h("button", { cls: "small", onclick: async () => {
        const tags = prompt("Tags, comma separated", (c.tags || []).join(", "));
        if (tags !== null) { await guard(() => api("/api/conversations/update", { id: c.id, tags: tags.split(",") }), "Saved."); loadHistory(); }
      } }, "Tags"),
      h("button", { cls: "small danger", onclick: async () => { await deleteConv(c.id); loadHistory(); } }, "Delete")));
  return row;
}
for (const id of ["#histSearch", "#histTag"]) $(id).oninput = () => { clearTimeout(loadHistory.t); loadHistory.t = setTimeout(loadHistory, 250); };
$("#histType").onchange = loadHistory;

// ------------------------------------------------------------- memory

async function loadMemory() {
  const q = $("#memSearch").value.trim(), cat = $("#memCat").value;
  const r = await guard(() => api("/api/memories?" + (q ? "q=" + encodeURIComponent(q) + "&" : "") + (cat ? "category=" + cat : "")));
  if (!r) return;
  if (!S.categories.length) {
    S.categories = r.categories;
    for (const c of r.categories) { $("#memCat").append(h("option", { value: c }, c)); $("#memNewCat").append(h("option", { value: c }, c)); }
    $("#memNewCat").value = "preference";
  }
  $("#memAuto").checked = !!r.auto_extract;
  $("#memLLM").checked = !!r.llm_extraction;
  const box = $("#memList");
  box.innerHTML = "";
  if (!r.memories.length) box.append(h("div", { cls: "dim" }, q ? "No matches." : "Nothing remembered yet."));
  for (const m of r.memories) box.append(memRow(m));
}

function memRow(m) {
  const row = h("div", { cls: "mem" + (m.pinned ? " pinned" : "") });
  const view = () => {
    row.innerHTML = "";
    row.append(
      h("div", { cls: "content" }, m.content),
      h("div", { cls: "meta" },
        h("span", { cls: "badge" }, m.category), " ", m.memory_type, " · importance " + (+m.importance).toFixed(2) +
        " · confidence " + (+m.confidence).toFixed(2) + " · " + (m.origin || "") + " · used " + (m.access_count || 0) + "×" +
        " · updated " + when(m.updated_at) + (m.expires_at ? " · expires " + when(m.expires_at) : "") + (m.pinned ? " · pinned" : "")),
      h("div", { cls: "row", style: "margin:6px 0 0" },
        h("button", { cls: "small", onclick: edit }, "Edit"),
        h("button", { cls: "small", onclick: async () => { await guard(() => api("/api/memories/update", { id: m.id, pinned: !m.pinned })); loadMemory(); } }, m.pinned ? "Unpin" : "Pin"),
        m.source_conversation_id ? h("button", { cls: "small", onclick: () => openConv(m.source_conversation_id) }, "Source: " + (m.source_title || "conversation")) : null,
        h("button", { cls: "small danger", onclick: async () => { if (confirm("Delete this memory?")) { await guard(() => api("/api/memories/delete", { id: m.id }), "Deleted."); loadMemory(); } } }, "Delete")));
  };
  const edit = () => {
    row.innerHTML = "";
    const ta = h("textarea"); ta.value = m.content;
    const cat = h("select", {}, S.categories.map(c => h("option", { value: c }, c))); cat.value = m.category;
    const imp = h("input", { type: "number", min: 0, max: 1, step: 0.05, value: m.importance, style: "width:80px" });
    row.append(ta, h("div", { cls: "row", style: "margin:6px 0 0" }, cat, h("label", { cls: "check" }, "importance", imp),
      h("button", { cls: "small primary", onclick: async () => {
        const r = await guard(() => api("/api/memories/update", { id: m.id, content: ta.value, category: cat.value, importance: +imp.value }), "Saved.");
        if (r) { m = r.memory; view(); }
      } }, "Save"),
      h("button", { cls: "small", onclick: view }, "Cancel")));
  };
  view();
  return row;
}

$("#memSearch").oninput = () => { clearTimeout(loadMemory.t); loadMemory.t = setTimeout(loadMemory, 250); };
$("#memCat").onchange = loadMemory;
$("#memAuto").onchange = e => guard(() => api("/api/settings", { changes: { MEMORY_AUTO_EXTRACT: e.target.checked } }), e.target.checked ? "Automatic memory on." : "Automatic memory off. Explicit 'remember …' still works.");
$("#memLLM").onchange = e => guard(() => api("/api/settings", { changes: { MEMORY_LLM_EXTRACTION: e.target.checked } }), "Saved.");
$("#memAdd").onclick = async () => {
  const content = $("#memNew").value.trim();
  if (!content) return;
  const r = await guard(() => api("/api/memories/create", { content, category: $("#memNewCat").value, allow_sensitive: $("#memNewSensitive").checked }));
  if (r) { $("#memNew").value = ""; $("#memNewSensitive").checked = false; toast(r.action === "updated" ? "Merged with an existing memory." : "Saved."); loadMemory(); }
};
$("#memExport").onclick = async () => {
  const data = await guard(() => api("/api/memories/export"));
  if (!data) return;
  const a = h("a", { href: URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })),
                     download: "vault-chat-memories.json" });
  document.body.append(a); a.click(); a.remove();
};
$("#memImport").onchange = async e => {
  const file = e.target.files[0];
  if (!file) return;
  let data;
  try { data = JSON.parse(await file.text()); } catch (err) { toast("That file is not JSON.", true); return; }
  const r = await guard(() => api("/api/memories/import", { data }));
  if (r) toast("Imported " + (r.created || 0) + ", merged " + (r.updated || 0) + ".");
  e.target.value = ""; loadMemory();
};
$("#memWipe").onclick = async () => {
  if (!confirm("Delete every memory? Conversations are kept. Consider exporting first.")) return;
  const r = await guard(() => api("/api/memories/delete_all", { confirm: true }));
  if (r) { toast("Deleted " + r.deleted + " memories."); loadMemory(); }
};

// ----------------------------------------------------------- projects

async function loadProjects() {
  const r = await guard(() => api("/api/projects"));
  const box = $("#projList");
  box.innerHTML = "";
  if (!r) return;
  if (!r.projects.length) box.append(h("div", { cls: "dim" }, "No projects. A project groups conversations and remembers its own context; give it a folder to let the coding assistant index the code."));
  for (const p of r.projects) {
    box.append(h("div", { cls: "proj" },
      h("b", {}, p.name), " ", h("span", { cls: "meta dim" }, p.repository_path || "no folder"),
      p.description ? h("div", { cls: "meta" }, p.description) : null,
      h("div", { cls: "row", style: "margin:6px 0 0" },
        h("button", { cls: "small", onclick: () => projectDetail(p.id) }, "Open"),
        h("button", { cls: "small", onclick: () => { S.projectId = p.id; show("coding"); } }, "Code with it"),
        h("button", { cls: "small danger", onclick: async () => {
          if (!confirm("Delete project " + p.name + "? Its index and project memories are removed; files on disk are not touched.")) return;
          await guard(() => api("/api/projects/delete", { id: p.id }), "Deleted."); if (S.projectId === p.id) S.projectId = null;
          $("#projDetail").innerHTML = ""; loadProjects();
        } }, "Delete"))));
  }
}

$("#projAdd").onclick = async () => {
  const r = await guard(() => api("/api/projects/create", { name: $("#projName").value, repository_path: $("#projPath").value, description: $("#projDesc").value }), "Created. Indexing starts in the background if it has a folder.");
  if (r) { for (const id of ["#projName", "#projPath", "#projDesc"]) $(id).value = ""; loadProjects(); projectDetail(r.id); }
};

async function projectDetail(id) {
  const r = await guard(() => api("/api/project?id=" + id));
  const box = $("#projDetail");
  box.innerHTML = "";
  if (!r) return;
  const p = r.project, idx = r.index || {};
  const idxText = idx.state === "running" ? "indexing… " + (idx.done || 0) + "/" + (idx.total || "?") + " files"
    : idx.state === "error" ? "index failed: " + idx.error
    : r.files + " files indexed" + (idx.result ? " (last run: " + (idx.result.added || 0) + " added, " + (idx.result.updated || 0) +
      " updated, " + (idx.result.removed || 0) + " removed)" : "");
  const memType = h("select", {}, r.memory_types.map(t => h("option", { value: t }, t)));
  const memText = h("input", { placeholder: "e.g. 'Billing schedules are created by the Order activation flow'" });
  box.append(h("div", { cls: "card" },
    h("h3", {}, p.name),
    h("div", { cls: "meta" }, (p.repository_path || "no folder") + " · " + idxText),
    h("div", { cls: "row", style: "margin:8px 0" },
      p.repository_path ? h("button", { cls: "small", onclick: async () => { await guard(() => api("/api/projects/index", { id }), "Re-indexing in the background."); setTimeout(() => projectDetail(id), 800); } }, "Re-index") : null,
      h("button", { cls: "small", onclick: async () => {
        const name = prompt("Name", p.name); if (name === null) return;
        const path = prompt("Repository folder (empty for none)", p.repository_path || ""); if (path === null) return;
        const desc = prompt("Description", p.description || ""); if (desc === null) return;
        await guard(() => api("/api/projects/update", { id, name, repository_path: path || null, description: desc }), "Saved.");
        loadProjects(); projectDetail(id);
      } }, "Edit")),
    h("h3", {}, "Project memory"),
    h("div", { cls: "row" }, memText, memType, h("button", { cls: "small primary", onclick: async () => {
      if (!memText.value.trim()) return;
      if (await guard(() => api("/api/project_memories/create", { project_id: id, content: memText.value, memory_type: memType.value }))) projectDetail(id);
    } }, "Add")),
    r.memories.length ? r.memories.map(m => h("div", { cls: "mem" },
      h("div", { cls: "content" }, m.content),
      h("div", { cls: "meta" }, h("span", { cls: "badge" }, m.memory_type), " " + (m.source || "") + " · " + when(m.updated_at)),
      h("div", { cls: "row", style: "margin:4px 0 0" },
        h("button", { cls: "small", onclick: async () => { const c = prompt("Edit", m.content); if (c && c.trim()) { await guard(() => api("/api/project_memories/update", { id: m.id, content: c })); projectDetail(id); } } }, "Edit"),
        h("button", { cls: "small danger", onclick: async () => { if (confirm("Delete?")) { await guard(() => api("/api/project_memories/delete", { id: m.id })); projectDetail(id); } } }, "Delete"))))
      : h("div", { cls: "dim" }, "Nothing yet. Decisions, standards and known issues you mention in this project's conversations are added here."),
    h("h3", {}, "Conversations"),
    r.conversations.length ? r.conversations.map(c => h("div", { cls: "conv", onclick: () => openConv(c.id) }, h("span", { cls: "t" }, c.title), h("span", { cls: "dim" }, c.conversation_type)))
      : h("div", { cls: "dim" }, "None yet.")));
  if (idx.state === "running") setTimeout(() => { if (S.view === "projects") projectDetail(id); }, 1500);
}

// ----------------------------------------------------------- settings

async function loadSettings() {
  renderColibri();
  const r = await guard(() => api("/api/settings"));
  if (!r) return;
  const form = $("#settingsForm");
  form.innerHTML = "";
  const inputs = {};
  for (const s of r.settings) {
    let el;
    if (Array.isArray(s.kind)) { el = h("select", {}, s.kind.map(k => h("option", { value: k }, k))); el.value = s.value; }
    else if (s.kind === "bool") { el = h("input", { type: "checkbox", checked: !!s.value }); }
    else el = h("input", { type: s.kind === "secret" ? "password" : (s.kind === "int" || s.kind === "float") ? "number" : "text",
                           step: s.kind === "float" ? "0.05" : null, value: s.value ?? "", placeholder: s.default === "" ? "" : String(s.default) });
    if (s.source === "environment") { el.disabled = true; el.title = "Set by an environment variable"; }
    inputs[s.key] = { el, s };
    form.append(h("div", { cls: "setting" }, h("label", {}, s.label, h("div", { cls: "src mono" }, s.key)), el, h("span", { cls: "src" }, s.source)));
  }
  form.append(h("div", { cls: "row", style: "margin-top:14px" },
    h("button", { cls: "primary", type: "submit" }, "Save settings"),
    h("span", { cls: "dim" }, "Changing the provider or model keeps every conversation and memory; they live in the local database, not in the model.")));
  form.onsubmit = async e => {
    e.preventDefault();
    const changes = {};
    for (const [k, { el, s }] of Object.entries(inputs)) {
      if (el.disabled) continue;
      const v = el.type === "checkbox" ? el.checked : el.value;
      if (String(v) !== String(s.value ?? "")) changes[k] = v;
    }
    if (!Object.keys(changes).length) { toast("Nothing changed."); return; }
    if (await guard(() => api("/api/settings", { changes }), "Saved.")) { loadSettings(); checkHealth(true); }
  };
}

async function renderColibri() {
  const card = $("#colibriCard");
  card.innerHTML = "";
  card.append(h("h3", { style: "margin-top:0" }, "colibri (inference server)"), h("div", { cls: "dim" }, "Checking…"));
  const st = await guard(() => api("/api/colibri/status"));
  card.innerHTML = "";
  card.append(h("h3", { style: "margin-top:0" }, "colibri (inference server)"));
  if (!st) return;
  if (!st.managed) {
    card.append(h("div", { cls: "warn" }, (st.problems || []).join(" ")),
      h("div", { cls: "dim" }, "vault-chat still talks to any colibri server at the base URL below; this panel only needs the checkout to start, stop and inspect it."));
  } else if (st.error) {
    card.append(h("div", { cls: "err" }, st.error));
  } else {
    card.append(h("div", { cls: "cards" },
      ["state", "model_id", "backend", "openai_base_url", "tokens_per_second", "model_dir"].map(k =>
        h("div", { cls: "card" }, h("b", {}, k.replace(/_/g, " ")), h("span", {}, st[k] == null ? "—" : String(st[k]))))));
  }
  const out = h("div");
  card.append(h("div", { cls: "row", style: "margin-top:10px" },
    st.managed ? h("button", { cls: "small", onclick: () => colibriAction("start") }, "Start (background)") : null,
    st.managed ? h("button", { cls: "small", onclick: () => colibriAction("stop") }, "Stop") : null,
    st.managed ? h("button", { cls: "small", onclick: async () => { const r = await guard(() => api("/api/colibri/logs?n=80")); if (r) { out.innerHTML = ""; out.append(h("pre", { cls: "logs" }, r.logs)); } } }, "Logs") : null,
    st.managed ? h("button", { cls: "small", onclick: async () => {
      out.innerHTML = '<div class="dim">Asking colibri what fits this machine…</div>';
      const r = await guard(() => api("/api/colibri/models"));
      out.innerHTML = "";
      if (r) out.append(h("pre", { cls: "logs" }, JSON.stringify(r.models, null, 1)));
    } }, "Models for this machine") : null,
    h("button", { cls: "small", onclick: async () => { const r = await guard(() => api("/api/colibri/guide")); if (r) { out.innerHTML = ""; out.append(h("pre", { cls: "logs" }, r.guide + (r.problems.length ? "\n! " + r.problems.join("\n! ") : ""))); } } }, "Setup guide")),
    h("div", { cls: "dim" }, "Start and stop never download anything. Installing a model is a separate step you run yourself (see the guide)."), out);
}

async function colibriAction(action) {
  toast(action === "start" ? "Starting colibri; loading the model can take a minute…" : "Stopping colibri…");
  const r = await guard(() => api("/api/colibri/" + action, {}));
  if (r) toast("colibri " + (action === "start" ? "started" : "stopped") + ".");
  checkHealth(true);
  if (S.view === "settings") renderColibri();
}

// ------------------------------------------------------------- health

async function loadHealth() {
  const hb = await checkHealth(true);
  const cards = $("#healthCards");
  cards.innerHTML = "";
  if (hb) {
    const c = hb.colibri || {};
    const items = [
      ["connection", hb.health.ok ? "ready" : (hb.health.state || "down") + (hb.health.detail ? ": " + hb.health.detail : "")],
      ["provider", hb.provider], ["base URL", hb.base_url], ["model", hb.model || "—"],
      ["context length", hb.context_length || "—"], ["compute", c.backend ? c.backend.toUpperCase() + (c.backend === "cpu" ? " only" : "") : "CPU (default)"],
      ["KV cache slots", hb.kv_slots || "off"], ["embeddings", hb.embeddings ? "on (local server)" : "off (keyword search)"],
      ["database", hb.database], ["schema version", hb.schema_version],
      ["conversations", hb.counts.conversations], ["messages", hb.counts.messages], ["memories", hb.counts.memories],
      ["projects", hb.counts.projects], ["open approvals", hb.pending_approvals], ["chars per token", hb.token_ratio.toFixed(2)],
    ];
    for (const [k, v] of items) cards.append(h("div", { cls: "card" }, h("b", {}, k), h("span", { cls: "mono" }, String(v))));
  }
  const m = await guard(() => api("/api/metrics?limit=60"));
  if (m) {
    const s = m.summary;
    $("#metricSummary").innerHTML = "";
    $("#metricSummary").append(h("table", { cls: "grid" },
      h("tr", {}, ["answers (7 days)", "succeeded", "avg first token", "avg tok/s", "avg total", "avg prompt tokens"].map(x => h("th", {}, x))),
      h("tr", {}, [s.n, s.ok || 0, s.ttft != null ? (s.ttft / 1000).toFixed(2) + " s" : "—", s.tps ?? "—",
                   s.total != null ? (s.total / 1000).toFixed(1) + " s" : "—", s.prompt ?? "—"].map(x => h("td", {}, String(x))))));
    const recent = m.recent.slice().reverse();
    $("#charts").innerHTML = "";
    $("#charts").append(chart("tokens / second", recent.map(r => r.tokens_per_second || 0), ""),
                        chart("time to first token (s)", recent.map(r => (r.ttft_ms || 0) / 1000), "s"),
                        chart("prompt tokens", recent.map(r => r.prompt_tokens || 0), ""));
  }
  loadBench();
}

function chart(title, values, unit) {
  const w = 640, ht = 90, n = Math.max(values.length, 1), max = Math.max(...values, 0.001);
  const bw = Math.max(2, Math.floor((w - 20) / n) - 2);
  let bars = "";
  values.forEach((v, i) => {
    const bh = Math.round((v / max) * (ht - 20));
    bars += '<rect x="' + (10 + i * (bw + 2)) + '" y="' + (ht - 6 - bh) + '" width="' + bw + '" height="' + bh +
            '" fill="#7aa2f7" opacity="0.8"><title>' + v.toFixed(2) + unit + "</title></rect>";
  });
  return h("div", { cls: "chart" }, h("div", { cls: "dim" }, title + (values.length ? " · last " + values.length + " · max " + max.toFixed(1) + unit : " · no data yet")),
    h("div", { html: '<svg width="' + w + '" height="' + ht + '" viewBox="0 0 ' + w + " " + ht + '">' + bars + "</svg>" }));
}

async function loadBench() {
  const r = await guard(() => api("/api/bench"));
  if (!r) return;
  $("#benchState").textContent = r.running ? "running… (this takes a few minutes on a CPU)" : r.error ? "last run failed: " + r.error : "";
  $("#benchRun").disabled = r.running;
  const box = $("#benchList");
  box.innerHTML = "";
  if (!r.runs.length) box.append(h("div", { cls: "dim" }, "No benchmark yet. Each provider must be running to be measured; results for the old runtimes (LM Studio, Ollama) appear side by side with colibri."));
  const head = h("tr", {}, ["when", "provider", "model", "prompt", "first token", "tok/s", "total", "prompt tok", "RAM", "CPU avg/peak", "disk", "init"].map(x => h("th", {}, x)));
  const table = h("table", { cls: "grid" }, head);
  for (const run of r.runs) {
    const res = run.results || {};
    if (res.error) { table.append(h("tr", {}, h("td", {}, when(run.created_at)), h("td", {}, run.provider), h("td", { colspan: 10, cls: "err" }, res.error))); continue; }
    for (const [label, p] of Object.entries(res.prompts || {})) {
      table.append(h("tr", {}, [when(run.created_at), run.provider, run.model, label,
        p.ttft_ms != null ? (p.ttft_ms / 1000).toFixed(2) + " s" : "—", p.tokens_per_second ?? "—",
        p.total_ms != null ? (p.total_ms / 1000).toFixed(1) + " s" : "—", p.prompt_tokens ?? "—",
        res.ram_mb != null ? res.ram_mb + " MB" : "—",
        res.cpu_percent_avg != null ? res.cpu_percent_avg + "% / " + res.cpu_percent_peak + "%" : "—",
        res.disk_gb != null ? res.disk_gb + " GB" : "—", res.model_init_seconds != null ? res.model_init_seconds + " s" : "—"].map(x => h("td", {}, String(x)))));
    }
  }
  box.append(table);
  if (r.running) setTimeout(() => { if (S.view === "health") loadBench(); }, 4000);
}

$("#benchRun").onclick = async () => {
  const providers = [...document.querySelectorAll(".benchP:checked")].map(x => x.value);
  if (!providers.length) return;
  if (await guard(() => api("/api/bench/run", { providers, runs: 1 }), "Benchmark started.")) loadBench();
};

// ---------------------------------------------------------- status bar

async function checkHealth(force) {
  let hb = null;
  try { hb = await api("/api/health" + (force ? "?force=1" : "")); }
  catch (e) {
    S.health = null;
    $("#stConn").className = "dot down";
    $("#stProvider").textContent = "vault-chat server unreachable";
    banner("The vault-chat server is not responding. Restart it with ./chat.py; your data is safe in the local database.", null);
    return null;
  }
  S.health = hb;
  const st = hb.health.ok ? "ok" : hb.health.state === "loading" || hb.health.state === "restarting" ? "loading" : "down";
  $("#stConn").className = "dot " + st;
  $("#stProvider").innerHTML = "<b>" + esc(hb.provider) + "</b> " + (st === "ok" ? "connected" : st === "loading" ? "loading model" : "not reachable");
  $("#stModel").innerHTML = "<b>model</b> " + esc(hb.model || "—") + (hb.context_length ? " · " + hb.context_length + " ctx" : "");
  const c = hb.colibri || {};
  $("#stCpu").innerHTML = "<b>compute</b> " + esc(c.backend && c.backend !== "unknown" ? c.backend.toUpperCase() : "CPU");
  $("#stPending").innerHTML = hb.pending_approvals ? '<b class="warn">' + hb.pending_approvals + " awaiting approval</b>" : "";
  if (st === "ok") banner(null);
  else {
    const detail = hb.health.detail || (hb.provider + " is not reachable at " + hb.base_url + ".");
    banner(detail + (hb.health.hint ? " " + hb.health.hint + "." : ""), c.managed);
  }
  return hb;
}

function banner(msg, managed) {
  const b = $("#banner");
  if (!msg) { b.hidden = true; return; }
  b.innerHTML = "";
  b.append(h("span", { cls: "msg" }, msg + " Conversations, history and memory still work."),
    h("button", { cls: "small", onclick: () => checkHealth(true) }, "Retry"),
    managed ? h("button", { cls: "small", onclick: () => colibriAction("start") }, "Start colibri") : null,
    h("button", { cls: "small", onclick: () => show("settings") }, "Settings"));
  b.hidden = false;
}

// --------------------------------------------------------------- boot

setStreaming(false);
show("chat");
checkHealth(true);
setInterval(() => { if (!S.busy) checkHealth(); }, 10000);
setInterval(pollReminders, 30000);
pollReminders();
