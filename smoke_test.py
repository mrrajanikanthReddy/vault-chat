"""Offline smoke test. Stubs out the model so it needs no runtime and no
model downloads — run it first on a new machine to prove the plumbing works.

    python3 smoke_test.py

Exercises indexing, incremental updates, deletion, retrieval ranking, the
per-file cap, review scheduling, agent profiles, and the tool allowlist.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VAULT = ROOT / ".smoketest-vault"

shutil.rmtree(VAULT, ignore_errors=True)
(VAULT / "notes").mkdir(parents=True)
os.environ["VAULT_DIR"] = str(VAULT)
os.environ["VAULT_DB"] = str(VAULT / ".index" / "vault.db")
os.environ["VAULT_BACKEND"] = "ollama"  # pinned so nothing probes the network
os.environ["VAULT_EMBED_MODEL"] = "stub-embed"
sys.path.insert(0, str(ROOT))

import numpy as np
import vaultdb as v

# --- stub the model ------------------------------------------------------
DIM = 96


def fake_embed(texts):
    """Hashed bag-of-words. Similar text really does land close together, so
    retrieval ordering is meaningfully exercised.

    crc32 rather than hash(): the built-in is salted per process, which would
    make scores — and therefore any threshold assertion — differ run to run.
    """
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for word in t.lower().split():
            word = "".join(c for c in word if c.isalnum())
            if word:
                out[i, zlib.crc32(word.encode()) % DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-9, None)


v.embed = fake_embed
v.chat_stream = lambda *a, **k: iter(["(stubbed answer) "])
v.chat = lambda *a, **k: "Q: What is the retry budget?\nA: Three attempts."

import agent
import ask
import index
import quiz

for module in (ask, index, agent):
    module.v.embed = fake_embed

failures = []


def check(label, condition, detail=""):
    mark = "PASS" if condition else "FAIL"
    print(f"  {mark}  {label}{(' — ' + detail) if detail and not condition else ''}")
    if not condition:
        failures.append(label)


def raises(fn) -> str | None:
    try:
        fn()
    except SystemExit as exc:
        return str(exc)
    return None


# --- fixture -------------------------------------------------------------
(VAULT / "notes" / "retries.md").write_text(
    "# Retry policy\n\n"
    "We retry idempotent requests three times with exponential backoff.\n"
    "Non-idempotent writes are never retried automatically.\n\n"
    "## Backoff\n\n"
    "Base delay is 200ms, doubling each attempt, with jitter.\n"
)
(VAULT / "notes" / "kafka.md").write_text(
    "# Kafka consumers\n\n"
    "Consumer lag alerts fire above 10000 messages.\n"
    + ("Partitions are rebalanced on deploy which causes a brief stall. " * 40)
)
(VAULT / "inbox.md").write_text("Random thought about gardening and tomatoes.\n")
(VAULT / ".index").mkdir(exist_ok=True)
(VAULT / ".index" / "should-be-ignored.md").write_text("# hidden\n\nmust not be indexed\n")

# --- 0. backend selection ------------------------------------------------
print("\n[0] backend configuration")
check("VAULT_BACKEND is respected without probing", v.backend() == "ollama", v.backend())
check("embed model honours the env override", v.embed_model() == "stub-embed")
check("ollama chat default", v.default_model("chat") == "qwen2.5-coder:7b")
check("lmstudio chat default differs",
      v.DEFAULT_MODELS["lmstudio"]["chat"] == "qwen2.5-coder-7b-instruct")
check("agent endpoint refuses non-lmstudio backends",
      "lmstudio" in (raises(lambda: v.agent_chat("hi", integrations=[{"id": "x"}])) or ""))

# --- 1. initial index ----------------------------------------------------
print("\n[1] initial index")
sys.argv = ["index.py", "--quiet"]
index.main()
conn = v.connect()
n_files = conn.execute("SELECT COUNT(*) n FROM files").fetchone()["n"]
n_chunks = conn.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
check("indexed 3 files", n_files == 3, f"got {n_files}")
check("produced chunks", n_chunks >= 5, f"got {n_chunks}")
check("dot-directory excluded",
      conn.execute("SELECT COUNT(*) n FROM chunks WHERE path LIKE '.index%'").fetchone()["n"] == 0)
check("long note was split",
      conn.execute("SELECT COUNT(*) n FROM chunks WHERE path='notes/kafka.md'").fetchone()["n"] > 1)
check("headings captured",
      conn.execute("SELECT COUNT(*) n FROM chunks WHERE heading='Backoff'").fetchone()["n"] == 1)

# --- 2. incremental ------------------------------------------------------
print("\n[2] incremental re-index (no changes)")
before = conn.execute("SELECT MAX(id) m FROM chunks").fetchone()["m"]
conn.close()
index.main()
conn = v.connect()
after = conn.execute("SELECT MAX(id) m FROM chunks").fetchone()["m"]
check("unchanged files not re-embedded", before == after, f"{before} -> {after}")

print("\n[3] edit one file")
(VAULT / "inbox.md").write_text("Updated thought about retries and timeouts.\n")
conn.close()
index.main()
conn = v.connect()
txt = conn.execute("SELECT text FROM chunks WHERE path='inbox.md'").fetchone()["text"]
check("edited file re-embedded", "timeouts" in txt, txt[:40])

print("\n[4] delete one file")
(VAULT / "notes" / "kafka.md").unlink()
conn.close()
index.main()
conn = v.connect()
check("removed file purged from index",
      conn.execute("SELECT COUNT(*) n FROM chunks WHERE path='notes/kafka.md'").fetchone()["n"] == 0)
check("file row purged too",
      conn.execute("SELECT COUNT(*) n FROM files WHERE path='notes/kafka.md'").fetchone()["n"] == 0)

# --- 5. retrieval --------------------------------------------------------
print("\n[5] retrieval")
hits = ask.retrieve(conn, "how many times do we retry requests", 6, 0.0)
check("retrieval returned hits", len(hits) > 0, f"got {len(hits)}")
if hits:
    check("most relevant note ranked first", hits[0][1]["path"] == "notes/retries.md",
          hits[0][1]["path"])
    check("scores descend", all(hits[i][0] >= hits[i + 1][0] for i in range(len(hits) - 1)))

context, included = ask.build_context(hits, 200)
check("context respects the char budget", len(context) <= 400, f"{len(context)} chars")
check("at least one passage always included", len(included) >= 1)
check("citation markers present", context.startswith("[1] "))

# --- 6. per-file cap -----------------------------------------------------
print("\n[6] per-file cap")
(VAULT / "notes" / "long.md").write_text(
    "# Retries everywhere\n\n"
    + "\n\n".join(f"Retry attempt paragraph {i} about retrying requests repeatedly." * 12
                  for i in range(8))
)
conn.close()
index.main()
conn = v.connect()
counts = {}
for _, row in ask.retrieve(conn, "retry retrying requests", 6, 0.0):
    counts[row["path"]] = counts.get(row["path"], 0) + 1
check("no file exceeds 2 passages", all(c <= 2 for c in counts.values()), str(counts))

# --- 7. quiz scheduling --------------------------------------------------
print("\n[7] quiz scheduling")
check("all notes due initially", len(quiz.due_notes(conn)) == 3, str(len(quiz.due_notes(conn))))
quiz.record(conn, "notes/retries.md", remembered=True)
check("first success schedules 1 day",
      conn.execute("SELECT interval_days d FROM reviews WHERE path='notes/retries.md'").fetchone()["d"] == 1.0)
quiz.record(conn, "notes/retries.md", remembered=True)
check("second success climbs the ladder",
      conn.execute("SELECT interval_days d FROM reviews WHERE path='notes/retries.md'").fetchone()["d"] == 3.0)
quiz.record(conn, "notes/retries.md", remembered=False)
check("a miss resets to 1 day",
      conn.execute("SELECT interval_days d FROM reviews WHERE path='notes/retries.md'").fetchone()["d"] == 1.0)
check("reviewed note no longer due", "notes/retries.md" not in quiz.due_notes(conn))
check("question parsing handles sloppy formatting",
      len(quiz.parse_pairs("Q: Budget?\nA: Three.\n**Q2:** Backoff?\n**A:** 200ms.")) == 2)

# --- 8. agent profiles and the tool allowlist ----------------------------
print("\n[8] agent profiles")
profiles = agent.load_profiles()
check("default profiles load", {"ask", "code", "tools"} <= set(profiles))
check("ask profile is vault-grounded", profiles["ask"]["vault"] is True)
check("code profile takes no tools", profiles["code"]["integrations"] == [])
check("shipped tools profile is allowlisted",
      all(i.get("allowed_tools") for i in profiles["tools"]["integrations"]))

check("bare-string integration is refused",
      "allowlist" in (raises(lambda: agent.check_allowlists({"integrations": ["mcp/github"]}, False)) or ""))
check("missing allowed_tools is refused",
      "allowed_tools" in (raises(lambda: agent.check_allowlists(
          {"integrations": [{"type": "plugin", "id": "mcp/github"}]}, False)) or ""))
check("--allow-any overrides the refusal",
      raises(lambda: agent.check_allowlists(
          {"integrations": [{"type": "plugin", "id": "mcp/github"}]}, True)) is None)
check("a proper allowlist passes",
      raises(lambda: agent.check_allowlists(
          {"integrations": [{"type": "plugin", "id": "mcp/github",
                             "allowed_tools": ["search_code"]}]}, False)) is None)

# --- 9. agent response parsing -------------------------------------------
print("\n[9] agent response parsing")
sample = {
    "output": [
        {"type": "reasoning", "content": "deciding which tool"},
        {"type": "tool_call", "tool": "search_code", "arguments": {"q": "retry"}, "output": "3 hits"},
        {"type": "message", "content": "You retry three times."},
    ],
    "stats": {"input_tokens": 419, "total_output_tokens": 362,
              "tokens_per_second": 27.6, "time_to_first_token_seconds": 1.4},
}
check("transcript yields only the message text",
      agent.render(sample, show_reasoning=False) == "You retry three times.")
check("openai-shaped fallback still parses",
      agent.render({"choices": [{"message": {"content": "fallback"}}]}, False) == "fallback")
check("unknown shape does not crash",
      isinstance(agent.render({"weird": True}, False), str))

# --- 10. prompt assembly and saving --------------------------------------
print("\n[10] prompt assembly")
args = argparse.Namespace(file=None, no_vault=False, k=6, min_score=0.0,
                          max_context_chars=6000, max_file_chars=12000)
prompt = agent.build_prompt("how many retries", profiles["ask"], args, conn)
check("vault notes are injected", "Notes:" in prompt and "retry" in prompt.lower())
check("question comes last", prompt.strip().endswith("Question: how many retries"))

args.no_vault = True
check("--no-vault skips retrieval",
      "Notes:" not in agent.build_prompt("how many retries", profiles["ask"], args, conn))

probe = VAULT / "probe.py"
probe.write_text("def handler():\n    return retry(3)\n")
args.file = [str(probe)]
check("attached files are embedded in the prompt",
      "def handler" in agent.build_prompt("explain", profiles["code"], args, conn))

agent.save_to_daily("a question", "an answer")
daily = list((VAULT / "daily").glob("*.md"))
check("--save writes a dated daily note", len(daily) == 1, str(daily))
check("daily note contains the exchange",
      "an answer" in daily[0].read_text() if daily else False)

# --- 11. embed-model guard -----------------------------------------------
print("\n[11] embed model change guard")
conn.close()
os.environ["VAULT_EMBED_MODEL"] = "some-other-model"
check("model change blocks indexing",
      "rebuild" in (raises(index.main) or "").lower())
os.environ["VAULT_EMBED_MODEL"] = "stub-embed"

# --- 12. chat server -----------------------------------------------------
print("\n[12] chat web server")
import threading
import urllib.request

import chat

chat.v.chat_stream = lambda *a, **k: iter(["Three ", "attempts."])

httpd = chat.Server(("127.0.0.1", 0), chat.Handler)
httpd.profile = "ask"
httpd.verbose = False
httpd.min_score = 0.0  # the stub embedder scores lower than real ones
port = httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{port}"


def get(path):
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return r.status, r.read().decode()


def sse(body):
    req = urllib.request.Request(
        base + "/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    events = []
    with urllib.request.urlopen(req, timeout=15) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
    return events


status, page = get("/")
check("serves the chat page", status == 200 and "<title>Vault Chat</title>" in page)
check("page has no external resources",
      "http://" not in page.split("<style>")[1].split("</script>")[0].replace(
          "http://www.w3.org", "") or "cdn" not in page.lower())

status, raw = get("/api/meta")
meta = json.loads(raw)
check("meta reports the backend", meta["backend"] == "ollama", meta["backend"])
check("meta lists profiles", "ask" in meta["profiles"] and "code" in meta["profiles"])
check("meta counts indexed passages", meta["chunks"] > 0, str(meta["chunks"]))

events = sse({"profile": "ask", "messages": [{"role": "user", "content": "how many retries"}]})
kinds = [e["type"] for e in events]
check("stream emits citations before tokens",
      "citations" in kinds and kinds.index("citations") < kinds.index("token"), str(kinds))
check("stream emits the answer in pieces", kinds.count("token") == 2, str(kinds))
check("answer reassembles correctly",
      "".join(e["text"] for e in events if e["type"] == "token") == "Three attempts.")
check("stream terminates with done", kinds[-1] == "done", str(kinds[-1]))
cites = [e for e in events if e["type"] == "citations"][0]["items"]
check("citations carry a source and score",
      bool(cites) and "where" in cites[0] and "score" in cites[0], str(cites[:1]))

events = sse({"profile": "code", "messages": [{"role": "user", "content": "hi"}]})
check("profiles without vault send no citations",
      "citations" not in [e["type"] for e in events])

events = sse({"profile": "nope", "messages": [{"role": "user", "content": "hi"}]})
check("unknown profile returns an error event",
      events and events[0]["type"] == "error", str(events[:1]))

events = sse({"profile": "ask", "messages": []})
check("empty message is rejected", events and events[0]["type"] == "error")

try:
    get("/nope")
    check("unknown path 404s", False, "no error raised")
except urllib.error.HTTPError as exc:
    check("unknown path 404s", exc.code == 404, str(exc.code))

# --- saved conversations over HTTP
def post(path, body, ctype="application/json"):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": ctype}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def saved_ids():
    return [x["id"] for x in json.loads(get("/api/sessions")[1])]


before = set(saved_ids())
events = sse({"profile": "ask", "save": True, "session": None,
              "messages": [{"role": "user", "content": "how many retries"}]})
announced = [e["id"] for e in events if e["type"] == "session"]
check("saving a chat announces its session id", len(announced) == 1, str([e["type"] for e in events]))
sid = announced[0] if announced else -1
check("saved chat appears in the history list", sid in saved_ids())
saved = json.loads(get(f"/api/session?id={sid}")[1])
check("stored turn is question then answer",
      [m["role"] for m in saved["messages"]] == ["user", "assistant"], str(saved["messages"]))
check("stored question is the plain question, not the notes prompt",
      saved["messages"][0]["content"] == "how many retries")
check("stored answer matches what streamed", saved["messages"][1]["content"] == "Three attempts.")
check("session title comes from the first question", saved["title"] == "how many retries")

seen = {}


def capture(msgs, **kw):
    seen["msgs"] = msgs
    return iter(["Same budget."])


previous_stream = chat.v.chat_stream
chat.v.chat_stream = capture
events = sse({"profile": "ask", "save": True, "session": sid, "messages":
              saved["messages"] + [{"role": "user", "content": "and for writes?"}]})
chat.v.chat_stream = previous_stream
check("follow-up stays in the same session",
      [e["id"] for e in events if e["type"] == "session"] == [sid])
check("model is sent the earlier turns",
      [m["role"] for m in seen.get("msgs", [])] == ["system", "user", "assistant", "user"],
      str([m["role"] for m in seen.get("msgs", [])]))
check("both turns are stored",
      len(json.loads(get(f"/api/session?id={sid}")[1])["messages"]) == 4)

events = sse({"profile": "ask", "save": True, "session": 987654,
              "messages": [{"role": "user", "content": "a fresh start"}]})
new_id = [e["id"] for e in events if e["type"] == "session"]
check("an unknown session id starts a new conversation",
      len(new_id) == 1 and new_id[0] != 987654)

count_before = len(saved_ids())
chat.v.chat_stream = lambda *a, **k: iter([])
events = sse({"profile": "ask", "save": True, "session": None,
              "messages": [{"role": "user", "content": "this gets no answer"}]})
chat.v.chat_stream = previous_stream
check("a conversation that produced nothing is not left behind",
      len(saved_ids()) == count_before)

events = sse({"profile": "ask", "messages": [{"role": "user", "content": "unsaved"}]})
check("requests without save stay stateless",
      "session" not in [e["type"] for e in events] and len(saved_ids()) == count_before)

try:
    post("/api/session/delete", {"id": sid}, ctype="text/plain")
    check("delete refuses a non-JSON content type", False, "no error raised")
except urllib.error.HTTPError as exc:
    check("delete refuses a non-JSON content type", exc.code == 415, str(exc.code))
check("refused delete changed nothing", sid in saved_ids())

status, body = post("/api/session/delete", {"id": sid})
check("delete removes the conversation", status == 200 and sid not in saved_ids())
try:
    get(f"/api/session?id={sid}")
    check("deleted session 404s", False, "no error raised")
except urllib.error.HTTPError as exc:
    check("deleted session 404s", exc.code == 404, str(exc.code))
conn = v.connect()
check("deleting a session removes its messages",
      conn.execute("SELECT COUNT(*) n FROM messages WHERE session_id=?", (sid,)).fetchone()["n"] == 0)
conn.close()

httpd.shutdown()
httpd.server_close()

# --- 13. installer helpers -----------------------------------------------
print("\n[13] installer")
import install

check("every launcher maps to a real script",
      all((install.HERE / s).exists() for s in install.COMMANDS.values()),
      str(install.COMMANDS))
check("venv python path is platform-correct",
      str(install.venv_python()).endswith("python.exe" if sys.platform == "win32" else "python"))
check("config dir matches the one vaultdb reads",
      install.config_dir() == v.config_dir(), f"{install.config_dir()} vs {v.config_dir()}")
check("runtime help covers all three platforms",
      {"Linux", "Darwin", "Windows"} == set(install.RUNTIME_HELP))
# The project was renamed from vault-tools; settings under the old name must
# keep working, and the installer must carry them across intact.
import tempfile
saved_env = {k: os.environ.get(k) for k in ("APPDATA", "XDG_CONFIG_HOME")}
with tempfile.TemporaryDirectory() as tmp:
    os.environ["APPDATA"] = os.environ["XDG_CONFIG_HOME"] = tmp
    check("fresh install uses the vault-chat folder",
          install.config_dir().name == "vault-chat" and v.config_dir() == install.config_dir())
    legacy = Path(tmp) / "vault-tools"
    legacy.mkdir()
    (legacy / "config.json").write_text('{"VAULT_DIR": "/kept"}')
    check("old settings folder is still honoured before migration",
          install.config_dir() == legacy and v.config_dir() == legacy)
    install.migrate_legacy_config()
    check("migration moves the folder to vault-chat",
          not legacy.exists() and install.config_dir().name == "vault-chat")
    check("migration keeps the settings intact",
          '"/kept"' in (install.config_dir() / "config.json").read_text())
    install.migrate_legacy_config()
    check("migrating twice is harmless", install.config_dir().name == "vault-chat")
for key, value in saved_env.items():
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value

check("scheduler names cover all three platforms",
      all(install._scheduler_name(p) != "the system scheduler"
          for p in ("Linux", "Darwin", "Windows")))

# --- 14. hardware detection and model choice -----------------------------
print("\n[14] hardware detection")
import hardware as hw

GB = hw.GB


def machine(ram_gb, bandwidth, cores=4, gpu_vram_gb=None, unified=False):
    gpus = []
    if gpu_vram_gb:
        gpus = [{"vendor": "nvidia", "name": "test", "vram": int(gpu_vram_gb * GB),
                 "unified": unified}]
    return {"ram_total": int(ram_gb * GB), "bandwidth_gbps": bandwidth,
            "physical_cores": cores, "logical_cores": cores * 2, "gpus": gpus}


check("catalog entries name a model for both runtimes",
      all(m.get("ollama") and m.get("lmstudio") and m.get("quants") for m in hw.CATALOG))
check("every quant has a size and an ordering",
      all(q in hw.QUANT_ORDER for m in hw.CATALOG for q in m["quants"]))

# Mixture-of-experts is the whole reason a 32 GB CPU box is viable: it should
# predict far faster than a dense model of the same file size.
moe = next(m for m in hw.CATALOG if m["key"] == "moe-large")
dense = next(m for m in hw.CATALOG if m["key"] == "medium")
moe_speed = hw.predict_tokens_per_second(moe, "Q4_K_M", 22.0)
dense_speed = hw.predict_tokens_per_second(dense, "Q4_K_M", 22.0)
check("MoE predicts faster than a larger-on-disk dense model",
      moe_speed > dense_speed * 2, f"moe {moe_speed} vs dense {dense_speed}")
check("bandwidth drives the prediction",
      hw.predict_tokens_per_second(dense, "Q4_K_M", 44.0) > dense_speed * 1.9)
check("no bandwidth means no fabricated number",
      hw.predict_tokens_per_second(dense, "Q4_K_M", None) is None)

big = hw.recommend(machine(32, 22.0, cores=8))
check("32 GB CPU box is given the MoE model", big["chat"]["key"] == "moe-large",
      big["chat"]["key"])
check("its recommendation clears the responsiveness bar",
      big["chat"]["tokens_per_second"] >= hw.RESPONSIVE,
      str(big["chat"]["tokens_per_second"]))
check("model fits inside the memory budget",
      big["chat"]["size_gb"] <= big["budget_gb"],
      f"{big['chat']['size_gb']} vs {big['budget_gb']}")
check("threads leave one core for the system", big["threads"] == 7, str(big["threads"]))

small = hw.recommend(machine(8, 12.0, cores=2))
check("8 GB box still gets a usable model", small["chat"] is not None)
check("8 GB box is not handed a 30B model", small["chat"]["size_gb"] < 5,
      str(small["chat"]["size_gb"]))
check("small core count keeps every core", small["threads"] == 2, str(small["threads"]))

# The flaw worth testing for: when nothing is fast, pick the quickest option
# rather than the biggest one that happens to fit.
slow = hw.recommend(machine(16, 4.0, cores=4))
alternatives = slow["alternatives"]
check("slow machine optimises for speed, not size",
      all((slow["chat"]["tokens_per_second"] or 0) >= (a["tokens_per_second"] or 0)
          for a in alternatives),
      f"{slow['chat']['label']} {slow['chat']['tokens_per_second']}")

gpu_box = hw.recommend(machine(16, 20.0, cores=8, gpu_vram_gb=24))
check("a discrete GPU switches placement", gpu_box["placement"] == "gpu")
check("GPU budget comes from VRAM, not system RAM", gpu_box["budget_gb"] > 16,
      str(gpu_box["budget_gb"]))
check("GPU box enables autocomplete", gpu_box["autocomplete"] is True)
check("CPU box with slow memory disables autocomplete",
      hw.recommend(machine(16, 6.0))["autocomplete"] is False)

check("context shrinks when memory is tight",
      hw.recommend(machine(8, 12.0))["context_length"] <= 8192)
check("context grows when memory is plentiful",
      hw.recommend(machine(128, 60.0, cores=16))["context_length"] >= 16384)
check("unified memory is not treated as separate VRAM",
      hw.recommend(machine(16, 50.0, gpu_vram_gb=16, unified=True))["placement"] == "ram")

probe_info = hw.collect(quick=True, disk_path=str(ROOT))
check("collect returns a full report without the benchmark",
      {"platform", "cpu", "ram_total", "gpus", "recommendation"} <= set(probe_info))
check("--quick really skips the benchmark", probe_info["bandwidth_gbps"] is None)
check("disk free space is reported", (probe_info["disk_free"] or 0) > 0)

# These probe the real machine. They exist because a thin PATH once made every
# macOS sysctl call fail silently, so RAM fell back to an assumed 8 GB and a
# 32 GB laptop was recommended a 3B model.
check("total RAM is actually detected", (probe_info["ram_total"] or 0) > 1 * GB,
      str(probe_info["ram_total"]))
check("RAM total is at least as large as RAM available",
      (probe_info["ram_total"] or 0) >= (probe_info["ram_available"] or 0),
      f"{probe_info['ram_total']} vs {probe_info['ram_available']}")
check("physical cores are detected", (probe_info["physical_cores"] or 0) > 0,
      str(probe_info["physical_cores"]))
check("physical cores do not exceed logical",
      (probe_info["physical_cores"] or 1) <= probe_info["logical_cores"])
check("cpu name is not a placeholder",
      probe_info["cpu"] not in ("i386", "unknown", ""), probe_info["cpu"])
check("a missing binary probe returns None rather than raising",
      hw._run(["definitely-not-a-real-binary-xyz"]) is None)
check("probe resolution finds binaries that exist",
      (hw._run(["ls", "/"]) is not None) if sys.platform != "win32" else True)
check("report renders without raising", hw.print_report(probe_info) is None)
check("hardware has a launcher", "vault-hardware" in install.COMMANDS)

# --- 15. conversation history ---------------------------------------------
print("\n[15] conversation history")
import contextlib
import io

conn = v.connect()
first = v.create_session(conn, "ask", "  a   long\nquestion  ")
check("title collapses whitespace", v.get_session(conn, first)["title"] == "a long question")
check("long titles are shortened", len(v.session_title("x" * 200)) <= 61)
v.add_messages(conn, first, [("user", "q1"), ("assistant", "a1")])
v.add_messages(conn, first, [("user", "q2"), ("assistant", "a2")])
check("messages come back in order",
      [m["content"] for m in v.session_messages(conn, first)] == ["q1", "a1", "q2", "a2"])
check("latest session is the one touched last", v.latest_session_id(conn) == first)

# The index can be rebuilt freely; the conversations must survive it.
conn.close()
sys.argv = ["index.py", "--rebuild", "--quiet"]
index.main()
conn = v.connect()
check("index --rebuild leaves conversations alone",
      len(v.session_messages(conn, first)) == 4)

turns = [{"role": r, "content": "x" * 100} for r in ("user", "assistant") * 5]
trimmed = agent.trim_history(turns, 250)
check("trim keeps the newest whole turn", len(trimmed) == 2 and trimmed[-1] is turns[-1])
check("trimmed history opens on a user message",
      all(agent.trim_history(turns, b)[:1] in ([], [turns[0]], [turns[2]], [turns[4]],
                                               [turns[6]], [turns[8]]) for b in (90, 150, 250, 450)))
check("a roomy budget keeps everything", len(agent.trim_history(turns, 10_000)) == 10)
check("nothing in, nothing out", agent.trim_history([], 100) == [])

convo = agent.Conversation(conn, "code", save=True)
convo.remember("why twice?", "Because of the retry wrapper.")
check("first turn creates a saved conversation", convo.id is not None)
resumed = agent.Conversation(conn, "code", save=True, resume=convo.id)
check("resuming loads the earlier turns", len(resumed.messages) == 2)
volatile = agent.Conversation(conn, "code", save=False)
volatile.remember("q", "a")
check("--no-history keeps turns in memory but not in the database",
      volatile.id is None and len(volatile.messages) == 2
      and v.latest_session_id(conn) == convo.id)

run_args = argparse.Namespace(file=None, no_vault=True, k=6, min_score=0.0,
                              max_context_chars=6000, max_file_chars=12000, model=None,
                              save=False, json=False, history_chars=4000,
                              context_length=8192, show_reasoning=False)
seen = {}


def capture(msgs, **kw):
    seen["msgs"] = msgs
    return iter(["ok"])


previous_stream = v.chat_stream
v.chat_stream = capture
with contextlib.redirect_stdout(io.StringIO()):
    agent.run("and now?", profiles["code"], run_args, conn, resumed)
v.chat_stream = previous_stream
check("agent sends earlier turns to the model",
      [m["role"] for m in seen["msgs"]] == ["system", "user", "assistant", "user"],
      str([m["role"] for m in seen.get("msgs", [])]))
run_args.history_chars = 0
v.chat_stream = capture
with contextlib.redirect_stdout(io.StringIO()):
    agent.run("and now?", profiles["code"], run_args, conn, resumed)
v.chat_stream = previous_stream
check("a zero history budget sends no earlier turns",
      [m["role"] for m in seen["msgs"]] == ["system", "user"])

# End to end through main(): continue the latest conversation with a follow-up.
v.chat_stream = lambda *a, **k: iter(["A follow-up answer."])
sys.argv = ["agent.py", "-c", "--no-vault", "what about timeouts?"]
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    agent.main()
v.chat_stream = previous_stream
check("-c appends to the latest conversation",
      len(v.session_messages(conn, convo.id)) == 4, str(len(v.session_messages(conn, convo.id))))
check("the follow-up was stored as asked",
      v.session_messages(conn, convo.id)[2]["content"] == "what about timeouts?")

sys.argv = ["agent.py", "--sessions"]
listing = io.StringIO()
with contextlib.redirect_stdout(listing):
    agent.main()
check("--sessions lists saved conversations", "turns" in listing.getvalue(), listing.getvalue()[:80])

sys.argv = ["agent.py", "--resume", "424242", "hello"]
check("resuming a missing conversation is a clear error",
      "No saved conversation" in (raises(agent.main) or ""))

v.delete_session(conn, first)
check("deleting a conversation removes it and its messages",
      v.get_session(conn, first) is None and v.session_messages(conn, first) == [])
conn.close()

shutil.rmtree(VAULT, ignore_errors=True)
print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all checks passed")
