"""Agent orchestration: one turn, from the user's message to a stored answer.

    message ─▶ conversation ─▶ persist user turn ─▶ memory hook ─▶ intent
            ─▶ actions (deterministic first, then a bounded tool loop)
            ─▶ context assembly ─▶ stream from the provider ─▶ persist reply
            ─▶ diff proposal (coding) ─▶ done

The application owns persistence, memory and approvals; the model only ever
produces text or tool requests. Every step reports progress as an event, so
the web UI and tests see exactly what happened.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import repo
import store as st
from context import AGENT_PROMPTS, ContextAssembler, TokenCounter
from memory import MemoryManager
from providers import ColibriProvider, EmbeddingProvider, GenParams, LLMProvider, ProviderError, make_provider
from tools import ToolContext, ToolError, builtin_registry, parse_when

# --------------------------------------------------------------- settings

SETTINGS = [
    # key, default, kind, label
    ("LLM_PROVIDER", "colibri", ["colibri", "lmstudio", "ollama", "openai"], "Inference provider"),
    ("COLIBRI_BASE_URL", "http://127.0.0.1:8000/v1", "url", "colibri OpenAI-compatible base URL"),
    ("COLIBRI_API_KEY", "", "secret", "colibri API key (only if COLI_API_KEY is set on the server)"),
    ("COLIBRI_KV_SLOTS", 0, "int", "KV slots the server was started with (--kv-slots); 0 or 1 = off"),
    ("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1", "url", "LM Studio base URL (optional provider)"),
    ("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1", "url", "Ollama base URL (optional provider)"),
    ("OPENAI_COMPAT_BASE_URL", "", "url", "Other OpenAI-compatible local server"),
    ("OPENAI_COMPAT_API_KEY", "", "secret", "API key for the other server"),
    ("LLM_MODEL", "", "text", "Model id (empty = whatever the server has loaded)"),
    ("LLM_TIMEOUT", 600, "int", "Seconds to wait for a reply"),
    ("TEMPERATURE", 0.2, "float", "Temperature (chat and agent)"),
    ("CODING_TEMPERATURE", 0.1, "float", "Temperature (coding)"),
    ("TOP_P", 0.0, "float", "top_p (0 = server default)"),
    ("MAX_TOKENS", 1024, "int", "Maximum output tokens"),
    ("CONTEXT_BUDGET_TOKENS", 6144, "int", "Prompt budget; smaller is faster on a CPU"),
    ("MEMORY_AUTO_EXTRACT", True, "bool", "Extract memories automatically from your messages"),
    ("MEMORY_LLM_EXTRACTION", False, "bool", "Also ask the model to extract memories (one extra call a turn)"),
    ("USE_NOTES", True, "bool", "Ground chat and agent answers in the notes vault"),
    ("VAULT_EMBED_URL", "", "url", "Optional embeddings server for semantic search (/v1 base URL)"),
    ("VAULT_EMBED_MODEL", "nomic-embed-text", "text", "Embedding model on that server"),
    ("COLIBRI_DIR", "", "text", "Path to your colibri checkout (for start/stop/status)"),
    ("COLIBRI_PYTHON", "", "text", "Python 3.10+ used to run colibri's `coli` (empty = auto)"),
]
SETTING_DEFAULTS = {k: d for k, d, _, _ in SETTINGS}
SECRET_KEYS = {k for k, _, kind, _ in SETTINGS if kind == "secret"}

AGENT_TOOLS = {
    "chat": [],
    "coding": ["search_code", "read_file", "run_command"],
    "agent": ["search_memory", "save_memory", "search_conversations", "add_task", "list_tasks",
              "complete_task"],
}
MODE_TYPE = {"chat": "chat", "coding": "coding", "agent": "agent"}


class Platform:
    """Long-lived services shared by every request."""

    def __init__(self, store_path: Path | str | None = None, notes_options: dict | None = None):
        self.store = st.Store(store_path)
        self.counter = TokenCounter(float(self.store.get_setting("TOKEN_RATIO", 4.0)))
        self.memory = MemoryManager(self.store, self._embedder())
        self.registry = builtin_registry()
        self.assembler = ContextAssembler(self.store, self.memory, self.counter)
        self.notes_options = {"k": 6, "min_score": 0.25, "max_context_chars": 6000, **(notes_options or {})}
        self._provider: LLMProvider | None = None
        self._provider_key: tuple | None = None
        self._lock = threading.Lock()
        self.memory.expire()

    def setting(self, key: str):
        return self.store.get_setting(key, SETTING_DEFAULTS.get(key))

    def settings(self) -> list[dict]:
        out = []
        for key, default, kind, label in SETTINGS:
            value = self.setting(key)
            if key in SECRET_KEYS and value:
                value = "********"
            out.append({"key": key, "value": value, "default": default, "kind": kind, "label": label,
                        "source": self.store.setting_source(key)})
        return out

    def update_settings(self, changes: dict) -> None:
        for key, value in changes.items():
            if key not in SETTING_DEFAULTS:
                raise ValueError(f"Unknown setting {key}")
            if key in SECRET_KEYS and value == "********":
                continue
            default = SETTING_DEFAULTS[key]
            if isinstance(default, bool):
                value = value in (True, "true", "1", 1, "on")
            elif isinstance(default, int):
                value = int(value or 0)
            elif isinstance(default, float):
                value = float(value or 0)
            self.store.set_setting(key, value)
        if "VAULT_EMBED_URL" in changes or "VAULT_EMBED_MODEL" in changes:
            self.memory.embedder = self._embedder()

    def _embedder(self) -> EmbeddingProvider | None:
        url = self.store.get_setting("VAULT_EMBED_URL", "")
        if not url:
            return None
        return EmbeddingProvider(url, self.store.get_setting("VAULT_EMBED_MODEL", "nomic-embed-text"))

    def provider(self) -> LLMProvider:
        name = self.setting("LLM_PROVIDER")
        url_key = {"colibri": "COLIBRI_BASE_URL", "lmstudio": "LMSTUDIO_BASE_URL",
                   "ollama": "OLLAMA_BASE_URL"}.get(name, "OPENAI_COMPAT_BASE_URL")
        key = (name, self.setting(url_key), self.setting("LLM_MODEL"),
               self.setting("COLIBRI_API_KEY") if name == "colibri" else self.setting("OPENAI_COMPAT_API_KEY"),
               self.setting("LLM_TIMEOUT"), self.setting("COLIBRI_KV_SLOTS"))
        with self._lock:
            if self._provider is None or key != self._provider_key:
                self._provider = make_provider(name, key[1], key[2], key[3], float(key[4]), int(key[5]))
                self._provider_key = key
            return self._provider

    def params(self, mode: str) -> GenParams:
        top_p = float(self.setting("TOP_P") or 0)
        temp = self.setting("CODING_TEMPERATURE") if mode == "coding" else self.setting("TEMPERATURE")
        return GenParams(temperature=float(temp), top_p=top_p or None,
                         max_tokens=int(self.setting("MAX_TOKENS") or 0) or None)

    def notes_retriever(self, question: str, k: int) -> list[dict]:
        """Passages from the notes vault, or [] when there is no index."""
        import ask
        import vaultdb as v
        try:
            conn = v.connect()
        except Exception:
            return []
        try:
            if conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0:
                return []
            hits = ask.retrieve(conn, question, min(k, self.notes_options["k"]),
                                self.notes_options["min_score"])
            _, included = ask.build_context(hits, self.notes_options["max_context_chars"])
        except (SystemExit, Exception):
            return []
        finally:
            conn.close()
        return [{"where": r["path"] + (f" › {r['heading']}" if r["heading"] else ""), "text": r["text"],
                 "score": round(score, 2)} for score, r in included]

    def remember_token_ratio(self) -> None:
        self.store.set_setting("TOKEN_RATIO", self.counter.ratio)


# ---------------------------------------------------------------- intent


@dataclass
class Intent:
    kind: str           # conversational, coding, action, recall, planning
    signals: list
    confidence: float

    def as_dict(self) -> dict:
        return {"kind": self.kind, "signals": self.signals, "confidence": self.confidence}


TASK_ADD = re.compile(r"^(?:please\s+)?(?:remind me (?:to|about)|add (?:a |an )?(?:task|todo|to-do|reminder)"
                      r"(?: to)?|create (?:a |an )?(?:task|todo|reminder)(?: to)?|new task[:]?|todo[:])\s*[:,-]?\s*"
                      r"(?P<rest>.+)$", re.I)
TASK_LIST = re.compile(r"\b(?:list|show|what are|what's on)\b.*\b(?:tasks|todos|to-dos|reminders|to do list)\b"
                       r"|^(?:my )?(?:tasks|todos|reminders)\??$", re.I)
TASK_DONE = re.compile(r"\b(?:mark|complete|finish|close|tick off)\s+(?:task\s+)?#?(?P<id>\d+)"
                       r"|\btask\s+#?(?P<id2>\d+)\s+(?:is\s+)?(?:done|complete|finished)", re.I)
RUN_CMD = re.compile(r"^(?:please\s+)?run\s+(?:the\s+command\s+)?`(?P<cmd>[^`]+)`|"
                     r"^(?:please\s+)?run\s+command[:\s]+(?P<cmd2>.+)$", re.I)
WHEN_TAIL = re.compile(r"\s+(?P<when>(?:in\s+\d+|in\s+an?\s|in\s+half|at\s+\d|on\s+\w+day|on\s+\d{4}-|"
                       r"tomorrow|today|tonight|next\s+\w+day|this\s+\w+day|by\s+\w+).*)$", re.I)

ACTION_WORDS = re.compile(r"\b(?:remind me|add (?:a )?task|create (?:a )?task|schedule|set (?:a )?reminder|"
                          r"mark .{1,40} (?:done|complete)|todo|to-do|remember that|save (?:this|that))\b", re.I)
RECALL_WORDS = re.compile(r"\b(?:what did (?:we|i|you)|did we (?:discuss|talk|decide|say)|remember when|last time|"
                          r"previous(?:ly)?|earlier (?:we|you|i)|what (?:project|am i|was i)|recap|"
                          r"what (?:have|were) (?:we|i)|our (?:previous|last|earlier) (?:discussion|conversation|chat)"
                          r"|catch me up|where (?:did|were) we)\b", re.I)
PLAN_WORDS = re.compile(r"\b(?:plan|roadmap|implementation plan|next steps|strategy|break (?:it|this) down|"
                        r"milestones?|timeline|approach for)\b", re.I)
CODE_WORDS = re.compile(r"```|Traceback \(most recent|Exception:|Error:|\bat [\w.$]+\([\w]+\.\w+:\d+\)|"
                        r"\b(?:function|class|method|refactor|bug|stack ?trace|compile|unit tests?|regex|apex|lwc|"
                        r"trigger|soql|python|javascript|typescript|java|sql|null pointer|exception|"
                        r"endpoint|api|debug)\b|\b\w+\.(?:py|js|ts|tsx|java|cls|trigger|html|css|sql)\b", re.I)


def detect_intent(text: str, mode: str) -> Intent:
    signals = []
    if TASK_ADD.search(text) or TASK_LIST.search(text) or TASK_DONE.search(text) or RUN_CMD.search(text):
        return Intent("action", ["task or command request"], 0.95)
    if ACTION_WORDS.search(text):
        signals.append("action words")
    if RECALL_WORDS.search(text):
        return Intent("recall", signals + ["refers to earlier work"], 0.8)
    if PLAN_WORDS.search(text):
        return Intent("planning", signals + ["planning request"], 0.75)
    if signals:
        return Intent("action", signals, 0.6)
    if mode == "coding" or CODE_WORDS.search(text):
        return Intent("coding", ["code signals"] if mode != "coding" else ["coding workspace"], 0.7)
    return Intent("conversational", [], 0.6)


def plan_actions(text: str, mode: str) -> list[tuple[str, dict]]:
    """Requests clear enough to execute without asking the model, which
    costs nothing and cannot pick the wrong tool."""
    m = RUN_CMD.search(text)
    if m:
        return [("run_command", {"command": (m.group("cmd") or m.group("cmd2")).strip()})]
    m = TASK_DONE.search(text)
    if m:
        return [("complete_task", {"task_id": int(m.group("id") or m.group("id2"))})]
    if TASK_LIST.search(text):
        return [("list_tasks", {"status": "open"})]
    m = TASK_ADD.search(text)
    if m:
        rest = m.group("rest").strip().rstrip(".")
        when = None
        tail = WHEN_TAIL.search(rest)
        if tail and parse_when(tail.group("when")):
            when = tail.group("when")
            rest = rest[: tail.start()].strip()
        args = {"title": rest[:1].upper() + rest[1:]}
        if when:
            args["due"] = when
            if text.lower().lstrip().startswith(("remind", "please remind")):
                args["remind"] = when
        return [("add_task", args)]
    return []


def describe_result(res: dict) -> str:
    name = res["tool"]
    if res["status"] == "pending_approval":
        return f"{name} is waiting for the user's approval (run #{res['run_id']})."
    if res["status"] == "failed":
        return f"{name} failed: {res.get('error')}"
    result = res.get("result")
    if isinstance(result, (list, dict)):
        result = json.dumps(result, default=str)[:1500]
    return f"{name}: {result}"


# ---------------------------------------------------------- orchestrator


TOOL_LOOP_PROMPT = """You can call tools to act for the user. Call a tool only when the request needs it; otherwise answer directly. Never invent tool results."""


class Orchestrator:
    def __init__(self, platform: Platform):
        self.p = platform

    # -- the bounded tool loop (personal agent)

    def _tool_loop(self, provider: LLMProvider, text: str, history: list[dict], ctx: ToolContext,
                   allowlist: list[str], max_steps: int = 3):
        messages = [{"role": "system", "content": AGENT_PROMPTS["agent"] + "\n\n" + TOOL_LOOP_PROMPT}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history[-4:]
                     if m["role"] in ("user", "assistant")]
        messages.append({"role": "user", "content": text})
        results: list[str] = []
        untrusted = False
        for _ in range(max_steps):
            tools = self.p.registry.schemas(allowlist, exclude_file_tools=untrusted, project=ctx.project)
            if not tools:
                break
            try:
                out = provider.complete(messages, GenParams(temperature=0.1, max_tokens=400), tools=tools)
            except ProviderError as exc:
                if exc.kind == "request":  # this engine does not do tool calls; answer without them
                    yield {"type": "status", "text": "Tool calling is not supported by this model; answering directly."}
                    return results, None
                raise
            calls = out.get("tool_calls") or []
            if not calls:
                return results, (out.get("content") or None) if not results else None
            messages.append({"role": "assistant", "content": out.get("content") or "", "tool_calls": calls})
            pending = False
            for call in calls[:3]:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                res = self.p.registry.invoke(fn.get("name") or "", args, ctx, allowlist=allowlist)
                yield self._action_event(res)
                tool = self.p.registry.get(res["tool"])
                if tool and tool.reads_untrusted and res["status"] == "succeeded":
                    untrusted = True
                results.append(describe_result(res))
                messages.append({"role": "tool", "tool_call_id": call.get("id") or res["tool"],
                                 "content": describe_result(res)})
                pending = pending or res["status"] == "pending_approval"
            if pending:
                break
        return results, None

    def _action_event(self, res: dict) -> dict:
        event = {"type": "approval" if res["status"] == "pending_approval" else "action",
                 "tool": res["tool"], "status": res["status"], "run_id": res["run_id"],
                 "arguments": res.get("arguments"), "summary": describe_result(res)}
        if res.get("warning"):
            event["warning"] = res["warning"]
        if res.get("result") is not None:
            event["result"] = res["result"]
        return event

    # -- one turn

    def run_turn(self, req: dict):
        s, p = self.p.store, self.p
        mode = req.get("mode") if req.get("mode") in MODE_TYPE else "chat"
        text = (req.get("message") or "").strip()
        regenerate = bool(req.get("regenerate"))

        conv = None
        if req.get("conversation_id"):
            try:
                conv = s.get_conversation(int(req["conversation_id"]))
            except (TypeError, ValueError):
                conv = None
        project = None
        pid = req.get("project_id") or (conv or {}).get("project_id")
        if pid:
            project = s.get_project(int(pid))

        if conv is None:
            if regenerate or not text:
                yield {"type": "error", "kind": "request", "message": "Nothing to answer.", "retryable": False}
                return
            cid = s.create_conversation(text, MODE_TYPE[mode], project["id"] if project else None)
            conv = s.get_conversation(cid)
            created = True
        else:
            cid, created = conv["id"], False
            if project and conv.get("project_id") != project["id"]:
                s.update_conversation(cid, project_id=project["id"])
        yield {"type": "conversation", "id": cid, "title": conv["title"], "created": created,
               "conversation_type": conv["conversation_type"], "project_id": project["id"] if project else None}

        history = s.messages(cid)
        memory_report = None
        if regenerate:
            while history and history[-1]["role"] != "user":
                s.delete_message(history.pop()["id"])
            if not history:
                yield {"type": "error", "kind": "request", "message": "No question to regenerate.",
                       "retryable": False}
                return
            user = history.pop()
            text, user_mid = user["content"], user["id"]
        else:
            if not text:
                yield {"type": "error", "kind": "request", "message": "Empty message.", "retryable": False}
                return
            user_mid = s.add_message(cid, "user", text, {"mode": mode})
            yield {"type": "user_message", "id": user_mid}
            memory_report = p.memory.process_user_message(
                text, cid, user_mid, project["id"] if project else None,
                auto=bool(p.setting("MEMORY_AUTO_EXTRACT")))

        intent = detect_intent(text, mode)
        yield {"type": "intent", "agent": mode, **intent.as_dict()}

        notes = p.notes_retriever if (p.setting("USE_NOTES") and mode in ("chat", "agent")) else None
        ctx = ToolContext(s, p.memory, project, cid, notes)
        action_results: list[str] = []
        if not regenerate:
            for name, args in plan_actions(text, mode):
                res = p.registry.invoke(name, args, ctx, origin="user")
                yield self._action_event(res)
                action_results.append(describe_result(res))

        provider = p.provider()
        params = p.params(mode)
        direct_answer = None
        try:
            if mode == "agent" and intent.kind == "action" and not action_results and not regenerate:
                yield {"type": "status", "text": "Choosing an action"}
                loop = self._tool_loop(provider, text, history, ctx, AGENT_TOOLS["agent"])
                action_results, direct_answer = yield from loop

            code_retriever = None
            if mode == "coding" and project and project.get("repository_path"):
                mentioned = [f for f in (req.get("files") or []) if isinstance(f, str)]
                code_retriever = (lambda q, k: repo.retrieve(s, project["id"], q, k, mentioned))
            attachments = []
            if project and project.get("repository_path"):
                for rel in (req.get("files") or [])[:5]:
                    try:
                        f = repo.read_file(project["repository_path"], rel, max_chars=20000)
                        attachments.append({"path": rel, "content": f["content"], "language": f["language"]})
                    except (repo.PathError, ValueError) as exc:
                        yield {"type": "status", "text": f"Skipped {rel}: {exc}"}
            if req.get("code"):
                attachments.append({"path": "(pasted code)", "content": str(req["code"])[:20000], "language": ""})

            if direct_answer is None:
                yield {"type": "status", "text": "Gathering context"}
                assembled = p.assembler.assemble(
                    conversation_id=cid, agent=mode, question=text, history=history,
                    context_length=provider.context_length(), max_tokens=params.max_tokens,
                    cap_tokens=int(p.setting("CONTEXT_BUDGET_TOKENS")), project=project,
                    code_retriever=code_retriever, notes_retriever=notes,
                    recall=intent.kind == "recall" or mode == "agent",
                    planning=intent.kind == "planning", action_results=action_results,
                    attachments=attachments or None, provider=provider, params_cls=GenParams)
                if assembled.summarized_now:
                    yield {"type": "status", "text": "Summarized earlier turns to fit the context window"}
                yield {"type": "context", "sources": assembled.sources, "prompt_tokens": assembled.prompt_tokens,
                       "budget": {"context_length": assembled.budget.context_length,
                                  "available": assembled.budget.available, "used": assembled.budget.used},
                       "history_kept": assembled.history_kept, "history_total": assembled.history_total,
                       "summary_used": assembled.summary_used}
        except ProviderError as exc:
            s.update_message(user_mid, metadata={"unanswered": True, "error": str(exc)})
            yield {"type": "error", **exc.as_dict()}
            return

        if memory_report and (memory_report["saved"] or memory_report["removed"] or memory_report["refused"]):
            yield {"type": "memory", **memory_report}

        parts: list[str] = []
        stream = None
        error: ProviderError | None = None
        stopped = False
        message_id = None
        stats: dict = {}
        try:
            if direct_answer:
                parts.append(direct_answer)
                yield {"type": "token", "text": direct_answer}
            else:
                yield {"type": "status", "text": "Generating"}
                if isinstance(provider, ColibriProvider):
                    stream = provider.stream_chat(assembled.messages, params, conversation_id=cid)
                else:
                    stream = provider.stream_chat(assembled.messages, params)
                for piece in stream:
                    parts.append(piece)
                    yield {"type": "token", "text": piece}
        except ProviderError as exc:
            error = exc
        except GeneratorExit:
            stopped = True
            raise
        finally:
            if stream is not None:
                stream.close()
                stats = stream.stats.as_dict()
                if not stopped and error is None:
                    self.p.counter.observe(stream.stats.prompt_chars, stream.stats.prompt_tokens)
            answer = "".join(parts).strip()
            meta = {"mode": mode, "intent": intent.kind, "provider": provider.name,
                    "model": stats.get("model") or provider.active_model(), "stats": stats,
                    "sources": [] if direct_answer else assembled.sources[:20]}
            if stopped:
                meta["stopped"] = True
            if error:
                meta["error"] = str(error)
            if answer:
                message_id = s.add_message(cid, "assistant", answer, meta,
                                           stats.get("completion_tokens") or None)
            else:
                s.update_message(user_mid, metadata={"unanswered": True,
                                                     "error": str(error) if error else "stopped"})
            if stats:
                s.record_metric("chat", stats, cid, ok=error is None and not stopped,
                                error=str(error) if error else None)
                p.remember_token_ratio()

        if error:
            yield {"type": "error", **error.as_dict(), "partial": bool(parts)}
        if stats:
            yield {"type": "stats", **stats}

        answer = "".join(parts)
        if mode == "coding" and project and project.get("repository_path") and answer:
            edits = repo.parse_edit_blocks(answer)
            if edits:
                try:
                    prop = repo.build_proposal(s, project["id"], edits, text[:200], cid, message_id)
                    yield {"type": "proposal", **prop}
                except (ValueError, repo.PathError) as exc:
                    yield {"type": "proposal_error", "message": str(exc)}

        if answer and not regenerate and p.setting("MEMORY_LLM_EXTRACTION"):
            threading.Thread(target=p.memory.llm_extract, args=(provider, text, cid), daemon=True).start()

        yield {"type": "done", "message_id": message_id, "conversation_id": cid}


def approve_tool(platform: Platform, run_id: int) -> dict:
    """Run a pending tool in the project and conversation it was requested in."""
    s = platform.store
    row = s.conn.execute("SELECT conversation_id, arguments FROM tool_runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise ToolError("No such tool run.")
    conv = s.get_conversation(row["conversation_id"]) if row["conversation_id"] else None
    pid = st.loads(row["arguments"]).get("_project_id") or (conv or {}).get("project_id")
    project = s.get_project(int(pid)) if pid else None
    ctx = ToolContext(s, platform.memory, project, conv["id"] if conv else None, platform.notes_retriever)
    return platform.registry.approve(run_id, ctx)
