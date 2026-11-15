"""The tool registry: every action the assistant can take, with its schema,
its risk flags, and an audit trail in `tool_runs`.

Rules the registry enforces, not the model:

  - a tool runs only if it is on the calling agent's allowlist (five or six
    tools at most: small models choose badly from long menus, and every
    schema costs prefill on every turn)
  - `requires_approval` tools are recorded as pending and run only when the
    user approves them in the UI; nothing external or destructive runs on
    the model's say-so
  - once a run has read untrusted text (notes, code, files), tools that
    touch files are withdrawn for the rest of that run
  - every invocation, approved, rejected or failed, is logged
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import repo
import store as st


@dataclass
class ToolContext:
    store: st.Store
    memory: Any
    project: dict | None = None
    conversation_id: int | None = None
    notes_retriever: Callable | None = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[[ToolContext, dict], Any]
    requires_approval: bool = False
    touches_files: bool = False
    reads_untrusted: bool = False
    needs_project: bool = False

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


class ToolError(Exception):
    pass


def _validate(params: dict, args: dict) -> dict:
    if not isinstance(args, dict):
        raise ToolError("arguments must be an object")
    props = params.get("properties", {})
    for key in params.get("required", []):
        if args.get(key) in (None, ""):
            raise ToolError(f"missing argument '{key}'")
    clean = {}
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            continue  # ignore invented arguments rather than failing
        kind = spec.get("type")
        try:
            if kind == "integer":
                value = int(value)
            elif kind == "number":
                value = float(value)
            elif kind == "string":
                value = str(value)
            elif kind == "boolean":
                value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
        except (TypeError, ValueError):
            raise ToolError(f"argument '{key}' should be {kind}") from None
        if "enum" in spec and value not in spec["enum"]:
            raise ToolError(f"argument '{key}' must be one of {spec['enum']}")
        clean[key] = value
    return clean


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def schemas(self, names: list[str], exclude_file_tools: bool = False,
                project: dict | None = None) -> list[dict]:
        out = []
        for n in names:
            t = self.tools.get(n)
            if not t or (exclude_file_tools and t.touches_files):
                continue
            if t.needs_project and not (project and project.get("repository_path")):
                continue
            out.append(t.schema())
        return out

    def describe(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "requires_approval": t.requires_approval,
                 "touches_files": t.touches_files, "reads_untrusted": t.reads_untrusted,
                 "parameters": t.parameters} for t in self.tools.values()]

    # -- execution

    def _log(self, ctx: ToolContext, name: str, args: dict, status: str, origin: str,
             requires_approval: bool, result=None, error: str | None = None) -> int:
        t = time.time()
        cur = ctx.store.conn.execute(
            "INSERT INTO tool_runs(conversation_id, tool, arguments, requires_approval, status, origin, "
            "result, error, created_at, finished_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ctx.conversation_id, name, json.dumps(args), int(requires_approval), status, origin,
             _dump(result), error, t, None if status == "pending_approval" else t))
        ctx.store.conn.commit()
        return int(cur.lastrowid)

    def invoke(self, name: str, args: dict, ctx: ToolContext, allowlist: list[str] | None = None,
               origin: str = "agent") -> dict:
        tool = self.tools.get(name)
        if tool is None or (allowlist is not None and name not in allowlist):
            err = f"Tool '{name}' is not available here."
            run_id = self._log(ctx, name, args if isinstance(args, dict) else {}, "failed", origin,
                               False, error=err)
            return {"run_id": run_id, "tool": name, "status": "failed", "error": err}
        try:
            clean = _validate(tool.parameters, args)
            if tool.needs_project and not (ctx.project and ctx.project.get("repository_path")):
                raise ToolError("select a project with a repository first")
            warning = check_command(clean.get("command", "")) if name == "run_command" else None
        except ToolError as exc:
            run_id = self._log(ctx, name, args if isinstance(args, dict) else {}, "failed", origin,
                               tool.requires_approval, error=str(exc))
            return {"run_id": run_id, "tool": name, "status": "failed", "error": str(exc)}
        if tool.needs_project:
            clean["_project_id"] = ctx.project["id"]  # so approval later runs in the same project
        if tool.requires_approval:
            run_id = self._log(ctx, name, clean, "pending_approval", origin, True)
            return {"run_id": run_id, "tool": name, "status": "pending_approval", "arguments": clean,
                    "warning": warning}
        return self._execute(tool, clean, ctx, origin)

    def _execute(self, tool: Tool, args: dict, ctx: ToolContext, origin: str, run_id: int | None = None) -> dict:
        try:
            result = tool.handler(ctx, args)
            status, error = "succeeded", None
        except (ToolError, repo.PathError, ValueError) as exc:
            result, status, error = None, "failed", str(exc)
        except Exception as exc:  # a broken tool must not take the turn down
            result, status, error = None, "failed", f"{type(exc).__name__}: {exc}"
        if run_id is None:
            run_id = self._log(ctx, tool.name, args, status, origin, tool.requires_approval, result, error)
        else:
            ctx.store.conn.execute("UPDATE tool_runs SET status = ?, result = ?, error = ?, finished_at = ? "
                                   "WHERE id = ?", (status, _dump(result), error, time.time(), run_id))
            ctx.store.conn.commit()
        out = {"run_id": run_id, "tool": tool.name, "status": status, "arguments": args}
        if error:
            out["error"] = error
        else:
            out["result"] = result
        return out

    def approve(self, run_id: int, ctx: ToolContext) -> dict:
        row = ctx.store.conn.execute("SELECT * FROM tool_runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            raise ToolError("No such tool run.")
        if row["status"] != "pending_approval":
            raise ToolError(f"Tool run is {row['status']}, not pending.")
        tool = self.tools.get(row["tool"])
        if tool is None:
            raise ToolError(f"Tool '{row['tool']}' no longer exists.")
        ctx.store.conn.execute("UPDATE tool_runs SET status = 'running', decided_at = ? WHERE id = ?",
                               (time.time(), run_id))
        ctx.store.conn.commit()
        if ctx.conversation_id is None:
            ctx.conversation_id = row["conversation_id"]
        return self._execute(tool, st.loads(row["arguments"]), ctx, row["origin"], run_id)

    def reject(self, run_id: int, s: st.Store) -> dict:
        cur = s.conn.execute("UPDATE tool_runs SET status = 'rejected', decided_at = ?, finished_at = ? "
                             "WHERE id = ? AND status = 'pending_approval'", (time.time(), time.time(), run_id))
        s.conn.commit()
        if cur.rowcount == 0:
            raise ToolError("No pending tool run with that id.")
        return {"run_id": run_id, "status": "rejected"}

    @staticmethod
    def runs(s: st.Store, status: str | None = None, conversation_id: int | None = None,
             limit: int = 100) -> list[dict]:
        sql, args = "SELECT * FROM tool_runs WHERE 1=1", []
        if status:
            sql += " AND status = ?"
            args.append(status)
        if conversation_id:
            sql += " AND conversation_id = ?"
            args.append(conversation_id)
        rows = s.conn.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["arguments"] = st.loads(d["arguments"])
            d["result"] = st.loads(d["result"], d["result"]) if d["result"] else None
            if d["tool"] == "run_command" and d["status"] == "pending_approval":
                d["warning"] = check_command(d["arguments"].get("command", ""))
            out.append(d)
        return out


def _dump(value) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, default=str)


# --------------------------------------------------------------- dates

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _clock(text: str) -> tuple[int, int] | None:
    m = re.search(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|\bat\s+(\d{1,2}):(\d{2})\b|"
                  r"\bat\s+(\d{1,2})\b(?!\s*(?:minutes?|hours?|days?|weeks?))", text, re.I)
    if not m:
        return None
    if m.group(1):
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
        if h == 12:
            h = 0
        if ap == "pm":
            h += 12
    elif m.group(4):
        h, mi = int(m.group(4)), int(m.group(5))
    else:
        h, mi = int(m.group(6)), 0
    if not (0 <= h < 24 and 0 <= mi < 60):
        return None
    return h, mi


def parse_when(text: str, now: dt.datetime | None = None) -> float | None:
    """Natural-language time to an epoch, or None. Covers 'in 2 hours',
    'tomorrow at 9am', 'today 5pm', 'tonight', 'next friday', 'on monday',
    'at 17:30', and ISO dates like '2026-11-20 14:00'."""
    if not text:
        return None
    now = now or dt.datetime.now()
    t = text.lower().strip()
    m = re.search(r"\bin\s+(\d+|an?|one|two|three|half an?)\s*(minute|min|hour|hr|day|week)s?\b", t)
    if m:
        words = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}
        qty = m.group(1)
        if qty.startswith("half"):
            n, unit = 0.5, m.group(2)
        else:
            n, unit = float(words.get(qty, qty)), m.group(2)
        seconds = {"minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400, "week": 604800}[unit]
        return (now + dt.timedelta(seconds=n * seconds)).timestamp()
    clock = _clock(t)
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})(?:[ t](\d{1,2}):(\d{2}))?", t)
    if m:
        h, mi = (int(m.group(4)), int(m.group(5))) if m.group(4) else (clock or (9, 0))
        try:
            return dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), h, mi).timestamp()
        except ValueError:
            return None
    day = None
    if "tomorrow" in t:
        day = now.date() + dt.timedelta(days=1)
    elif "tonight" in t:
        day = now.date()
        clock = clock or (20, 0)
    elif "today" in t:
        day = now.date()
    else:
        for i, name in enumerate(_WEEKDAYS):
            if re.search(rf"\b(?:next|on|this)?\s*{name}\b", t):
                ahead = (i - now.weekday()) % 7 or 7
                day = now.date() + dt.timedelta(days=ahead)
                break
    if day is None and clock is None:
        return None
    if day is None:
        h, mi = clock
        candidate = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        if candidate <= now:
            candidate += dt.timedelta(days=1)
        return candidate.timestamp()
    h, mi = clock or (9, 0)
    return dt.datetime(day.year, day.month, day.day, h, mi).timestamp()


def fmt_time(ts: float | None) -> str:
    if not ts:
        return ""
    return dt.datetime.fromtimestamp(ts).strftime("%a %d %b %H:%M")


# -------------------------------------------------------- run_command

DESTRUCTIVE = re.compile(
    r"\brm\s+-[a-z]*[rf]|\bgit\s+(?:reset\s+--hard|clean\s+-[a-z]*f|push\s+(?:.*\s)?(?:-f|--force)|"
    r"checkout\s+--\s|branch\s+-D)|\bdrop\s+(?:table|database)\b|\btruncate\b|\bmkfs|\bdd\s+if=|"
    r"\bchmod\s+-R|\bchown\s+-R|>\s*/dev/|\bsf\s+(?:org\s+delete|project\s+deploy)|\bsfdx\s+force:org:delete|"
    r"\bkubectl\s+delete|\bterraform\s+(?:destroy|apply)|\bnpm\s+publish|\bshutdown|\breboot|\bkill(?:all)?\b",
    re.I)
SHELL_META = re.compile(r"[|;&`$<>]|\$\(")


def check_command(command: str) -> str | None:
    """A warning for the approval dialog, or None for ordinary commands."""
    if DESTRUCTIVE.search(command or ""):
        return "This command can delete or overwrite data, or change something outside this machine."
    return None


def _is_program(name: str) -> bool:
    if os.sep in name or name.startswith("."):
        return os.path.isfile(name) and os.access(name, os.X_OK)
    return shutil.which(name) is not None


def _split_command(command: str) -> list[str]:
    """Split one command into arguments. An unquoted program path that itself
    contains spaces (common on macOS) is kept as one argument when that path
    names a real program."""
    argv = shlex.split(command)
    if not argv or _is_program(argv[0]):
        return argv
    for end in range(1, len(argv)):
        candidate = " ".join(argv[: end + 1])
        if _is_program(candidate):
            return [candidate, *argv[end + 1 :]]
    return argv


def _run_command(ctx: ToolContext, args: dict) -> dict:
    command = args["command"].strip()
    if SHELL_META.search(command):
        raise ToolError("Shell operators (| ; & > $ `) are not supported; run one program at a time.")
    try:
        argv = _split_command(command)
    except ValueError as exc:
        raise ToolError(f"Cannot parse the command: {exc}") from None
    if not argv:
        raise ToolError("Empty command.")
    if argv[0] in ("sudo", "su", "doas"):
        raise ToolError("Commands that need elevated privileges are refused.")
    root = Path(ctx.project["repository_path"]).expanduser().resolve()
    if not root.is_dir():
        raise ToolError(f"Repository folder not found: {root}")
    started = time.time()
    try:
        proc = subprocess.run(argv, cwd=root, capture_output=True, text=True,
                              timeout=int(args.get("timeout") or 120), stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        raise ToolError(f"Program not found: {argv[0]}") from None
    except subprocess.TimeoutExpired:
        raise ToolError("The command timed out.") from None
    out = (proc.stdout or "")[-12000:]
    err = (proc.stderr or "")[-6000:]
    return {"exit_code": proc.returncode, "stdout": out, "stderr": err,
            "seconds": round(time.time() - started, 2), "cwd": str(root)}


# ------------------------------------------------------------ builtins


def _search_memory(ctx, a):
    hits = ctx.memory.search(a["query"], limit=6)
    return [{"id": m["id"], "category": m["category"], "content": m["content"]} for m in hits] or "No matching memories."


def _save_memory(ctx, a):
    res = ctx.memory.add(a["content"], a.get("category", "fact"), origin="explicit",
                         source_conversation_id=ctx.conversation_id)
    if res["action"] == "refused":
        raise ToolError(f"Not saved: {res['reason']}")
    return f"Memory {res['action']}: {res['content']}"


def _search_conversations(ctx, a):
    hits = ctx.store.search_conversations(a["query"], limit=6, exclude_id=ctx.conversation_id)
    return [{"conversation_id": h["conversation_id"], "title": h["title"], "snippet": h["snippet"]}
            for h in hits] or "No matching conversations."


def _search_notes(ctx, a):
    if not ctx.notes_retriever:
        return "No notes index is available."
    return ctx.notes_retriever(a["query"], 5) or "No matching notes."


def _add_task(ctx, a):
    due = parse_when(a.get("due", "")) if a.get("due") else None
    remind = parse_when(a.get("remind", "")) if a.get("remind") else due
    tid = ctx.store.add_task(a["title"], a.get("notes", ""), due, remind,
                             (ctx.project or {}).get("id"))
    when = f" (due {fmt_time(due)})" if due else ""
    return f"Task #{tid} added: {a['title']}{when}"


def _list_tasks(ctx, a):
    tasks = ctx.store.list_tasks(a.get("status", "open"), limit=30)
    if not tasks:
        return "No tasks."
    return [{"id": t["id"], "title": t["title"], "status": t["status"],
             "due": fmt_time(t["due_at"]) or None} for t in tasks]


def _complete_task(ctx, a):
    task = ctx.store.get_task(a["task_id"])
    if not task:
        raise ToolError(f"No task #{a['task_id']}.")
    ctx.store.update_task(a["task_id"], status="done")
    return f"Task #{task['id']} marked done: {task['title']}"


def _search_code(ctx, a):
    hits = repo.retrieve(ctx.store, ctx.project["id"], a["query"], k=5)
    return [{"path": h["path"], "lines": f"{h['start_line']}-{h['end_line']}", "symbol": h["symbol"],
             "content": h["content"][:1500]} for h in hits] or "No matching code."


def _read_file(ctx, a):
    return repo.read_file(ctx.project["repository_path"], a["path"], int(a.get("start_line") or 1),
                          int(a["end_line"]) if a.get("end_line") else None, max_chars=12000)


def builtin_registry() -> ToolRegistry:
    r = ToolRegistry()
    s = {"type": "string"}
    r.register(Tool("search_memory", "Search what is remembered about the user (preferences, projects, "
                    "decisions).", {"type": "object", "properties": {"query": s}, "required": ["query"]},
                    _search_memory))
    r.register(Tool("save_memory", "Remember a durable fact the user stated about themselves or their work.",
                    {"type": "object", "properties": {"content": s, "category": {
                        "type": "string", "enum": ["preference", "project", "decision", "goal", "technical",
                                                   "fact"]}}, "required": ["content"]}, _save_memory))
    r.register(Tool("search_conversations", "Find earlier conversations about a topic.",
                    {"type": "object", "properties": {"query": s}, "required": ["query"]},
                    _search_conversations))
    r.register(Tool("search_notes", "Search the user's markdown notes vault.",
                    {"type": "object", "properties": {"query": s}, "required": ["query"]},
                    _search_notes, reads_untrusted=True))
    r.register(Tool("add_task", "Create a task or reminder. `due`/`remind` accept phrases like "
                    "'tomorrow 9am' or 'in 2 hours'.",
                    {"type": "object", "properties": {"title": s, "due": s, "remind": s, "notes": s},
                     "required": ["title"]}, _add_task))
    r.register(Tool("list_tasks", "List the user's tasks.",
                    {"type": "object", "properties": {"status": {"type": "string", "enum": ["open", "done"]}}},
                    _list_tasks))
    r.register(Tool("complete_task", "Mark a task as done by its number.",
                    {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]},
                    _complete_task))
    r.register(Tool("search_code", "Search the active project's repository.",
                    {"type": "object", "properties": {"query": s}, "required": ["query"]},
                    _search_code, reads_untrusted=True, needs_project=True))
    r.register(Tool("read_file", "Read part of a file in the active project's repository.",
                    {"type": "object", "properties": {"path": s, "start_line": {"type": "integer"},
                                                      "end_line": {"type": "integer"}}, "required": ["path"]},
                    _read_file, reads_untrusted=True, touches_files=True, needs_project=True))
    r.register(Tool("run_command", "Run a single program (tests, linters, builds) in the project folder. "
                    "Always needs the user's approval.",
                    {"type": "object", "properties": {"command": s, "timeout": {"type": "integer"}},
                     "required": ["command"]},
                    _run_command, requires_approval=True, touches_files=True, needs_project=True))
    return r
