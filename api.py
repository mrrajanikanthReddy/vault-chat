"""HTTP routes for the web UI, independent of the server plumbing in chat.py.

GET routes read; every change is a POST with a JSON body (chat.py enforces
the content type, which forces browsers to preflight cross-site requests that
this server never approves). Approvals are separate endpoints, so nothing the
model writes can approve its own action.
"""

from __future__ import annotations

import threading
import time

import bench
import repo
import store as st
from colibri import BUNDLED, Colibri, ColibriError, GUIDE, summarize_status
from memory import CATEGORIES, PROJECT_TYPES
from orchestrator import Orchestrator, Platform, approve_tool
from tools import ToolContext, ToolError, fmt_time, parse_when


class NotFound(Exception):
    pass


class BadRequest(Exception):
    pass


def _int(value, name: str = "id") -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise BadRequest(f"'{name}' must be a number") from None


def _q(query: dict, key: str, default=None):
    value = query.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    return default if value in (None, "") else value


class API:
    def __init__(self, platform: Platform):
        self.p = platform
        self.o = Orchestrator(platform)
        self._colibri_cache: tuple[float, dict] | None = None
        self._bench_state = {"running": False, "error": None}

    @property
    def s(self) -> st.Store:
        return self.p.store

    # ---------------------------------------------------------- GET

    def get(self, route: str, q: dict):
        handler = {
            "/api/health": self.health,
            "/api/settings": lambda q: {"settings": self.p.settings()},
            "/api/conversations": self.conversations,
            "/api/conversation": self.conversation,
            "/api/memories": self.memories,
            "/api/memories/export": lambda q: self.p.memory.export(),
            "/api/topics": lambda q: {"topics": self.s.frequent_topics(12)},
            "/api/projects": lambda q: {"projects": self.s.list_projects()},
            "/api/project": self.project,
            "/api/project/files": lambda q: {"files": repo.list_files(self.s, _int(_q(q, "id")), _q(q, "q", ""))},
            "/api/project/file": self.project_file,
            "/api/proposals": lambda q: {"proposals": repo.list_proposals(
                self.s, _q(q, "project_id") and _int(_q(q, "project_id")), _q(q, "status"))},
            "/api/proposal": self.proposal,
            "/api/tools": lambda q: {"tools": self.p.registry.describe()},
            "/api/tools/runs": lambda q: {"runs": self.p.registry.runs(
                self.s, _q(q, "status"), _q(q, "conversation_id") and _int(_q(q, "conversation_id")),
                int(_q(q, "limit", 100)))},
            "/api/tasks": lambda q: {"tasks": [self._task(t) for t in self.s.list_tasks(_q(q, "status"), 200)]},
            "/api/reminders/due": self.due_reminders,
            "/api/metrics": lambda q: {"recent": self.s.recent_metrics(int(_q(q, "limit", 60))),
                                       "summary": self.s.metric_summary()},
            "/api/bench": lambda q: {"runs": self.s.list_bench(), **self._bench_state},
            "/api/colibri/status": lambda q: self.colibri_status(force=True),
            "/api/colibri/models": lambda q: {"models": self._colibri().models()},
            "/api/colibri/hardware": lambda q: self._colibri().hardware(),
            "/api/colibri/logs": lambda q: {"logs": self._colibri().logs(int(_q(q, "n", 80)),
                                                                          _q(q, "which", "server"))},
            "/api/colibri/guide": lambda q: self.colibri_guide(),
        }.get(route)
        if handler is None:
            raise NotFound(route)
        return handler(q)

    # ---------------------------------------------------------- POST

    def post(self, route: str, b: dict):
        handler = {
            "/api/settings": self.save_settings,
            "/api/conversations/create": lambda b: {"id": self.s.create_conversation(
                b.get("title", ""), b.get("type", "chat"), b.get("project_id"))},
            "/api/conversations/update": self.update_conversation,
            "/api/conversations/delete": lambda b: {"ok": self.s.delete_conversation(_int(b.get("id")))},
            "/api/memories/create": self.create_memory,
            "/api/memories/update": self.update_memory,
            "/api/memories/delete": lambda b: {"ok": self.p.memory.delete(_int(b.get("id")))},
            "/api/memories/delete_all": self.delete_all_memories,
            "/api/memories/import": lambda b: self.p.memory.import_(b.get("data") or {}),
            "/api/projects/create": self.create_project,
            "/api/projects/update": self.update_project,
            "/api/projects/delete": lambda b: {"ok": self.s.delete_project(_int(b.get("id")))},
            "/api/projects/index": self.index_project,
            "/api/project_memories/create": self.create_project_memory,
            "/api/project_memories/update": lambda b: {"ok": self.p.memory.update_project_memory(
                _int(b.get("id")), **{k: b[k] for k in ("content", "memory_type") if k in b})},
            "/api/project_memories/delete": lambda b: {"ok": self.p.memory.delete_project_memory(_int(b.get("id")))},
            "/api/proposals/apply": lambda b: repo.apply_proposal(self.s, _int(b.get("id"))),
            "/api/proposals/reject": lambda b: repo.reject_proposal(self.s, _int(b.get("id"))),
            "/api/proposals/revert": lambda b: repo.revert_proposal(self.s, _int(b.get("id"))),
            "/api/tools/run": self.run_tool,
            "/api/tools/approve": lambda b: approve_tool(self.p, _int(b.get("run_id"), "run_id")),
            "/api/tools/reject": lambda b: self.p.registry.reject(_int(b.get("run_id"), "run_id"), self.s),
            "/api/tasks/create": self.create_task,
            "/api/tasks/update": self.update_task,
            "/api/tasks/delete": lambda b: {"ok": self.s.delete_task(_int(b.get("id")))},
            "/api/bench/run": self.run_bench,
            "/api/colibri/start": lambda b: self._colibri_action("start"),
            "/api/colibri/stop": lambda b: self._colibri_action("stop"),
        }.get(route)
        if handler is None:
            raise NotFound(route)
        try:
            return handler(b)
        except (ToolError, repo.PathError, ColibriError) as exc:
            raise BadRequest(str(exc)) from exc
        except ValueError as exc:
            raise BadRequest(str(exc)) from exc

    def turn(self, body: dict):
        return self.o.run_turn(body)

    # ------------------------------------------------------ handlers

    def health(self, q: dict) -> dict:
        provider = self.p.provider()
        h = provider.health(force=_q(q, "force") == "1")
        info = {
            "provider": provider.name, "base_url": getattr(provider, "base_url", ""), "health": h,
            "model": h.get("model") or provider.active_model() if h.get("ok") else h.get("model"),
            "context_length": h.get("context_length"),
            "kv_slots": getattr(provider, "kv_slots", 0),
            "database": str(self.s.path), "schema_version": self.s.schema_version(),
            "counts": {k: self.s.conn.execute(f"SELECT COUNT(*) FROM {k}").fetchone()[0]
                       for k in ("conversations", "messages", "memories", "projects", "tasks")},
            "pending_approvals": self.s.conn.execute(
                "SELECT COUNT(*) FROM tool_runs WHERE status = 'pending_approval'").fetchone()[0]
                + self.s.conn.execute("SELECT COUNT(*) FROM change_proposals WHERE status = 'pending'").fetchone()[0],
            "embeddings": bool(self.p.memory.embedder),
            "token_ratio": self.p.counter.ratio,
        }
        if provider.name == "colibri":
            info["colibri"] = self.colibri_status()
        return info

    def colibri_guide(self) -> dict:
        c = self._colibri()
        return {"guide": GUIDE.format(py=c.python or "python3.12", root=c.dir or BUNDLED),
                "problems": c.problems()}

    def _colibri(self) -> Colibri:
        return Colibri(self.p.setting("COLIBRI_DIR") or "", self.p.setting("COLIBRI_PYTHON") or "")

    def colibri_status(self, force: bool = False) -> dict:
        now = time.time()
        if not force and self._colibri_cache and now - self._colibri_cache[0] < 20:
            return self._colibri_cache[1]
        c = self._colibri()
        if not c.ready:
            out = {"managed": False, "problems": c.problems()}
        else:
            try:
                out = {"managed": True, **summarize_status(c.status())}
            except ColibriError as exc:
                out = {"managed": True, "error": str(exc)}
        self._colibri_cache = (now, out)
        return out

    def _colibri_action(self, action: str) -> dict:
        c = self._colibri()
        result = c.start() if action == "start" else c.stop()
        self._colibri_cache = None
        self.p.provider().health(force=True)
        return {"ok": True, "result": result}

    def save_settings(self, b: dict) -> dict:
        self.p.update_settings(b.get("changes") or {})
        return {"settings": self.p.settings()}

    def conversations(self, q: dict) -> dict:
        query = _q(q, "q", "")
        if query:
            return {"results": self.s.search_conversations(query, int(_q(q, "limit", 40)))}
        pid = _q(q, "project_id")
        return {"conversations": self.s.list_conversations(
            _q(q, "type"), _int(pid) if pid else None, _q(q, "tag"), _q(q, "category"),
            int(_q(q, "limit", 100)), int(_q(q, "offset", 0)))}

    def conversation(self, q: dict) -> dict:
        cid = _int(_q(q, "id"))
        conv = self.s.get_conversation(cid)
        if not conv:
            raise NotFound("conversation")
        return {"conversation": conv, "messages": self.s.messages(cid),
                "summary": self.s.latest_summary(cid),
                "proposals": [p for p in repo.list_proposals(self.s, limit=200) if p["conversation_id"] == cid],
                "runs": self.p.registry.runs(self.s, conversation_id=cid, limit=50)}

    def update_conversation(self, b: dict) -> dict:
        cid = _int(b.get("id"))
        if not self.s.get_conversation(cid):
            raise NotFound("conversation")
        self.s.update_conversation(cid, **{k: b[k] for k in ("title", "category", "tags", "project_id") if k in b})
        return {"conversation": self.s.get_conversation(cid)}

    def memories(self, q: dict) -> dict:
        items = self.p.memory.list(_q(q, "category"), _q(q, "type"), _q(q, "q", ""), int(_q(q, "limit", 300)))
        for m in items:
            conv = self.s.get_conversation(m["source_conversation_id"]) if m.get("source_conversation_id") else None
            m["source_title"] = conv["title"] if conv else None
        return {"memories": items, "categories": list(CATEGORIES),
                "auto_extract": self.p.setting("MEMORY_AUTO_EXTRACT"),
                "llm_extraction": self.p.setting("MEMORY_LLM_EXTRACTION")}

    def create_memory(self, b: dict) -> dict:
        res = self.p.memory.add(b.get("content", ""), b.get("category", "fact"), b.get("memory_type", "long_term"),
                                b.get("importance"), 1.0, "manual", pinned=bool(b.get("pinned")),
                                allow_sensitive=bool(b.get("allow_sensitive")))
        if res["action"] == "refused":
            raise BadRequest(f"Not saved: {res['reason']}. Tick 'store anyway' to keep it.")
        return res

    def update_memory(self, b: dict) -> dict:
        mid = _int(b.get("id"))
        fields = {k: b[k] for k in ("content", "category", "importance", "pinned", "memory_type") if k in b}
        if not self.p.memory.update(mid, **fields):
            raise NotFound("memory")
        return {"memory": self.p.memory.get(mid)}

    def delete_all_memories(self, b: dict) -> dict:
        if b.get("confirm") is not True:
            raise BadRequest("Send {\"confirm\": true} to delete every memory.")
        return {"deleted": self.p.memory.delete_all()}

    def project(self, q: dict) -> dict:
        pid = _int(_q(q, "id"))
        project = self.s.get_project(pid)
        if not project:
            raise NotFound("project")
        return {"project": project, "memories": self.p.memory.list_project_memories(pid),
                "memory_types": list(PROJECT_TYPES),
                "index": repo.job_status(pid),
                "files": self.s.conn.execute("SELECT COUNT(*) FROM code_files WHERE project_id = ?",
                                             (pid,)).fetchone()[0],
                "conversations": self.s.list_conversations(project_id=pid, limit=30)}

    def project_file(self, q: dict) -> dict:
        project = self.s.get_project(_int(_q(q, "id")))
        if not project or not project.get("repository_path"):
            raise NotFound("project")
        return repo.read_file(project["repository_path"], _q(q, "path", ""))

    def create_project(self, b: dict) -> dict:
        name = (b.get("name") or "").strip()
        if not name:
            raise BadRequest("A project needs a name.")
        path = (b.get("repository_path") or "").strip() or None
        if path:
            from pathlib import Path
            if not Path(path).expanduser().is_dir():
                raise BadRequest(f"Folder not found: {path}")
            path = str(Path(path).expanduser().resolve())
        if self.s.find_project(name):
            raise BadRequest(f"A project called {name} already exists.")
        pid = self.s.create_project(name, b.get("description", ""), path)
        if path:
            repo.index_in_background(self.s.path, pid)
        return {"id": pid}

    def update_project(self, b: dict) -> dict:
        pid = _int(b.get("id"))
        fields = {k: b[k] for k in ("name", "description", "repository_path") if k in b}
        if fields.get("repository_path"):
            from pathlib import Path
            if not Path(fields["repository_path"]).expanduser().is_dir():
                raise BadRequest(f"Folder not found: {fields['repository_path']}")
        self.s.update_project(pid, **fields)
        return {"project": self.s.get_project(pid)}

    def index_project(self, b: dict) -> dict:
        pid = _int(b.get("id"))
        if not self.s.get_project(pid):
            raise NotFound("project")
        started = repo.index_in_background(self.s.path, pid)
        return {"started": started, "status": repo.job_status(pid)}

    def create_project_memory(self, b: dict) -> dict:
        res = self.p.memory.add_project_memory(_int(b.get("project_id"), "project_id"), b.get("content", ""),
                                               b.get("memory_type", "context"), "manual")
        if res["action"] == "refused":
            raise BadRequest(f"Not saved: {res['reason']}")
        return res

    def proposal(self, q: dict) -> dict:
        p = repo.get_proposal(self.s, _int(_q(q, "id")))
        if not p:
            raise NotFound("proposal")
        return {"proposal": p}

    def run_tool(self, b: dict) -> dict:
        """A tool the user asked for directly (e.g. a test command from the
        coding workspace). Approval-gated tools still wait for approval."""
        project = self.s.get_project(_int(b["project_id"])) if b.get("project_id") else None
        ctx = ToolContext(self.s, self.p.memory, project,
                          _int(b["conversation_id"]) if b.get("conversation_id") else None,
                          self.p.notes_retriever)
        return self.p.registry.invoke(b.get("tool", ""), b.get("arguments") or {}, ctx, origin="user")

    def _task(self, t: dict) -> dict:
        return {**t, "due_label": fmt_time(t.get("due_at")), "remind_label": fmt_time(t.get("remind_at"))}

    def create_task(self, b: dict) -> dict:
        title = (b.get("title") or "").strip()
        if not title:
            raise BadRequest("A task needs a title.")
        due = parse_when(b["due"]) if b.get("due") else None
        if b.get("due") and due is None:
            raise BadRequest(f"Could not understand the time '{b['due']}'.")
        tid = self.s.add_task(title, b.get("notes", ""), due, due if b.get("remind", True) else None,
                              b.get("project_id"))
        return {"task": self._task(self.s.get_task(tid))}

    def update_task(self, b: dict) -> dict:
        tid = _int(b.get("id"))
        fields = {k: b[k] for k in ("title", "notes", "status") if k in b}
        if "due" in b:
            fields["due_at"] = parse_when(b["due"]) if b["due"] else None
            fields["remind_at"] = fields["due_at"]
        if not self.s.update_task(tid, **fields):
            raise NotFound("task")
        return {"task": self._task(self.s.get_task(tid))}

    def due_reminders(self, q: dict) -> dict:
        due = self.s.due_reminders()
        for t in due:
            self.s.mark_reminded(t["id"])
        return {"due": [self._task(t) for t in due]}

    def run_bench(self, b: dict) -> dict:
        if self._bench_state["running"]:
            raise BadRequest("A benchmark is already running.")
        providers = [x for x in (b.get("providers") or [self.p.setting("LLM_PROVIDER")])
                     if x in ("colibri", "lmstudio", "ollama", "openai")]
        urls = {"colibri": self.p.setting("COLIBRI_BASE_URL"), "lmstudio": self.p.setting("LMSTUDIO_BASE_URL"),
                "ollama": self.p.setting("OLLAMA_BASE_URL"), "openai": self.p.setting("OPENAI_COMPAT_BASE_URL")}
        self._bench_state.update(running=True, error=None, started=time.time())
        store_path = self.s.path

        def work():
            s = st.Store(store_path)
            try:
                for name in providers:
                    r = bench.bench_provider(name, urls.get(name, ""), runs=int(b.get("runs", 1)),
                                             max_tokens=int(b.get("max_tokens", 200)))
                    s.save_bench(name, r.get("model") or "", r, "dashboard")
            except Exception as exc:
                self._bench_state["error"] = str(exc)
            finally:
                self._bench_state["running"] = False
                s.close()

        threading.Thread(target=work, daemon=True, name="bench").start()
        return {"started": True, "providers": providers}
