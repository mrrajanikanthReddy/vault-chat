"""Persistent storage for the assistant platform: conversations, messages,
memories, projects, tasks, approvals and metrics, in one local SQLite file.

It lives apart from the notes index (`<vault>/.index/vault.db`) on purpose:
the index can be thrown away and rebuilt, this cannot. Nothing here depends
on which model or provider is answering, so memory survives model changes,
provider restarts and upgrades.

Schema changes are numbered migrations recorded in `schema_migrations`;
`Store()` applies any that are missing, in order, inside a transaction.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path

import vaultdb as v

# ----------------------------------------------------------------- paths


def data_dir() -> Path:
    """Where the platform database lives. VAULT_DATA_DIR overrides it."""
    configured = v.setting("VAULT_DATA_DIR", "")
    if configured:
        return Path(configured).expanduser()
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", "~")).expanduser() / "vault-chat"
    return Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser() / "vault-chat"


def db_path() -> Path:
    return Path(v.setting("VAULT_PLATFORM_DB", str(data_dir() / "assistant.db"))).expanduser()


# ------------------------------------------------------------ migrations

FTS = "tokenize='porter unicode61'"


def _fts_triggers(table: str, cols: list[str]) -> str:
    c = ", ".join(cols)
    new = ", ".join(f"new.{x}" for x in cols)
    old = ", ".join(f"old.{x}" for x in cols)
    return f"""
CREATE TRIGGER {table}_fts_ai AFTER INSERT ON {table} BEGIN
  INSERT INTO {table}_fts(rowid, {c}) VALUES (new.id, {new});
END;
CREATE TRIGGER {table}_fts_ad AFTER DELETE ON {table} BEGIN
  INSERT INTO {table}_fts({table}_fts, rowid, {c}) VALUES ('delete', old.id, {old});
END;
CREATE TRIGGER {table}_fts_au AFTER UPDATE ON {table} BEGIN
  INSERT INTO {table}_fts({table}_fts, rowid, {c}) VALUES ('delete', old.id, {old});
  INSERT INTO {table}_fts(rowid, {c}) VALUES (new.id, {new});
END;
"""


MIGRATION_1 = f"""
CREATE TABLE users (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    preferences TEXT NOT NULL DEFAULT '{{}}',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE projects (
    id              INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    repository_path TEXT,
    metadata        TEXT NOT NULL DEFAULT '{{}}',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    UNIQUE (user_id, name)
);

CREATE TABLE conversations (
    id                INTEGER PRIMARY KEY,
    user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title             TEXT NOT NULL,
    conversation_type TEXT NOT NULL DEFAULT 'chat'
                      CHECK (conversation_type IN ('chat', 'coding', 'agent')),
    project_id        INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    category          TEXT,
    metadata          TEXT NOT NULL DEFAULT '{{}}',
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL
);
CREATE INDEX conversations_recent ON conversations(user_id, updated_at DESC);
CREATE INDEX conversations_project ON conversations(project_id);

CREATE TABLE conversation_tags (
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    tag             TEXT NOT NULL,
    PRIMARY KEY (conversation_id, tag)
);
CREATE INDEX conversation_tags_by_tag ON conversation_tags(tag);

CREATE TABLE messages (
    id              INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content         TEXT NOT NULL,
    metadata        TEXT NOT NULL DEFAULT '{{}}',
    token_count     INTEGER,
    created_at      REAL NOT NULL
);
CREATE INDEX messages_by_conversation ON messages(conversation_id, id);
CREATE VIRTUAL TABLE messages_fts USING fts5(content, content='messages', content_rowid='id', {FTS});
{_fts_triggers("messages", ["content"])}

CREATE TABLE conversation_summaries (
    id                        INTEGER PRIMARY KEY,
    conversation_id           INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    covers_through_message_id INTEGER NOT NULL,
    summary                   TEXT NOT NULL,
    token_count               INTEGER,
    model                     TEXT,
    created_at                REAL NOT NULL
);
CREATE INDEX summaries_by_conversation ON conversation_summaries(conversation_id, id DESC);

CREATE TABLE memories (
    id                     INTEGER PRIMARY KEY,
    user_id                INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    memory_type            TEXT NOT NULL CHECK (memory_type IN ('short_term', 'long_term')),
    category               TEXT NOT NULL,
    content                TEXT NOT NULL,
    normalized             TEXT NOT NULL,
    source_conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    source_message_id      INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    origin                 TEXT NOT NULL DEFAULT 'auto' CHECK (origin IN ('auto', 'explicit', 'manual', 'llm')),
    importance             REAL NOT NULL DEFAULT 0.5,
    confidence             REAL NOT NULL DEFAULT 0.7,
    access_count           INTEGER NOT NULL DEFAULT 0,
    pinned                 INTEGER NOT NULL DEFAULT 0,
    vec                    BLOB,
    expires_at             REAL,
    created_at             REAL NOT NULL,
    updated_at             REAL NOT NULL,
    last_accessed_at       REAL,
    UNIQUE (user_id, normalized)
);
CREATE INDEX memories_by_category ON memories(user_id, category);
CREATE INDEX memories_by_expiry ON memories(expires_at) WHERE expires_at IS NOT NULL;
CREATE VIRTUAL TABLE memories_fts USING fts5(content, category, content='memories', content_rowid='id', {FTS});
{_fts_triggers("memories", ["content", "category"])}

CREATE TABLE project_memories (
    id               INTEGER PRIMARY KEY,
    project_id       INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    memory_type      TEXT NOT NULL CHECK (memory_type IN
                     ('architecture', 'decision', 'standard', 'documentation', 'issue',
                      'resolution', 'context')),
    content          TEXT NOT NULL,
    normalized       TEXT NOT NULL,
    source           TEXT NOT NULL DEFAULT '',
    importance       REAL NOT NULL DEFAULT 0.6,
    access_count     INTEGER NOT NULL DEFAULT 0,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    last_accessed_at REAL,
    UNIQUE (project_id, normalized)
);
CREATE INDEX project_memories_by_project ON project_memories(project_id, memory_type);
CREATE VIRTUAL TABLE project_memories_fts USING fts5(content, memory_type, content='project_memories', content_rowid='id', {FTS});
{_fts_triggers("project_memories", ["content", "memory_type"])}

CREATE TABLE code_files (
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    sha        TEXT NOT NULL,
    language   TEXT,
    size       INTEGER NOT NULL,
    indexed_at REAL NOT NULL,
    PRIMARY KEY (project_id, path)
);
CREATE TABLE code_chunks (
    id         INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    symbol     TEXT,
    language   TEXT,
    content    TEXT NOT NULL
);
CREATE INDEX code_chunks_by_file ON code_chunks(project_id, path);
CREATE VIRTUAL TABLE code_chunks_fts USING fts5(path, symbol, content, content='code_chunks', content_rowid='id', {FTS});
{_fts_triggers("code_chunks", ["path", "symbol", "content"])}

CREATE TABLE change_proposals (
    id              INTEGER PRIMARY KEY,
    project_id      INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    message_id      INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'applied', 'rejected', 'failed', 'stale', 'reverted')),
    summary         TEXT NOT NULL DEFAULT '',
    edits           TEXT NOT NULL,
    diff            TEXT NOT NULL,
    result          TEXT,
    created_at      REAL NOT NULL,
    decided_at      REAL
);
CREATE INDEX proposals_by_status ON change_proposals(status, created_at DESC);

CREATE TABLE tool_runs (
    id                INTEGER PRIMARY KEY,
    conversation_id   INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    tool              TEXT NOT NULL,
    arguments         TEXT NOT NULL DEFAULT '{{}}',
    requires_approval INTEGER NOT NULL DEFAULT 0,
    status            TEXT NOT NULL CHECK (status IN
                      ('pending_approval', 'rejected', 'running', 'succeeded', 'failed')),
    origin            TEXT NOT NULL DEFAULT 'agent',
    result            TEXT,
    error             TEXT,
    created_at        REAL NOT NULL,
    decided_at        REAL,
    finished_at       REAL
);
CREATE INDEX tool_runs_by_status ON tool_runs(status, created_at DESC);

CREATE TABLE tasks (
    id           INTEGER PRIMARY KEY,
    user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id   INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    title        TEXT NOT NULL,
    notes        TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done')),
    due_at       REAL,
    remind_at    REAL,
    reminded_at  REAL,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    completed_at REAL
);
CREATE INDEX tasks_open ON tasks(user_id, status, due_at);
CREATE INDEX tasks_reminders ON tasks(remind_at) WHERE remind_at IS NOT NULL;

CREATE TABLE settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE inference_metrics (
    id                INTEGER PRIMARY KEY,
    conversation_id   INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    kind              TEXT NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    prompt_chars      INTEGER,
    ttft_ms           REAL,
    total_ms          REAL,
    tokens_per_second REAL,
    queue_wait_ms     REAL,
    ok                INTEGER NOT NULL,
    error             TEXT,
    created_at        REAL NOT NULL
);
CREATE INDEX metrics_recent ON inference_metrics(created_at DESC);

CREATE TABLE bench_runs (
    id         INTEGER PRIMARY KEY,
    provider   TEXT NOT NULL,
    model      TEXT NOT NULL,
    label      TEXT NOT NULL DEFAULT '',
    results    TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

MIGRATION_3_VOCAB = """
CREATE VIRTUAL TABLE messages_vocab USING fts5vocab(messages_fts, 'row');
"""


def _migration_2_import_legacy(conn: sqlite3.Connection) -> None:
    """Carry conversations saved by the old chat screen (`sessions` in the
    notes index) into the new tables, once. The old rows are left untouched
    so the command-line agent keeps reading them."""
    legacy = v.DB_PATH
    if not legacy.exists() or legacy.resolve() == db_path().resolve():
        return
    try:
        old = sqlite3.connect(f"file:{legacy}?mode=ro", uri=True)
        old.row_factory = sqlite3.Row
        sessions = old.execute("SELECT * FROM sessions ORDER BY id").fetchall()
    except sqlite3.Error:
        return
    profile_type = {"code": "coding", "tools": "agent"}
    for s in sessions:
        cur = conn.execute(
            "INSERT INTO conversations(user_id, title, conversation_type, metadata, created_at, "
            "updated_at) VALUES (1, ?, ?, ?, ?, ?)",
            (s["title"], profile_type.get(s["profile"], "chat"),
             json.dumps({"imported_from": "sessions", "legacy_id": s["id"],
                         "profile": s["profile"]}),
             s["created_at"], s["updated_at"]),
        )
        cid = cur.lastrowid
        for m in old.execute("SELECT role, content, created_at FROM messages "
                             "WHERE session_id = ? ORDER BY id", (s["id"],)):
            conn.execute(
                "INSERT INTO messages(conversation_id, role, content, token_count, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (cid, m["role"], m["content"], estimate_tokens(m["content"]), m["created_at"]))
    old.close()


MIGRATIONS = [
    (1, "initial schema", MIGRATION_1),
    (2, "import legacy chat sessions", _migration_2_import_legacy),
    (3, "message vocabulary for topic tracking", MIGRATION_3_VOCAB),
]


# --------------------------------------------------------------- helpers


def now() -> float:
    return time.time()


def estimate_tokens(text: str) -> int:
    """About four characters a token for English and code. context.py
    calibrates this against the usage counts the server reports."""
    return max(1, (len(text or "") + 3) // 4)


def normalize(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return " ".join(text.split())


def title_from(text: str) -> str:
    line = " ".join((text or "").split())
    return (line[:60] + ("…" if len(line) > 60 else "")) or "New conversation"


def loads(raw, default=None):
    if raw is None or raw == "":
        return {} if default is None else default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {} if default is None else default


def row_dict(row: sqlite3.Row | None, json_fields: tuple = ("metadata", "preferences")) -> dict | None:
    if row is None:
        return None
    out = dict(row)
    for key in json_fields:
        if key in out and isinstance(out[key], str):
            out[key] = loads(out[key])
    out.pop("vec", None)
    out.pop("normalized", None)
    return out


# ------------------------------------------------------------------ store


DEFAULT_USER = 1


class Store:
    """One connection per thread; SQLite in WAL mode handles the rest."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._migrate()

    # -- connection

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _migrate(self) -> None:
        c = self.conn
        c.execute("CREATE TABLE IF NOT EXISTS schema_migrations ("
                  "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at REAL NOT NULL)")
        c.commit()
        done = {r[0] for r in c.execute("SELECT version FROM schema_migrations")}
        previous = c.isolation_level
        c.isolation_level = None  # explicit BEGIN/COMMIT around each migration
        try:
            self._apply(c, done)
        finally:
            c.isolation_level = previous

    def _apply(self, c: sqlite3.Connection, done: set) -> None:
        for version, name, step in MIGRATIONS:
            if version in done:
                continue
            try:
                c.execute("BEGIN")
                if callable(step):
                    step(c)
                else:
                    for statement in _split_sql(step):
                        c.execute(statement)
                if version == 1:
                    t = now()
                    c.execute("INSERT INTO users(id, name, created_at, updated_at) "
                              "VALUES (?, 'me', ?, ?)", (DEFAULT_USER, t, t))
                c.execute("INSERT INTO schema_migrations VALUES (?, ?, ?)", (version, name, now()))
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise

    def schema_version(self) -> int:
        return self.conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] or 0

    # -- settings: environment > saved in the UI > config.json > default

    def get_setting(self, key: str, default=None):
        env = os.environ.get(key)
        if env not in (None, ""):
            return _coerce(env, default)
        row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if row is not None:
            return loads(row["value"], default)
        file_value = v._CONFIG.get(key)
        if file_value not in (None, ""):
            return _coerce(file_value, default)
        return default

    def set_setting(self, key: str, value) -> None:
        self.conn.execute(
            "INSERT INTO settings(key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, json.dumps(value), now()))
        self.conn.commit()

    def setting_source(self, key: str) -> str:
        if os.environ.get(key) not in (None, ""):
            return "environment"
        if self.conn.execute("SELECT 1 FROM settings WHERE key = ?", (key,)).fetchone():
            return "saved"
        if v._CONFIG.get(key) not in (None, ""):
            return "config file"
        return "default"

    # -- conversations

    def create_conversation(self, title: str = "", conversation_type: str = "chat",
                            project_id: int | None = None, category: str | None = None,
                            metadata: dict | None = None) -> int:
        t = now()
        cur = self.conn.execute(
            "INSERT INTO conversations(user_id, title, conversation_type, project_id, category, "
            "metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (DEFAULT_USER, title_from(title), conversation_type, project_id, category,
             json.dumps(metadata or {}), t, t))
        self.conn.commit()
        return int(cur.lastrowid)

    def get_conversation(self, cid: int) -> dict | None:
        row = row_dict(self.conn.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone())
        if row:
            row["tags"] = self.conversation_tags(cid)
        return row

    def update_conversation(self, cid: int, **fields) -> None:
        allowed = {"title", "category", "project_id", "conversation_type", "metadata"}
        sets, args = [], []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "metadata":
                value = json.dumps(value or {})
            if key == "title":
                value = title_from(value)
            sets.append(f"{key} = ?")
            args.append(value)
        if "tags" in fields:
            self.set_tags(cid, fields["tags"] or [])
        if sets:
            sets.append("updated_at = ?")
            args += [now(), cid]
            self.conn.execute(f"UPDATE conversations SET {', '.join(sets)} WHERE id = ?", args)
        self.conn.commit()

    def touch_conversation(self, cid: int) -> None:
        self.conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now(), cid))

    def delete_conversation(self, cid: int) -> bool:
        cur = self.conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))
        self.conn.commit()
        return cur.rowcount > 0

    def conversation_tags(self, cid: int) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT tag FROM conversation_tags WHERE conversation_id = ? ORDER BY tag", (cid,))]

    def set_tags(self, cid: int, tags: list[str]) -> None:
        clean = sorted({t.strip().lower()[:40] for t in tags if t and t.strip()})
        self.conn.execute("DELETE FROM conversation_tags WHERE conversation_id = ?", (cid,))
        self.conn.executemany("INSERT INTO conversation_tags VALUES (?, ?)", [(cid, t) for t in clean])

    def list_conversations(self, conversation_type: str | None = None, project_id: int | None = None,
                           tag: str | None = None, category: str | None = None,
                           limit: int = 100, offset: int = 0) -> list[dict]:
        where, args = ["c.user_id = ?"], [DEFAULT_USER]
        if conversation_type:
            where.append("c.conversation_type = ?")
            args.append(conversation_type)
        if project_id:
            where.append("c.project_id = ?")
            args.append(project_id)
        if category:
            where.append("c.category = ?")
            args.append(category)
        if tag:
            where.append("EXISTS (SELECT 1 FROM conversation_tags t WHERE t.conversation_id = c.id "
                         "AND t.tag = ?)")
            args.append(tag.lower())
        rows = self.conn.execute(
            f"SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS n_messages, "
            f"(SELECT GROUP_CONCAT(tag, ',') FROM conversation_tags t WHERE t.conversation_id = c.id) AS tag_list "
            f"FROM conversations c WHERE {' AND '.join(where)} "
            f"ORDER BY c.updated_at DESC, c.id DESC LIMIT ? OFFSET ?",
            (*args, limit, offset)).fetchall()
        out = []
        for r in rows:
            d = row_dict(r)
            d["tags"] = [t for t in (d.pop("tag_list") or "").split(",") if t]
            out.append(d)
        return out

    def search_conversations(self, query: str, limit: int = 30,
                             exclude_id: int | None = None) -> list[dict]:
        """Full-text search over message bodies and titles, best match first,
        one row per conversation with the matching snippet."""
        fts = v.fts_query(query)
        results: dict[int, dict] = {}
        if fts:
            rows = self.conn.execute(
                "SELECT m.id AS message_id, m.conversation_id, m.role, m.created_at, "
                "snippet(messages_fts, 0, '[', ']', '…', 18) AS snippet, bm25(messages_fts) AS rank "
                "FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid "
                "WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?", (fts, limit * 4)).fetchall()
            for r in rows:
                cid = r["conversation_id"]
                if cid == exclude_id or cid in results:
                    continue
                results[cid] = {"conversation_id": cid, "message_id": r["message_id"],
                                "snippet": r["snippet"], "role": r["role"], "rank": r["rank"]}
        like = f"%{query.strip()}%"
        for r in self.conn.execute("SELECT id FROM conversations WHERE title LIKE ? "
                                   "ORDER BY updated_at DESC LIMIT ?", (like, limit)):
            if r["id"] != exclude_id and r["id"] not in results:
                results[r["id"]] = {"conversation_id": r["id"], "message_id": None,
                                    "snippet": "", "role": None, "rank": 0.0}
        out = []
        for cid, hit in list(results.items())[:limit]:
            conv = self.get_conversation(cid)
            if conv:
                out.append({**hit, "id": cid, "title": conv["title"], "conversation_type": conv["conversation_type"],
                            "updated_at": conv["updated_at"], "project_id": conv["project_id"]})
        return out

    # -- messages

    def add_message(self, cid: int, role: str, content: str, metadata: dict | None = None,
                    token_count: int | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages(conversation_id, role, content, metadata, token_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cid, role, content, json.dumps(metadata or {}),
             token_count if token_count is not None else estimate_tokens(content), now()))
        self.touch_conversation(cid)
        self.conn.commit()
        return int(cur.lastrowid)

    def update_message(self, mid: int, content: str | None = None, metadata: dict | None = None) -> None:
        if content is not None:
            self.conn.execute("UPDATE messages SET content = ?, token_count = ? WHERE id = ?",
                              (content, estimate_tokens(content), mid))
        if metadata is not None:
            row = self.conn.execute("SELECT metadata FROM messages WHERE id = ?", (mid,)).fetchone()
            merged = {**loads(row["metadata"] if row else None), **metadata}
            self.conn.execute("UPDATE messages SET metadata = ? WHERE id = ?", (json.dumps(merged), mid))
        self.conn.commit()

    def delete_message(self, mid: int) -> None:
        self.conn.execute("DELETE FROM messages WHERE id = ?", (mid,))
        self.conn.commit()

    def messages(self, cid: int, after_id: int = 0, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM messages WHERE conversation_id = ? AND id > ? ORDER BY id"
        args: list = [cid, after_id]
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return [row_dict(r) for r in self.conn.execute(sql, args)]

    def get_message(self, mid: int) -> dict | None:
        return row_dict(self.conn.execute("SELECT * FROM messages WHERE id = ?", (mid,)).fetchone())

    def latest_summary(self, cid: int) -> dict | None:
        return row_dict(self.conn.execute(
            "SELECT * FROM conversation_summaries WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
            (cid,)).fetchone())

    def add_summary(self, cid: int, through_id: int, summary: str, model: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO conversation_summaries(conversation_id, covers_through_message_id, summary, "
            "token_count, model, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (cid, through_id, summary, estimate_tokens(summary), model, now()))
        self.conn.commit()
        return int(cur.lastrowid)

    def frequent_topics(self, limit: int = 10) -> list[dict]:
        """Terms that recur across many conversations' messages."""
        stop = v.STOPWORDS | {"like", "use", "using", "get", "make", "one", "way", "thing", "think",
                              "would", "could", "should", "here", "now", "work", "code", "file"}
        rows = self.conn.execute(
            "SELECT term, doc FROM messages_vocab WHERE length(term) > 3 "
            "ORDER BY doc DESC LIMIT ?", (limit * 6,)).fetchall()
        out = []
        for r in rows:
            if r["term"] in stop or r["term"].isdigit():
                continue
            out.append({"term": self._surface_form(r["term"]), "messages": r["doc"]})
            if len(out) >= limit:
                break
        return out

    def _surface_form(self, stem: str) -> str:
        """The vocabulary holds Porter stems ("schedul"); show a real word."""
        row = self.conn.execute(
            "SELECT m.content FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid "
            "WHERE messages_fts MATCH ? LIMIT 1", (f'"{stem}"',)).fetchone()
        if row:
            match = re.search(rf"\b{re.escape(stem)}\w*", row["content"], re.IGNORECASE)
            if match:
                return match.group(0).lower()
        return stem

    # -- projects

    def create_project(self, name: str, description: str = "", repository_path: str | None = None,
                       metadata: dict | None = None) -> int:
        t = now()
        cur = self.conn.execute(
            "INSERT INTO projects(user_id, name, description, repository_path, metadata, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (DEFAULT_USER, name.strip(), description, repository_path, json.dumps(metadata or {}), t, t))
        self.conn.commit()
        return int(cur.lastrowid)

    def get_project(self, pid: int) -> dict | None:
        return row_dict(self.conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone())

    def find_project(self, name: str) -> dict | None:
        return row_dict(self.conn.execute("SELECT * FROM projects WHERE lower(name) = lower(?)",
                                          (name.strip(),)).fetchone())

    def list_projects(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT p.*, (SELECT COUNT(*) FROM code_files f WHERE f.project_id = p.id) AS n_files, "
            "(SELECT COUNT(*) FROM project_memories pm WHERE pm.project_id = p.id) AS n_memories, "
            "(SELECT COUNT(*) FROM conversations c WHERE c.project_id = p.id) AS n_conversations "
            "FROM projects p WHERE p.user_id = ? ORDER BY p.updated_at DESC", (DEFAULT_USER,))
        return [row_dict(r) for r in rows]

    def update_project(self, pid: int, **fields) -> None:
        allowed = {"name", "description", "repository_path", "metadata"}
        sets, args = [], []
        for key, value in fields.items():
            if key in allowed:
                sets.append(f"{key} = ?")
                args.append(json.dumps(value or {}) if key == "metadata" else value)
        if sets:
            sets.append("updated_at = ?")
            args += [now(), pid]
            self.conn.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id = ?", args)
            self.conn.commit()

    def delete_project(self, pid: int) -> bool:
        cur = self.conn.execute("DELETE FROM projects WHERE id = ?", (pid,))
        self.conn.commit()
        return cur.rowcount > 0

    # -- tasks

    def add_task(self, title: str, notes: str = "", due_at: float | None = None,
                 remind_at: float | None = None, project_id: int | None = None) -> int:
        t = now()
        cur = self.conn.execute(
            "INSERT INTO tasks(user_id, project_id, title, notes, due_at, remind_at, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (DEFAULT_USER, project_id, title.strip()[:300], notes, due_at, remind_at, t, t))
        self.conn.commit()
        return int(cur.lastrowid)

    def list_tasks(self, status: str | None = "open", limit: int = 100) -> list[dict]:
        sql, args = "SELECT * FROM tasks WHERE user_id = ?", [DEFAULT_USER]
        if status:
            sql += " AND status = ?"
            args.append(status)
        sql += " ORDER BY status, due_at IS NULL, due_at, id DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def get_task(self, tid: int) -> dict | None:
        r = self.conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
        return dict(r) if r else None

    def update_task(self, tid: int, **fields) -> bool:
        allowed = {"title", "notes", "status", "due_at", "remind_at", "project_id"}
        sets, args = [], []
        for key, value in fields.items():
            if key in allowed:
                sets.append(f"{key} = ?")
                args.append(value)
        if fields.get("status") == "done":
            sets.append("completed_at = ?")
            args.append(now())
        elif fields.get("status") == "open":
            sets.append("completed_at = NULL")
        if not sets:
            return False
        sets.append("updated_at = ?")
        args += [now(), tid]
        cur = self.conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", args)
        self.conn.commit()
        return cur.rowcount > 0

    def delete_task(self, tid: int) -> bool:
        cur = self.conn.execute("DELETE FROM tasks WHERE id = ?", (tid,))
        self.conn.commit()
        return cur.rowcount > 0

    def due_reminders(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM tasks WHERE status = 'open' AND remind_at IS NOT NULL AND remind_at <= ? "
            "AND reminded_at IS NULL ORDER BY remind_at", (now(),)).fetchall()
        return [dict(r) for r in rows]

    def mark_reminded(self, tid: int) -> None:
        self.conn.execute("UPDATE tasks SET reminded_at = ? WHERE id = ?", (now(), tid))
        self.conn.commit()

    # -- metrics

    def record_metric(self, kind: str, stats: dict, conversation_id: int | None = None,
                      ok: bool = True, error: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO inference_metrics(conversation_id, provider, model, kind, prompt_tokens, "
            "completion_tokens, prompt_chars, ttft_ms, total_ms, tokens_per_second, queue_wait_ms, ok, "
            "error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, stats.get("provider") or "", stats.get("model") or "", kind,
             stats.get("prompt_tokens"), stats.get("completion_tokens"), stats.get("prompt_chars"),
             stats.get("ttft_ms"), stats.get("total_ms"), stats.get("tokens_per_second"),
             stats.get("queue_wait_ms"), int(ok), error, now()))
        self.conn.commit()

    def recent_metrics(self, limit: int = 50) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM inference_metrics ORDER BY id DESC LIMIT ?", (limit,))]

    def metric_summary(self, hours: float = 24 * 7) -> dict:
        r = self.conn.execute(
            "SELECT COUNT(*) n, SUM(ok) ok, AVG(ttft_ms) ttft, AVG(tokens_per_second) tps, "
            "AVG(total_ms) total, AVG(prompt_tokens) prompt FROM inference_metrics "
            "WHERE created_at > ? AND kind = 'chat'", (now() - hours * 3600,)).fetchone()
        return {k: (round(r[k], 1) if isinstance(r[k], float) else r[k]) for k in r.keys()}

    def save_bench(self, provider: str, model: str, results: dict, label: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO bench_runs(provider, model, label, results, created_at) VALUES (?, ?, ?, ?, ?)",
            (provider, model, label, json.dumps(results), now()))
        self.conn.commit()
        return int(cur.lastrowid)

    def list_bench(self, limit: int = 20) -> list[dict]:
        return [row_dict(r, ("results",)) for r in self.conn.execute(
            "SELECT * FROM bench_runs ORDER BY id DESC LIMIT ?", (limit,))]


def _split_sql(script: str) -> list[str]:
    """Split a migration into statements, keeping trigger bodies whole."""
    out, buf, depth = [], [], 0
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped and not buf:
            continue
        buf.append(line)
        upper = stripped.upper()
        if upper.startswith("CREATE TRIGGER"):
            depth += 1
        if depth and upper == "END;":
            depth -= 1
            out.append("\n".join(buf))
            buf = []
        elif not depth and stripped.endswith(";"):
            out.append("\n".join(buf))
            buf = []
    if "".join(buf).strip():
        out.append("\n".join(buf))
    return out


def _coerce(value, default):
    """Environment and config values arrive as strings; match the default's type."""
    if isinstance(value, str):
        if isinstance(default, bool):
            return value.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(default, int):
            try:
                return int(value)
            except ValueError:
                return default
        if isinstance(default, float):
            try:
                return float(value)
            except ValueError:
                return default
    return value
