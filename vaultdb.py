"""Shared plumbing for vault-chat: config, SQLite storage, model calls.

Three backends are supported and auto-detected, in this order:

  colibri   colibri's OpenAI-compatible gateway on :8000 (the default). It has
            no embeddings endpoint, so notes are searched by keyword unless
            VAULT_EMBED_URL points at a server that embeds.
  lmstudio  OpenAI-compatible server on :1234, plus the /api/v1/chat endpoint
            that executes MCP tools server-side (LM Studio 0.4.0+). Optional.
  ollama    The native /api/* endpoints on :11434. Optional.

Everything here is stdlib plus numpy. The only network destinations are the
loopback addresses above, or URLs the user configured.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required:  python3 -m pip install --user numpy")


def config_root() -> Path:
    """The platform's per-user settings folder."""
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", "~")).expanduser()
    return Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()


def config_dir() -> Path:
    """Where settings live, following each platform's convention.

    The project used to be called vault-tools. Settings written under that
    name keep working until the installer moves them, so renaming the project
    never silently drops a configured model or vault path."""
    new, old = config_root() / "vault-chat", config_root() / "vault-tools"
    return old if old.is_dir() and not new.exists() else new


CONFIG_PATH = config_dir() / "config.json"

try:
    _CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    _CONFIG = {}


def setting(key: str, default: str) -> str:
    """Environment wins, then the config file written by the installer,
    then the built-in default."""
    value = os.environ.get(key) or _CONFIG.get(key)
    return str(value) if value else default


OLLAMA_HOST = setting("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
LMSTUDIO_HOST = setting("LMSTUDIO_HOST", "http://127.0.0.1:1234").rstrip("/")
LM_API_TOKEN = setting("LM_API_TOKEN", "")
COLIBRI_BASE_URL = setting("COLIBRI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
COLIBRI_API_KEY = setting("COLIBRI_API_KEY", "")
# Optional OpenAI-compatible /v1/embeddings server for semantic note search
# when the chat backend is colibri, which does not embed.
EMBED_URL = setting("VAULT_EMBED_URL", "").rstrip("/")

VAULT = Path(setting("VAULT_DIR", "~/vault")).expanduser()
DB_PATH = Path(setting("VAULT_DB", str(VAULT / ".index" / "vault.db"))).expanduser()
NUM_CTX = int(setting("VAULT_NUM_CTX", "8192"))
# 0 means "let the runtime decide". install.py writes the physical core count
# here, because hyper-threaded siblings share a memory path and using both
# usually makes CPU inference slower, not faster.
NUM_THREADS = int(setting("VAULT_THREADS", "0"))

# "auto" probes colibri first, then LM Studio, then Ollama.
BACKEND = setting("VAULT_BACKEND", "auto").lower()

DEFAULT_MODELS = {
    "colibri": {
        "embed": "",  # no embeddings endpoint; see EMBED_URL
        "chat": "",   # whatever model the running colibri server reports
    },
    "lmstudio": {
        "embed": "text-embedding-nomic-embed-text-v1.5",
        "chat": "qwen2.5-coder-7b-instruct",
    },
    "ollama": {
        "embed": "nomic-embed-text",
        "chat": "qwen2.5-coder:7b",
    },
}

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path       TEXT PRIMARY KEY,
    sha        TEXT NOT NULL,
    mtime      REAL NOT NULL,
    indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    path    TEXT NOT NULL,
    ord     INTEGER NOT NULL,
    heading TEXT,
    text    TEXT NOT NULL,
    vec     BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_path ON chunks(path);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    path          TEXT PRIMARY KEY,
    last_review   REAL NOT NULL,
    interval_days REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    profile    TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_session ON messages(session_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    heading, text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, heading, text) VALUES (new.id, new.heading, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, heading, text)
    VALUES ('delete', old.id, old.heading, old.text);
END;
"""


# --------------------------------------------------------------- backend


_resolved: str | None = None


def _probe(url: str) -> bool:
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=2)
        return True
    except urllib.error.HTTPError:
        return True  # reachable, just unhappy with the request — good enough
    except Exception:
        return False


def backend() -> str:
    """Which runtime we are talking to. Probed once, then cached."""
    global _resolved
    if _resolved:
        return _resolved
    if BACKEND in ("colibri", "lmstudio", "ollama"):
        _resolved = BACKEND
    elif _probe(f"{COLIBRI_BASE_URL}/models"):
        _resolved = "colibri"
    elif _probe(f"{LMSTUDIO_HOST}/v1/models"):
        _resolved = "lmstudio"
    elif _probe(f"{OLLAMA_HOST}/api/tags"):
        _resolved = "ollama"
    else:
        sys.exit(
            f"No model runtime found.\n"
            f"  colibri expected at   {COLIBRI_BASE_URL}  (coli start --background)\n"
            f"  LM Studio expected at {LMSTUDIO_HOST}  (lms server start)\n"
            f"  Ollama expected at    {OLLAMA_HOST}  (ollama serve)\n"
            f"Set VAULT_BACKEND to force one."
        )
    return _resolved


def host() -> str:
    return {"colibri": COLIBRI_BASE_URL, "lmstudio": LMSTUDIO_HOST}.get(backend(), OLLAMA_HOST)


def default_model(role: str) -> str:
    return DEFAULT_MODELS[backend()][role]


def has_embedder() -> bool:
    """Whether semantic search is possible. With colibri it needs a separate
    embeddings server; without one, retrieval falls back to keyword search."""
    return backend() != "colibri" or bool(EMBED_URL)


def embed_model() -> str:
    """The embed model's name, or "none" when retrieval is keyword-only.
    index.py stores it, so switching between the two forces --rebuild."""
    if not has_embedder():
        return "none"
    return setting("VAULT_EMBED_MODEL", default_model("embed") or "nomic-embed-text")


def chat_model() -> str:
    configured = setting("VAULT_CHAT_MODEL", default_model("chat"))
    if configured or backend() != "colibri":
        return configured
    return _colibri().active_model()


_colibri_provider = None


def _colibri():
    """The shared colibri provider; the model stays loaded server-side."""
    global _colibri_provider
    if _colibri_provider is None:
        from providers import ColibriProvider
        _colibri_provider = ColibriProvider(
            COLIBRI_BASE_URL, setting("VAULT_CHAT_MODEL", ""), COLIBRI_API_KEY,
            timeout=float(setting("VAULT_TIMEOUT", "600")))
    return _colibri_provider


# ------------------------------------------------------------------ http


def _request(path: str, payload: dict | None = None, stream: bool = False, method: str = "POST"):
    headers = {"Content-Type": "application/json"}
    if LM_API_TOKEN and backend() == "lmstudio":
        headers["Authorization"] = f"Bearer {LM_API_TOKEN}"
    req = urllib.request.Request(
        f"{host()}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
        method=method,
    )
    try:
        resp = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:600]
        raise SystemExit(f"{host()}{path} returned {exc.code}:\n{body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cannot reach {host()} ({exc.reason}).") from exc
    return resp if stream else json.loads(resp.read())


# -------------------------------------------------------------- embedding


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch and return L2-normalised rows, so cosine similarity
    reduces to a plain dot product later."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    if backend() == "colibri":
        if not EMBED_URL:
            raise SystemExit("No embeddings server configured (colibri does not embed). "
                             "Set VAULT_EMBED_URL, or use keyword search.")
        from providers import EmbeddingProvider, ProviderError
        try:
            vectors = EmbeddingProvider(EMBED_URL, embed_model()).embed(texts)
        except ProviderError as exc:
            raise SystemExit(str(exc)) from exc
    elif backend() == "lmstudio":
        data = _request("/v1/embeddings", {"model": embed_model(), "input": texts})
        vectors = [row["embedding"] for row in data["data"]]
    else:
        try:
            vectors = _request("/api/embed", {"model": embed_model(), "input": texts})["embeddings"]
        except (SystemExit, KeyError):
            # Older Ollama builds only expose the single-input endpoint.
            vectors = [
                _request("/api/embeddings", {"model": embed_model(), "prompt": t})["embedding"]
                for t in texts
            ]

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


# ------------------------------------------------------------------ chat


def chat_stream(messages: list[dict], model: str | None = None, temperature: float = 0.2):
    """Yield response text as it is generated, so words appear instead of
    you staring at nothing for a minute."""
    if backend() == "colibri":
        from providers import GenParams, ProviderError
        try:
            yield from _colibri().stream_chat(messages, GenParams(temperature=temperature),
                                              model=model or None)
        except ProviderError as exc:
            raise SystemExit(f"{exc}\n{exc.hint}".strip()) from exc
        return

    model = model or chat_model()

    if backend() == "lmstudio":
        resp = _request(
            "/v1/chat/completions",
            {"model": model, "messages": messages, "stream": True,
             "temperature": temperature, "ttl": 3600},
            stream=True,
        )
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                continue
            for choice in event.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    yield piece
        return

    options = {"num_ctx": NUM_CTX, "temperature": temperature}
    if NUM_THREADS:
        options["num_thread"] = NUM_THREADS
    resp = _request(
        "/api/chat",
        {"model": model, "messages": messages, "stream": True, "options": options},
        stream=True,
    )
    for raw in resp:
        line = raw.strip()
        if not line:
            continue
        event = json.loads(line)
        piece = event.get("message", {}).get("content", "")
        if piece:
            yield piece
        if event.get("done"):
            break


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.2) -> str:
    return "".join(chat_stream(messages, model=model, temperature=temperature))


def agent_chat(
    user_input: str,
    integrations: list | None = None,
    model: str | None = None,
    system: str | None = None,
    context_length: int = NUM_CTX,
    temperature: float = 0.2,
) -> dict:
    """LM Studio's /api/v1/chat — it runs the whole tool-calling loop itself,
    including executing MCP servers, and returns the transcript.

    Note the security shape: tools execute server-side, so there is no
    per-call approval prompt. The `allowed_tools` allowlist on each
    integration is what keeps a run inside its lane.
    """
    if backend() != "lmstudio":
        raise SystemExit("MCP tool use requires the LM Studio backend (VAULT_BACKEND=lmstudio).")

    payload: dict = {
        "model": model or chat_model(),
        "input": (
            [{"role": "system", "content": system}, {"role": "user", "content": user_input}]
            if system
            else user_input
        ),
        "context_length": context_length,
        "temperature": temperature,
        "ttl": 3600,
    }
    if integrations:
        payload["integrations"] = integrations
    return _request("/api/v1/chat", payload)


# --------------------------------------------------------------- storage


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # Indexes built before the keyword table existed get it filled once.
    if get_meta(conn, "fts_version") != "1":
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        set_meta(conn, "fts_version", "1")
        conn.commit()
    return conn


STOPWORDS = frozenset(
    "a an and are as at be but by can could did do does for from had has have how i if in "
    "into is it its me my no not of on or our so than that the their them then there these "
    "they this to was we were what when where which who why will with would you your about "
    "please tell show give help need want know just also any some all more most".split()
)


def fts_query(text: str, max_terms: int = 12) -> str:
    """Turn free text into a safe FTS5 OR-query. Every term is quoted, so
    user input can never inject FTS operators."""
    import re
    seen: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*", text):
        for word in [raw, *re.split(r"[_.\-]", raw)]:
            w = word.lower().strip(".-_")
            if len(w) < 2 or w in STOPWORDS or w in seen:
                continue
            seen.append(w)
    return " OR ".join(f'"{w}"' for w in seen[:max_terms])


def keyword_search_chunks(conn: sqlite3.Connection, question: str, limit: int) -> list[tuple[float, sqlite3.Row]]:
    """BM25 over note passages, scored 0..1 relative to the best hit."""
    query = fts_query(question)
    if not query:
        return []
    rows = conn.execute(
        "SELECT c.id, c.path, c.ord, c.heading, c.text, bm25(chunks_fts) AS rank "
        "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
        (query, limit),
    ).fetchall()
    if not rows:
        return []
    best = min(r["rank"] for r in rows) or -1.0
    return [(float(r["rank"] / best) if best else 0.0, r) for r in rows]


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def pack(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def load_matrix(conn: sqlite3.Connection) -> tuple[np.ndarray, list[sqlite3.Row]]:
    """Pull every chunk into one matrix. Brute force is the right call here:
    a personal vault is thousands of chunks, not millions, and an exact scan
    of a few megabytes beats maintaining an approximate index."""
    rows = conn.execute(
        "SELECT id, path, ord, heading, text, vec FROM chunks ORDER BY id"
    ).fetchall()
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.vstack([unpack(r["vec"]) for r in rows]), rows


# ------------------------------------------------------------- sessions
#
# Conversation history lives in the same SQLite file as the index but in its
# own tables, so `index.py --rebuild` never touches it. Only the plain
# question and answer are stored — not the retrieved notes that were pasted
# into the prompt — which keeps history small and keeps old note text from
# being replayed to the model as if it were still current.


def session_title(text: str) -> str:
    """One short line, suitable for a list."""
    line = " ".join(text.split())
    return line[:60] + ("…" if len(line) > 60 else "") or "(untitled)"


def create_session(conn: sqlite3.Connection, profile: str, title: str) -> int:
    now = time.time()
    cur = conn.execute(
        "INSERT INTO sessions(title, profile, created_at, updated_at) VALUES(?,?,?,?)",
        (session_title(title), profile, now, now),
    )
    conn.commit()
    return int(cur.lastrowid)


def add_messages(conn: sqlite3.Connection, session_id: int, pairs: list[tuple[str, str]]) -> None:
    """Append (role, content) pairs in order and bump the session's clock."""
    now = time.time()
    conn.executemany(
        "INSERT INTO messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
        [(session_id, role, content, now + i * 1e-6) for i, (role, content) in enumerate(pairs)],
    )
    conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
    conn.commit()


def session_messages(conn: sqlite3.Connection, session_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def get_session(conn: sqlite3.Connection, session_id: int):
    return conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()


def list_sessions(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT s.id, s.title, s.profile, s.created_at, s.updated_at, "
        "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n "
        "FROM sessions s ORDER BY s.updated_at DESC, s.id DESC LIMIT ?",
        (limit,),
    ).fetchall()


def latest_session_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT id FROM sessions ORDER BY updated_at DESC, id DESC LIMIT 1").fetchone()
    return int(row["id"]) if row else None


def delete_session(conn: sqlite3.Connection, session_id: int) -> None:
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()


# ----------------------------------------------------------------- misc


def walk_vault(root: Path | None = None):
    """Every text file in the vault, skipping dot-directories (which is how
    the index itself stays out of its own corpus)."""
    root = root or VAULT
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        yield path


def relative(path: Path) -> str:
    """Vault-relative path, always with forward slashes.

    as_posix() matters on Windows: str() would give 'notes\\retries.md', so
    the same note would be stored under a different key than on Linux or
    macOS, and lookups written with '/' would silently miss.
    """
    try:
        return Path(path).resolve().relative_to(VAULT.resolve()).as_posix()
    except ValueError:
        return str(path)


def ago(timestamp: float) -> str:
    days = (time.time() - timestamp) / 86400
    if days < 1:
        return "today"
    if days < 2:
        return "yesterday"
    if days < 60:
        return f"{int(days)} days ago"
    return f"{int(days / 30)} months ago"
