"""Repository awareness for the coding assistant.

    index      walk a project's repository, split source into symbol-sized
               chunks, keep them in SQLite FTS5; only changed files are redone
    retrieve   rank chunks for a question by identifiers, words and paths, so
               the model sees a handful of relevant sections, never the repo
    propose    turn the model's SEARCH/REPLACE blocks into a reviewable diff
    apply      write the files only after the user approves, refusing paths
               outside the repository and files changed since the diff

Nothing here executes code or shells out.
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import store as st
import vaultdb as v

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__", "dist",
             "build", "target", "out", ".idea", ".vscode", ".next", ".nuxt", "coverage", ".sfdx", ".sf",
             ".gradle", ".mypy_cache", ".pytest_cache", ".tox", "bower_components", "vendor", ".index",
             ".cache", ".terraform", "Pods", "DerivedData"}

LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".cls": "apex", ".trigger": "apex", ".apex": "apex", ".soql": "sql", ".sql": "sql",
    ".html": "html", ".htm": "html", ".css": "css", ".scss": "css", ".less": "css",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php", ".c": "c", ".h": "c", ".cpp": "cpp",
    ".cc": "cpp", ".hpp": "cpp", ".cs": "csharp", ".swift": "swift", ".m": "objc",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ps1": "powershell",
    ".md": "markdown", ".rst": "text", ".txt": "text", ".json": "json", ".yml": "yaml", ".yaml": "yaml",
    ".toml": "toml", ".ini": "ini", ".xml": "xml", ".gradle": "groovy", ".vue": "vue", ".svelte": "svelte",
    ".tf": "hcl", ".dockerfile": "docker",
}
SPECIAL_FILES = {"Dockerfile": "docker", "Makefile": "make", "README": "text", "Procfile": "text"}

MAX_FILE_BYTES = 400_000
CHUNK_LINES = 80
OVERLAP_LINES = 12

SYMBOL = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:public|private|protected|global|static|abstract|final|override|"
    r"virtual|async|with sharing|without sharing|inherited sharing|\s)*\s*(?:def|class|interface|enum|trait|struct|func|fn|function|module|"
    r"trigger|record|object|impl)\s+([A-Za-z_][\w]*)"
    r"|^\s*(?:public|private|protected|global)\s+(?:static\s+)?(?:with sharing\s+|without sharing\s+)?"
    r"[\w<>\[\],\s]+?\s+([A-Za-z_]\w*)\s*\("
    r"|^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\(?[^=]*=>"
    r"|^#{1,4}\s+(.+)$",
)


def language_of(path: Path) -> str | None:
    if path.name in SPECIAL_FILES:
        return SPECIAL_FILES[path.name]
    if path.name.endswith("-meta.xml"):
        return "xml"
    return LANGUAGES.get(path.suffix.lower())


def _gitignore_patterns(root: Path) -> list[str]:
    try:
        lines = (root / ".gitignore").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [ln.strip().rstrip("/") for ln in lines if ln.strip() and not ln.startswith(("#", "!"))]


def _ignored(rel: str, patterns: list[str]) -> bool:
    name = rel.rsplit("/", 1)[-1]
    for p in patterns:
        p = p.lstrip("/")
        if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(name, p) or rel.startswith(p + "/"):
            return True
    return False


def walk_repo(root: Path):
    """Source files worth indexing, as (relative posix path, absolute path)."""
    root = root.resolve()
    patterns = _gitignore_patterns(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
                             and not _ignored(f"{rel_dir}/{d}".lstrip("./"), patterns))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if name.startswith(".") or language_of(path) is None or _ignored(rel, patterns):
                continue
            try:
                if path.is_symlink() or path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield rel, path


def read_text(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    return raw.decode("utf-8", errors="replace")


def chunk_source(text: str, language: str | None) -> list[tuple[int, int, str | None, str]]:
    """(start_line, end_line, symbol, content), 1-based inclusive lines.
    Breaks at top-level symbol starts where possible, else fixed windows."""
    lines = text.splitlines()
    if not lines:
        return []
    starts = []
    for i, line in enumerate(lines):
        m = SYMBOL.match(line)
        if m and (language == "markdown" or not line.startswith((" " * 8, "\t\t"))):
            starts.append((i, next(g for g in m.groups() if g)))
    chunks = []
    boundaries = [s for s, _ in starts] or [0]
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    names = {s: n for s, n in starts}
    boundaries.append(len(lines))
    for a, b in zip(boundaries, boundaries[1:]):
        if b <= a:
            continue
        symbol = names.get(a)
        pos = a
        while pos < b:
            end = min(pos + CHUNK_LINES, b)
            body = "\n".join(lines[pos:end]).strip("\n")
            if body.strip():
                chunks.append((pos + 1, end, symbol, body))
            if end >= b:
                break
            pos = end - OVERLAP_LINES
    # Merge tiny neighbours so one-line symbols do not each cost a row.
    merged: list[list] = []
    for c in chunks:
        if merged and (c[1] - merged[-1][0] + 1) <= CHUNK_LINES // 2 and merged[-1][1] + 1 >= c[0]:
            prev = merged[-1]
            extra = "\n".join(lines[prev[1]:c[1]])
            merged[-1] = [prev[0], c[1], prev[2] or c[2], (prev[3] + "\n" + extra).strip("\n")]
        else:
            merged.append(list(c))
    return [tuple(m) for m in merged]


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


# ----------------------------------------------------------- indexing

_jobs: dict[int, dict] = {}
_jobs_lock = threading.Lock()


def job_status(project_id: int) -> dict:
    with _jobs_lock:
        return dict(_jobs.get(project_id) or {"state": "idle"})


def index_project(s: st.Store, project_id: int, progress=None) -> dict:
    project = s.get_project(project_id)
    if not project or not project.get("repository_path"):
        raise ValueError("The project has no repository path.")
    root = Path(project["repository_path"]).expanduser()
    if not root.is_dir():
        raise ValueError(f"Repository not found: {root}")
    started = time.time()
    conn = s.conn
    known = {r["path"]: r["sha"] for r in conn.execute(
        "SELECT path, sha FROM code_files WHERE project_id = ?", (project_id,))}
    seen, added, updated, skipped, chunks = set(), 0, 0, 0, 0
    files = list(walk_repo(root))
    for n, (rel, path) in enumerate(files, 1):
        seen.add(rel)
        text = read_text(path)
        if text is None:
            continue
        sha = sha_text(text)
        if known.get(rel) == sha:
            skipped += 1
            continue
        lang = language_of(path)
        pieces = chunk_source(text, lang)
        conn.execute("DELETE FROM code_chunks WHERE project_id = ? AND path = ?", (project_id, rel))
        conn.executemany(
            "INSERT INTO code_chunks(project_id, path, start_line, end_line, symbol, language, content) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(project_id, rel, a, b, sym, lang, body) for a, b, sym, body in pieces])
        conn.execute(
            "INSERT INTO code_files(project_id, path, sha, language, size, indexed_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(project_id, path) DO UPDATE SET sha = excluded.sha, language = excluded.language, "
            "size = excluded.size, indexed_at = excluded.indexed_at",
            (project_id, rel, sha, lang, len(text), time.time()))
        chunks += len(pieces)
        if rel in known:
            updated += 1
        else:
            added += 1
        if n % 50 == 0:
            conn.commit()
            if progress:
                progress(n, len(files))
    removed = [p for p in known if p not in seen]
    for rel in removed:
        conn.execute("DELETE FROM code_chunks WHERE project_id = ? AND path = ?", (project_id, rel))
        conn.execute("DELETE FROM code_files WHERE project_id = ? AND path = ?", (project_id, rel))
    meta = {**(project.get("metadata") or {}), "last_indexed": time.time(),
            "indexed_files": len(seen)}
    s.update_project(project_id, metadata=meta)
    conn.commit()
    return {"files": len(seen), "added": added, "updated": updated, "removed": len(removed),
            "unchanged": skipped, "chunks": chunks, "seconds": round(time.time() - started, 2)}


def index_in_background(store_path: Path, project_id: int) -> bool:
    """Index on a worker thread with its own connection. Returns False when a
    job for this project is already running."""
    with _jobs_lock:
        if (_jobs.get(project_id) or {}).get("state") == "running":
            return False
        _jobs[project_id] = {"state": "running", "done": 0, "total": 0, "started": time.time()}

    def progress(done, total):
        with _jobs_lock:
            _jobs[project_id].update(done=done, total=total)

    def work():
        s = st.Store(store_path)
        try:
            result = index_project(s, project_id, progress)
            with _jobs_lock:
                _jobs[project_id] = {"state": "done", "result": result, "finished": time.time()}
        except Exception as exc:
            with _jobs_lock:
                _jobs[project_id] = {"state": "error", "error": str(exc), "finished": time.time()}
        finally:
            s.close()

    threading.Thread(target=work, daemon=True, name=f"index-{project_id}").start()
    return True


# ---------------------------------------------------------- retrieval

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _identifier_terms(text: str) -> list[str]:
    """Identifiers and their camelCase / snake_case parts."""
    terms = []
    for ident in IDENT.findall(text):
        terms.append(ident)
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", ident)
        terms += [p for p in parts if len(p) > 2]
    return terms


def retrieve(s: st.Store, project_id: int, question: str, k: int = 8,
             mentioned: list[str] | None = None) -> list[dict]:
    """Best chunks for a question: BM25 over path, symbol and content, a
    boost for files the user named, and at most two chunks per file."""
    conn = s.conn
    paths = [r["path"] for r in conn.execute("SELECT path FROM code_files WHERE project_id = ?",
                                             (project_id,))]
    named = set(mentioned or [])
    for p in paths:
        base = p.rsplit("/", 1)[-1]
        if p in question or (len(base) > 3 and re.search(rf"(?<![\w/]){re.escape(base)}\b", question)):
            named.add(p)
    query = v.fts_query(" ".join(_identifier_terms(question)) + " " + question, max_terms=16)
    scored: dict[int, dict] = {}
    if query:
        rows = conn.execute(
            "SELECT c.*, bm25(code_chunks_fts, 4.0, 3.0, 1.0) AS rank FROM code_chunks_fts "
            "JOIN code_chunks c ON c.id = code_chunks_fts.rowid "
            "WHERE code_chunks_fts MATCH ? AND c.project_id = ? ORDER BY rank LIMIT ?",
            (query, project_id, k * 6)).fetchall()
        best = min((r["rank"] for r in rows), default=0.0) or -1.0
        for r in rows:
            d = dict(r)
            d["score"] = round(float(r["rank"] / best), 3) if best else 0.0
            scored[r["id"]] = d
    for p in named:
        for r in conn.execute("SELECT * FROM code_chunks WHERE project_id = ? AND path = ? "
                              "ORDER BY start_line LIMIT 3", (project_id, p)):
            d = scored.get(r["id"]) or {**dict(r), "score": 0.0}
            d["score"] = round(d["score"] + 1.0, 3)
            scored[r["id"]] = d
    ranked = sorted(scored.values(), key=lambda d: d["score"], reverse=True)
    out, per_file = [], {}
    for d in ranked:
        cap = 3 if d["path"] in named else 2
        if per_file.get(d["path"], 0) >= cap:
            continue
        per_file[d["path"]] = per_file.get(d["path"], 0) + 1
        d.pop("rank", None)
        out.append(d)
        if len(out) >= k:
            break
    return out


def list_files(s: st.Store, project_id: int, query: str = "", limit: int = 500) -> list[dict]:
    sql = "SELECT path, language, size FROM code_files WHERE project_id = ?"
    args: list = [project_id]
    if query:
        sql += " AND path LIKE ?"
        args.append(f"%{query}%")
    return [dict(r) for r in s.conn.execute(sql + " ORDER BY path LIMIT ?", (*args, limit))]


# ---------------------------------------------------- safe file access


class PathError(ValueError):
    pass


def safe_path(root: str | Path, rel: str) -> Path:
    """Resolve `rel` inside `root`, refusing anything that escapes it
    (.., absolute paths, symlinks pointing outside, the .git folder)."""
    base = Path(root).expanduser().resolve()
    rel = (rel or "").strip().replace("\\", "/")
    if not rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        raise PathError(f"Not a relative path inside the repository: {rel!r}")
    target = (base / rel).resolve()
    if target != base and base not in target.parents:
        raise PathError(f"Path escapes the repository: {rel}")
    if ".git" in Path(rel).parts:
        raise PathError("Refusing to touch the .git folder.")
    return target


def read_file(root: str | Path, rel: str, start: int = 1, end: int | None = None,
              max_chars: int = 60_000) -> dict:
    path = safe_path(root, rel)
    text = read_text(path)
    if text is None:
        raise PathError(f"Cannot read {rel} (missing or binary).")
    lines = text.splitlines()
    end = min(end or len(lines), len(lines))
    body = "\n".join(lines[max(0, start - 1):end])
    return {"path": rel, "start_line": start, "end_line": end, "total_lines": len(lines),
            "language": language_of(path), "content": body[:max_chars],
            "truncated": len(body) > max_chars}


# ------------------------------------------------------- edit proposals

EDIT_BLOCK = re.compile(
    r"(?:^|\n)(?:```[\w+-]*\n)?(?P<path>[^\n`<>=]+?)\s*\n(?:```[\w+-]*\n)?"
    r"<{5,9} SEARCH\s*\n(?P<search>.*?)\n?={5,9}\s*\n(?P<replace>.*?)\n?>{5,9} REPLACE",
    re.S)


@dataclass
class Edit:
    path: str
    search: str
    replace: str


def parse_edit_blocks(text: str) -> list[Edit]:
    edits = []
    for m in EDIT_BLOCK.finditer(text or ""):
        path = m.group("path").strip().strip("`*:").strip()
        path = re.sub(r"^(?:file|path)\s*[:=]\s*", "", path, flags=re.I).strip()
        if not path or " " in path and "/" not in path and "." not in path:
            continue
        edits.append(Edit(path, m.group("search"), m.group("replace")))
    return edits


def _locate(content: str, search: str) -> tuple[int, int] | None:
    """Exact match first; then a match that ignores trailing whitespace and
    indentation differences line by line, which small models often get wrong."""
    if not search.strip():
        return None
    idx = content.find(search)
    if idx != -1 and content.find(search, idx + 1) == -1:
        return idx, idx + len(search)
    if idx != -1:
        raise ValueError("The SEARCH text appears more than once; include more surrounding lines.")
    want = [ln.strip() for ln in search.strip("\n").splitlines()]
    lines = content.splitlines(keepends=True)
    hits = []
    for i in range(len(lines) - len(want) + 1):
        if all(lines[i + j].strip() == want[j] for j in range(len(want))):
            hits.append(i)
    if len(hits) == 1:
        i = hits[0]
        start = sum(len(x) for x in lines[:i])
        end = start + sum(len(x) for x in lines[i:i + len(want)])
        return start, end
    if len(hits) > 1:
        raise ValueError("The SEARCH text appears more than once; include more surrounding lines.")
    return None


def build_proposal(s: st.Store, project_id: int, edits: list[Edit], summary: str = "",
                   conversation_id: int | None = None, message_id: int | None = None) -> dict:
    """Apply edits in memory and store the result as a pending proposal with
    a unified diff. Nothing on disk changes here."""
    project = s.get_project(project_id)
    if not project or not project.get("repository_path"):
        raise ValueError("The project has no repository path.")
    root = project["repository_path"]
    files: dict[str, dict] = {}
    problems = []
    for e in edits:
        try:
            path = safe_path(root, e.path)
        except PathError as exc:
            problems.append(str(exc))
            continue
        entry = files.get(e.path)
        if entry is None:
            original = read_text(path) if path.exists() else ""
            if original is None:
                problems.append(f"{e.path}: binary or unreadable file")
                continue
            entry = files[e.path] = {"path": e.path, "original": original, "new": original,
                                     "existed": path.exists()}
        if not e.search.strip():
            if entry["existed"] and entry["original"].strip():
                problems.append(f"{e.path}: empty SEARCH on an existing file (would replace it entirely)")
                continue
            entry["new"] = e.replace if e.replace.endswith("\n") else e.replace + "\n"
            continue
        try:
            span = _locate(entry["new"], e.search)
        except ValueError as exc:
            problems.append(f"{e.path}: {exc}")
            continue
        if span is None:
            problems.append(f"{e.path}: SEARCH text not found")
            continue
        a, b = span
        replacement = e.replace
        if entry["new"][a:b].endswith("\n") and not replacement.endswith("\n"):
            replacement += "\n"
        entry["new"] = entry["new"][:a] + replacement + entry["new"][b:]

    changed = [f for f in files.values() if f["new"] != f["original"]]
    if not changed:
        raise ValueError("No applicable changes. " + "; ".join(problems))
    diff = "".join(
        "".join(difflib.unified_diff(f["original"].splitlines(keepends=True),
                                     f["new"].splitlines(keepends=True),
                                     fromfile=f"a/{f['path']}" if f["existed"] else "/dev/null",
                                     tofile=f"b/{f['path']}"))
        + ("\n" if not f["new"].endswith("\n") else "")
        for f in changed)
    stored = [{"path": f["path"], "original_sha": sha_text(f["original"]) if f["existed"] else None,
               "original": f["original"], "new": f["new"], "existed": f["existed"]} for f in changed]
    cur = s.conn.execute(
        "INSERT INTO change_proposals(project_id, conversation_id, message_id, status, summary, edits, diff, "
        "result, created_at) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?)",
        (project_id, conversation_id, message_id, summary[:500], json.dumps(stored), diff,
         json.dumps({"problems": problems}) if problems else None, time.time()))
    s.conn.commit()
    return {"id": int(cur.lastrowid), "diff": diff, "files": [f["path"] for f in changed],
            "problems": problems, "status": "pending"}


def get_proposal(s: st.Store, pid: int, with_content: bool = False) -> dict | None:
    row = s.conn.execute("SELECT * FROM change_proposals WHERE id = ?", (pid,)).fetchone()
    if not row:
        return None
    d = dict(row)
    edits = st.loads(d.pop("edits"), [])
    d["files"] = [e["path"] for e in edits]
    d["result"] = st.loads(d.get("result"), None)
    if with_content:
        d["edits"] = edits
    return d


def list_proposals(s: st.Store, project_id: int | None = None, status: str | None = None,
                   limit: int = 50) -> list[dict]:
    sql, args = "SELECT id FROM change_proposals WHERE 1=1", []
    if project_id:
        sql += " AND project_id = ?"
        args.append(project_id)
    if status:
        sql += " AND status = ?"
        args.append(status)
    ids = [r[0] for r in s.conn.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit))]
    return [get_proposal(s, i) for i in ids]


def _set_status(s: st.Store, pid: int, status: str, result: dict) -> None:
    s.conn.execute("UPDATE change_proposals SET status = ?, result = ?, decided_at = ? WHERE id = ?",
                   (status, json.dumps(result), time.time(), pid))
    s.conn.commit()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.vault-chat-tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def apply_proposal(s: st.Store, pid: int) -> dict:
    """Write an approved proposal. Every file is checked before any is
    written, so a stale file aborts the whole change, not half of it."""
    p = get_proposal(s, pid, with_content=True)
    if not p:
        raise ValueError("No such proposal.")
    if p["status"] != "pending":
        raise ValueError(f"Proposal is {p['status']}, not pending.")
    project = s.get_project(p["project_id"])
    root = project["repository_path"]
    targets = []
    for e in p["edits"]:
        path = safe_path(root, e["path"])
        current = read_text(path) if path.exists() else None
        if e["existed"] and (current is None or sha_text(current) != e["original_sha"]):
            _set_status(s, pid, "stale", {"error": f"{e['path']} changed after the diff was made."})
            raise ValueError(f"{e['path']} changed since the diff was made; ask for a fresh change.")
        if not e["existed"] and path.exists():
            _set_status(s, pid, "stale", {"error": f"{e['path']} now exists."})
            raise ValueError(f"{e['path']} was created since the diff was made.")
        targets.append((path, e))
    written = []
    try:
        for path, e in targets:
            _atomic_write(path, e["new"])
            written.append(e["path"])
    except OSError as exc:
        _set_status(s, pid, "failed", {"error": str(exc), "written": written})
        raise ValueError(f"Write failed after {len(written)} file(s): {exc}") from exc
    _set_status(s, pid, "applied", {"written": written})
    return {"status": "applied", "written": written}


def reject_proposal(s: st.Store, pid: int) -> dict:
    p = get_proposal(s, pid)
    if not p:
        raise ValueError("No such proposal.")
    if p["status"] != "pending":
        raise ValueError(f"Proposal is {p['status']}, not pending.")
    _set_status(s, pid, "rejected", {})
    return {"status": "rejected"}


def revert_proposal(s: st.Store, pid: int) -> dict:
    """Undo an applied proposal, if the files still hold what it wrote."""
    p = get_proposal(s, pid, with_content=True)
    if not p or p["status"] != "applied":
        raise ValueError("Only applied proposals can be reverted.")
    root = s.get_project(p["project_id"])["repository_path"]
    for e in p["edits"]:
        path = safe_path(root, e["path"])
        current = read_text(path) if path.exists() else None
        if current != e["new"]:
            raise ValueError(f"{e['path']} was edited after the change was applied; revert by hand.")
    for e in p["edits"]:
        path = safe_path(root, e["path"])
        if e["existed"]:
            _atomic_write(path, e["original"])
        elif path.exists():
            path.unlink()
    _set_status(s, pid, "reverted", {"reverted": [e["path"] for e in p["edits"]]})
    return {"status": "reverted"}
