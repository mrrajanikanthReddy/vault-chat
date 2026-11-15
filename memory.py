"""The memory subsystem. It lives entirely outside the inference engine, so
memories survive restarts, model swaps and provider changes.

Three levels:

  short-term   the current conversation (context.py reads it from messages),
               open tasks, and `short_term` memories that expire after days
  long-term    preferences, goals, decisions, technical and project context,
               and personal facts the user explicitly asked to keep
  project      per-project architecture, decisions, standards, known issues
               and resolutions (`project_memories`)

Extraction is rule-based by default: instant, free, and it never invents a
fact. Optional LLM extraction can be switched on; it costs one model call per
turn, which matters on a CPU. Only user messages are mined, never model
output, so the assistant cannot "remember" its own hallucinations.

Retrieval scores = keyword relevance (BM25) blended with importance, recency
and how often a memory proved useful; cosine similarity joins in when an
embeddings server is configured. Nothing injects the whole database.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass

import numpy as np

import store as st
import vaultdb as v

DAY = 86400.0

CATEGORIES = ("preference", "personal", "project", "decision", "goal", "technical",
              "topic", "task_context", "fact")
PROJECT_TYPES = ("architecture", "decision", "standard", "documentation", "issue",
                 "resolution", "context")

# Long-term categories do not expire; these do, unless pinned or reinforced.
EXPIRY_DAYS = {"task_context": 7, "goal": 180, "topic": 90}
DEFAULT_IMPORTANCE = {"preference": 0.75, "personal": 0.8, "project": 0.8, "decision": 0.8,
                      "goal": 0.7, "technical": 0.65, "topic": 0.4, "task_context": 0.5,
                      "fact": 0.6}


# ---------------------------------------------------------- sensitivity

_SENSITIVE = [
    ("email address", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("phone number", re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){9,14}\d(?!\d)")),
    ("card number", re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")),
    ("government id", re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b(?:ssn|passport|aadhaar|pan card|"
                                 r"social security|national insurance)\b", re.I)),
    ("bank details", re.compile(r"\b(?:iban|swift|routing number|account number|sort code)\b", re.I)),
    ("credential", re.compile(r"\b(?:password|passcode|passwd|pin code|secret key|api[_ -]?key|"
                              r"private key|access token|otp)\b", re.I)),
    ("secret token", re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
                                r"xox[bpas]-[A-Za-z0-9-]{10,}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+)")),
    ("health information", re.compile(r"\b(?:diagnos\w*|medication|prescription|therapy|"
                                      r"illness|disease|disorder|pregnan\w*|hiv|cancer)\b", re.I)),
    ("financial details", re.compile(r"\b(?:salary|income|net worth|debt|loan amount|"
                                     r"credit score|tax return)\b", re.I)),
    ("home address", re.compile(r"\b(?:i live at|my address is|home address)\b", re.I)),
    ("beliefs or orientation", re.compile(r"\b(?:my religion|i am (?:gay|lesbian|bisexual|"
                                          r"transgender)|my political)\b", re.I)),
]


def sensitivity(text: str) -> str | None:
    """The kind of sensitive data the text appears to hold, or None."""
    for label, pattern in _SENSITIVE:
        if pattern.search(text or ""):
            return label
    return None


# ------------------------------------------------------------ extraction


@dataclass
class Candidate:
    content: str
    category: str
    memory_type: str = "long_term"
    importance: float = 0.6
    confidence: float = 0.7
    origin: str = "auto"
    project_type: str | None = None  # set when it belongs in project memory


_CLAUSE_END = re.compile(r"(?<=[.!?])\s+|\n+")

_RULES = [
    # (pattern, category, project_type, rewrite)
    (re.compile(r"\b(?:i am|i'm|im|we are|we're)\s+(?:currently\s+|now\s+)?(?:working on|building|"
                r"implementing|developing|migrating|designing|setting up)\s+(?P<x>.{6,200})", re.I),
     "project", "context", "Working on {x}"),
    (re.compile(r"\bmy (?:current |main )?project is\s+(?P<x>.{4,200})", re.I),
     "project", "context", "Current project: {x}"),
    (re.compile(r"\b(?:i|we)\s+(?:always |usually |generally )?(?:prefer|like to use|rather use)\s+"
                r"(?P<x>.{3,200})", re.I),
     "preference", None, "Prefers {x}"),
    (re.compile(r"\b(?:please\s+)?(?:always|never)\s+(?P<x>(?:use|write|reply|answer|respond|format|"
                r"include|add|show|give|explain)\b.{3,200})", re.I),
     "preference", None, "Instruction: {m}"),
    (re.compile(r"\bi (?:want|would like|need) (?:you|answers|responses) to\s+(?P<x>.{3,200})", re.I),
     "preference", None, "Wants responses to {x}"),
    (re.compile(r"\b(?:we|i)\s+(?:have\s+)?(?:decided|agreed|chose|chosen|settled on)\s+"
                r"(?:to\s+|on\s+|that\s+)?(?P<x>.{4,200})", re.I),
     "decision", "decision", "Decided: {x}"),
    (re.compile(r"\bthe decision (?:is|was) (?:to\s+)?(?P<x>.{4,200})", re.I),
     "decision", "decision", "Decided: {x}"),
    (re.compile(r"\bmy (?:goal|objective|aim) is (?:to\s+)?(?P<x>.{4,200})", re.I),
     "goal", None, "Goal: {x}"),
    (re.compile(r"\bi(?:'m| am) (?:trying|planning|aiming) to\s+(?P<x>.{4,200})", re.I),
     "goal", None, "Goal: {x}"),
    (re.compile(r"\b(?:our|my|the) (?:tech )?stack (?:is|uses|includes)\s+(?P<x>.{3,200})", re.I),
     "technical", "architecture", "Tech stack: {x}"),
    (re.compile(r"\bwe use\s+(?P<x>.{3,120}?\s+for\s+.{3,120})", re.I),
     "technical", "architecture", "Uses {x}"),
    (re.compile(r"\b(?:our|the) (?:coding )?(?:standard|convention|guideline) is\s+(?P<x>.{4,200})", re.I),
     "technical", "standard", "Standard: {x}"),
    (re.compile(r"\b(?:the )?(?:issue|bug|problem|root cause) (?:was|is)\s+(?P<x>.{6,200})", re.I),
     "fact", "issue", "Issue: {x}"),
    (re.compile(r"\b(?:it was |this was |we )?(?:fixed|resolved|solved) (?:it |this )?(?:by|with)\s+"
                r"(?P<x>.{4,200})", re.I),
     "fact", "resolution", "Resolved by {x}"),
    (re.compile(r"\btoday i(?:'m| am)\s+(?P<x>.{4,200})", re.I),
     "task_context", None, "Today: {x}"),
    (re.compile(r"\bi(?:'m| am) (?:currently )?(?:debugging|investigating|troubleshooting|fixing)\s+"
                r"(?P<x>.{4,200})", re.I),
     "task_context", "issue", "Investigating {x}"),
]

_EXPLICIT = re.compile(r"^\s*(?:please\s+)?(?:remember|note|keep in mind|don't forget)"
                       r"(?:\s+that)?[\s,:]+(?P<x>.{3,500})", re.I)
_FORGET = re.compile(r"^\s*(?:please\s+)?forget(?:\s+that|\s+about)?\s+(?P<x>.{3,300})", re.I)
_PERSONAL = re.compile(r"\b(?:my name is|i live in|i work (?:at|for)|my role is|i am a|i'm a|"
                       r"my (?:wife|husband|partner|son|daughter|kid)|my birthday)\b", re.I)


def _clean(fragment: str) -> str:
    fragment = re.split(r"(?<=[.!?;])\s|\s+(?:but|because|so that|and then)\s", fragment, maxsplit=1)[0]
    return fragment.strip(" .,;:!?\"'`").strip()


def classify(text: str) -> str:
    for pattern, category, _, _ in _RULES:
        if pattern.search(text):
            return category
    if _PERSONAL.search(text):
        return "personal"
    return "fact"


def extract(text: str) -> tuple[list[Candidate], list[str], list[str]]:
    """Mine one user message. Returns (candidates, forget_requests, refusals)
    where refusals explain what was deliberately not stored."""
    candidates: list[Candidate] = []
    forget: list[str] = []
    refused: list[str] = []
    text = (text or "").strip()
    if not text or text.startswith("```") or len(text) > 4000:
        return candidates, forget, refused

    m = _FORGET.match(text)
    if m:
        forget.append(_clean(m.group("x")))
        return candidates, forget, refused

    m = _EXPLICIT.match(text)
    if m:
        content = _clean(m.group("x")) or m.group("x").strip()
        kind = sensitivity(content)
        if kind:
            refused.append(f"Not remembered automatically: it looks like {kind}. "
                           "Add it in the Memory Manager if you really want it kept.")
        else:
            category = classify(content)
            candidates.append(Candidate(content[0].upper() + content[1:], category,
                                        importance=max(0.8, DEFAULT_IMPORTANCE.get(category, 0.6)),
                                        confidence=1.0, origin="explicit"))
        return candidates, forget, refused

    for clause in _CLAUSE_END.split(text):
        if len(clause) < 8 or clause.rstrip().endswith("?"):
            continue
        kind = sensitivity(clause)
        if kind:
            refused.append(f"Skipped a statement that looks like {kind}.")
            continue
        if _PERSONAL.search(clause):
            continue  # personal facts are kept only when the user says "remember"
        for pattern, category, project_type, template in _RULES:
            match = pattern.search(clause)
            if not match:
                continue
            x = _clean(match.group("x"))
            if len(x) < 3:
                continue
            content = template.format(x=x, m=_clean(match.group(0)))
            content = content[0].upper() + content[1:]
            memory_type = "short_term" if category == "task_context" else "long_term"
            candidates.append(Candidate(content, category, memory_type,
                                        DEFAULT_IMPORTANCE.get(category, 0.6), 0.7, "auto",
                                        project_type))
            break
    return candidates, forget, refused


LLM_EXTRACTION_PROMPT = """Extract durable facts about the USER from the message below that would help in future conversations: preferences, goals, decisions, projects, technical context.
Rules: only facts the user stated; no guesses; no sensitive data (credentials, health, finances, contact details, ids); at most 3 items; each under 25 words.
Reply with JSON only: [{"content": "...", "category": "preference|project|decision|goal|technical|fact"}] or [].

Message:
"""


def parse_llm_candidates(raw: str) -> list[Candidate]:
    match = re.search(r"\[.*\]", raw or "", re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except ValueError:
        return []
    out = []
    for item in items[:3] if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        category = item.get("category") if item.get("category") in CATEGORIES else "fact"
        if 6 <= len(content) <= 300 and not sensitivity(content):
            out.append(Candidate(content, category, importance=DEFAULT_IMPORTANCE.get(category, 0.6),
                                 confidence=0.6, origin="llm"))
    return out


# ------------------------------------------------------------- manager


def _tokens(text: str) -> set[str]:
    return {w for w in st.normalize(text).split() if len(w) > 2 and w not in v.STOPWORDS}


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


DUPLICATE_THRESHOLD = 0.6


class MemoryManager:
    def __init__(self, store: st.Store, embedder=None):
        self.store = store
        self.embedder = embedder  # optional providers.EmbeddingProvider

    @property
    def conn(self):
        return self.store.conn

    # -- writes

    def _vector(self, text: str) -> bytes | None:
        if not self.embedder:
            return None
        try:
            vec = np.asarray(self.embedder.embed([text])[0], dtype=np.float32)
            return (vec / max(float(np.linalg.norm(vec)), 1e-9)).tobytes()
        except Exception:
            return None  # semantic search is optional; never block a write on it

    def find_duplicate(self, content: str, category: str | None = None) -> dict | None:
        norm = st.normalize(content)
        row = self.conn.execute("SELECT * FROM memories WHERE user_id = ? AND normalized = ?",
                                (st.DEFAULT_USER, norm)).fetchone()
        if row:
            return dict(row)
        query = v.fts_query(content)
        if not query:
            return None
        rows = self.conn.execute(
            "SELECT m.* FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid "
            "WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts) LIMIT 8", (query,)).fetchall()
        best, best_score = None, 0.0
        for r in rows:
            if category and r["category"] != category and r["category"] != "fact":
                continue
            score = jaccard(content, r["content"])
            if score > best_score:
                best, best_score = r, score
        return dict(best) if best is not None and best_score >= DUPLICATE_THRESHOLD else None

    def add(self, content: str, category: str = "fact", memory_type: str = "long_term",
            importance: float | None = None, confidence: float = 0.7, origin: str = "manual",
            source_conversation_id: int | None = None, source_message_id: int | None = None,
            expires_in_days: float | None = None, pinned: bool = False,
            allow_sensitive: bool = False) -> dict:
        """Insert or reinforce a memory. Returns {"id", "action", ...} where
        action is created, updated or refused."""
        content = " ".join((content or "").split())[:1000]
        if not content:
            return {"action": "refused", "reason": "empty"}
        if not allow_sensitive and sensitivity(content):
            return {"action": "refused", "reason": f"looks like {sensitivity(content)}"}
        category = category if category in CATEGORIES else "fact"
        importance = DEFAULT_IMPORTANCE.get(category, 0.6) if importance is None else importance
        t = st.now()
        if expires_in_days is None:
            expires_in_days = EXPIRY_DAYS.get(category)
        expires_at = None if (pinned or not expires_in_days) else t + expires_in_days * DAY

        dup = self.find_duplicate(content, category)
        if dup:
            # Reinforce: newer wording wins, confidence and importance only rise.
            new_conf = min(1.0, max(dup["confidence"], confidence) + 0.1)
            new_imp = max(dup["importance"], importance)
            keep_content = content if origin in ("explicit", "manual") or len(content) >= len(dup["content"]) \
                else dup["content"]
            new_expiry = None if dup["expires_at"] is None and dup["memory_type"] == "long_term" else expires_at
            try:
                self.conn.execute(
                    "UPDATE memories SET content = ?, normalized = ?, confidence = ?, importance = ?, "
                    "updated_at = ?, expires_at = ?, memory_type = CASE WHEN ? = 'long_term' THEN "
                    "'long_term' ELSE memory_type END, vec = COALESCE(?, vec) WHERE id = ?",
                    (keep_content, st.normalize(keep_content), new_conf, new_imp, t, new_expiry,
                     memory_type, self._vector(keep_content) if keep_content != dup["content"] else None,
                     dup["id"]))
            except Exception:
                self.conn.execute("UPDATE memories SET confidence = ?, importance = ?, updated_at = ? "
                                  "WHERE id = ?", (new_conf, new_imp, t, dup["id"]))
            self.conn.commit()
            return {"id": dup["id"], "action": "updated", "content": keep_content, "category": dup["category"]}

        cur = self.conn.execute(
            "INSERT INTO memories(user_id, memory_type, category, content, normalized, "
            "source_conversation_id, source_message_id, origin, importance, confidence, pinned, vec, "
            "expires_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (st.DEFAULT_USER, memory_type, category, content, st.normalize(content),
             source_conversation_id, source_message_id, origin, importance, confidence, int(pinned),
             self._vector(content), expires_at, t, t))
        self.conn.commit()
        return {"id": int(cur.lastrowid), "action": "created", "content": content, "category": category}

    def update(self, mid: int, **fields) -> bool:
        allowed = {"content", "category", "importance", "confidence", "pinned", "memory_type", "expires_at"}
        sets, args = [], []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "content":
                value = " ".join(str(value).split())[:1000]
                sets.append("normalized = ?")
                args.append(st.normalize(value))
                vec = self._vector(value)
                if vec is not None:
                    sets.append("vec = ?")
                    args.append(vec)
            if key == "pinned" and value:
                sets.append("expires_at = NULL")
            sets.append(f"{key} = ?")
            args.append(value)
        if not sets:
            return False
        sets.append("updated_at = ?")
        args += [st.now(), mid]
        cur = self.conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", args)
        self.conn.commit()
        return cur.rowcount > 0

    def delete(self, mid: int) -> bool:
        cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
        self.conn.commit()
        return cur.rowcount > 0

    def delete_all(self) -> int:
        cur = self.conn.execute("DELETE FROM memories WHERE user_id = ?", (st.DEFAULT_USER,))
        self.conn.commit()
        return cur.rowcount

    def forget(self, description: str) -> list[dict]:
        """Delete memories matching a "forget ..." request."""
        removed = []
        for m in self.search(description, limit=5, touch=False):
            if m["relevance"] >= 0.5 or jaccard(description, m["content"]) >= 0.3:
                self.delete(m["id"])
                removed.append(m)
        return removed

    def expire(self) -> int:
        cur = self.conn.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ? "
                                "AND pinned = 0", (st.now(),))
        self.conn.commit()
        return cur.rowcount

    # -- reads

    def get(self, mid: int) -> dict | None:
        return st.row_dict(self.conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone())

    def list(self, category: str | None = None, memory_type: str | None = None, query: str = "",
             limit: int = 200, offset: int = 0) -> list[dict]:
        if query:
            return self.search(query, limit=limit, touch=False, category=category)
        where, args = ["user_id = ?"], [st.DEFAULT_USER]
        if category:
            where.append("category = ?")
            args.append(category)
        if memory_type:
            where.append("memory_type = ?")
            args.append(memory_type)
        rows = self.conn.execute(
            f"SELECT * FROM memories WHERE {' AND '.join(where)} ORDER BY pinned DESC, updated_at DESC "
            f"LIMIT ? OFFSET ?", (*args, limit, offset))
        return [st.row_dict(r) for r in rows]

    def search(self, query: str, limit: int = 6, touch: bool = True, category: str | None = None,
               min_score: float = 0.0, min_relevance: float = 0.0) -> list[dict]:
        """Rank memories for a query. Each result carries `score` (final) and
        `relevance` (text match only, 0..1)."""
        t = st.now()
        candidates: dict[int, dict] = {}
        fts = v.fts_query(query)
        if fts:
            sql = ("SELECT m.*, bm25(memories_fts) AS rank FROM memories_fts "
                   "JOIN memories m ON m.id = memories_fts.rowid WHERE memories_fts MATCH ? "
                   "AND (m.expires_at IS NULL OR m.expires_at > ?)")
            args: list = [fts, t]
            if category:
                sql += " AND m.category = ?"
                args.append(category)
            rows = self.conn.execute(sql + " ORDER BY rank LIMIT 40", args).fetchall()
            best = min((r["rank"] for r in rows), default=0.0) or -1.0
            for r in rows:
                d = dict(r)
                d["relevance"] = float(r["rank"] / best) if best else 0.0
                candidates[r["id"]] = d

        if self.embedder:
            self._blend_semantic(query, candidates, t, category)

        for d in candidates.values():
            recency = math.exp(-(t - (d["updated_at"] or t)) / (60 * DAY))
            use = min(1.0, (d["access_count"] or 0) / 10)
            d["score"] = round(0.55 * d["relevance"] + 0.2 * d["importance"] + 0.15 * recency
                               + 0.05 * use + 0.05 * d["confidence"] + (0.1 if d["pinned"] else 0.0), 4)
        ranked = sorted(candidates.values(), key=lambda d: d["score"], reverse=True)
        ranked = [d for d in ranked
                  if d["score"] >= min_score and d["relevance"] >= min_relevance][:limit]
        if touch and ranked:
            self.conn.executemany(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
                [(t, d["id"]) for d in ranked])
            self.conn.commit()
        return [{k: val for k, val in d.items() if k not in ("vec", "normalized", "rank")} for d in ranked]

    def _blend_semantic(self, query: str, candidates: dict, t: float, category: str | None) -> None:
        try:
            q = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        except Exception:
            return
        q /= max(float(np.linalg.norm(q)), 1e-9)
        sql = "SELECT * FROM memories WHERE vec IS NOT NULL AND (expires_at IS NULL OR expires_at > ?)"
        args: list = [t]
        if category:
            sql += " AND category = ?"
            args.append(category)
        for r in self.conn.execute(sql, args):
            vec = np.frombuffer(r["vec"], dtype=np.float32)
            if vec.shape != q.shape:
                continue
            cos = float(vec @ q)
            if cos < 0.35 and r["id"] not in candidates:
                continue
            d = candidates.get(r["id"]) or {**dict(r), "relevance": 0.0}
            d["relevance"] = max(d["relevance"], cos)
            candidates[r["id"]] = d

    def profile(self, limit: int = 8) -> list[dict]:
        """Always-relevant memories (explicit preferences, pinned items),
        kept short: these ride along with every new conversation."""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE user_id = ? AND (pinned = 1 OR (category = 'preference' AND "
            "confidence >= 0.8)) AND (expires_at IS NULL OR expires_at > ?) "
            "ORDER BY pinned DESC, importance DESC, updated_at DESC LIMIT ?",
            (st.DEFAULT_USER, st.now(), limit))
        return [st.row_dict(r) for r in rows]

    def export(self) -> dict:
        projects = {p["id"]: p["name"] for p in self.store.list_projects()}
        return {
            "exported_at": st.now(),
            "schema_version": self.store.schema_version(),
            "memories": self.list(limit=100000),
            "project_memories": [
                {**pm, "project": projects.get(pm["project_id"])}
                for pm in (st.row_dict(r) for r in self.conn.execute(
                    "SELECT * FROM project_memories ORDER BY project_id, id"))
            ],
        }

    def import_(self, data: dict) -> dict:
        created = updated = 0
        for m in data.get("memories") or []:
            res = self.add(m.get("content", ""), m.get("category", "fact"), m.get("memory_type", "long_term"),
                           m.get("importance"), m.get("confidence", 0.7), "manual",
                           pinned=bool(m.get("pinned")))
            created += res["action"] == "created"
            updated += res["action"] == "updated"
        return {"created": created, "updated": updated}

    # -- project memory

    def add_project_memory(self, project_id: int, content: str, memory_type: str = "context",
                           source: str = "", importance: float = 0.6) -> dict:
        content = " ".join((content or "").split())[:2000]
        if not content:
            return {"action": "refused", "reason": "empty"}
        if sensitivity(content) in ("credential", "secret token"):
            return {"action": "refused", "reason": "looks like a secret"}
        memory_type = memory_type if memory_type in PROJECT_TYPES else "context"
        norm = st.normalize(content)
        t = st.now()
        existing = self.conn.execute("SELECT id FROM project_memories WHERE project_id = ? AND normalized = ?",
                                     (project_id, norm)).fetchone()
        if existing is None:
            for r in self.conn.execute("SELECT id, content FROM project_memories WHERE project_id = ? "
                                       "AND memory_type = ?", (project_id, memory_type)):
                if jaccard(content, r["content"]) >= DUPLICATE_THRESHOLD:
                    existing = r
                    break
        if existing is not None:
            self.conn.execute("UPDATE project_memories SET content = ?, normalized = ?, updated_at = ?, "
                              "importance = MAX(importance, ?) WHERE id = ?",
                              (content, norm, t, importance, existing["id"]))
            self.conn.commit()
            return {"id": existing["id"], "action": "updated", "content": content}
        cur = self.conn.execute(
            "INSERT INTO project_memories(project_id, memory_type, content, normalized, source, importance, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (project_id, memory_type, content, norm, source, importance, t, t))
        self.conn.commit()
        return {"id": int(cur.lastrowid), "action": "created", "content": content}

    def list_project_memories(self, project_id: int, memory_type: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM project_memories WHERE project_id = ?", [project_id]
        if memory_type:
            sql += " AND memory_type = ?"
            args.append(memory_type)
        return [st.row_dict(r) for r in self.conn.execute(sql + " ORDER BY updated_at DESC", args)]

    def update_project_memory(self, pmid: int, **fields) -> bool:
        sets, args = [], []
        if "content" in fields:
            sets += ["content = ?", "normalized = ?"]
            args += [fields["content"], st.normalize(fields["content"])]
        if fields.get("memory_type") in PROJECT_TYPES:
            sets.append("memory_type = ?")
            args.append(fields["memory_type"])
        if not sets:
            return False
        sets.append("updated_at = ?")
        args += [st.now(), pmid]
        cur = self.conn.execute(f"UPDATE project_memories SET {', '.join(sets)} WHERE id = ?", args)
        self.conn.commit()
        return cur.rowcount > 0

    def delete_project_memory(self, pmid: int) -> bool:
        cur = self.conn.execute("DELETE FROM project_memories WHERE id = ?", (pmid,))
        self.conn.commit()
        return cur.rowcount > 0

    def search_project(self, project_id: int, query: str, limit: int = 6) -> list[dict]:
        fts = v.fts_query(query)
        t = st.now()
        results: dict[int, dict] = {}
        if fts:
            rows = self.conn.execute(
                "SELECT pm.*, bm25(project_memories_fts) AS rank FROM project_memories_fts "
                "JOIN project_memories pm ON pm.id = project_memories_fts.rowid "
                "WHERE project_memories_fts MATCH ? AND pm.project_id = ? ORDER BY rank LIMIT 30",
                (fts, project_id)).fetchall()
            best = min((r["rank"] for r in rows), default=0.0) or -1.0
            for r in rows:
                d = st.row_dict(r)
                d["relevance"] = float(r["rank"] / best) if best else 0.0
                results[r["id"]] = d
        # Decisions and standards stay relevant to any question about the project.
        for r in self.conn.execute(
                "SELECT * FROM project_memories WHERE project_id = ? AND memory_type IN "
                "('decision', 'standard', 'architecture') ORDER BY importance DESC, updated_at DESC LIMIT 4",
                (project_id,)):
            results.setdefault(r["id"], {**st.row_dict(r), "relevance": 0.3})
        for d in results.values():
            recency = math.exp(-(t - d["updated_at"]) / (90 * DAY))
            d["score"] = round(0.6 * d["relevance"] + 0.25 * d["importance"] + 0.15 * recency, 4)
        ranked = sorted(results.values(), key=lambda d: d["score"], reverse=True)[:limit]
        if ranked:
            self.conn.executemany("UPDATE project_memories SET access_count = access_count + 1, "
                                  "last_accessed_at = ? WHERE id = ?", [(t, d["id"]) for d in ranked])
            self.conn.commit()
        for d in ranked:
            d.pop("rank", None)
        return ranked

    # -- the per-turn hook

    def process_user_message(self, text: str, conversation_id: int | None = None,
                             message_id: int | None = None, project_id: int | None = None,
                             auto: bool = True) -> dict:
        """Called after each user turn. Explicit "remember"/"forget" always
        work; automatic extraction only when `auto` is on."""
        candidates, forget, refused = extract(text)
        saved, removed = [], []
        for description in forget:
            removed += self.forget(description)
        for c in candidates:
            if c.origin != "explicit" and not auto:
                continue
            if c.project_type and project_id:
                res = self.add_project_memory(project_id, c.content, c.project_type,
                                              source=f"conversation:{conversation_id}")
                if res["action"] != "refused":
                    saved.append({**res, "scope": "project", "category": c.project_type})
                if c.category not in ("project", "decision", "technical"):
                    continue
            res = self.add(c.content, c.category, c.memory_type, c.importance, c.confidence, c.origin,
                           conversation_id, message_id)
            if res["action"] != "refused":
                saved.append({**res, "scope": "user"})
            else:
                refused.append(f"Not stored: {res['reason']}.")
        return {"saved": saved, "removed": [{"id": m["id"], "content": m["content"]} for m in removed],
                "refused": refused}

    def llm_extract(self, provider, text: str, conversation_id: int | None = None) -> list[dict]:
        """Optional model-assisted extraction (setting MEMORY_LLM_EXTRACTION)."""
        from providers import GenParams
        try:
            raw = provider.complete([{"role": "user", "content": LLM_EXTRACTION_PROMPT + text[:2000]}],
                                    GenParams(temperature=0.0, max_tokens=200))["content"]
        except Exception:
            return []
        saved = []
        for c in parse_llm_candidates(raw):
            res = self.add(c.content, c.category, "long_term", c.importance, c.confidence, "llm",
                           conversation_id)
            if res["action"] != "refused":
                saved.append(res)
        return saved


def time_ago(ts: float | None) -> str:
    if not ts:
        return "never"
    return v.ago(ts)


def now() -> float:
    return time.time()
