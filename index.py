#!/usr/bin/env python3
"""Index a markdown vault into a local vector store.

Incremental: only files whose contents actually changed are re-embedded, so
the nightly run over an unchanged vault costs a few hundred milliseconds.

    ./index.py                 # refresh changed files
    ./index.py --rebuild       # throw the index away and start over
    ./index.py --stats         # what is in the index right now
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import vaultdb as v

CHUNK_CHARS = 900
OVERLAP_CHARS = 150
BATCH = 16

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def chunk_markdown(text: str) -> list[tuple[str | None, str]]:
    """Split into (heading, passage) pairs.

    Headings are tracked so a citation can say *where* in a note an answer
    came from, which matters more than it sounds once notes get long.
    """
    chunks: list[tuple[str | None, str]] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        if len(body) <= CHUNK_CHARS:
            chunks.append((heading, body))
            return
        start = 0
        while start < len(body):
            end = min(start + CHUNK_CHARS, len(body))
            if end < len(body):  # prefer to break at a paragraph or sentence
                window = body[start:end]
                for sep in ("\n\n", "\n", ". "):
                    cut = window.rfind(sep)
                    if cut > CHUNK_CHARS // 2:
                        end = start + cut + len(sep)
                        break
            piece = body[start:end].strip()
            if piece:
                chunks.append((heading, piece))
            if end >= len(body):
                break
            start = max(end - OVERLAP_CHARS, start + 1)

    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            heading = match.group(2)
            continue
        buffer.append(line)
    flush()
    return chunks


def sha_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def reindex_file(conn, path: Path, body: str) -> int:
    rel = v.relative(path)
    pieces = chunk_markdown(body)
    conn.execute("DELETE FROM chunks WHERE path = ?", (rel,))
    if not pieces:
        return 0

    for start in range(0, len(pieces), BATCH):
        batch = pieces[start : start + BATCH]
        # Prefixing the heading gives the embedding a little context about
        # where the passage sits, which measurably improves retrieval.
        payload = [f"{h}\n\n{t}" if h else t for h, t in batch]
        # Keyword-only indexes (no embedder) store empty vectors; the FTS
        # table filled by trigger is what gets searched.
        vectors = v.embed(payload) if v.embed_model() != "none" else [b""] * len(batch)
        conn.executemany(
            "INSERT INTO chunks(path, ord, heading, text, vec) VALUES(?,?,?,?,?)",
            [
                (rel, start + i, batch[i][0], batch[i][1],
                 vectors[i] if isinstance(vectors[i], bytes) else v.pack(vectors[i]))
                for i in range(len(batch))
            ],
        )
    return len(pieces)


def show_stats(conn) -> None:
    files = conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"]
    chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    model = v.get_meta(conn, "embed_model") or "-"
    print(f"vault        {v.VAULT}")
    print(f"database     {v.DB_PATH}")
    print(f"embed model  {model}")
    print(f"last index   {v.ago(float(v.get_meta(conn, 'last_index') or 0))}")
    print(f"indexed      {files} files, {chunks} chunks")
    if chunks:
        rows = conn.execute(
            "SELECT path, COUNT(*) AS n FROM chunks GROUP BY path "
            "ORDER BY n DESC LIMIT 5"
        ).fetchall()
        print("largest      " + ", ".join(f"{r['path']} ({r['n']})" for r in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="re-embed everything from scratch")
    parser.add_argument("--stats", action="store_true", help="print index contents and exit")
    parser.add_argument("--quiet", action="store_true", help="only print the summary line")
    args = parser.parse_args()

    if not v.VAULT.is_dir():
        sys.exit(f"Vault not found: {v.VAULT}  (set VAULT_DIR)")

    conn = v.connect()

    if args.stats:
        show_stats(conn)
        return 0

    # Embeddings from different models are not comparable, so a model change
    # invalidates the whole index rather than silently poisoning results.
    previous = v.get_meta(conn, "embed_model")
    current = v.embed_model()
    if previous and previous != current and not args.rebuild:
        sys.exit(
            f"Index was built with '{previous}' but the embed model is now "
            f"'{current}'.\nVectors from different models cannot be "
            "compared — re-run with --rebuild."
        )

    if args.rebuild:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM files")

    started = time.time()
    seen: set[str] = set()
    added = updated = skipped = chunk_count = 0

    for path in v.walk_vault():
        rel = v.relative(path)
        seen.add(rel)
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"  skip  {rel}: {exc}", file=sys.stderr)
            continue

        sha = sha_of(body)
        row = conn.execute("SELECT sha FROM files WHERE path = ?", (rel,)).fetchone()
        if row and row["sha"] == sha:
            skipped += 1
            continue

        n = reindex_file(conn, path, body)
        chunk_count += n
        conn.execute(
            "INSERT INTO files(path, sha, mtime, indexed_at) VALUES(?,?,?,?) "
            "ON CONFLICT(path) DO UPDATE SET sha=excluded.sha, "
            "mtime=excluded.mtime, indexed_at=excluded.indexed_at",
            (rel, sha, path.stat().st_mtime, time.time()),
        )
        conn.commit()
        if row:
            updated += 1
        else:
            added += 1
        if not args.quiet:
            print(f"  {'update' if row else 'add   '}  {rel}  ({n} chunks)")

    gone = [r["path"] for r in conn.execute("SELECT path FROM files")] 
    removed = [p for p in gone if p not in seen]
    for rel in removed:
        conn.execute("DELETE FROM chunks WHERE path = ?", (rel,))
        conn.execute("DELETE FROM files WHERE path = ?", (rel,))
        if not args.quiet:
            print(f"  remove  {rel}")

    v.set_meta(conn, "embed_model", current)
    v.set_meta(conn, "last_index", str(time.time()))
    conn.commit()

    print(
        f"{added} added, {updated} updated, {len(removed)} removed, "
        f"{skipped} unchanged — {chunk_count} chunks embedded "
        f"in {time.time() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
