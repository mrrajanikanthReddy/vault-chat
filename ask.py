#!/usr/bin/env python3
"""Ask a question against your own notes, with citations.

Retrieval keeps the prompt small, which is the difference between a reply in
seconds and a reply in minutes on a CPU — so the design that makes answers
traceable is the same one that makes them fast.

    ./ask.py "what did I decide about retries?"
    ./ask.py --search "terraform"        # just show matches, no model call
    ./ask.py -k 8 --show-context "..."   # see exactly what was sent
    ./ask.py                             # interactive loop
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import vaultdb as v

MAX_CONTEXT_CHARS = 6000
MIN_SCORE = 0.25

SYSTEM = """You answer questions using only the notes provided below.

Rules:
- Base every statement on the numbered notes. Do not add outside knowledge.
- Cite the note number inline, like [2], for each claim.
- If the notes do not contain the answer, say so plainly and stop. Do not guess.
- Be concise. Prefer the author's own wording where it is clear."""


def _cap_per_file(ranked, k: int, min_score: float) -> list:
    hits: list = []
    per_file: dict[str, int] = {}
    for score, row in ranked:
        if score < min_score or len(hits) >= k:
            break
        # Allow at most two passages per file so one long note cannot
        # crowd out everything else.
        if per_file.get(row["path"], 0) >= 2:
            continue
        per_file[row["path"]] = per_file.get(row["path"], 0) + 1
        hits.append((score, row))
    return hits


def retrieve(conn, question: str, k: int, min_score: float):
    if conn.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"] == 0:
        sys.exit("Index is empty. Run ./index.py first.")

    # An index built without an embedder holds no vectors: search by keyword.
    if v.get_meta(conn, "embed_model") == "none" or not v.has_embedder():
        return _cap_per_file(v.keyword_search_chunks(conn, question, k * 3), k, min_score)

    matrix, rows = v.load_matrix(conn)
    query = v.embed([question])[0]
    if query.shape[0] != matrix.shape[1]:
        sys.exit(
            "Query and index have different embedding sizes — the embed model "
            "changed. Re-run ./index.py --rebuild."
        )

    scores = matrix @ query  # both sides are normalised, so this is cosine
    order = np.argsort(-scores)[: k * 3]
    return _cap_per_file(((float(scores[i]), rows[i]) for i in order), k, min_score)


def build_context(hits, budget: int) -> tuple[str, list]:
    blocks, included, spent = [], [], 0
    for n, (score, row) in enumerate(hits, start=1):
        where = row["path"] + (f" › {row['heading']}" if row["heading"] else "")
        block = f"[{n}] {where}\n{row['text']}"
        if spent + len(block) > budget and included:
            break
        blocks.append(block)
        included.append((score, row))
        spent += len(block)
    return "\n\n".join(blocks), included


def answer(conn, question: str, args) -> None:
    hits = retrieve(conn, question, args.k, args.min_score)
    if not hits:
        print("Nothing in the vault looks relevant. Either the note does not "
              "exist yet, or try different wording.")
        return

    context, included = build_context(hits, args.max_context_chars)

    if args.show_context:
        print(context)
        print("-" * 60)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Notes:\n\n{context}\n\nQuestion: {question}"},
    ]
    for piece in v.chat_stream(messages, model=args.model):
        print(piece, end="", flush=True)
    print("\n")

    print("Sources")
    for n, (score, row) in enumerate(included, start=1):
        where = row["path"] + (f" › {row['heading']}" if row["heading"] else "")
        print(f"  [{n}] {where}   ({score:.2f})")


def search_only(conn, question: str, args) -> None:
    hits = retrieve(conn, question, args.k, args.min_score)
    if not hits:
        print("No matches above the score threshold.")
        return
    for score, row in hits:
        where = row["path"] + (f" › {row['heading']}" if row["heading"] else "")
        snippet = " ".join(row["text"].split())[:160]
        print(f"{score:.2f}  {where}\n      {snippet}…\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*", help="what to ask")
    parser.add_argument("-k", type=int, default=6, help="passages to retrieve (default 6)")
    parser.add_argument("--search", action="store_true", help="show matches without calling the model")
    parser.add_argument("--show-context", action="store_true", help="print what was sent to the model")
    parser.add_argument("--model", default=None, help="override the chat model")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE, help="similarity floor (default 0.25)")
    parser.add_argument("--max-context-chars", type=int, default=MAX_CONTEXT_CHARS,
                        help="prompt budget; raising this slows the reply (default 6000)")
    args = parser.parse_args()

    conn = v.connect()
    run = search_only if args.search else answer

    if args.question:
        run(conn, " ".join(args.question), args)
        return 0

    print(f"Asking {args.model or v.chat_model()} about {v.VAULT} "
          f"via {v.backend()}. Ctrl-D to quit.\n")
    while True:
        try:
            question = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question:
            run(conn, question, args)
            print()


if __name__ == "__main__":
    raise SystemExit(main())
