#!/usr/bin/env python3
"""The local assistant: chat, coding and personal agent, in one web UI.

Standard library only, no build step and no CDN, so it works with the
network cable unplugged. Inference comes from the configured provider
(colibri by default, a separate process this server never starts or stops);
conversations, memories, projects and metrics live in a local SQLite file.
The original /api/chat, /api/meta and /api/session* endpoints still work.

    ./chat.py                 # then open http://127.0.0.1:8737
    ./chat.py --open          # and launch the browser for you
    ./chat.py --port 9000

It binds to 127.0.0.1 deliberately. See --host before changing that.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import agent
import ask
import vaultdb as v

WEB = Path(__file__).resolve().parent / "web"


def page() -> bytes:
    """One self-contained page: the stylesheet and script are inlined, so the
    UI makes no request beyond this server and works with the network off."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "app.css").read_text(encoding="utf-8")
    js = (WEB / "app.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    return (html.replace("/*__CSS__*/", css).replace("//__JS__", js)).encode("utf-8")


# ------------------------------------------------------------------ server


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "VaultChat"

    def log_message(self, fmt, *args):
        if self.server.verbose:  # type: ignore[attr-defined]
            sys.stderr.write("  %s\n" % (fmt % args))

    # -- helpers

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # No CORS header on purpose: only this page, served from this origin,
        # may call the API.
        self.end_headers()
        self.wfile.write(body)

    def _event(self, payload: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
        self.wfile.flush()

    # -- routes

    def _json_body(self):
        """Parse a JSON POST body, or send the error and return None.

        Requiring application/json matters: a web page on another site can
        send a "simple" cross-origin POST to localhost without any preflight,
        but only with a form-ish content type. Insisting on JSON makes the
        browser preflight it, which this server never approves."""
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            self._send(415, b"expected application/json", "text/plain")
            return None
        length = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._send(400, b"bad json", "text/plain")
            return None
        if not isinstance(data, dict):
            self._send(400, b"expected an object", "text/plain")
            return None
        return data

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _host_ok(self) -> bool:
        """Refuse requests addressed to any other name. Without this, a web
        page could rebind its own domain to 127.0.0.1 and read the API."""
        host = (self.headers.get("Host") or "").strip().lower()
        name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        allowed = {"127.0.0.1", "localhost", "[::1]", self.server.server_address[0]}  # type: ignore[attr-defined]
        if name in allowed:
            return True
        self._send(403, b"unexpected Host header", "text/plain")
        return False

    def _platform_api(self):
        return self.server.api()  # type: ignore[attr-defined]

    def _api_get(self, route: str, query: dict) -> None:
        import api
        try:
            self._json(200, self._platform_api().get(route, query))
        except api.NotFound:
            self._send(404, b"not found", "text/plain")
        except api.BadRequest as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # report, never kill the server
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _api_post(self, route: str) -> None:
        import api
        payload = self._json_body()
        if payload is None:
            return
        if route == "/api/turn":
            self._stream_turn(payload)
            return
        try:
            self._json(200, self._platform_api().post(route, payload))
        except api.NotFound:
            self._send(404, b"not found", "text/plain")
        except api.BadRequest as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _stream_turn(self, payload: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        events = self._platform_api().turn(payload)
        try:
            for event in events:
                self._event(event)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Stop: closing the generator below saves the partial answer
        except Exception as exc:
            try:
                self._event({"type": "error", "kind": "server", "message": f"{type(exc).__name__}: {exc}"})
            except OSError:
                pass
        finally:
            events.close()

    # -- routes

    def do_GET(self) -> None:
        if not self._host_ok():
            return
        parts = urlsplit(self.path)
        route = parts.path
        if route == "/":
            try:
                body = page()
            except OSError as exc:
                self._send(500, f"web UI files missing: {exc}".encode(), "text/plain")
                return
            self._send(200, body, "text/html; charset=utf-8")
        elif route.startswith("/api/") and route not in ("/api/meta", "/api/sessions", "/api/session"):
            self._api_get(route, parse_qs(parts.query))
        elif route == "/api/meta":
            self._json(200, self.meta())
        elif route == "/api/sessions":
            conn = v.connect()
            try:
                rows = v.list_sessions(conn, 60)
            finally:
                conn.close()
            self._json(200, [
                {"id": r["id"], "title": r["title"], "profile": r["profile"],
                 "updated": r["updated_at"], "turns": r["n"] // 2}
                for r in rows
            ])
        elif route == "/api/session":
            try:
                sid = int(parse_qs(parts.query).get("id", [""])[0])
            except ValueError:
                self._send(400, b"bad id", "text/plain")
                return
            conn = v.connect()
            try:
                row = v.get_session(conn, sid)
                messages = v.session_messages(conn, sid) if row else []
            finally:
                conn.close()
            if not row:
                self._send(404, b"not found", "text/plain")
                return
            self._json(200, {"id": row["id"], "title": row["title"],
                             "profile": row["profile"], "messages": messages})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if not self._host_ok():
            return
        route = urlsplit(self.path).path
        if route.startswith("/api/") and route not in ("/api/session/delete", "/api/chat"):
            self._api_post(route)
            return
        if route == "/api/session/delete":
            payload = self._json_body()
            if payload is None:
                return
            try:
                sid = int(payload.get("id"))
            except (TypeError, ValueError):
                self._send(400, b"bad id", "text/plain")
                return
            conn = v.connect()
            try:
                v.delete_session(conn, sid)
            finally:
                conn.close()
            self._json(200, {"ok": True})
            return

        if route != "/api/chat":
            self._send(404, b"not found", "text/plain")
            return
        payload = self._json_body()
        if payload is None:
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.stream_reply(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser hit Stop; nothing to clean up

    # -- work

    def meta(self) -> dict:
        profiles = agent.load_profiles()
        conn = v.connect()
        chunks = conn.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
        conn.close()
        return {
            "backend": v.backend(),
            "model": v.chat_model(),
            "vault": str(v.VAULT),
            "chunks": chunks,
            "default_profile": self.server.profile,  # type: ignore[attr-defined]
            "profiles": {
                name: {
                    "description": body.get("description", ""),
                    "tools": [
                        t
                        for i in body.get("integrations") or []
                        if isinstance(i, dict)
                        for t in (i.get("allowed_tools") or [])
                    ],
                }
                for name, body in profiles.items()
            },
        }

    def open_session(self, raw_id, profile_name: str, question: str) -> tuple[int, bool]:
        """The session this exchange belongs to, and whether we just made it.
        An id that does not exist (deleted in another tab, say) starts afresh
        rather than failing."""
        conn = v.connect()
        try:
            try:
                sid = int(raw_id) if raw_id is not None else None
            except (TypeError, ValueError):
                sid = None
            if sid is not None and v.get_session(conn, sid) is not None:
                return sid, False
            return v.create_session(conn, profile_name, question), True
        finally:
            conn.close()

    def close_session(self, sid, created: bool, question: str, answer: str) -> None:
        """Runs even when the browser hit Stop, so a partial answer is kept —
        the page keeps it in view, and the two should not disagree. A session
        we created that produced nothing is removed rather than left empty."""
        if sid is None:
            return
        conn = v.connect()
        try:
            if answer.strip():
                v.add_messages(conn, sid, [("user", question), ("assistant", answer.strip())])
            elif created:
                v.delete_session(conn, sid)
        finally:
            conn.close()

    def stream_reply(self, payload: dict) -> None:
        profiles = agent.load_profiles()
        name = payload.get("profile") or self.server.profile  # type: ignore[attr-defined]
        profile = profiles.get(name)
        if not profile:
            self._event({"type": "error", "text": f"unknown profile '{name}'"})
            return

        messages = [m for m in payload.get("messages", []) if m.get("content")]
        if not messages:
            self._event({"type": "error", "text": "empty message"})
            return
        question = messages[-1]["content"]

        # Saving is opt-in per request, so scripted callers stay stateless.
        sid, created = (None, False)
        if payload.get("save"):
            sid, created = self.open_session(payload.get("session"), name, question)

        answer_parts: list[str] = []

        def token(text: str) -> None:
            answer_parts.append(text)
            self._event({"type": "token", "text": text})

        try:
            if sid is not None:
                self._event({"type": "session", "id": sid})

            # Ground the newest question in the vault, exactly as the CLI does.
            prompt = question
            if profile.get("vault"):
                conn = v.connect()
                try:
                    hits = ask.retrieve(
                        conn, question,
                        self.server.k,          # type: ignore[attr-defined]
                        self.server.min_score,  # type: ignore[attr-defined]
                    )
                finally:
                    conn.close()
                if hits:
                    context, included = ask.build_context(
                        hits, self.server.max_context_chars  # type: ignore[attr-defined]
                    )
                    prompt = f"Notes:\n\n{context}\n\nQuestion: {question}"
                    self._event({
                        "type": "citations",
                        "items": [
                            {
                                "n": n,
                                "where": row["path"] + (f" › {row['heading']}" if row["heading"] else ""),
                                "score": f"{score:.2f}",
                            }
                            for n, (score, row) in enumerate(included, start=1)
                        ],
                    })

            integrations = profile.get("integrations") or []
            temperature = profile.get("temperature", 0.2)

            try:
                if integrations:
                    agent.check_allowlists(profile, False)
                    data = v.agent_chat(
                        prompt,
                        integrations=integrations,
                        system=profile.get("system"),
                        temperature=temperature,
                    )
                    for item in data.get("output") or []:
                        if item.get("type") == "tool_call":
                            self._event({"type": "tool", "text": f"called {item.get('tool')}"})
                    token(agent.render(data, False))
                    stats = data.get("stats") or {}
                    if stats.get("tokens_per_second"):
                        self._event({
                            "type": "stats",
                            "text": f"{stats['tokens_per_second']:.1f} tok/s · "
                                    f"{stats.get('time_to_first_token_seconds', 0):.1f}s to first token",
                        })
                else:
                    # Earlier turns, trimmed to a budget: the browser sends the
                    # whole conversation, but a small model cannot afford it.
                    convo = [{"role": "system", "content": profile.get("system", "")}]
                    convo += agent.trim_history(messages[:-1], agent.HISTORY_CHARS)
                    convo.append({"role": "user", "content": prompt})
                    for piece in v.chat_stream(convo, temperature=temperature):
                        token(piece)
            except SystemExit as exc:
                self._event({"type": "error", "text": str(exc)})
            except Exception as exc:  # a dead runtime should not kill the server
                self._event({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
        except SystemExit as exc:  # e.g. "Index is empty" during retrieval
            self._event({"type": "error", "text": str(exc)})
        finally:
            self.close_session(sid, created, question, "".join(answer_parts))

        self._event({"type": "done"})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    profile = "ask"
    verbose = False
    k = 6
    min_score = 0.25
    max_context_chars = 6000
    store_path = None
    _api = None
    _api_lock = threading.Lock()

    def api(self):
        """Built on first use, so the legacy endpoints never touch the
        platform database."""
        with self._api_lock:
            if self._api is None:
                import api
                from orchestrator import Platform
                self._api = api.API(Platform(self.store_path, {
                    "k": self.k, "min_score": self.min_score, "max_context_chars": self.max_context_chars}))
            return self._api


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8737)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address — anything but 127.0.0.1 exposes an "
                             "unauthenticated assistant to your network")
    parser.add_argument("-p", "--profile", default="ask", help="profile selected on load")
    parser.add_argument("--open", action="store_true", help="open a browser once serving")
    parser.add_argument("--verbose", action="store_true", help="log every request")
    parser.add_argument("-k", type=int, default=6, help="notes to retrieve (default 6)")
    parser.add_argument("--min-score", type=float, default=0.25,
                        help="similarity floor for notes (default 0.25)")
    parser.add_argument("--max-context-chars", type=int, default=6000,
                        help="prompt budget for notes; raising it slows replies")
    args = parser.parse_args()

    if args.profile not in agent.load_profiles():
        sys.exit(f"Unknown profile '{args.profile}'. Try ./agent.py --list.")

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"!  Binding to {args.host}. This server has no authentication, so\n"
            f"!  anyone who can reach that address can use your model and read\n"
            f"!  answers drawn from your notes.\n",
            file=sys.stderr,
        )

    httpd = Server((args.host, args.port), Handler)
    httpd.profile = args.profile
    httpd.verbose = args.verbose
    httpd.k = args.k
    httpd.min_score = args.min_score
    httpd.max_context_chars = args.max_context_chars
    url = f"http://{args.host}:{args.port}"
    print(f"Vault Chat on {url}   (Ctrl-C to stop)")

    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
