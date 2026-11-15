"""A stand-in for colibri's gateway, shaped after docs/api.md: GET /health,
GET /v1/models (with context_length), POST /v1/chat/completions with SSE
streaming and a usage chunk, tool calls, 503 while "restarting" and 429 when
the queue is full. Tests drive its behaviour through `FakeColibri.state`.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    @property
    def state(self) -> dict:
        return self.server.state  # type: ignore[attr-defined]

    def _json(self, code: int, obj, headers: dict | None = None) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            if self.state["restarting"]:
                self._json(503, {"status": "restarting"}, {"Retry-After": "5"})
            else:
                self._json(200, {"status": "ok", "active": 0, "queued": 0,
                                 "max_tokens": None})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": self.state["model"], "object": "model",
                 "context_length": self.state["context_length"]}]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.state["requests"].append(body)
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return
        if self.state["fail_next"]:
            code = self.state["fail_next"].pop(0)
            if code == 503:
                self._json(503, {"error": {"message": "restarting"}}, {"Retry-After": "1"})
            elif code == 429:
                self._json(429, {"error": {"message": "queue full", "code": "queue_full"}})
            elif code == 400:
                self._json(400, {"error": {"message": "prompt too long",
                                           "code": "context_length_exceeded"}})
            else:
                self._json(code, {"error": {"message": "boom"}})
            return
        if body.get("model") != self.state["model"]:
            self._json(404, {"error": {"message": f"model {body.get('model')} not loaded",
                                       "code": "model_not_found"}})
            return

        reply = self.state["reply"]
        if callable(reply):
            reply = reply(body)
        tool_calls = self.state["tool_calls"].pop(0) if self.state["tool_calls"] else None
        prompt_tokens = sum(len(str(m.get("content") or "")) for m in body["messages"]) // 4

        if not body.get("stream"):
            message = {"role": "assistant", "content": "" if tool_calls else reply}
            if tool_calls:
                message["tool_calls"] = tool_calls
            self._json(200, {
                "id": "x", "object": "chat.completion", "model": body["model"],
                "choices": [{"index": 0, "message": message,
                             "finish_reason": "tool_calls" if tool_calls else "stop"}],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": len(reply.split())}})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("x-colibri-queue-wait-ms", "3")
        self.send_header("Connection", "close")
        self.end_headers()
        words = reply.split(" ")
        try:
            for i, word in enumerate(words):
                piece = word + (" " if i < len(words) - 1 else "")
                chunk = {"choices": [{"index": 0, "delta": {"content": piece},
                                      "finish_reason": None}]}
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()
                if self.state["delay"]:
                    time.sleep(self.state["delay"])
            done = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            self.wfile.write(b"data: " + json.dumps(done).encode() + b"\n\n")
            if (body.get("stream_options") or {}).get("include_usage"):
                usage = {"choices": [], "usage": {"prompt_tokens": prompt_tokens,
                                                  "completion_tokens": len(words)}}
                self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.state["cancelled"] += 1


class FakeColibri:
    def __init__(self, model: str = "qwen3-coder-30b", context_length: int = 16384):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.state = self.state = {  # type: ignore[attr-defined]
            "model": model, "context_length": context_length, "restarting": False,
            "reply": "Hello from colibri.", "delay": 0.0, "fail_next": [],
            "tool_calls": [], "requests": [], "cancelled": 0,
        }
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
