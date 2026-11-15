"""The inference layer: every model call in the platform goes through an
`LLMProvider`, so nothing above this file knows which runtime is answering.

    ColibriProvider     default; colibri's OpenAI-compatible gateway on :8000
    LMStudioProvider    optional; LM Studio's OpenAI-compatible server on :1234
    OllamaProvider      optional; Ollama's OpenAI-compatible endpoints on :11434
    OpenAICompatibleProvider   any other local server speaking /v1/chat/completions

Colibri keeps one model loaded in a persistent process and queues requests,
so a provider never starts, loads or reloads a model. It only talks HTTP.

Embeddings are separate (`EmbeddingProvider`) because colibri does not serve
/v1/embeddings: semantic search is optional, keyword search is the default.

Standard library only. Every destination is a URL the user configured.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator


DEFAULT_URLS = {
    "colibri": "http://127.0.0.1:8000/v1",
    "lmstudio": "http://127.0.0.1:1234/v1",
    "ollama": "http://127.0.0.1:11434/v1",
}


class ProviderError(Exception):
    """A failure the UI can explain. `kind` is one of: unavailable, loading,
    busy, timeout, model, context, request, server."""

    def __init__(self, kind: str, message: str, hint: str = "", status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.hint = hint
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.kind in ("unavailable", "loading", "busy", "timeout")

    def as_dict(self) -> dict:
        return {"kind": self.kind, "message": str(self), "hint": self.hint,
                "retryable": self.retryable}


@dataclass
class GenParams:
    temperature: float = 0.2
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    extra: dict = field(default_factory=dict)  # provider extensions, passed through

    def body(self) -> dict:
        out: dict = {"temperature": self.temperature}
        if self.top_p is not None:
            out["top_p"] = self.top_p
        if self.max_tokens:
            out["max_tokens"] = int(self.max_tokens)
        if self.stop:
            out["stop"] = self.stop[:4]  # colibri accepts up to four
        out.update(self.extra)
        return out


@dataclass
class StreamStats:
    provider: str
    model: str
    prompt_chars: int = 0
    started: float = 0.0
    first_token_at: float | None = None
    finished: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    pieces: int = 0
    queue_wait_ms: float | None = None
    finish_reason: str | None = None
    compaction: dict | None = None

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return (self.first_token_at - self.started) * 1000

    @property
    def total_ms(self) -> float | None:
        return None if self.finished is None else (self.finished - self.started) * 1000

    @property
    def output_tokens(self) -> int:
        # Without a usage block, each streamed delta is close to one token.
        return self.completion_tokens if self.completion_tokens is not None else self.pieces

    @property
    def tokens_per_second(self) -> float | None:
        """Decode speed: output tokens over the time after the first token."""
        if self.first_token_at is None or self.finished is None:
            return None
        span = self.finished - self.first_token_at
        tokens = self.output_tokens
        if tokens <= 1 or span <= 0:
            return None
        return (tokens - 1) / span

    def as_dict(self) -> dict:
        def r(x):
            return None if x is None else round(x, 2)
        return {
            "provider": self.provider, "model": self.model,
            "ttft_ms": r(self.ttft_ms), "total_ms": r(self.total_ms),
            "tokens_per_second": r(self.tokens_per_second),
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.output_tokens,
            "prompt_chars": self.prompt_chars, "queue_wait_ms": self.queue_wait_ms,
            "finish_reason": self.finish_reason,
        }


class ChatStream:
    """Iterate for text pieces; `stats` is filled in as it goes. `close()`
    drops the HTTP connection, which is how a server-side generation is
    cancelled (colibri checks for a closed client and frees the slot)."""

    def __init__(self, gen: Iterator[str], stats: StreamStats, closer):
        self._gen = gen
        self.stats = stats
        self._closer = closer

    def __iter__(self):
        return self._gen

    def close(self) -> None:
        try:
            self._closer()
        except Exception:
            pass


# ------------------------------------------------------------------ http


def _http(method: str, url: str, body: dict | None, headers: dict, timeout: float):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **headers},
        method=method,
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _classify_http_error(exc: urllib.error.HTTPError, url: str) -> ProviderError:
    try:
        raw = exc.read().decode("utf-8", "replace")[:2000]
    except Exception:
        raw = ""
    message, code = raw, ""
    try:
        err = json.loads(raw).get("error", {})
        if isinstance(err, dict):
            message = err.get("message") or raw
            code = str(err.get("code") or err.get("type") or "")
        elif isinstance(err, str):
            message = err
    except (ValueError, AttributeError):
        pass
    status = exc.code
    if status == 503:
        return ProviderError("loading", f"The model server is loading or restarting ({url}).",
                             "Wait a few seconds and retry.", status)
    if status == 429:
        return ProviderError("busy", "The model server queue is full.",
                             "Another generation is running. Retry shortly.", status)
    if "context_length" in code or "context_length" in message:
        return ProviderError("context", "The prompt does not fit the model's context window.",
                             "Start a new conversation or lower the context budget.", status)
    if status == 404 or "model" in code:
        return ProviderError("model", f"Model not available: {message[:300]}",
                             "Check the model id in Model Settings.", status)
    if status in (401, 403):
        return ProviderError("request", "The model server refused the API key.",
                             "Set the provider API key in Model Settings.", status)
    if 400 <= status < 500:
        return ProviderError("request", f"Request rejected ({status}): {message[:300]}", "", status)
    return ProviderError("server", f"Model server error ({status}): {message[:300]}", "", status)


# ------------------------------------------------------------- providers


class LLMProvider:
    """The contract the rest of the application depends on."""

    name = "base"

    def health(self, force: bool = False) -> dict:
        raise NotImplementedError

    def models(self) -> list[dict]:
        raise NotImplementedError

    def active_model(self) -> str:
        raise NotImplementedError

    def context_length(self) -> int | None:
        return None

    def stream_chat(self, messages: list[dict], params: GenParams | None = None,
                    model: str | None = None) -> ChatStream:
        raise NotImplementedError

    def complete(self, messages: list[dict], params: GenParams | None = None,
                 model: str | None = None, tools: list[dict] | None = None) -> dict:
        """Non-streaming call: {"content", "tool_calls", "usage", "finish_reason"}."""
        raise NotImplementedError

    def chat(self, messages: list[dict], params: GenParams | None = None,
             model: str | None = None) -> str:
        return "".join(self.stream_chat(messages, params, model))


class OpenAICompatibleProvider(LLMProvider):
    name = "openai"

    def __init__(self, base_url: str, model: str = "", api_key: str = "",
                 timeout: float = 600.0, connect_timeout: float = 5.0,
                 retries: int = 2, health_ttl: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.retries = retries
        self.health_ttl = health_ttl
        self._health_cache: tuple[float, dict] | None = None
        self._models_cache: tuple[float, list] | None = None
        self._lock = threading.Lock()

    # -- plumbing

    @property
    def root(self) -> str:
        """The server root, for endpoints outside /v1 such as /health."""
        return self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _open(self, method: str, url: str, body: dict | None, timeout: float):
        """Open a request, retrying only failures that happen before any
        output: a loading engine (503) or a full queue (429)."""
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                return _http(method, url, body, self._headers(), timeout)
            except urllib.error.HTTPError as exc:
                err = _classify_http_error(exc, url)
                if err.kind in ("loading", "busy") and attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise err from exc
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", exc)
                if isinstance(reason, (socket.timeout, TimeoutError)):
                    raise ProviderError("timeout", f"Timed out connecting to {self.base_url}.",
                                        "The server may be busy loading a model.") from exc
                raise ProviderError(
                    "unavailable", f"Cannot reach {self.name} at {self.base_url} ({reason}).",
                    self._start_hint()) from exc
            except (socket.timeout, TimeoutError) as exc:
                raise ProviderError("timeout", f"No response from {self.base_url} within {timeout:.0f}s.",
                                    "Lower max tokens or the context budget.") from exc
        raise ProviderError("server", "unreachable")  # pragma: no cover

    def _get_json(self, url: str, timeout: float | None = None) -> dict:
        with self._open("GET", url, None, timeout or self.connect_timeout) as resp:
            return json.loads(resp.read() or b"{}")

    def _start_hint(self) -> str:
        return "Start the model server, then retry."

    # -- discovery

    def models(self) -> list[dict]:
        now = time.time()
        if self._models_cache and now - self._models_cache[0] < 30:
            return self._models_cache[1]
        data = self._get_json(f"{self.base_url}/models").get("data") or []
        models = [m for m in data if isinstance(m, dict) and m.get("id")]
        self._models_cache = (now, models)
        return models

    def active_model(self) -> str:
        if self.model:
            return self.model
        try:
            models = self.models()
        except ProviderError:
            return ""
        return models[0]["id"] if models else ""

    def context_length(self) -> int | None:
        try:
            models = self.models()
        except ProviderError:
            return None
        want = self.active_model()
        for m in models:
            if m.get("id") == want:
                value = m.get("context_length") or m.get("max_model_len")
                return int(value) if value else None
        return None

    def health(self, force: bool = False) -> dict:
        now = time.time()
        if not force and self._health_cache and now - self._health_cache[0] < self.health_ttl:
            return self._health_cache[1]
        started = time.time()
        result = {"provider": self.name, "base_url": self.base_url, "ok": False,
                  "state": "down", "model": self.model or "", "context_length": None,
                  "detail": "", "latency_ms": None}
        try:
            models = self.models() if force is False else self._fresh_models()
            result.update(ok=True, state="ready",
                          model=self.model or (models[0]["id"] if models else ""),
                          models=[m["id"] for m in models])
            result["context_length"] = self.context_length()
            if not models:
                result.update(ok=False, state="no-model", detail="The server lists no models.")
        except ProviderError as exc:
            result.update(state="loading" if exc.kind == "loading" else "down",
                          detail=str(exc), hint=exc.hint)
        result["latency_ms"] = round((time.time() - started) * 1000, 1)
        self._health_cache = (now, result)
        return result

    def _fresh_models(self) -> list[dict]:
        self._models_cache = None
        return self.models()

    # -- generation

    def _payload(self, messages, params: GenParams, model: str | None, stream: bool) -> dict:
        body = {"model": model or self.active_model(), "messages": messages, "stream": stream}
        body.update(params.body())
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def stream_chat(self, messages, params=None, model=None) -> ChatStream:
        params = params or GenParams()
        body = self._payload(messages, params, model, stream=True)
        stats = StreamStats(provider=self.name, model=body["model"],
                            prompt_chars=sum(len(str(m.get("content") or "")) for m in messages),
                            started=time.time())
        resp = self._open("POST", f"{self.base_url}/chat/completions", body, self.timeout)
        wait = resp.headers.get("x-colibri-queue-wait-ms") if hasattr(resp, "headers") else None
        if wait:
            try:
                stats.queue_wait_ms = float(wait)
            except ValueError:
                pass

        def gen():
            try:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if event.get("error"):
                        err = event["error"]
                        msg = err.get("message") if isinstance(err, dict) else str(err)
                        raise ProviderError("server", f"Generation failed: {msg}")
                    if event.get("compaction"):
                        stats.compaction = event["compaction"]
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        stats.prompt_tokens = usage.get("prompt_tokens", stats.prompt_tokens)
                        stats.completion_tokens = usage.get("completion_tokens",
                                                            stats.completion_tokens)
                    for choice in event.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content")
                        if choice.get("finish_reason"):
                            stats.finish_reason = choice["finish_reason"]
                        if piece:
                            if stats.first_token_at is None:
                                stats.first_token_at = time.time()
                            stats.pieces += 1
                            yield piece
            except (socket.timeout, TimeoutError) as exc:
                raise ProviderError("timeout", "The model stopped sending tokens.",
                                    "Retry, or lower max tokens.") from exc
            finally:
                stats.finished = time.time()
                try:
                    resp.close()
                except Exception:
                    pass

        return ChatStream(gen(), stats, resp.close)

    def complete(self, messages, params=None, model=None, tools=None) -> dict:
        params = params or GenParams()
        body = self._payload(messages, params, model, stream=False)
        if tools:
            body["tools"] = tools
        started = time.time()
        with self._open("POST", f"{self.base_url}/chat/completions", body, self.timeout) as resp:
            data = json.loads(resp.read() or b"{}")
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return {
            "content": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "model": data.get("model") or body["model"],
            "total_ms": round((time.time() - started) * 1000, 1),
        }


class ColibriProvider(OpenAICompatibleProvider):
    """colibri's gateway (`coli serve` / `coli start`). Uses `GET /health` for
    liveness, which answers 503 `restarting` while the engine reloads, and
    the optional `cache_slot` field to keep one KV context per conversation
    when the server was started with `--kv-slots N`."""

    name = "colibri"

    def __init__(self, base_url: str = DEFAULT_URLS["colibri"], model: str = "",
                 api_key: str = "", kv_slots: int = 0, compaction: bool = True, **kw):
        super().__init__(base_url, model, api_key, **kw)
        self.kv_slots = max(0, int(kv_slots or 0))
        self.compaction = compaction

    def _start_hint(self) -> str:
        return "Start colibri: python3 <colibri>/c/coli start --background --no-browser"

    def health(self, force: bool = False) -> dict:
        now = time.time()
        if not force and self._health_cache and now - self._health_cache[0] < self.health_ttl:
            return self._health_cache[1]
        started = time.time()
        result = {"provider": self.name, "base_url": self.base_url, "ok": False,
                  "state": "down", "model": self.model or "", "context_length": None,
                  "detail": "", "latency_ms": None, "server": {}}
        try:
            req = urllib.request.Request(f"{self.root}/health", headers=self._headers())
            with urllib.request.urlopen(req, timeout=self.connect_timeout) as resp:
                result["server"] = json.loads(resp.read() or b"{}")
            if force:
                self._models_cache = None
            models = self.models()
            result.update(ok=bool(models), state="ready" if models else "no-model",
                          model=self.model or (models[0]["id"] if models else ""),
                          models=[m["id"] for m in models],
                          context_length=self.context_length())
        except urllib.error.HTTPError as exc:
            try:
                result["server"] = json.loads(exc.read() or b"{}")
            except ValueError:
                pass
            state = (result["server"] or {}).get("status") or ""
            if exc.code == 503:
                result.update(state="loading" if state != "restarting" else "restarting",
                              detail="colibri is loading the model.", hint="Wait, then retry.")
            elif exc.code == 401:
                result.update(state="auth", detail="colibri requires an API key.",
                              hint="Set COLIBRI_API_KEY.")
            else:
                result.update(detail=f"/health returned {exc.code}")
        except urllib.error.URLError as exc:
            result.update(detail=f"Cannot reach colibri at {self.root} ({exc.reason}).",
                          hint=self._start_hint())
        except ProviderError as exc:
            result.update(detail=str(exc), hint=exc.hint)
        except (socket.timeout, TimeoutError, ValueError) as exc:
            result.update(detail=f"Health check failed: {exc}")
        result["latency_ms"] = round((time.time() - started) * 1000, 1)
        self._health_cache = (now, result)
        return result

    def slot_for(self, conversation_id: int | None) -> int | None:
        if self.kv_slots <= 1 or conversation_id is None:
            return None
        return int(conversation_id) % self.kv_slots

    def stream_chat(self, messages, params=None, model=None, conversation_id=None) -> ChatStream:
        params = params or GenParams()
        extra = dict(params.extra)
        slot = self.slot_for(conversation_id)
        if slot is not None:
            extra["cache_slot"] = slot
        if self.compaction:
            # A safety net only: context.py already fits the prompt to the
            # window, so this fires only if our token estimate was short.
            extra.setdefault("context_compaction", "auto")
        params = GenParams(params.temperature, params.top_p, params.max_tokens, params.stop, extra)
        return super().stream_chat(messages, params, model)


class LMStudioProvider(OpenAICompatibleProvider):
    name = "lmstudio"

    def _start_hint(self) -> str:
        return "Start LM Studio's server: lms server start"


class OllamaProvider(OpenAICompatibleProvider):
    name = "ollama"

    def _start_hint(self) -> str:
        return "Start Ollama: ollama serve"


PROVIDERS = {
    "colibri": ColibriProvider,
    "lmstudio": LMStudioProvider,
    "ollama": OllamaProvider,
    "openai": OpenAICompatibleProvider,
}


def make_provider(name: str, base_url: str = "", model: str = "", api_key: str = "",
                  timeout: float = 600.0, kv_slots: int = 0) -> LLMProvider:
    name = (name or "colibri").lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}'. Choose one of: {', '.join(PROVIDERS)}")
    url = base_url or DEFAULT_URLS.get(name, "")
    if not url:
        raise ValueError(f"Provider '{name}' needs a base URL.")
    if name == "colibri":
        return ColibriProvider(url, model, api_key, kv_slots=kv_slots, timeout=timeout)
    return PROVIDERS[name](url, model, api_key, timeout=timeout)


# ------------------------------------------------------------- embeddings


class EmbeddingProvider:
    """Optional. Any server with an OpenAI-compatible /v1/embeddings works
    (LM Studio, Ollama, llama.cpp's server). Disabled by default because
    colibri has no embeddings endpoint; keyword search covers that case."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            with _http("POST", f"{self.base_url}/embeddings",
                       {"model": self.model, "input": texts}, headers, self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise _classify_http_error(exc, self.base_url) from exc
        except urllib.error.URLError as exc:
            raise ProviderError("unavailable", f"Cannot reach the embedding server at "
                                f"{self.base_url} ({exc.reason}).") from exc
        rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
        return [r["embedding"] for r in rows]
