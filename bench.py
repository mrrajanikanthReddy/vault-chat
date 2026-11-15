#!/usr/bin/env python3
"""Benchmark the inference provider, and compare it with the old runtimes.

    ./bench.py                                   # colibri, standard prompts
    ./bench.py -p colibri -p ollama -p lmstudio  # side by side (each must be running)
    ./bench.py --cold                            # also time colibri stop -> start -> ready
    ./bench.py --runs 3 --json results.json

Per prompt it records time to first token, decode tokens/second, total time,
prompt and completion tokens; per provider, the server's resident memory and
CPU while generating (summed over its process tree), and the model's size on
disk where that can be found. Results are stored for the Performance
dashboard. Do not assume a number from one machine holds on another: run it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from providers import DEFAULT_URLS, GenParams, ProviderError, make_provider

PROMPTS = {
    "short": "Explain what a mutex is in two sentences.",
    "code": "Write a Python function `parse_duration(text)` that turns strings like '1h30m' or '45s' into "
            "seconds, with three unit tests using unittest.",
    "long": ("Summarize the following design notes in five bullet points.\n\n"
             + " ".join(f"Note {i}: the billing service retries idempotent writes with exponential backoff, "
                        f"records every attempt, and escalates after the third failure to an operator queue."
                        for i in range(40))),
}


# ---------------------------------------------------- process sampling


def listener_pid(port: int) -> int | None:
    exe = shutil.which("lsof") or ("/usr/sbin/lsof" if Path("/usr/sbin/lsof").exists() else None)
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True,
                             text=True, timeout=10).stdout.split()
        return int(out[0]) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def process_tree(root: int) -> dict[int, tuple[int, float]]:
    """pid -> (rss KB, %cpu) for root and every descendant (colibri's gateway
    runs the C engine as a child process, so both count)."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss=,pcpu="], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    rows, children = {}, {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            pid, ppid, rss, cpu = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
        except ValueError:
            continue
        rows[pid] = (rss, cpu)
        children.setdefault(ppid, []).append(pid)
    tree, stack = {}, [root]
    while stack:
        pid = stack.pop()
        if pid in rows and pid not in tree:
            tree[pid] = rows[pid]
            stack += children.get(pid, [])
    return tree


class Sampler(threading.Thread):
    def __init__(self, pid: int | None, interval: float = 0.5):
        super().__init__(daemon=True)
        self.pid, self.interval = pid, interval
        self.rss: list[int] = []
        self.cpu: list[float] = []
        self._stop = threading.Event()

    def run(self):
        while self.pid and not self._stop.is_set():
            tree = process_tree(self.pid)
            if tree:
                self.rss.append(sum(r for r, _ in tree.values()))
                self.cpu.append(sum(c for _, c in tree.values()))
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        self.join(timeout=2)
        if not self.rss:
            return {"ram_mb": None, "cpu_percent_avg": None, "cpu_percent_peak": None}
        return {"ram_mb": round(max(self.rss) / 1024, 1),
                "cpu_percent_avg": round(statistics.mean(self.cpu), 1),
                "cpu_percent_peak": round(max(self.cpu), 1)}


def disk_usage_gb(path: str | Path | None) -> float | None:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        return None
    try:
        out = subprocess.run(["du", "-sk", str(p)], capture_output=True, text=True, timeout=120).stdout
        return round(int(out.split()[0]) / 1024 / 1024, 2)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def model_dir_for(provider: str) -> str | None:
    if provider == "colibri":
        try:
            from colibri import Colibri, summarize_status
            c = Colibri()
            if c.ready:
                return summarize_status(c.status()).get("model_dir") or str(Path("~/colibri-models").expanduser())
        except Exception:
            return str(Path("~/colibri-models").expanduser())
    return {"ollama": os.environ.get("OLLAMA_MODELS", "~/.ollama/models"),
            "lmstudio": "~/.lmstudio/models"}.get(provider)


# --------------------------------------------------------------- runs


def cold_start(base_url: str, timeout: float = 900) -> float | None:
    """colibri only: stop, start in the background, time until /health is ready."""
    from colibri import Colibri
    from providers import ColibriProvider
    c = Colibri()
    if not c.ready:
        print("  cold start skipped: " + " ".join(c.problems()), file=sys.stderr)
        return None
    c.stop()
    started = time.time()
    c.start()
    probe = ColibriProvider(base_url)
    while time.time() - started < timeout:
        if probe.health(force=True)["ok"]:
            return round(time.time() - started, 2)
        time.sleep(1)
    return None


def bench_provider(name: str, base_url: str = "", model: str = "", runs: int = 2, max_tokens: int = 256,
                   prompts: dict | None = None, cold: bool = False) -> dict:
    provider = make_provider(name, base_url, model)
    result: dict = {"provider": name, "base_url": provider.base_url, "started": time.time(), "prompts": {}}
    if cold and name == "colibri":
        result["model_init_seconds"] = cold_start(provider.base_url)
    health = provider.health(force=True)
    result["health"] = {k: health.get(k) for k in ("ok", "state", "model", "context_length", "detail")}
    if not health["ok"]:
        result["error"] = health.get("detail") or "provider not ready"
        return result
    result["model"] = provider.active_model()
    port = urlsplit(provider.base_url).port or 80
    pid = listener_pid(port)
    result["server_pid"] = pid
    sampler = Sampler(pid)
    sampler.start()
    params = GenParams(temperature=0.0, max_tokens=max_tokens)
    for label, prompt in (prompts or PROMPTS).items():
        rows = []
        for _ in range(runs):
            try:
                stream = provider.stream_chat([{"role": "user", "content": prompt}], params)
                for _piece in stream:
                    pass
                rows.append(stream.stats.as_dict())
            except ProviderError as exc:
                rows.append({"error": str(exc)})
        ok = [r for r in rows if "error" not in r]

        def med(key):
            values = [r[key] for r in ok if r.get(key) is not None]
            return round(statistics.median(values), 2) if values else None

        result["prompts"][label] = {"runs": rows, "ttft_ms": med("ttft_ms"),
                                    "tokens_per_second": med("tokens_per_second"),
                                    "total_ms": med("total_ms"), "prompt_tokens": med("prompt_tokens"),
                                    "completion_tokens": med("completion_tokens"),
                                    "errors": len(rows) - len(ok)}
    result.update(sampler.stop())
    result["disk_gb"] = disk_usage_gb(model_dir_for(name))
    result["finished"] = time.time()
    return result


def print_table(results: list[dict]) -> None:
    print(f"\n{'provider':<10} {'model':<28} {'prompt':<6} {'ttft ms':>9} {'tok/s':>7} {'total ms':>9} "
          f"{'p.tok':>6} {'c.tok':>6}")
    for r in results:
        if r.get("error"):
            print(f"{r['provider']:<10} {'-':<28} {'-':<6}  unavailable: {r['error'][:60]}")
            continue
        for label, p in r["prompts"].items():
            print(f"{r['provider']:<10} {str(r.get('model'))[:28]:<28} {label:<6} {str(p['ttft_ms']):>9} "
                  f"{str(p['tokens_per_second']):>7} {str(p['total_ms']):>9} {str(p['prompt_tokens']):>6} "
                  f"{str(p['completion_tokens']):>6}")
        print(f"{'':<10} RAM {r.get('ram_mb')} MB · CPU avg {r.get('cpu_percent_avg')}% peak "
              f"{r.get('cpu_percent_peak')}% · disk {r.get('disk_gb')} GB"
              + (f" · init {r['model_init_seconds']}s" if r.get("model_init_seconds") else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--provider", action="append", choices=list(DEFAULT_URLS) + ["openai"],
                        help="provider to benchmark (repeatable; default colibri)")
    parser.add_argument("--url", action="append", default=[], help="base URL per provider, in the same order")
    parser.add_argument("--model", default="", help="model id (default: whatever is loaded)")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--cold", action="store_true", help="time a colibri restart until ready")
    parser.add_argument("--json", help="also write results to this file")
    parser.add_argument("--no-save", action="store_true", help="do not store results for the dashboard")
    args = parser.parse_args()

    providers = args.provider or ["colibri"]
    results = []
    for i, name in enumerate(providers):
        url = args.url[i] if i < len(args.url) else ""
        print(f"benchmarking {name} …", file=sys.stderr)
        results.append(bench_provider(name, url, args.model, args.runs, args.max_tokens, cold=args.cold))
    print_table(results)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    if not args.no_save:
        import store
        s = store.Store()
        for r in results:
            s.save_bench(r["provider"], r.get("model") or "", r, "cli")
    return 0 if all(not r.get("error") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
