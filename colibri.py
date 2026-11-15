#!/usr/bin/env python3
"""colibri lifecycle helpers, following docs/AI_SETUP.md exactly: every call
is one of colibri's own commands with a --json result.

    ./colibri.py status              # coli status --json
    ./colibri.py hardware            # c/setup_hw.py --json
    ./colibri.py models              # coli setup --list --json (what fits here)
    ./colibri.py start | stop        # coli start --background --no-browser / coli stop
    ./colibri.py logs [-n 50]
    ./colibri.py guide               # print the setup steps for this machine
    ./colibri.py install MODEL       # asks first; then coli setup --yes --model MODEL --no-start

The inference server is its own process: starting or stopping vault-chat
never starts or stops colibri, and the model stays loaded between requests.
Nothing here downloads a model or installs a package without an explicit yes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import vaultdb as v

MIN_PYTHON = (3, 10)


def colibri_dir(configured: str = "") -> Path | None:
    for candidate in (configured, v.setting("COLIBRI_DIR", ""), "~/colibri", "~/src/colibri",
                      "~/Documents/colibri", "~/code/colibri"):
        if candidate:
            path = Path(candidate).expanduser()
            if (path / "c" / "coli").exists():
                return path
    return None


def _python_version(exe: str) -> tuple[int, int] | None:
    try:
        out = subprocess.run([exe, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
                             capture_output=True, text=True, timeout=10)
        major, minor = out.stdout.split()
        return int(major), int(minor)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def find_python(configured: str = "") -> str | None:
    """colibri needs Python 3.10+; vault-chat itself runs on 3.9."""
    names = [configured, v.setting("COLIBRI_PYTHON", "")]
    names += [f"python3.{m}" for m in range(14, 9, -1)] + ["python3", "python"]
    extra_dirs = ["/opt/homebrew/bin", "/usr/local/bin", str(Path.home() / ".pyenv/shims")]
    for name in names:
        if not name:
            continue
        candidates = [name] if os.path.isabs(name) else (
            [shutil.which(name)] + [str(Path(d) / name) for d in extra_dirs])
        for exe in candidates:
            if exe and Path(exe).exists():
                version = _python_version(exe)
                if version and version >= MIN_PYTHON:
                    return exe
    return None


class ColibriError(RuntimeError):
    pass


class Colibri:
    def __init__(self, directory: str = "", python: str = ""):
        self.dir = colibri_dir(directory)
        self.python = find_python(python)

    @property
    def ready(self) -> bool:
        return bool(self.dir and self.python)

    def problems(self) -> list[str]:
        out = []
        if not self.dir:
            out.append("colibri checkout not found. Clone it (git clone https://github.com/JustVugg/colibri) "
                       "and set COLIBRI_DIR in Model Settings.")
        if not self.python:
            out.append("colibri needs Python 3.10 or newer, and none was found. On macOS: brew install python@3.12 "
                       "(then set COLIBRI_PYTHON if it is not on PATH).")
        return out

    def _run(self, args: list[str], timeout: float = 60, json_out: bool = True):
        if not self.ready:
            raise ColibriError(" ".join(self.problems()))
        cmd = [self.python, str(self.dir / "c" / "coli"), *args]
        try:
            proc = subprocess.run(cmd, cwd=self.dir, capture_output=True, text=True, timeout=timeout,
                                  stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            raise ColibriError(f"`coli {' '.join(args)}` timed out after {timeout:.0f}s") from exc
        if not json_out:
            return {"exit_code": proc.returncode, "output": (proc.stdout + proc.stderr)[-20000:]}
        try:
            return json.loads(proc.stdout)
        except ValueError:
            raise ColibriError(f"`coli {' '.join(args)}` failed ({proc.returncode}): "
                               f"{(proc.stderr or proc.stdout)[-800:]}") from None

    def status(self) -> dict:
        return self._run(["status", "--json"], timeout=30)

    def models(self) -> dict | list:
        return self._run(["setup", "--list", "--json"], timeout=120)

    def hardware(self) -> dict:
        if not self.ready:
            raise ColibriError(" ".join(self.problems()))
        proc = subprocess.run([self.python, str(self.dir / "c" / "setup_hw.py"), "--json"], cwd=self.dir,
                              capture_output=True, text=True, timeout=60)
        try:
            return json.loads(proc.stdout)
        except ValueError:
            raise ColibriError(f"setup_hw.py failed: {proc.stderr[-800:]}") from None

    def start(self) -> dict:
        self._run(["start", "--background", "--no-browser"], timeout=120, json_out=False)
        return self.status()

    def stop(self) -> dict:
        return self._run(["stop"], timeout=60, json_out=False)

    def logs(self, lines: int = 50, which: str = "server") -> str:
        args = ["logs", "-n", str(lines)] + (["--install"] if which == "install" else [])
        return self._run(args, timeout=30, json_out=False)["output"]

    def install_command(self, model: str) -> list[str]:
        return [self.python or "python3", str((self.dir or Path("colibri")) / "c" / "coli"), "setup", "--yes",
                "--model", model, "--no-start"]


def summarize_status(status: dict) -> dict:
    """The fields the UI shows, from `coli status --json` (AI_SETUP.md §3)."""
    server = status.get("server") or {}
    config = status.get("config") or status.get("configuration") or {}
    install = status.get("install") or {}
    backend = (config.get("backend") or status.get("backend") or "").lower()
    return {
        "state": server.get("state") or "unknown",
        "model_id": server.get("model_id"),
        "openai_base_url": (server.get("urls") or {}).get("openai_base_url"),
        "browser_url": (server.get("urls") or {}).get("browser"),
        "tokens_per_second": server.get("tokens_per_second"),
        "backend": backend or "unknown",
        "cpu_only": backend in ("cpu", "") if backend else None,
        "model": config.get("model"),
        "model_dir": config.get("model_dir") or config.get("model_path"),
        "install_phase": install.get("phase"),
    }


GUIDE = """colibri setup (from docs/AI_SETUP.md). Nothing below runs automatically.

  0. Python 3.10+ and the code
       brew install python@3.12            # macOS, only if you have no 3.10+
       git clone https://github.com/JustVugg/colibri ~/colibri
  1. Look at the machine
       {py} ~/colibri/c/setup_hw.py --json
       {py} ~/colibri/c/coli setup --list --json
  2. Install the model you chose (downloads 19 GB or more; asks nothing)
       {py} ~/colibri/c/coli setup --yes --model qwen3-coder-30b --no-start
     CPU only:  add --backend cpu
  3. Start it as a background service, once
       {py} ~/colibri/c/coli start --background --no-browser
       {py} ~/colibri/c/coli status --json     # wait for server.state == ready
  4. Point vault-chat at it (the default already is)
       COLIBRI_BASE_URL=http://127.0.0.1:8000/v1
  Stop it with:  {py} ~/colibri/c/coli stop
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "hardware", "models", "start", "stop", "logs", "guide",
                                            "install"])
    parser.add_argument("model", nargs="?", help="model id for install")
    parser.add_argument("-n", type=int, default=50, help="log lines")
    parser.add_argument("--dir", default="", help="colibri checkout (default: COLIBRI_DIR or ~/colibri)")
    parser.add_argument("--python", default="", help="Python 3.10+ to run coli with")
    args = parser.parse_args()

    c = Colibri(args.dir, args.python)
    if args.command == "guide":
        print(GUIDE.format(py=c.python or "python3.12"))
        for p in c.problems():
            print(f"!  {p}")
        return 0
    if args.command == "install":
        if not args.model:
            sys.exit("Give the model id, from `./colibri.py models`.")
        cmd = c.install_command(args.model)
        print("This downloads the model (19 GB to hundreds of GB) and builds the engine:\n  " + " ".join(cmd))
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Nothing done.")
            return 1
        if not c.ready:
            sys.exit(" ".join(c.problems()))
        return subprocess.call(cmd, cwd=c.dir)
    try:
        if args.command == "logs":
            print(c.logs(args.n))
            return 0
        result = getattr(c, args.command)()
    except ColibriError as exc:
        sys.exit(str(exc))
    print(json.dumps(result, indent=2) if not isinstance(result, str) else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
