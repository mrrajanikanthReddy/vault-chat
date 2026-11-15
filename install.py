#!/usr/bin/env python3
"""Set up vault-chat on Linux, macOS or Windows.

Everything it does is reversible and nothing is installed system-wide: a
private virtual environment, a config file, a notes folder, and launcher
scripts in ./bin. It never edits your PATH or installs a model runtime
behind your back — where those are needed it tells you the command to run.

    python3 install.py              # interactive
    python3 install.py --yes        # accept defaults, no prompts
    python3 install.py --schedule   # also set up nightly re-indexing
    python3 install.py --check      # report what is set up, change nothing
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
VENV = HERE / ".venv"
BIN = HERE / "bin"
COMMANDS = {
    "vault-chat": "chat.py",
    "vault-ask": "ask.py",
    "vault-agent": "agent.py",
    "vault-index": "index.py",
    "vault-quiz": "quiz.py",
    "vault-hardware": "hardware.py",
    "vault-colibri": "colibri.py",
    "vault-bench": "bench.py",
}
MIN_PYTHON = (3, 9)

OK, WARN, BAD = "  ok   ", "  note ", "  !!   "


def say(mark: str, text: str) -> None:
    print(f"{mark} {text}")


def ask_yes(question: str, assume: bool) -> bool:
    if assume:
        return True
    try:
        return input(f"       {question} [Y/n] ").strip().lower() in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


# --------------------------------------------------------------- platform


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def config_root() -> Path:
    if IS_WIN:
        return Path(os.environ.get("APPDATA", "~")).expanduser()
    return Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()


def config_dir() -> Path:
    # Must agree with vaultdb.config_dir(), which the tools read at runtime.
    new, old = config_root() / "vault-chat", config_root() / "vault-tools"
    return old if old.is_dir() and not new.exists() else new


def migrate_legacy_config() -> None:
    """The project used to be called vault-tools. Move its settings across
    once, so config.json and profiles.json carry over untouched."""
    new, old = config_root() / "vault-chat", config_root() / "vault-tools"
    if old.is_dir() and not new.exists():
        try:
            old.rename(new)
            say(OK, f"moved settings from {old.name}/ to {new.name}/")
        except OSError as exc:
            say(WARN, f"could not move {old} to {new} ({exc}); still using the old folder")


def probe(url: str) -> bool:
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=2)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def detect_runtime() -> str | None:
    if probe("http://127.0.0.1:8000/health"):
        return "colibri"
    if probe("http://127.0.0.1:1234/v1/models"):
        return "lmstudio"
    if probe("http://127.0.0.1:11434/api/tags"):
        return "ollama"
    return None


_COLIBRI = ("the engine is c/coli in this project. Follow "
            "`./colibri.py guide` (Python 3.10+; the model download is a step you run yourself)")
RUNTIME_HELP = {
    "Linux": _COLIBRI,
    "Darwin": "brew install python@3.12 (if you have no Python 3.10+); " + _COLIBRI,
    "Windows": "install Python 3.10+ from python.org; " + _COLIBRI,
}


# ----------------------------------------------------------------- steps


def step_python() -> None:
    if sys.version_info < MIN_PYTHON:
        sys.exit(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, found "
            f"{sys.version.split()[0]}."
        )
    say(OK, f"Python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")


def step_venv(assume: bool) -> None:
    if venv_python().exists():
        say(OK, f"virtual environment already at {VENV.name}/")
    else:
        say(WARN, f"creating virtual environment in {VENV.name}/")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)

    probe_numpy = subprocess.run(
        [str(venv_python()), "-c", "import numpy, sys; sys.stdout.write(numpy.__version__)"],
        capture_output=True, text=True,
    )
    if probe_numpy.returncode == 0:
        say(OK, f"numpy {probe_numpy.stdout.strip()}")
        return

    say(WARN, "installing numpy (the only dependency)")
    result = subprocess.run(
        [str(venv_python()), "-m", "pip", "install", "--quiet", "numpy"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        say(BAD, "numpy install failed — are you offline? Install it manually:")
        print(f"       {venv_python()} -m pip install numpy")
        print(result.stderr.strip()[:400])
        sys.exit(1)
    say(OK, "numpy installed")


def step_vault(vault: Path, assume: bool) -> None:
    if vault.exists():
        say(OK, f"vault already at {vault}")
    else:
        if not ask_yes(f"create a notes vault at {vault}?", assume):
            say(WARN, "skipped; set VAULT_DIR in the config file yourself")
            return
        for sub in ("notes", "daily", "refs"):
            (vault / sub).mkdir(parents=True, exist_ok=True)
        say(OK, f"created {vault} with notes/, daily/, refs/")

    inbox = vault / "inbox.md"
    if not inbox.exists():
        inbox.write_text(
            "# Inbox\n\nEverything lands here first. Distil it into notes/ weekly.\n",
            encoding="utf-8",
        )
        say(OK, "created inbox.md")


def step_hardware(quick: bool) -> dict | None:
    """Measure the machine and let it choose the models, rather than shipping
    one set of defaults that is wrong on most hardware."""
    scratch = HERE / ".hardware.json"
    cmd = [str(venv_python()), str(HERE / "hardware.py"), "--save", str(scratch)]
    if quick:
        cmd.append("--quick")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0 or not scratch.exists():
        say(WARN, "hardware detection failed; falling back to built-in defaults")
        return None
    try:
        info = json.loads(scratch.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    finally:
        scratch.unlink(missing_ok=True)
    return info


def step_config(vault: Path, backend: str | None, info: dict | None) -> None:
    path = config_dir() / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            say(WARN, f"{path} was not valid JSON; replacing it")

    tuned: dict[str, str] = {}
    if info:
        rec = info.get("recommendation") or {}
        # Model identifiers differ between runtimes, so only write them once a
        # backend is known. Without one, the tools fall back to their defaults.
        if backend and rec.get("chat"):
            tuned["VAULT_CHAT_MODEL"] = rec["chat"][backend]
            tuned["VAULT_EMBED_MODEL"] = rec["embed"][backend]
        if rec.get("context_length"):
            tuned["VAULT_NUM_CTX"] = str(rec["context_length"])
        if rec.get("threads"):
            tuned["VAULT_THREADS"] = str(rec["threads"])

    # Anything already in the file wins: re-running the installer must not
    # silently undo a choice you made by hand.
    merged = {"VAULT_DIR": str(vault), **tuned, **existing}
    if backend:
        merged.setdefault("VAULT_BACKEND", backend)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    say(OK, f"config at {path}")
    if tuned and not existing:
        say(OK, "settings tuned to this machine: "
                + ", ".join(f"{k.replace('VAULT_', '').lower()}={v}"
                            for k, v in tuned.items()))


def step_launchers() -> None:
    BIN.mkdir(exist_ok=True)
    python = venv_python()
    for name, script in COMMANDS.items():
        target = HERE / script
        if IS_WIN:
            launcher = BIN / f"{name}.cmd"
            launcher.write_text(f'@echo off\r\n"{python}" "{target}" %*\r\n', encoding="utf-8")
        else:
            launcher = BIN / name
            launcher.write_text(f'#!/bin/sh\nexec "{python}" "{target}" "$@"\n', encoding="utf-8")
            launcher.chmod(0o755)
    say(OK, f"{len(COMMANDS)} launchers in {BIN}")

    # Offer a symlink into a directory that is usually already on PATH,
    # rather than editing shell startup files behind the user's back.
    if not IS_WIN:
        local = Path("~/.local/bin").expanduser()
        if local.is_dir():
            for name in COMMANDS:
                link = local / name
                try:
                    if link.is_symlink() or link.exists():
                        link.unlink()
                    link.symlink_to(BIN / name)
                except OSError:
                    say(WARN, f"could not link {link}")
                    break
            else:
                say(OK, f"linked into {local}")
                return
    print(f"       add to PATH:  {BIN}")


def step_schedule(assume: bool) -> None:
    system = platform.system()
    if not ask_yes(f"set up nightly re-indexing via {_scheduler_name(system)}?", assume):
        say(WARN, "skipped scheduling")
        return
    try:
        if system == "Linux":
            _schedule_systemd()
        elif system == "Darwin":
            _schedule_launchd()
        elif system == "Windows":
            _schedule_task()
        else:
            say(WARN, f"no scheduler recipe for {system}")
    except (OSError, subprocess.SubprocessError) as exc:
        say(BAD, f"scheduling failed: {exc}")


def _scheduler_name(system: str) -> str:
    return {"Linux": "systemd", "Darwin": "launchd", "Windows": "Task Scheduler"}.get(
        system, "the system scheduler"
    )


def _schedule_systemd() -> None:
    units = Path("~/.config/systemd/user").expanduser()
    units.mkdir(parents=True, exist_ok=True)
    for unit in ("vault-index.service", "vault-index.timer"):
        shutil.copy(HERE / "systemd" / unit, units / unit)
    # The shipped unit points at index.py directly; use the venv launcher so
    # numpy resolves without a system-wide install. Either folder name is
    # accepted, since the project was renamed from vault-tools.
    service = units / "vault-index.service"
    text = service.read_text()
    for shipped in ("%h/vault-chat/index.py", "%h/vault-tools/index.py"):
        text = text.replace(shipped, f"{BIN / 'vault-index'}")
    service.write_text(text, encoding="utf-8")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "vault-index.timer"], check=True)
    say(OK, "systemd timer enabled (03:00 nightly)")
    print("       keep it running after logout:  loginctl enable-linger $USER")


def _schedule_launchd() -> None:
    label = "local.vault.index"
    plist = Path(f"~/Library/LaunchAgents/{label}.plist").expanduser()
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{BIN / 'vault-index'}</string>
    <string>--quiet</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>3</integer><key>Minute</key><integer>0</integer></dict>
  <key>RunAtLoad</key><false/>
  <key>StandardErrorPath</key><string>/tmp/{label}.log</string>
</dict>
</plist>
""",
        encoding="utf-8",
    )
    subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
    subprocess.run(["launchctl", "load", str(plist)], check=True)
    say(OK, f"launchd agent loaded ({plist.name}, 03:00 nightly)")


def _schedule_task() -> None:
    subprocess.run(
        ["schtasks", "/Create", "/F", "/SC", "DAILY", "/ST", "03:00",
         "/TN", "VaultIndex", "/TR", f'"{BIN / "vault-index.cmd"}" --quiet'],
        check=True, capture_output=True,
    )
    say(OK, "scheduled task 'VaultIndex' created (03:00 nightly)")


def step_runtime(info: dict | None) -> str | None:
    found = detect_runtime()
    if found:
        say(OK, f"model runtime detected: {found}")
        return found

    say(WARN, "no inference server running yet — the assistant needs one (colibri by default)")
    print(f"       {RUNTIME_HELP.get(platform.system(), RUNTIME_HELP['Linux'])}")
    print("       then start it once:  python3.12 c/coli start --background --no-browser")

    rec = (info or {}).get("recommendation") or {}
    chat = rec.get("chat")
    if chat:
        print(f"       LM Studio and Ollama still work as optional providers:")
        print(f"         lms load {chat['lmstudio']} --yes")
        print(f"         lms load {rec['embed']['lmstudio']} --yes")
        print(f"       or:")
        print(f"         ollama pull {chat['ollama']}")
        print(f"         ollama pull {rec['embed']['ollama']}")
    return None


def step_verify() -> bool:
    result = subprocess.run(
        [str(venv_python()), str(HERE / "smoke_test.py")],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        passed = result.stdout.count("PASS")
        say(OK, f"self-test passed ({passed} checks)")
        return True
    say(BAD, "self-test failed")
    print(result.stdout[-1500:])
    print(result.stderr[-800:])
    return False


# ------------------------------------------------------------------ main


def do_check(vault: Path) -> int:
    say(OK if venv_python().exists() else BAD, f"virtual environment  {VENV}")
    say(OK if vault.is_dir() else BAD, f"vault                {vault}")
    cfg = config_dir() / "config.json"
    say(OK if cfg.exists() else BAD, f"config               {cfg}")
    say(OK if BIN.is_dir() else BAD, f"launchers            {BIN}")
    found = detect_runtime()
    say(OK if found else BAD, f"runtime              {found or 'not running'}")
    if venv_python().exists():
        db = subprocess.run(
            [str(venv_python()), str(HERE / "index.py"), "--stats"],
            capture_output=True, text=True,
        )
        print(("\n" + db.stdout.strip()) if db.returncode == 0 else "")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--yes", "-y", action="store_true", help="accept every prompt")
    parser.add_argument("--vault", default=None, help="where notes live (default ~/vault)")
    parser.add_argument("--schedule", action="store_true", help="set up nightly re-indexing")
    parser.add_argument("--no-verify", action="store_true", help="skip the self-test")
    parser.add_argument("--check", action="store_true", help="report status and exit")
    parser.add_argument("--quick", action="store_true",
                        help="skip the memory bandwidth benchmark (about a second)")
    args = parser.parse_args()

    vault = Path(args.vault).expanduser() if args.vault else Path("~/vault").expanduser()

    if args.check:
        return do_check(vault)

    print(f"\nInstalling vault-chat from {HERE}\n")
    step_python()
    migrate_legacy_config()
    step_venv(args.yes)
    step_vault(vault, args.yes)
    info = step_hardware(args.quick)
    backend = step_runtime(info)
    step_config(vault, backend, info)
    step_launchers()
    if args.schedule or (not args.yes and ask_yes("set up nightly re-indexing?", False)):
        step_schedule(True)
    if not args.no_verify and not step_verify():
        return 1

    launcher = "vault-chat" if not IS_WIN else "vault-chat.cmd"
    print(f"""
Done. Next:

  1. Make sure a model runtime is up, with a chat model and an embedding
     model loaded.
  2. Index your notes:        {BIN / 'vault-index'}
  3. Open the chat screen:    {BIN / launcher} --open
  4. Or ask from a terminal:  {BIN / 'vault-ask'} "what did I write about X?"

Nothing here talks to the internet. Everything lives under {HERE}
and {vault}, and uninstalling means deleting those two folders.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
