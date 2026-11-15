"""Point every test at a throwaway data directory before any project module
reads its configuration. Import this first in each test module."""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="vault-chat-tests-"))
atexit.register(shutil.rmtree, TMP, True)

os.environ.update({
    "VAULT_DIR": str(TMP / "vault"),
    "VAULT_DB": str(TMP / "vault" / ".index" / "vault.db"),
    "VAULT_DATA_DIR": str(TMP / "data"),
    "VAULT_BACKEND": "colibri",
    "XDG_CONFIG_HOME": str(TMP / "config"),
})
for key in ("VAULT_EMBED_URL", "VAULT_PLATFORM_DB", "COLIBRI_BASE_URL", "VAULT_CHAT_MODEL"):
    os.environ.pop(key, None)
(TMP / "vault").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

_counter = [0]


def fresh_db() -> Path:
    _counter[0] += 1
    return TMP / "data" / f"test-{_counter[0]}.db"


def stable_id(text: str) -> int:
    """Deterministic across processes, unlike hash()."""
    return zlib.crc32(text.encode())
