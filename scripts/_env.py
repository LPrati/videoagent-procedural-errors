"""Minimal ``.env`` loader for the scripts (no python-dotenv dependency).

API keys live in the repo-root ``.env`` (git-ignored) so a run only needs
the file, never a key on the command line or in a transcript.
"""

import os
from pathlib import Path

DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
"""Repo root, i.e. the directory holding this project's ``.gitignore``."""


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read ``KEY=VALUE`` lines into ``os.environ`` and return what the file held.

    Existing environment variables win, so an explicit ``KEY=... uv run ...``
    overrides the file. A missing file is not an error: the keys may already
    be exported.
    """
    file = DEFAULT_ENV_FILE if path is None else Path(path)
    if not file.exists():
        return {}
    values: dict[str, str] = {}
    for line in file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue
        values[key] = _unquote(value.strip())
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values
