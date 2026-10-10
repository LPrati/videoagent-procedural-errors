"""Stage-wise persistence for memory construction.

Each video gets one directory holding one file per pipeline stage plus a
``meta.json`` recording data lineage: which model and parameters produced
each artifact, when, and from which code version. Stages can then run (and
re-run) independently — for example, embeddings while the GPU is free and
captioning while the model server holds it — and assembly reads whatever is
already there.

Writes are atomic (temp file + rename) so a crash mid-write never leaves a
truncated artifact that a resumed run would trust.
"""

import json
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

META_FILE = "meta.json"


class VideoArtifacts:
    """Artifact directory for one video."""

    def __init__(self, root: Path, video_name: str) -> None:
        self.directory = Path(root) / video_name
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, filename: str) -> Path:
        return self.directory / filename

    def has(self, filename: str) -> bool:
        return self.path(filename).exists()

    def _atomic_write(self, filename: str, write: Callable[[Path], None]) -> None:
        with tempfile.NamedTemporaryFile(
            dir=self.directory, prefix=f".{filename}.", delete=False
        ) as handle:
            temp = Path(handle.name)
        try:
            write(temp)
            temp.replace(self.path(filename))
        except BaseException:
            temp.unlink(missing_ok=True)
            raise

    def save_json(self, filename: str, obj: Any) -> None:
        self._atomic_write(
            filename, lambda p: p.write_text(json.dumps(obj, indent=1))
        )

    def load_json(self, filename: str) -> Any:
        return json.loads(self.path(filename).read_text())

    def save_array(self, filename: str, array: np.ndarray) -> None:
        def write(p: Path) -> None:
            with open(p, "wb") as f:
                np.save(f, array)

        self._atomic_write(filename, write)

    def load_array(self, filename: str) -> np.ndarray:
        return np.load(self.path(filename))

    def save_pickle(self, filename: str, obj: Any) -> None:
        def write(p: Path) -> None:
            with open(p, "wb") as f:
                pickle.dump(obj, f)

        self._atomic_write(filename, write)

    def load_pickle(self, filename: str) -> Any:
        with open(self.path(filename), "rb") as f:
            return pickle.load(f)

    def record_stage(
        self,
        stage: str,
        *,
        artifact: str,
        model: str | None = None,
        params: dict | None = None,
        code_version: str | None = None,
    ) -> None:
        """Append this stage's lineage to ``meta.json`` (last write wins)."""
        meta = self.load_json(META_FILE) if self.has(META_FILE) else {}
        meta[stage] = {
            "artifact": artifact,
            "model": model,
            "params": params or {},
            "code_version": code_version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.save_json(META_FILE, meta)

    def stage_meta(self, stage: str) -> dict | None:
        if not self.has(META_FILE):
            return None
        return self.load_json(META_FILE).get(stage)
