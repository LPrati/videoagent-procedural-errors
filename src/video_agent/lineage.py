"""Provenance of a saved output: which code and which libraries wrote it.

Every artifact this project saves carries the commit it was produced by, so a
report found on disk later can be traced back to the code that made its
numbers. The helpers live in the package rather than in each script so that
every script that saves an output records its provenance the same way.

Nothing here raises and nothing here imports a model library: a run must not
fail because it could not name itself, and an offline code path must not
pull ``torch`` into the process just to report which ``torch`` is installed.
"""

import subprocess
from importlib import metadata
from pathlib import Path
from typing import Iterable

PACKAGE_ROOT = Path(__file__).resolve().parent
"""Where the running code lives, and so which checkout to ask about."""

LIBRARIES: tuple[str, ...] = ("torch", "transformers", "open_clip")
"""Libraries whose version can move a number in a report.

The three that hold the model implementations: the aligner towers come from
``transformers`` or from ``open_clip`` depending on the model, and both run
on ``torch``.
"""


def code_version(path: str | Path | None = None) -> str | None:
    """Short SHA of ``HEAD``, or ``None`` outside a git checkout.

    Asks about the checkout that contains this package by default, not about
    the process's working directory: a script launched from a dataset folder
    still reports the code that is running. ``path`` overrides that, which is
    what makes the "no checkout" answer testable.

    Never raises. A missing ``git``, an exported tarball with no ``.git`` and
    a repository with no commit yet all mean the same thing — the version is
    unknown — and an unknown version is a value, not a failure.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PACKAGE_ROOT if path is None else path),
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def library_version(name: str) -> str | None:
    """Installed version of the importable ``name``, or ``None``.

    Read from the distribution metadata, so asking costs no import. The
    import name and the distribution name are not always the same
    (``open_clip`` ships as ``open-clip-torch``), hence the second lookup:
    without it an installed library would be reported as missing.
    """
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
    for distribution in metadata.packages_distributions().get(name, ()):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return None


def library_versions(names: Iterable[str] = LIBRARIES) -> dict[str, str | None]:
    """Version of each library, ``None`` for the ones that are not installed.

    Every name always gets a key: "this run had no ``open_clip``" is itself
    part of the provenance, and a missing key would read as a report that
    never recorded the library at all.
    """
    return {name: library_version(name) for name in names}
