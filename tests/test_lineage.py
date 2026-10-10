"""Provenance of a saved output: the commit and the library versions.

Every report the project writes has to name the code that produced it, so
these are the guarantees the writers rely on: the version is the short SHA of
the checkout the *package* lives in, whatever directory the run was started
from, and neither helper ever raises — a run outside a git tree or without
``torch`` still has to finish and still has to say so.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from video_agent.lineage import (
    LIBRARIES,
    code_version,
    library_version,
    library_versions,
)

SHORT_SHA = re.compile(r"^[0-9a-f]{7,}$")
"""``git rev-parse --short`` gives 7 hex characters, more only on collision."""


class TestCodeVersion:
    def test_inside_this_checkout_it_is_the_short_sha_of_head(self) -> None:
        version = code_version()

        assert version is not None
        assert SHORT_SHA.match(version), version

    def test_it_names_the_checkout_of_the_package_not_of_the_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Preprocessing runs are started from a dataset folder; the version
        # they record must still be the version of the code that is running.
        monkeypatch.chdir(tmp_path)

        assert code_version() is not None

    def test_outside_a_checkout_the_version_is_unknown_rather_than_an_error(
        self, tmp_path: Path
    ) -> None:
        assert code_version(tmp_path) is None

    def test_a_path_that_does_not_exist_is_unknown_too(self, tmp_path: Path) -> None:
        assert code_version(tmp_path / "nowhere") is None


class TestLibraryVersions:
    def test_every_library_of_the_reports_gets_a_key(self) -> None:
        versions = library_versions()

        assert set(versions) == {"torch", "transformers", "open_clip"}
        assert set(versions) == set(LIBRARIES)

    def test_a_library_that_is_not_installed_reads_as_none(self) -> None:
        assert library_version("no_such_module_anywhere") is None
        assert library_versions(["no_such_module_anywhere"]) == {
            "no_such_module_anywhere": None
        }

    def test_a_distribution_named_unlike_its_import_is_still_found(self) -> None:
        # ``cv2`` ships as ``opencv-python-headless`` and ``open_clip`` as
        # ``open-clip-torch``; looking the import name up in the metadata
        # alone would report an installed library as missing.
        version = library_version("cv2")

        assert version is not None and version[0].isdigit()

    def test_asking_never_imports_the_library(self) -> None:
        # An offline code path reports which torch is installed; it must not
        # load a GPU stack to do so. A fresh interpreter is the only
        # honest check — this one may already have imported anything.
        snippet = (
            "import sys; from video_agent.lineage import library_versions; "
            "library_versions(); print('torch' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, check=True,
        )

        assert completed.stdout.strip() == "False"
