"""The detector registry: which model runs and what its memory is called.

The adapter itself cannot be exercised here — no GPU, no checkpoint, and it
imports torch inside its methods for exactly that reason. What is testable
offline is the part that decides *which* detector a command builds and *how
its artifacts are named*, which is where a mistake would be silent: a
drifting slug does not crash, it writes a second memory nobody reads.
"""

import pytest

from video_agent.adapters.detector import (
    DEFAULT_DETECTOR,
    DETECTORS,
    build_detector,
    detector_slug,
    detector_spec,
    known_detectors,
    object_memory_slug,
)
from video_agent.memory.store import DEFAULT_DETECTOR_SLUG


class TestSlugs:
    def test_the_default_keeps_the_slug_the_store_names_artifacts_with(self) -> None:
        # ``memory.store`` spells this slug itself rather than importing an
        # adapter; the two have to say the same thing or a default run would
        # start writing a suffixed file next to the unsuffixed one.
        assert detector_slug(DEFAULT_DETECTOR) == "rtdetr-v2-r50"
        assert detector_slug(DEFAULT_DETECTOR) == DEFAULT_DETECTOR_SLUG

    def test_every_detector_has_its_own_filesystem_safe_slug(self) -> None:
        slugs = [spec.slug for spec in DETECTORS.values()]

        assert len(slugs) == len(set(slugs))
        for slug in slugs:
            assert slug and slug == slug.lower()
            assert set(slug) <= set("abcdefghijklmnopqrstuvwxyz0123456789-")

    def test_the_default_detector_carries_its_score_threshold(self) -> None:
        assert detector_spec(DEFAULT_DETECTOR).score_threshold == 0.25

    def test_an_unknown_id_lists_the_known_ones(self) -> None:
        with pytest.raises(ValueError) as error:
            detector_slug("nobody/nothing")

        message = str(error.value)
        assert "nobody/nothing" in message
        assert DEFAULT_DETECTOR in message

    def test_known_detectors_lists_every_id_sorted(self) -> None:
        ids = known_detectors()

        assert ids == sorted(ids)
        assert set(ids) == set(DETECTORS)


class TestObjectMemorySlug:
    def test_a_detector_names_its_memory_with_its_slug(self) -> None:
        assert object_memory_slug(DEFAULT_DETECTOR) == "rtdetr-v2-r50"


class TestBuildDetector:
    def test_an_unknown_id_is_refused_before_anything_is_loaded(self) -> None:
        with pytest.raises(ValueError, match="unknown detector"):
            build_detector("nobody/nothing", device="cpu", batch_size=1)
