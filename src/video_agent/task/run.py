"""One recording in, one run record out: the unit every run logs.

Every configuration knob that can change a verdict is frozen into a
``TaskConfig`` and hashed into a ``config_id``, so two records can be
compared without trusting a directory name. The record itself carries the
question, the raw Final Answer, the normalized verdicts, the whole
transcript and the cost counters: none of it can be reconstructed
afterwards, and without the transcript error analysis is impossible.

The record is a plain dict rather than a dataclass because it is written
straight to disk with ``json.dumps``; keeping it JSON-native means a reader
in the evaluation stage never has to import this package.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from video_agent.adapters.aligner import aligner_slug
from video_agent.agent.config import ReActConfig, ToolConfig
from video_agent.agent.runner import AgentResult, AgentRunner
from video_agent.agent.tools import ToolKit
from video_agent.agent.transcript import Transcript
from video_agent.media import Video
from video_agent.memory.ablation import (
    MEMORY_FULL,
    MEMORY_SHUFFLED,
    PERMUTATION_BASE_SEED,
    SegmentPermutation,
)
from video_agent.memory.artifacts import VideoArtifacts
from video_agent.memory.store import (
    DEFAULT_DETECTOR_SLUG,
    object_memory_file,
    visual_embeddings_file,
)
from video_agent.memory.types import Segment
from video_agent.models.types import Usage
from video_agent.task.annotations import Sop, TaskGraph
from video_agent.task.prompt import (
    build_question,
    final_answer_candidates,
    parse_final_answer,
)
from video_agent.task.verdicts import normalize

RUN_SCHEMA_VERSION = 1
"""Bumped when the shape of a run record changes; readers check it."""

RUN_FILE = "run.json"
"""Name of the record inside a recording's directory of a run."""

PERMUTATION_FILE = "permutation.json"
"""Name of the shuffled condition's permutation, beside the record it produced.

A shuffled-memory run draws one permutation per recording; it is
written here, next to that recording's ``run.json``, so the run carries what
it would take to read its own transcript — which caption the agent saw when
it asked for segment 40 — without redrawing anything.
"""

CONFIG_ID_LENGTH = 16
"""Hex characters of the configuration digest kept; enough to name a run."""

PACKAGE_TEMPLATE = "package"
"""``*_template_source`` of a run that used the template shipped in the code."""


@dataclass(frozen=True)
class TaskConfig:
    """Everything about a run that could change its verdicts.

    The prompts are recorded by digest rather than by text: they are long,
    they are already stored as package data, and what a comparison needs is
    whether two runs used the same wording.

    The two ``*_template_source`` fields say where each template was read
    from — a path, or ``PACKAGE_TEMPLATE`` for the shipped one. They are
    provenance and not identity: the digest above already says *which*
    wording ran, while the path of the same file differs between machines,
    so they stay out of ``to_dict`` and therefore out of ``config_id``
    (``template_sources`` puts them in the record).
    """

    orchestrator_model: str
    chat_backend: str
    thinking: bool  # whether the orchestrator was asked to reason first
    vqa_backend: str
    vqa_style: str
    caption_variant: str
    use_objects: bool
    seed: int | None
    temperature: float
    react: ReActConfig
    tools: ToolConfig
    prompt_sha256: str
    question_template_sha256: str
    code_version: str | None
    prompt_template_source: str = PACKAGE_TEMPLATE
    question_template_source: str = PACKAGE_TEMPLATE
    # Which memory the agent ran on: the recording's own (``full``), none at
    # all (``empty``) or the same one with its segments permuted
    # (``shuffled``), which measures how much the answers owe to the prior
    # rather than to the memory; see ``memory.ablation``.
    memory: str = MEMORY_FULL
    memory_seed: int = PERMUTATION_BASE_SEED
    # Model that answers the visual questions when it is served by a separate
    # endpoint (``--vqa-base-url``); ``None`` when the orchestrator's own
    # endpoint answers them.
    vqa_endpoint_model: str | None = None

    def to_dict(self) -> dict:
        """The identifying half of the configuration, as ``config_id`` hashes it.

        ``prompt_template_source`` and ``question_template_source`` are left
        out on purpose: two checkouts running the very same custom template
        from two different paths are the same configuration, and hashing the
        path would give them two ids. See ``template_sources``.

        ``memory`` is written down only when it is not ``full``, and its base
        seed only when it is ``shuffled``, so the ``config_id`` of a
        full-memory configuration does not depend on the memory options at
        all: an absent key reads as ``full``. This keeps the ids of
        full-memory runs stable and directly comparable with their empty- and
        shuffled-memory counterparts. ``vqa_endpoint_model`` follows the
        same rule: it is written only when a separate VQA endpoint was used.
        """
        data = {
            "orchestrator_model": self.orchestrator_model,
            "chat_backend": self.chat_backend,
            "thinking": self.thinking,
            "vqa_backend": self.vqa_backend,
            "vqa_style": self.vqa_style,
            "caption_variant": self.caption_variant,
            "use_objects": self.use_objects,
            "seed": self.seed,
            "temperature": self.temperature,
            "react": self.react.to_dict(),
            "tools": self.tools.to_dict(),
            "prompt_sha256": self.prompt_sha256,
            "question_template_sha256": self.question_template_sha256,
            "code_version": self.code_version,
        }
        if self.memory != MEMORY_FULL:
            data["memory"] = self.memory
            if self.memory == MEMORY_SHUFFLED:
                # The permutation of one recording is not part of the
                # configuration — every recording of the run has its own, and
                # they share one id — but the seed they are all drawn from is.
                data["memory_seed"] = self.memory_seed
        if self.vqa_endpoint_model is not None:
            data["vqa_endpoint_model"] = self.vqa_endpoint_model
        return data


def config_id(config: TaskConfig) -> str:
    """Short digest naming a configuration.

    Keys are sorted so the id depends on the values and not on the order the
    dataclass happens to serialize them in.
    """
    payload = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:CONFIG_ID_LENGTH]


def visual_memory_names(config: TaskConfig) -> dict[str, str]:
    """Which aligner's artifact this configuration reads, by slug and by name.

    Both follow from ``tools.aligner_model``, so neither adds anything to
    ``config_id``; they are written into the record because a recording's
    directory may hold one ``segment_visual_embeddings*.npy`` per aligner,
    and a reader asking which of them produced these verdicts
    should not have to know the registry by heart.
    """
    slug = aligner_slug(config.tools.aligner_model)
    return {
        "aligner_slug": slug,
        "visual_embeddings_file": visual_embeddings_file(slug),
    }


def template_sources(config: TaskConfig) -> dict[str, str]:
    """Where this run read each of its two templates from.

    The counterpart of the digests in ``to_dict``: those say which wording
    ran, this says which file it came from — a path when the run overrode a
    template, ``PACKAGE_TEMPLATE`` otherwise. Kept out of ``config_id`` for
    the reason ``TaskConfig`` gives, and written into the record like the
    memory names below, so a reader can find the file again.
    """
    return {
        "prompt_template_source": config.prompt_template_source,
        "question_template_source": config.question_template_source,
    }


def object_memory_names(config: TaskConfig) -> dict[str, str]:
    """Which detector's object memory this configuration reads.

    The sibling of ``visual_memory_names`` for the other memory. Both keys
    are derived from ``tools.object_memory_slug``, which ``config_id``
    already hashes, so they add nothing to the identity of a run — they are
    written because a recording's directory may hold one
    ``object_memory*.pkl`` per detector and a reader asking which of them answered the
    sub-agent should not have to know the registry by heart. ``None`` is
    spelled out as the default slug rather than left null, so the
    field always names a detector.
    """
    slug = config.tools.object_memory_slug
    return {
        "object_memory_slug": slug or DEFAULT_DETECTOR_SLUG,
        "object_memory_file": object_memory_file(slug),
    }


def permutation_name(permutation: SegmentPermutation | None) -> dict[str, str]:
    """Which shuffle produced this record, by digest; nothing when there was none.

    The sibling of ``visual_memory_names`` for the shuffled memory, and the one
    per-recording entry of the configuration block: the permutation belongs
    to this recording, while ``config_id`` names the run. Its digest is here
    so a record and the ``permutation.json`` beside it can be checked against
    each other, and the whole list is in that file rather than in the record,
    which has to stay readable.
    """
    if permutation is None:
        return {}
    return {"memory_permutation_sha256": permutation.sha256()}


def total_usage(transcript: Transcript) -> Usage:
    """Add up the parent episode's cost and every sub-agent episode inside it.

    ``Transcript.usage`` counts only its own loop, so a run whose agent asked
    the object-memory sub-agent anything would under-report its tokens if the
    nested transcripts were not walked.
    """
    total = transcript.usage
    for step in transcript.steps:
        if step.action is not None and step.action.sub_transcript is not None:
            total = total + total_usage(step.action.sub_transcript)
    return total


def _now() -> str:
    """UTC timestamp at second resolution, as the record stores it."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_recording(
    *,
    recording_id: str,
    sop: Sop,
    graph: TaskGraph,
    toolkit: ToolKit,
    runner: AgentRunner,
    prompt_template: str,
    question_template: str,
    config: TaskConfig,
    permutation: SegmentPermutation | None = None,
) -> dict:
    """Run one ReAct episode over one recording and return its run record.

    This is the reference system: the reproduced agent used as it is (one
    episode, the original tools, the original prompt) with only the
    question and the answer format changed. A failure of the agent is
    recorded, not raised: a batch of recordings must not stop because one
    video's tools blew up, and a record with ``status`` ``"error"`` is itself
    a result — one that keeps the steps the agent had already run, when the
    runner hands its partial transcript over with the exception.
    """
    segments = toolkit.temporal.segments
    n_segments = toolkit.n_segments
    question = build_question(sop, graph, n_segments, question_template)
    started_at = _now()
    started = time.perf_counter()

    error: str | None = None
    result = None
    partial: Transcript | None = None
    try:
        result = runner.run(
            question, toolkit.main_tools(), prompt_template=prompt_template
        )
    except Exception as exception:  # the batch outlives one broken recording
        error = f"{type(exception).__name__}: {exception}"
        # A runner that kept the episode it had built hands it over on the
        # exception (``ReActError.transcript``). Duck-typed rather than
        # imported: what this function needs is the ``AgentRunner`` protocol,
        # not one implementation of it.
        carried = getattr(exception, "transcript", None)
        partial = carried if isinstance(carried, Transcript) else None

    return make_record(
        recording_id=recording_id,
        sop=sop,
        segments=segments,
        video=toolkit.video,
        question=question,
        config=config,
        started_at=started_at,
        wall_s=time.perf_counter() - started,
        result=result,
        error=error,
        partial=partial,
        permutation=permutation,
    )


def make_record(
    *,
    recording_id: str,
    sop: Sop,
    segments: Sequence[Segment],
    video: Video,
    question: str,
    config: TaskConfig,
    started_at: str,
    wall_s: float,
    result: AgentResult | None,
    error: str | None,
    partial: Transcript | None,
    permutation: SegmentPermutation | None = None,
) -> dict:
    """Turn one episode — finished, failed or still running — into a record.

    Split out of ``run_recording`` because the episode and the record are two
    different things: a caller that steps a ``ReActSession`` by hand still
    has to show, and be able to save, exactly the record a batch run would
    have written. Nothing here touches a model or a tool, so a record can be
    rebuilt from a transcript alone.

    An unfinished episode is not an error: a ``result`` whose status is
    ``"running"`` falls through to a record with that status and the
    unclassified verdicts an absent answer earns, as for any run that
    produced no answer.
    """
    if result is None:
        status = "error"
        raw = None
        final_answer_text = None
        n_answer_candidates = 0
        transcript = partial
        usage = Usage() if partial is None else partial.usage
        usage_sum = Usage() if partial is None else total_usage(partial)
    else:
        final_answer_text = result.answer
        # Counted apart from the accepted answer: a model that wrote several
        # objects, or one empty one, is a run to look at even when the reader
        # recovered something (the record keeps what cannot be reconstructed
        # later).
        n_answer_candidates = len(final_answer_candidates(final_answer_text or ""))
        raw = parse_final_answer(final_answer_text or "")
        if raw is not None:
            status = "ok"
        elif result.status == "final_answer":
            # The agent finished, but not with the JSON it was asked for.
            status = "parse_failed"
        else:
            status = result.status
        transcript = result.transcript
        usage = result.usage
        usage_sum = total_usage(result.transcript)

    verdicts = normalize(raw, sop, segments)
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "config": {
            **config.to_dict(),
            "config_id": config_id(config),
            **template_sources(config),
            **visual_memory_names(config),
            **object_memory_names(config),
            **permutation_name(permutation),
        },
        "recording_id": recording_id,
        "activity_id": sop.activity_id,
        "started_at": started_at,
        "finished_at": _now(),
        "wall_s": wall_s,
        "status": status,
        "error": error,
        "n_segments": len(segments),
        # The evaluation reads the grid in seconds (pairwise order,
        # localization) and must not
        # have to reopen the video or the artifacts directory to get it.
        "video": {
            "fps": float(video.fps),
            "frame_count": int(video.frame_count),
            "n_segments": len(segments),
        },
        "question": question,
        "final_answer_text": final_answer_text,
        "n_answer_candidates": n_answer_candidates,
        "verdicts": [verdict.to_dict() for verdict in verdicts],
        "transcript": None if transcript is None else transcript.to_dict(),
        "usage": usage.to_dict(),
        "usage_total": usage_sum.to_dict(),
    }


class RunStore:
    """The records of one run, one directory per recording.

    Laid out as ``<root>/<run_name>/<recording_id>/run.json`` through
    ``VideoArtifacts``, which writes atomically: a run is resumed by asking
    which records exist, and a half-written file would be indistinguishable
    from a finished one.
    """

    def __init__(self, root: Path, run_name: str) -> None:
        self.root = Path(root)
        self.run_name = run_name
        self.directory = self.root / run_name

    def _artifacts(self, recording_id: str) -> VideoArtifacts:
        return VideoArtifacts(self.directory, recording_id)

    def has(self, recording_id: str) -> bool:
        """Whether ``recording_id`` already has a record.

        Reads the path directly instead of through ``VideoArtifacts``, whose
        constructor creates the directory: asking a question should not leave
        an empty directory that ``recording_ids`` would then have to filter.
        """
        return (self.directory / recording_id / RUN_FILE).is_file()

    def load(self, recording_id: str) -> dict:
        return self._artifacts(recording_id).load_json(RUN_FILE)

    def save(self, recording_id: str, record: dict) -> None:
        self._artifacts(recording_id).save_json(RUN_FILE, record)

    def save_permutation(
        self, recording_id: str, permutation: SegmentPermutation
    ) -> None:
        """Write the shuffle this recording ran under, beside its record.

        Before the record and not after: the permutation is what the episode
        is about to happen on, so a run interrupted mid-episode still leaves
        behind the memory its transcript refers to.
        """
        self._artifacts(recording_id).save_json(
            PERMUTATION_FILE, permutation.to_dict()
        )

    def load_permutation(self, recording_id: str) -> SegmentPermutation:
        return SegmentPermutation.from_dict(
            self._artifacts(recording_id).load_json(PERMUTATION_FILE)
        )

    def has_permutation(self, recording_id: str) -> bool:
        """Whether this recording's shuffle was written; see ``has``."""
        return (self.directory / recording_id / PERMUTATION_FILE).is_file()

    def recording_ids(self) -> list[str]:
        """The recordings this run has records for, sorted."""
        if not self.directory.is_dir():
            return []
        return sorted(
            path.parent.name for path in self.directory.glob(f"*/{RUN_FILE}")
        )
