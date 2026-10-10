"""Memory controls: withholding the memory, or shuffling it.

Two counterfactual conditions run the agent over the same recordings. With
the memory **withheld** (empty) every tool answers that there is none, so
whatever the agent still gets right came from the recipe and not from the
video (a pure-prior control). With the memory **shuffled** the content is
exactly the content of the ordered run — same captions, same vectors, same
objects — under other segment names, so what survives measures how much of
the verdict used the order of the video. Under shuffling, the anchors the
agent cites are mapped back to the real time spans of the segments they name,
so the computed order is read on the content the agent actually saw.

Both are *views*: nothing here rewrites an artifact. A shuffled run reads
the very files an ordered run reads and renames their rows, which is what
makes the two comparable at all.

The permutation is written down as ``perm[new] = old``: segment ``k`` of the
shuffled memory carries what segment ``perm[k]`` of the recording holds, and
``inverse()[old] = new`` renames an index the other way, which is what the
object memory's segment indices need. One permutation per recording, drawn
from ``random.Random(PERMUTATION_BASE_SEED + ordinal)`` where ``ordinal`` is
the recording's position in its split: the seed is a property of the split
and of the recording, so two machines sharding the same run shuffle each
recording the same way, and the permutation is saved beside the run record
anyway (``task.run.RunStore.save_permutation``).
"""

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from video_agent.memory.types import (
    ObjectMemory,
    ObjectRecord,
    Segment,
    TemporalMemory,
)

MEMORY_FULL = "full"
MEMORY_EMPTY = "empty"
MEMORY_SHUFFLED = "shuffled"
MEMORY_MODES = (MEMORY_FULL, MEMORY_EMPTY, MEMORY_SHUFFLED)
"""The three memory conditions, as ``--memory`` names them.

``full`` is the normal agent and the flag's default, so a configuration that
does not set the flag is unaffected by it.
"""

PERMUTATION_BASE_SEED = 20260903
"""Base seed of the shuffled condition; the recording's ordinal is added to it.

The same fixed value ``--seed`` carries into the runs. It is part of what
``config_id`` hashes for a shuffled run, so a shuffle drawn with another base
seed is another configuration.
"""

WITHHELD_CAPTION = ""
"""Caption of a withheld memory's segment: there is none, and nothing reads it.

The tools of a withheld memory return ``MEMORY UNAVAILABLE`` before they
look at a caption (``agent.tools``); the list exists so the grid still has a
row per segment and the agent is still told how long the video is.
"""


@dataclass(frozen=True)
class SegmentPermutation:
    """One recording's segment permutation, ``perm[new] = old``.

    Carries the seed and the ordinal it was drawn from because the record on
    disk has to say not only *which* permutation ran but also *why* that one:
    a shuffled run that could not be redrawn from its own record would not be
    reproducible.
    """

    perm: tuple[int, ...]
    seed: int
    ordinal: int

    def __post_init__(self) -> None:
        if sorted(self.perm) != list(range(len(self.perm))):
            raise ValueError(
                f"perm is not a permutation of 0..{len(self.perm) - 1}: {self.perm}"
            )

    @property
    def n_segments(self) -> int:
        return len(self.perm)

    def inverse(self) -> tuple[int, ...]:
        """``inverse()[old] = new``: the new name of an original segment."""
        inverse = [0] * len(self.perm)
        for new, old in enumerate(self.perm):
            inverse[old] = new
        return tuple(inverse)

    def to_dict(self) -> dict:
        return {
            "perm": list(self.perm),
            "seed": self.seed,
            "ordinal": self.ordinal,
            "n_segments": self.n_segments,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SegmentPermutation":
        return cls(
            perm=tuple(int(value) for value in data["perm"]),
            seed=int(data["seed"]),
            ordinal=int(data["ordinal"]),
        )

    def payload(self) -> str:
        """Canonical serialization, the bytes ``sha256`` is taken over.

        Canonical and not the file's own text: the file is written with
        indentation like every other artifact, and a digest that depended on
        the indentation would change whenever the writer's formatting did. Reading the
        file back through ``from_dict`` reproduces this string exactly.
        """
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def sha256(self) -> str:
        """Digest of the permutation, for the run record's configuration.

        It says which shuffle a record was produced under; it is deliberately
        *not* part of ``config_id``, because every recording of a shuffled run
        has its own permutation and they all share one configuration.
        """
        return hashlib.sha256(self.payload().encode("utf-8")).hexdigest()


def segment_permutation(
    n_segments: int, ordinal: int, base_seed: int = PERMUTATION_BASE_SEED
) -> SegmentPermutation:
    """Draw the permutation of a recording at position ``ordinal`` in its split.

    Uniform over the ``n!`` orders (``random.shuffle`` is a Fisher-Yates
    shuffle) and reproducible from the seed alone, which is what lets a
    sharded run produce one permutation per recording without any worker
    talking to another.
    """
    if n_segments < 0:
        raise ValueError(f"a memory cannot have {n_segments} segments")
    order = list(range(n_segments))
    seed = base_seed + int(ordinal)
    random.Random(seed).shuffle(order)
    return SegmentPermutation(perm=tuple(order), seed=seed, ordinal=int(ordinal))


def _check_size(permutation: SegmentPermutation, n_segments: int, what: str) -> None:
    if permutation.n_segments != n_segments:
        raise ValueError(
            f"permutation of {permutation.n_segments} segments cannot rename "
            f"{what} of {n_segments}"
        )


def shuffle_segments(
    segments: Sequence[Segment], permutation: SegmentPermutation
) -> list[Segment]:
    """Rename the grid: row ``k`` is the interval of original ``perm[k]``.

    The times and frames stay the ones the video really has — they are what
    the captions of that row describe — while ``index`` becomes the new name,
    so the memory a shuffled run holds is self-consistent: ``segments[k].index
    == k``. An anchor the agent cites is therefore read back as the time span
    of the content it saw, which is the only reading of a shuffled anchor that
    means anything (``task.verdicts``).
    """
    _check_size(permutation, len(segments), "a grid")
    return [
        Segment(
            index=new,
            start_s=segments[old].start_s,
            end_s=segments[old].end_s,
            frame_range=segments[old].frame_range,
        )
        for new, old in enumerate(permutation.perm)
    ]


def shuffle_temporal_memory(
    memory: TemporalMemory, permutation: SegmentPermutation
) -> TemporalMemory:
    """Reindex a temporal memory: row ``k`` is the old row ``perm[k]``.

    Captions and both embedding matrices move together, so the multiset of
    captions and the set of vectors are exactly the ordered memory's — only
    the names change. ``np.take`` copies, and the copy is what the view owns:
    the caller's memory is never edited.
    """
    _check_size(permutation, len(memory.segments), "a temporal memory")
    order = list(permutation.perm)
    return TemporalMemory(
        segments=shuffle_segments(memory.segments, permutation),
        captions=[memory.captions[old] for old in order],
        caption_embeddings=np.take(memory.caption_embeddings, order, axis=0),
        visual_embeddings=np.take(memory.visual_embeddings, order, axis=0),
    )


def shuffle_object_memory(
    memory: ObjectMemory, permutation: SegmentPermutation
) -> ObjectMemory:
    """Rename the segments an object appears in, by the same permutation.

    Here the map runs the other way: a record says where an object was seen
    in the *original* grid, and the shuffled memory has to call that segment
    by its new name (``inverse()[old]``). An index outside the grid is left
    as it is — ``ObjectDatabase`` drops it either way, and inventing a name
    for it would put a pair in the shuffled database that the ordered one
    does not have.
    """
    inverse = permutation.inverse()
    n_segments = permutation.n_segments
    records = [
        ObjectRecord(
            object_id=record.object_id,
            category=record.category,
            segment_indices=tuple(
                sorted(
                    inverse[index] if 0 <= index < n_segments else index
                    for index in record.segment_indices
                )
            ),
            feature=record.feature,
            tracklets=record.tracklets,
        )
        for record in memory.records
    ]
    return ObjectMemory(records=records)


@dataclass(frozen=True)
class ShuffledMemories:
    """The views one recording's agent runs on under ``--memory shuffled``.

    ``video_segments`` is the grid the *video* is still cut on, kept beside
    the renamed memory because the visual-question-answering tool shows
    pixels: asked about shuffled segment ``k`` it has to play original
    ``perm[k]``, so that the clip the agent watches is the clip the caption
    it just read describes.
    """

    permutation: SegmentPermutation
    temporal: TemporalMemory
    objects: ObjectMemory | None
    video_segments: tuple[Segment, ...]


def shuffle_memories(
    temporal: TemporalMemory,
    objects: ObjectMemory | None,
    permutation: SegmentPermutation,
) -> ShuffledMemories:
    """Apply one permutation to everything a recording's memory indexes."""
    return ShuffledMemories(
        permutation=permutation,
        temporal=shuffle_temporal_memory(temporal, permutation),
        objects=None if objects is None else shuffle_object_memory(objects, permutation),
        video_segments=tuple(temporal.segments),
    )


def withheld_temporal_memory(segments: Sequence[Segment]) -> TemporalMemory:
    """A memory with the recording's grid and nothing in it (``--memory empty``).

    The grid survives because the prompt still tells the agent how many
    segments the video has — the control withholds the memory, not the length
    of the video — and because the record and the verdict anchors are read on
    that grid. The rows are empty and the embedding matrices have no columns:
    every tool of a withheld memory answers ``MEMORY UNAVAILABLE`` before it
    would look at one, and building anything wider would be pretending to
    have read an artifact this condition never opens.
    """
    rows = len(segments)
    return TemporalMemory(
        segments=list(segments),
        captions=[WITHHELD_CAPTION] * rows,
        caption_embeddings=np.zeros((rows, 0), dtype=np.float32),
        visual_embeddings=np.zeros((rows, 0), dtype=np.float32),
    )


class WithheldVideoQA:
    """Stand-in for the video-question-answering model of a withheld memory.

    The control has to leave the *shape* of the agent untouched — the same four
    tools with the same descriptions, so that the prompt is the prompt the
    full run used — while none of them answers; the toolkit is therefore
    still handed something in the model's place. It is never called:
    ``ToolKit.visual_question_answering`` returns ``MEMORY UNAVAILABLE``
    before it reaches a model, and this raises rather than answer if that
    ever stops being true.
    """

    def answer(self, clips: Sequence[object], questions: Sequence[str]) -> list[str]:
        raise RuntimeError(
            "the memory is withheld: no clip may be decoded and no VQA model asked"
        )
