"""Relational view of the object memory and open-vocabulary object lookup.

The paper keeps, inside the object memory, "a relational database with three
fields: object ID, object category, and indices of video segments where the
object has appeared", queried with SQL by the LLM, plus a feature table
matched against text descriptions. This module is that database (SQLite,
in memory, one per video) and that lookup; the agent layer only wraps them
as tools.
"""

import sqlite3
from typing import Sequence

import numpy as np

from video_agent.memory.retrieval import cosine_scores
from video_agent.memory.types import ObjectMemory
from video_agent.models.interfaces import TextAlignedImageEmbedder

# Verbatim from the reference ``database.py``; the sub-agent prompt describes
# exactly these tables, so the schema is part of the prompt contract.
SCHEMA = """
CREATE TABLE Objects(
    object_id INT,
    category VARCHAR(255),
    PRIMARY KEY (object_id)
);
CREATE TABLE Segments(
    segment_id INT,
    PRIMARY KEY (segment_id)
);
CREATE TABLE Objects_Segments(
    object_id INT,
    segment_id INT,
    PRIMARY KEY (object_id, segment_id),
    FOREIGN KEY (object_id) REFERENCES Objects(object_id),
    FOREIGN KEY (segment_id) REFERENCES Segments(segment_id)
);
"""

DEFAULT_QUERY_TEMPLATE = "a photo of a {}."  # reference CLIP prompt
DEFAULT_SIMILARITY_THRESHOLD = 0.26  # reference, calibrated for CLIP ViT-B/32


class ObjectDatabase:
    """In-memory SQLite database over an ``ObjectMemory``.

    ``read_only`` blocks writes (``PRAGMA query_only``) — a deliberate
    deviation from the reference, which executed the LLM's SQL unguarded.
    """

    def __init__(self, memory: ObjectMemory, n_segments: int, *, read_only: bool = True) -> None:
        # ``check_same_thread=False``: an interactive caller may build a
        # toolkit in one thread and query or close it from another. Access
        # is still serialised (one live operation at a time), which is the
        # condition SQLite's own docs put on sharing a connection this way;
        # the batch runners are single-threaded and unaffected.
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        cursor = self.connection.cursor()
        cursor.executescript(SCHEMA)
        cursor.executemany(
            "INSERT INTO Objects (object_id, category) VALUES (?, ?)",
            [(record.object_id, record.category) for record in memory.records],
        )
        cursor.executemany(
            "INSERT INTO Segments (segment_id) VALUES (?)",
            [(i,) for i in range(n_segments)],
        )
        pairs = sorted(
            {
                (record.object_id, index)
                for record in memory.records
                for index in record.segment_indices
                if 0 <= index < n_segments
            }
        )
        cursor.executemany(
            "INSERT INTO Objects_Segments (object_id, segment_id) VALUES (?, ?)", pairs
        )
        self.connection.commit()
        if read_only:
            cursor.execute("PRAGMA query_only = ON")
        cursor.close()

    def query(self, program: str) -> str:
        """Run ``program`` and return ``str(rows)``, or the error text, like the reference."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(program)
            return str(cursor.fetchall())
        except sqlite3.Error as error:
            return str(error)
        finally:
            cursor.close()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ObjectDatabase":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def retrieve_candidate_objects(
    memory: ObjectMemory,
    description: str,
    embedder: TextAlignedImageEmbedder,
    *,
    text_columns: slice,
    template: str = DEFAULT_QUERY_TEMPLATE,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[int]:
    """Object ids whose text-aligned feature matches ``description`` at ``threshold``.

    ``text_columns`` selects the text-aligned member of the concatenated
    ``ObjectRecord.feature`` (see ``store.crop_embedding_widths``).
    """
    if not memory.records:
        return []
    query = embedder.embed_text([template.format(description)])[0]
    features = np.stack([record.feature[text_columns] for record in memory.records])
    scores = cosine_scores(query, features)
    return [
        record.object_id
        for record, score in zip(memory.records, scores)
        if score >= threshold
    ]


def text_columns_for(widths: Sequence[int], member: int = 0) -> slice:
    """Column slice of member ``member`` in a concatenated feature of ``widths``."""
    if not 0 <= member < len(widths):
        raise ValueError(f"member {member} out of range for widths {tuple(widths)}")
    start = sum(widths[:member])
    return slice(start, start + widths[member])
