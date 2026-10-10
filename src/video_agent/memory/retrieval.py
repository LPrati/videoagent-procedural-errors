"""Query the temporal memory by text: the scoring behind segment localization.

Reference (``tools.py`` of the official repo): cosine similarity of the query
against caption embeddings (textual channel) and against video embeddings via
the aligner's text tower (visual channel), combined as
``18 * visual + 11 * textual`` and ranked top-5. The weights were found by
grid search for ViCLIP + text-embedding-3-large; they are parameters here
because the embedders here are different; 18 and 11 are used (see Section
5.1 of the monograph).
"""

import numpy as np

from video_agent.memory.types import TemporalMemory


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query against every row of ``matrix``.

    Both sides are normalized here so callers never depend on whether an
    embedder returns unit vectors (X-CLIP and SigLIP do not).
    """
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
        raise ValueError(f"matrix shape {matrix.shape} incompatible with query of dim {query.shape[0]}")
    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    denominator = row_norms * query_norm
    scores = matrix @ query
    return np.divide(scores, denominator, out=np.zeros_like(scores), where=denominator > 0)


def ensemble_scores(
    visual: np.ndarray, textual: np.ndarray, *, visual_weight: float, textual_weight: float
) -> np.ndarray:
    if visual.shape != textual.shape:
        raise ValueError(f"channel shapes differ: {visual.shape} vs {textual.shape}")
    return visual_weight * visual + textual_weight * textual


def top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the ``k`` highest scores, best first; ties as ``np.argsort`` orders them."""
    k = min(int(scores.shape[0]), k)
    if k <= 0:
        return []
    return [int(i) for i in np.argsort(scores)[-k:][::-1]]


def localize_segments(
    memory: TemporalMemory,
    *,
    text_query: np.ndarray,
    visual_query: np.ndarray,
    visual_weight: float,
    textual_weight: float,
    k: int,
) -> list[tuple[int, float]]:
    """Rank segments for a query embedded in both channels; returns ``(index, score)``."""
    textual = cosine_scores(text_query, memory.caption_embeddings)
    visual = cosine_scores(visual_query, memory.visual_embeddings)
    scores = ensemble_scores(
        visual, textual, visual_weight=visual_weight, textual_weight=textual_weight
    )
    return [(memory.segments[i].index, float(scores[i])) for i in top_k_indices(scores, k)]
