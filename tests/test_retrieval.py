import numpy as np
import pytest

from video_agent.memory.retrieval import (
    cosine_scores,
    ensemble_scores,
    localize_segments,
    top_k_indices,
)

from tests.fakes import make_temporal_memory


def test_cosine_ignores_the_row_norms_a_dot_product_would_follow() -> None:
    memory = make_temporal_memory(3, dim=2)
    # X-CLIP embeddings are not unit vectors: a long row wins a raw dot
    # product against a query it barely points at.
    memory.visual_embeddings = np.array([[10.0, 1.0], [0.0, 0.5], [0.3, 0.4]])
    memory.caption_embeddings = np.zeros((3, 2))
    query = np.array([0.0, 1.0])
    assert int(np.argmax(memory.visual_embeddings @ query)) == 0

    ranked = localize_segments(
        memory,
        text_query=query,
        visual_query=query,
        visual_weight=1.0,
        textual_weight=0.0,
        k=3,
    )

    assert [index for index, _ in ranked] == [1, 2, 0]
    assert ranked[0][1] == pytest.approx(1.0)
    assert ranked[2][1] == pytest.approx(1 / np.sqrt(101))


def test_weights_decide_which_channel_wins() -> None:
    memory = make_temporal_memory(2, dim=2)
    memory.visual_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    memory.caption_embeddings = np.array([[0.0, 1.0], [1.0, 0.0]])
    query = np.array([1.0, 0.0])

    reference = localize_segments(
        memory,
        text_query=query,
        visual_query=query,
        visual_weight=18.0,
        textual_weight=11.0,
        k=2,
    )
    assert [index for index, _ in reference] == [0, 1]
    assert [score for _, score in reference] == pytest.approx([18.0, 11.0])

    textual_only = localize_segments(
        memory,
        text_query=query,
        visual_query=query,
        visual_weight=0.0,
        textual_weight=1.0,
        k=2,
    )
    assert [index for index, _ in textual_only] == [1, 0]


def test_ties_are_ranked_as_numpy_sorts_them() -> None:
    memory = make_temporal_memory(3, dim=2)
    memory.visual_embeddings = np.ones((3, 2))
    memory.caption_embeddings = np.ones((3, 2))
    query = np.ones(2)

    ranked = localize_segments(
        memory,
        text_query=query,
        visual_query=query,
        visual_weight=18.0,
        textual_weight=11.0,
        k=2,
    )

    tied = np.full(3, 29.0)
    assert [index for index, _ in ranked] == [int(i) for i in np.argsort(tied)[-2:][::-1]]
    assert [index for index, _ in ranked] == [2, 1]


def test_asking_for_more_segments_than_exist_returns_them_all() -> None:
    memory = make_temporal_memory(3)
    query = np.zeros(4)
    query[1] = 1.0

    ranked = localize_segments(
        memory,
        text_query=query,
        visual_query=query,
        visual_weight=18.0,
        textual_weight=11.0,
        k=10,
    )

    assert len(ranked) == 3
    assert ranked[0][0] == 1  # the segment the one-hot query names


def test_top_k_indices_handles_degenerate_k() -> None:
    scores = np.array([0.1, 0.9, 0.5])
    assert top_k_indices(scores, 0) == []
    assert top_k_indices(scores, -1) == []
    assert top_k_indices(scores, 99) == [1, 2, 0]


def test_scoring_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        cosine_scores(np.ones(3), np.ones((2, 4)))
    with pytest.raises(ValueError):
        ensemble_scores(np.ones(3), np.ones(2), visual_weight=1.0, textual_weight=1.0)
