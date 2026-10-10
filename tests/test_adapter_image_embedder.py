from typing import Sequence

import numpy as np
import pytest

from video_agent.adapters.image_embedder import (
    SIGLIP2_TEXT_TOKENS,
    ConcatImageEmbedder,
    siglip_text_kwargs,
)
from video_agent.media import Image


class FakeEmbedder:
    def __init__(self, dim: int, value: float) -> None:
        self.embedding_dim = dim
        self.value = value

    def embed(self, images: Sequence[Image]) -> np.ndarray:
        return np.full((len(images), self.embedding_dim), self.value)


def test_concat_lays_members_side_by_side() -> None:
    concat = ConcatImageEmbedder([FakeEmbedder(2, 1.0), FakeEmbedder(3, 2.0)])
    assert concat.widths == (2, 3)
    embeddings = concat.embed([np.zeros((4, 4, 3))] * 2)
    assert embeddings.shape == (2, 5)
    assert embeddings[0].tolist() == [1.0, 1.0, 2.0, 2.0, 2.0]


class LyingEmbedder:
    embedding_dim = 7  # declared width disagrees with actual output

    def embed(self, images: Sequence[Image]) -> np.ndarray:
        return np.zeros((len(images), 2))


def test_concat_rejects_width_drift() -> None:
    concat = ConcatImageEmbedder([LyingEmbedder()])
    with pytest.raises(ValueError):
        concat.embed([np.zeros((4, 4, 3))])


def test_concat_requires_members() -> None:
    with pytest.raises(ValueError):
        ConcatImageEmbedder([])


def test_text_queries_are_padded_and_truncated_to_the_text_tower_length() -> None:
    """A query longer than the tower's 64 tokens must not kill the batch.

    Without truncation the processor returns that row at its own length and
    ``convert_to_tensors`` raises "expected sequence of length 64 at dim 1",
    which a long step description used as a query would trigger.
    """
    kwargs = siglip_text_kwargs(["garlic clove", "a step description " * 20])

    assert SIGLIP2_TEXT_TOKENS == 64
    assert kwargs["padding"] == "max_length"
    assert kwargs["truncation"] is True
    assert kwargs["max_length"] == SIGLIP2_TEXT_TOKENS


def test_the_texts_are_passed_through_as_a_list_in_order() -> None:
    kwargs = siglip_text_kwargs(("bowl", "whisk"))

    assert kwargs["text"] == ["bowl", "whisk"]
    assert kwargs["return_tensors"] == "pt"
