"""Text embedding via sentence-transformers models."""

from typing import Sequence

import numpy as np


class SentenceTransformerEmbedder:
    """TextEmbedder backed by any sentence-transformers checkpoint.

    The same instance must embed captions at build time and queries at
    inference time so both land in one vector space.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        *,
        batch_size: int = 32,
        device: str | None = None,
        model_kwargs: dict | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.batch_size = batch_size
        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"dtype": "bfloat16", **(model_kwargs or {})},
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return embeddings of shape (len(texts), D)."""
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
