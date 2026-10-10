"""Crop embedding for re-identification: SigLIP2, DINOv2 and their concat.

The reference re-ID ensemble scores CLIP and DINOv2 similarities separately
(see ``memory.reid``). ``ConcatImageEmbedder`` lays member embeddings side by
side and exposes their widths so ``ScoreChannel`` slices can be wired to it.
"""

from typing import Protocol, Sequence

import numpy as np

from video_agent.media import Image


class _Embedder(Protocol):
    embedding_dim: int

    def embed(self, images: Sequence[Image]) -> np.ndarray: ...


SIGLIP2_TEXT_TOKENS = 64
"""Fixed sequence length of the SigLIP2 text tower.

The processor is asked to pad to it (``padding="max_length"``), which is what
the model expects; without ``truncation`` a text longer than that comes back
at its own length and the whole batch dies in ``convert_to_tensors``
("expected sequence of length 64 at dim 1 (got 65)"). The tool's real queries
are short nouns ("garlic clove"), so truncation only ever touches long
queries such as whole step descriptions.
"""


def siglip_text_kwargs(texts: Sequence[str]) -> dict:
    """The processor call for a batch of open-vocabulary text queries.

    A function rather than an inline literal because it is the only part of
    ``embed_text`` that can be checked without a GPU and a model download.
    """
    return {
        "text": list(texts),
        "padding": "max_length",
        "truncation": True,
        "max_length": SIGLIP2_TEXT_TOKENS,
        "return_tensors": "pt",
    }


def _pooled(features) -> np.ndarray:
    # transformers v5 returns a model-output object; v4 returned the tensor.
    tensor = getattr(features, "pooler_output", features)
    return tensor.cpu().numpy()


class SiglipImageEmbedder:
    """Text-aligned crop embedder (supports querying objects by name)."""

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        *,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.batch_size = batch_size
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim: int = self.model.config.vision_config.hidden_size

    def embed(self, images: Sequence[Image]) -> np.ndarray:
        import torch

        parts = []
        for start in range(0, len(images), self.batch_size):
            batch = list(images[start : start + self.batch_size])
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                features = self.model.get_image_features(**inputs)
            parts.append(_pooled(features))
        return np.concatenate(parts, axis=0)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """Embed object descriptions into the same space (open-vocab queries)."""
        import torch

        inputs = self.processor(**siglip_text_kwargs(texts)).to(self.device)
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return _pooled(features)


class Dinov2ImageEmbedder:
    """Instance-sensitive crop embedder (no text alignment)."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        *,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.batch_size = batch_size
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim: int = self.model.config.hidden_size

    def embed(self, images: Sequence[Image]) -> np.ndarray:
        import torch

        parts = []
        for start in range(0, len(images), self.batch_size):
            batch = list(images[start : start + self.batch_size])
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
            parts.append(outputs.pooler_output.cpu().numpy())
        return np.concatenate(parts, axis=0)


class ConcatImageEmbedder:
    """Concatenates member embeddings; ``widths`` maps members to columns.

    Widths are known at construction time so ``ScoreChannel`` slices for the
    re-ID merger can be wired before the pipeline runs.
    """

    def __init__(self, embedders: Sequence[_Embedder]) -> None:
        if not embedders:
            raise ValueError("at least one embedder is required")
        self.embedders = tuple(embedders)
        self.widths: tuple[int, ...] = tuple(e.embedding_dim for e in embedders)

    def embed(self, images: Sequence[Image]) -> np.ndarray:
        parts = [embedder.embed(images) for embedder in self.embedders]
        actual = tuple(part.shape[1] for part in parts)
        if actual != self.widths:
            raise ValueError(f"embedding widths {actual} != declared {self.widths}")
        return np.concatenate(parts, axis=1)
