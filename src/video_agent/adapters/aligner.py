"""Video-text alignment: one adapter per visual aligner, behind one registry.

``segment_localization`` scores a segment as ``18*cos(video, text) +
11*cos(caption, text)``; the video side is a ``VideoTextAligner``. The agent
uses PE-Core L by default (``agent.config.DEFAULT_ALIGNER_MODEL``); X-CLIP,
ViCLIP and SigLIP 2 are kept as alternatives that can be compared under the
same weights. Every adapter below answers the same two calls over the same
``Clip`` objects, so any script that encodes clips picks one by model id and
changes nothing else.

``build_aligner`` maps a model id to an adapter and ``aligner_slug`` maps it
to the short name the artifacts are suffixed with, so several aligners'
embeddings coexist in one artifact root.

| model id | adapter | slug | frames | dim | licence | loads with |
|---|---|---|---|---|---|---|
| ``microsoft/xclip-base-patch16`` | ``XCLIPAligner`` | ``xclip-base-patch16`` | 8 | 512 | MIT | ``transformers.XCLIPModel`` |
| ``timm/PE-Core-L-14-336`` | ``PECoreAligner`` | ``pe-core-l`` | 8 | 1024 | Apache-2.0 | ``open_clip`` (``hf-hub:``) |
| ``timm/PE-Core-bigG-14-448`` | ``PECoreAligner`` | ``pe-core-g`` | 8 | 1280 | Apache-2.0 | ``open_clip`` (``hf-hub:``) |
| ``timm/PE-Core-B-16`` | ``PECoreAligner`` | ``pe-core-b`` | 8 | 1024 | Apache-2.0 | ``open_clip`` (``hf-hub:``) |
| ``OpenGVLab/ViCLIP-L-14-hf`` | ``ViCLIPAligner`` | ``viclip-l-14`` | 8 | 768 | none declared on the Hub (InternVideo is Apache-2.0) | ``transformers`` + ``trust_remote_code`` |
| ``google/siglip2-so400m-patch16-384`` | ``Siglip2FrameMeanAligner`` | ``siglip2-so400m-mean`` | 8 | 1152 | Apache-2.0 | ``transformers.AutoModel`` |
| ``google/siglip2-base-patch16-224`` | ``Siglip2FrameMeanAligner`` | ``siglip2-base-mean`` | 8 | 768 | Apache-2.0 | ``transformers.AutoModel`` |
| ``facebookresearch/LaViLa-TSF-L`` | not implemented | ``lavila-tsf-l`` | 4 | 256 | MIT (code) | standalone repo |
| ``OpenGVLab/InternVideo2-Stage2_1B-224p-f4`` | not implemented | ``internvideo2-1b`` | 4 | 512 | Apache-2.0, gated | standalone repo |

Meta publishes PE-Core three ways and only one of them carries both towers:
``facebook/PE-Core-*`` is a bare ``.pt`` for Meta's own ``perception_models``
package, ``timm/vit_pe_core_*`` is the image tower alone, and
``timm/PE-Core-*`` is the ``open_clip`` remap with image *and* text. The
``facebook/`` ids are accepted here as aliases of the ``timm/`` ones because
they are Meta's canonical names; they load the ``open_clip`` repository
either way. ``transformers`` has no Perception Encoder dual
encoder (only ``PerceptionLM``, a different model), which is why this is the
one adapter that does not go through ``AutoModel``.

Normalization: every adapter but ``XCLIPAligner`` returns unit rows, so a dot
product is a cosine. X-CLIP is left un-normalized, as the model produces it;
``memory.retrieval.cosine_scores`` and ``memory.store.l2_normalize``
normalize on the way in, so readers do not depend on the stored scale.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from video_agent.adapters.frames import evenly_spaced
from video_agent.media import Clip, Image

DEFAULT_ALIGNER = "microsoft/xclip-base-patch16"
"""The aligner whose artifacts carry no slug suffix.

Naming rule: X-CLIP embeddings are stored under the unsuffixed file names
(``memory.store.DEFAULT_ALIGNER_SLUG`` is its slug), and every other aligner
appends its slug. This is not the aligner the agent runs by default — that
is ``agent.config.DEFAULT_ALIGNER_MODEL``, PE-Core L.
"""

FRAMES_PER_CLIP = 8
"""Frames every implemented aligner sees, so a comparison holds the input fixed.

X-CLIP's checkpoint fixes 8 by architecture; the image-tower aligners
(PE-Core, SigLIP2) would take any number and are held to the same 8; ViCLIP
was pre-trained with 8 and its temporal embedding has that many rows.
"""


def _projected(features) -> np.ndarray:
    # transformers v5 returns a model-output object; v4 returned the tensor.
    tensor = getattr(features, "pooler_output", features)
    return tensor.cpu().numpy()


def pad_to_length(frames: list[Image], count: int) -> list[Image]:
    """Repeat the last frame until ``frames`` has ``count`` entries."""
    if not frames:
        raise ValueError("cannot pad an empty clip")
    return frames + [frames[-1]] * (count - len(frames))


def clip_frames(clip: Clip, count: int) -> list[Image]:
    """The ``count`` frames an aligner sees: evenly spaced, last-frame padded.

    One rule for every adapter, so two aligners scored against each other
    differ by the model and not by which frames reached it.
    """
    frames = [frame.data for frame in clip.frames]
    return pad_to_length(evenly_spaced(frames, count), count)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length; a zero row stays zero rather than becoming NaN.

    The write-side twin of ``memory.store.l2_normalize`` (which normalizes on
    load). Adapters normalize here so an embedding file is a cosine table on
    its own, whatever reads it.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected (n, D) embeddings, got {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def frame_mean(embeddings: np.ndarray, *, frames_per_clip: int) -> np.ndarray:
    """Pool per-frame embeddings into one unit vector per clip.

    The video recipe of the image-tower aligners: PE-Core and SigLIP2 have
    no temporal attention, so a clip is the *frame average* of its own frames
    (the recipe the Perception Encoder paper reports its video numbers with),
    L2-normalized afterwards. Rows arrive clip-major — the ``frames_per_clip``
    frames of clip 0, then those of clip 1 — and come back as ``(n, D)``.

    The average is taken over the raw frame embeddings, not over normalized
    ones: a frame the tower is confident about therefore weighs more. The
    alternative (normalize, average, normalize) is what CLIP4Clip's mean
    pooling does.
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected (n*frames, D) embeddings, got {matrix.shape}")
    if frames_per_clip <= 0:
        raise ValueError("frames_per_clip must be positive")
    if matrix.shape[0] % frames_per_clip:
        raise ValueError(
            f"{matrix.shape[0]} frame embeddings do not divide into clips of "
            f"{frames_per_clip}"
        )
    pooled = matrix.reshape(-1, frames_per_clip, matrix.shape[1]).mean(axis=1)
    return l2_normalize(pooled)


class XCLIPAligner:
    """VideoTextAligner backed by an X-CLIP checkpoint.

    X-CLIP consumes a fixed number of frames per clip (8 for the released
    checkpoints); clips are evenly subsampled or last-frame padded to fit.
    Outputs are returned as the model produces them, un-normalized; the
    readers normalize.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ALIGNER,
        *,
        batch_size: int = 16,
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import AutoProcessor, XCLIPModel

        self.batch_size = batch_size
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = XCLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.num_frames = self.model.config.vision_config.num_frames
        self.embedding_dim: int = self.model.config.projection_dim

    def _clip_frames(self, clip: Clip) -> list[Image]:
        return clip_frames(clip, self.num_frames)

    def encode_video(self, clips: Sequence[Clip]) -> np.ndarray:
        """Return clip embeddings of shape (len(clips), D)."""
        import torch

        parts = []
        for start in range(0, len(clips), self.batch_size):
            batch = clips[start : start + self.batch_size]
            inputs = self.processor(
                videos=[self._clip_frames(clip) for clip in batch],
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                features = self.model.get_video_features(**inputs)
            parts.append(_projected(features))
        return np.concatenate(parts, axis=0)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return text embeddings of shape (len(texts), D), same space as videos."""
        import torch

        parts = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            inputs = self.processor(
                text=batch, padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                features = self.model.get_text_features(**inputs)
            parts.append(_projected(features))
        return np.concatenate(parts, axis=0)


def _open_clip_reference(model_name: str) -> str:
    """``open_clip``'s name for a Hub repository: the ``hf-hub:`` schema."""
    return model_name if ":" in model_name else f"hf-hub:{model_name}"


def _open_clip_embed_dim(model, tokenizer, device) -> int:
    """Measure the joint embedding width by encoding one token.

    Not every ``open_clip`` model class exposes ``embed_dim`` (PE-Core loads
    as ``CustomTextCLIP``, which does not), so the width is read off a real
    text embedding instead of an attribute that may be missing.
    """
    import torch

    with torch.inference_mode():
        features = model.encode_text(tokenizer(["a"]).to(device))
    return int(features.shape[-1])


class PECoreAligner:
    """VideoTextAligner over Meta's Perception Encoder, frame-averaged.

    PE-Core is an image-text dual encoder with no temporal module, so a clip
    is encoded frame by frame and pooled by ``frame_mean``. Its weights ship
    for ``open_clip`` rather than for ``transformers`` — see the module
    docstring for which of the three PE repositories carries the text tower.

    The text context is short (32 tokens for L and B, 72 for G); open_clip's
    tokenizer truncates, so a step description longer than that loses its
    tail. That is a property of the checkpoint, recorded here rather than
    worked around.
    """

    def __init__(
        self,
        model_name: str = "timm/PE-Core-L-14-336",
        *,
        batch_size: int = 16,
        device: str = "cuda",
        num_frames: int = FRAMES_PER_CLIP,
    ) -> None:
        import open_clip
        import torch

        self.batch_size = batch_size
        self.num_frames = num_frames
        self.device = torch.device(device)
        reference = _open_clip_reference(model_name)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            reference, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(reference)
        self.embedding_dim: int = _open_clip_embed_dim(
            self.model, self.tokenizer, self.device
        )

    def _pixels(self, clips: Sequence[Clip]):
        import torch
        from PIL import Image as PILImage

        return torch.stack(
            [
                self.preprocess(PILImage.fromarray(frame))
                for clip in clips
                for frame in clip_frames(clip, self.num_frames)
            ]
        ).to(self.device)

    def encode_video(self, clips: Sequence[Clip]) -> np.ndarray:
        """Return unit clip embeddings of shape (len(clips), D)."""
        import torch

        parts = []
        for start in range(0, len(clips), self.batch_size):
            batch = clips[start : start + self.batch_size]
            with torch.inference_mode():
                features = self.model.encode_image(self._pixels(batch))
            parts.append(features.float().cpu().numpy())
        return frame_mean(
            np.concatenate(parts, axis=0), frames_per_clip=self.num_frames
        )

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return unit text embeddings of shape (len(texts), D)."""
        import torch

        parts = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            tokens = self.tokenizer(batch).to(self.device)
            with torch.inference_mode():
                features = self.model.encode_text(tokens)
            parts.append(features.float().cpu().numpy())
        return l2_normalize(np.concatenate(parts, axis=0))


VICLIP_RESOLUTION = 224
VICLIP_MEAN = (0.485, 0.456, 0.406)
VICLIP_STD = (0.229, 0.224, 0.225)
VICLIP_TOKENIZER_FILE = "bpe_simple_vocab_16e6.txt.gz"
VICLIP_WEIGHTS_FILE = "model.safetensors"
"""ViCLIP's preprocessing, from the ``demo.ipynb`` of its model card.

Square resize to 224 (no crop, no aspect preserved) and the ImageNet
statistics — not CLIP's own — applied to ``pixel/255``. The tokenizer is a
gzipped BPE table shipped inside the repository; ``config.json`` points at it
with a path relative to the working directory, which is why the adapter
rewrites it to the downloaded file.
"""


def viclip_video_array(
    clips: Sequence[Sequence[Image]], *, resolution: int = VICLIP_RESOLUTION
) -> np.ndarray:
    """Frames as ViCLIP wants them: ``(B, T, C, H, W)`` float32, normalized.

    Kept out of the adapter's forward pass because it is the part of the
    ViCLIP path that can be checked without the checkpoint: every clip must
    carry the same number of frames, RGB stays RGB (the notebook's ``[::-1]``
    undoes OpenCV's BGR, which ``media`` has already done), and the values
    leave as ``(x/255 - mean) / std``.
    """
    import cv2

    if not clips:
        raise ValueError("cannot encode an empty batch of clips")
    lengths = {len(frames) for frames in clips}
    if len(lengths) != 1:
        raise ValueError(f"clips carry different frame counts: {sorted(lengths)}")
    mean = np.asarray(VICLIP_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(VICLIP_STD, dtype=np.float32).reshape(1, 1, 3)
    batch = np.stack(
        [
            np.stack(
                [
                    (
                        cv2.resize(
                            np.asarray(frame), (resolution, resolution)
                        ).astype(np.float32)
                        / 255.0
                        - mean
                    )
                    / std
                    for frame in frames
                ]
            )
            for frames in clips
        ]
    )
    return np.ascontiguousarray(batch.transpose(0, 1, 4, 2, 3))


class ViCLIPAligner:
    """VideoTextAligner over ViCLIP-L-14, the aligner of the original VideoAgent.

    The checkpoint carries its own modelling code (``trust_remote_code``) and
    its own BPE tokenizer; ``config.json`` names the tokenizer with a path
    relative to the process's working directory, so the adapter resolves it
    to the file the Hub actually downloaded.

    Text goes through ``encode_text`` rather than the checkpoint's own
    ``get_text_features``, which takes one string at a time and memoizes into
    a mutable default argument shared by every caller; normalization is done
    here instead, on the batch.
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/ViCLIP-L-14-hf",
        *,
        batch_size: int = 16,
        device: str = "cuda",
        num_frames: int = FRAMES_PER_CLIP,
    ) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self.batch_size = batch_size
        self.num_frames = num_frames
        self.device = torch.device(device)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.tokenizer_path = hf_hub_download(model_name, VICLIP_TOKENIZER_FILE)
        # The checkpoint's modelling code predates transformers 5: it never
        # calls ``post_init``, so ``from_pretrained`` dies looking for
        # ``all_tied_weights_keys`` while finalising the load. Build the
        # module ourselves and load the safetensors file into it directly;
        # the file was written by ``save_pretrained`` of this same class, so
        # a strict load is the right check.
        try:
            model_class = get_class_from_dynamic_module(
                config.auto_map["AutoModel"], model_name
            )
        except ImportError as error:  # the remote code's own dependencies
            raise ImportError(
                f"{model_name}'s remote code could not be imported ({error}). "
                "It needs einops, timm and fvcore, and its text tower still "
                "does `from pkg_resources import packaging`, which setuptools "
                "81+ no longer ships."
            ) from error
        self.model = model_class(config)
        weights = load_file(hf_hub_download(model_name, VICLIP_WEIGHTS_FILE))
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device)
        self.model.eval()
        # The remote code may or may not expose the width; measure it when not.
        width = getattr(self.model, "embed_dim", None)
        if width is None:
            with torch.inference_mode():
                width = self.model.encode_text(["a"]).shape[-1]
        self.embedding_dim: int = int(width)

    def encode_video(self, clips: Sequence[Clip]) -> np.ndarray:
        """Return unit clip embeddings of shape (len(clips), D)."""
        import torch

        parts = []
        for start in range(0, len(clips), self.batch_size):
            batch = clips[start : start + self.batch_size]
            pixels = viclip_video_array(
                [clip_frames(clip, self.num_frames) for clip in batch]
            )
            with torch.inference_mode():
                features = self.model.get_vid_features(
                    torch.from_numpy(pixels).to(self.device)
                )
            parts.append(features.float().cpu().numpy())
        return l2_normalize(np.concatenate(parts, axis=0))

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return unit text embeddings of shape (len(texts), D)."""
        import torch

        parts = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            with torch.inference_mode():
                features = self.model.encode_text(batch)
            parts.append(features.float().cpu().numpy())
        return l2_normalize(np.concatenate(parts, axis=0))


class Siglip2FrameMeanAligner:
    """VideoTextAligner over SigLIP2 image features, averaged across frames.

    A strong image encoder with no notion of time at all, so whatever it
    scores is what the visual channel gets from appearance alone.
    Built on ``SiglipImageEmbedder`` — the same tower the object memory
    queries — so the text side is the one already used by the crop channel,
    truncated at ``SIGLIP2_TEXT_TOKENS``.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch16-384",
        *,
        batch_size: int = 16,
        device: str = "cuda",
        num_frames: int = FRAMES_PER_CLIP,
    ) -> None:
        from video_agent.adapters.image_embedder import SiglipImageEmbedder

        self.batch_size = batch_size
        self.num_frames = num_frames
        # One image batch is one clip batch: the tower sees frames, and
        # ``batch_size`` counts clips everywhere else in this module.
        self.frames = SiglipImageEmbedder(
            model_name, batch_size=batch_size * num_frames, device=device
        )
        self.embedding_dim: int = self.frames.embedding_dim

    def encode_video(self, clips: Sequence[Clip]) -> np.ndarray:
        """Return unit clip embeddings of shape (len(clips), D)."""
        frames = [
            frame
            for clip in clips
            for frame in clip_frames(clip, self.num_frames)
        ]
        return frame_mean(
            self.frames.embed(frames), frames_per_clip=self.num_frames
        )

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return unit text embeddings of shape (len(texts), D)."""
        return l2_normalize(self.frames.embed_text(list(texts)))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


LAVILA_NOTE = (
    "LaViLa (TimeSformer-L dual encoder, Ego4D) has no Hugging Face package: "
    "clone github.com/facebookresearch/LaViLa (MIT), fetch "
    "clip_openai_timesformer_large.narrator_rephraser.ep_0003.pth from "
    "dl.fbaipublicfiles.com, and build it with "
    "lavila.models.models.CLIP_OPENAI_TIMESFORMER_LARGE(num_frames=...). Its "
    "requirements.txt pins torch 1.10.1, timm 0.5.4 and transformers 4.27, "
    "so the modelling code has to be vendored and ported before it runs on "
    "a current torch 2.x; it also wants decord and einops. Zero-shot eval "
    "uses 4 frames per clip (16 improves it), 224px, and a 256-d joint space."
)

INTERNVIDEO2_NOTE = (
    "InternVideo2-Stage2 1B is a bare .pt checkpoint behind a gated Hugging "
    "Face repository (Apache-2.0, contact details required), not a "
    "trust_remote_code model: it is built from "
    "github.com/OpenGVLab/InternVideo (InternVideo2/multi_modality) with "
    "demo/internvideo2_stage2_config.py, and the text side is a separate "
    "BERT-large checkpoint. The repo's own install guide asks for ninja, "
    "flash-attn built from source (FusedMLP and DropoutLayerNorm CUDA "
    "extensions) and deepspeed, and warns the wheel build can take hours. "
    "The published checkpoint is f4: 4 frames per clip at 224p."
)


@dataclass(frozen=True)
class AlignerSpec:
    """One row of the registry table: how a model id is loaded and named.

    ``kind`` is the adapter to build; the unimplemented entries have none and
    carry ``note`` instead, which is what ``build_aligner`` raises. ``frames``
    and ``embedding_dim`` are documentation — the adapters read the truth
    from the checkpoint — and let a cost estimate be made before anything is
    downloaded.
    """

    slug: str
    kind: str | None
    checkpoint: str
    frames: int
    embedding_dim: int
    licence: str
    note: str = ""


ALIGNERS: dict[str, AlignerSpec] = {
    DEFAULT_ALIGNER: AlignerSpec(
        slug="xclip-base-patch16",
        kind="xclip",
        checkpoint=DEFAULT_ALIGNER,
        frames=8,
        embedding_dim=512,
        licence="MIT",
    ),
    "timm/PE-Core-L-14-336": AlignerSpec(
        slug="pe-core-l",
        kind="pe-core",
        checkpoint="timm/PE-Core-L-14-336",
        frames=FRAMES_PER_CLIP,
        embedding_dim=1024,
        licence="Apache-2.0",
    ),
    "timm/PE-Core-bigG-14-448": AlignerSpec(
        slug="pe-core-g",
        kind="pe-core",
        checkpoint="timm/PE-Core-bigG-14-448",
        frames=FRAMES_PER_CLIP,
        embedding_dim=1280,
        licence="Apache-2.0",
    ),
    "timm/PE-Core-B-16": AlignerSpec(
        slug="pe-core-b",
        kind="pe-core",
        checkpoint="timm/PE-Core-B-16",
        frames=FRAMES_PER_CLIP,
        embedding_dim=1024,
        licence="Apache-2.0",
    ),
    "OpenGVLab/ViCLIP-L-14-hf": AlignerSpec(
        slug="viclip-l-14",
        kind="viclip",
        checkpoint="OpenGVLab/ViCLIP-L-14-hf",
        frames=FRAMES_PER_CLIP,
        embedding_dim=768,
        licence="none declared on the Hub (InternVideo is Apache-2.0)",
    ),
    "google/siglip2-so400m-patch16-384": AlignerSpec(
        slug="siglip2-so400m-mean",
        kind="siglip2-mean",
        checkpoint="google/siglip2-so400m-patch16-384",
        frames=FRAMES_PER_CLIP,
        embedding_dim=1152,
        licence="Apache-2.0",
    ),
    "google/siglip2-base-patch16-224": AlignerSpec(
        slug="siglip2-base-mean",
        kind="siglip2-mean",
        checkpoint="google/siglip2-base-patch16-224",
        frames=FRAMES_PER_CLIP,
        embedding_dim=768,
        licence="Apache-2.0",
    ),
    "facebookresearch/LaViLa-TSF-L": AlignerSpec(
        slug="lavila-tsf-l",
        kind=None,
        checkpoint="clip_openai_timesformer_large.narrator_rephraser.ep_0003",
        frames=4,
        embedding_dim=256,
        licence="MIT (code); Ego4D terms apply to the data it was trained on",
        note=LAVILA_NOTE,
    ),
    "OpenGVLab/InternVideo2-Stage2_1B-224p-f4": AlignerSpec(
        slug="internvideo2-1b",
        kind=None,
        checkpoint="OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
        frames=4,
        embedding_dim=512,
        licence="Apache-2.0 (gated repository)",
        note=INTERNVIDEO2_NOTE,
    ),
}
"""Model id -> how to load it. See the table in the module docstring."""

ALIASES: dict[str, str] = {
    "facebook/PE-Core-L14-336": "timm/PE-Core-L-14-336",
    "facebook/PE-Core-G14-448": "timm/PE-Core-bigG-14-448",
    "facebook/PE-Core-B16-224": "timm/PE-Core-B-16",
}
"""Meta's own PE-Core ids, which hold no text tower, mapped to the open_clip ones.

A command that spells the ``facebook/`` repositories keeps working and lands
on the same weights and the same slug.
"""


def known_aligners() -> list[str]:
    """Every model id ``build_aligner`` accepts, aliases included, sorted."""
    return sorted([*ALIGNERS, *ALIASES])


def aligner_spec(model_name: str) -> AlignerSpec:
    """The registry row for ``model_name``; unknown ids name the known ones."""
    resolved = ALIASES.get(model_name, model_name)
    spec = ALIGNERS.get(resolved)
    if spec is None:
        raise ValueError(
            f"unknown aligner {model_name!r}; known ids are "
            + ", ".join(known_aligners())
        )
    return spec


def aligner_slug(model_name: str) -> str:
    """Filesystem-safe short name of an aligner, used as artifact suffix."""
    return aligner_spec(model_name).slug


def build_aligner(model_name: str, *, device: str, batch_size: int = 16):
    """The ``VideoTextAligner`` for a model id.

    Some entries are in the registry but not implemented; they raise
    ``NotImplementedError`` carrying what it would take, so a command that
    names one gets the answer instead of a traceback.
    """
    spec = aligner_spec(model_name)
    if spec.kind is None:
        raise NotImplementedError(f"{model_name} is not implemented: {spec.note}")
    if spec.kind == "xclip":
        return XCLIPAligner(spec.checkpoint, device=device, batch_size=batch_size)
    if spec.kind == "pe-core":
        return PECoreAligner(spec.checkpoint, device=device, batch_size=batch_size)
    if spec.kind == "viclip":
        return ViCLIPAligner(spec.checkpoint, device=device, batch_size=batch_size)
    if spec.kind == "siglip2-mean":
        return Siglip2FrameMeanAligner(
            spec.checkpoint, device=device, batch_size=batch_size
        )
    raise ValueError(f"registry entry {model_name!r} names no adapter: {spec.kind!r}")
