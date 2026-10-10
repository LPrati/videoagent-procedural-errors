"""Per-frame object detection: one adapter per detector, behind one registry.

The detector is RT-DETRv2 over the 80 COCO classes, the successor of the
RT-DETR used by the original VideoAgent. ``build_detector`` maps a model id
to an adapter, and ``detector_slug``/``object_memory_slug`` map it to the
short name a non-default detector's artifacts would be suffixed with.

| model id | adapter | slug | classes | licence |
|---|---|---|---|---|
| ``PekingU/rtdetr_v2_r50vd`` | ``RTDetrV2Detector`` | ``rtdetr-v2-r50`` | fixed COCO-80 | Apache-2.0 |

Neither ``torch`` nor ``transformers`` is imported at module level: the
registry is read by argument parsing, by the store and by tests that run on
a machine with no GPU, and none of them should pay for a model library to
learn what a file is called.
"""

from dataclasses import dataclass
from typing import Sequence

from video_agent.media import Frame
from video_agent.models.types import Detection

DEFAULT_DETECTOR = "PekingU/rtdetr_v2_r50vd"
"""The default detector, whose artifacts carry no slug suffix.

``memory.store.DEFAULT_DETECTOR_SLUG`` is its slug, spelled there so the
store never imports an adapter; a test ties the two together.
"""

@dataclass(frozen=True)
class DetectorSpec:
    """One row of the registry: how a model id is loaded, named and defaulted.

    ``score_threshold`` is the adapter's default detection cut.
    """

    slug: str
    kind: str  # "rtdetr"
    checkpoint: str
    licence: str
    score_threshold: float
    note: str = ""


DETECTORS: dict[str, DetectorSpec] = {
    DEFAULT_DETECTOR: DetectorSpec(
        slug="rtdetr-v2-r50",
        kind="rtdetr",
        checkpoint=DEFAULT_DETECTOR,
        licence="Apache-2.0",
        score_threshold=0.25,
        note="default detector; 80 COCO classes",
    ),
}
"""Model id -> how to load it. See the table in the module docstring."""


def known_detectors() -> list[str]:
    """Every model id ``build_detector`` accepts, sorted."""
    return sorted(DETECTORS)


def detector_spec(model_name: str) -> DetectorSpec:
    """The registry row for ``model_name``; unknown ids name the known ones."""
    spec = DETECTORS.get(model_name)
    if spec is None:
        raise ValueError(
            f"unknown detector {model_name!r}; known ids are "
            + ", ".join(known_detectors())
        )
    return spec


def detector_slug(model_name: str) -> str:
    """Filesystem-safe short name of a detector."""
    return detector_spec(model_name).slug


def object_memory_slug(model_name: str) -> str:
    """What the object-memory artifacts of this detector are suffixed with."""
    return detector_spec(model_name).slug


def build_detector(
    model_name: str,
    *,
    device: str,
    batch_size: int,
    score_threshold: float | None = None,
):
    """The ``ObjectDetector`` for a model id."""
    spec = detector_spec(model_name)
    threshold = spec.score_threshold if score_threshold is None else score_threshold
    if spec.kind == "rtdetr":
        return RTDetrV2Detector(
            spec.checkpoint,
            device=device,
            batch_size=batch_size,
            score_threshold=threshold,
        )
    raise ValueError(f"registry entry {model_name!r} names no adapter: {spec.kind!r}")


class RTDetrV2Detector:
    """ObjectDetector backed by a transformers RT-DETRv2 checkpoint.

    Runs in bfloat16 with preprocessing on the GPU (fast image processor);
    both are throughput decisions, roughly 3x faster than the naive fp32/CPU
    path on a recent GPU.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DETECTOR,
        *,
        score_threshold: float = 0.25,
        batch_size: int = 32,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

        self.model_name = model_name
        self.score_threshold = score_threshold
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(
            model_name, dtype=dtype
        ).to(self.device)
        self.model.eval()

    def detect(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        """Return detections per frame, aligned with ``frames``."""
        import torch

        results: list[list[Detection]] = []
        for start in range(0, len(frames), self.batch_size):
            batch = frames[start : start + self.batch_size]
            inputs = self.processor(
                images=[frame.data for frame in batch],
                return_tensors="pt",
                device=self.device,  # fast processor: resize/normalize on GPU
            )
            pixel_values = inputs["pixel_values"].to(self.model.dtype)
            with torch.inference_mode():
                outputs = self.model(pixel_values=pixel_values)
            processed = self.processor.post_process_object_detection(
                outputs,
                target_sizes=[frame.data.shape[:2] for frame in batch],
                threshold=self.score_threshold,
            )
            for frame, frame_output in zip(batch, processed):
                detections = [
                    Detection(
                        frame_index=frame.index,
                        box_xyxy=tuple(float(v) for v in box.tolist()),
                        category=self.model.config.id2label[int(label)],
                        score=float(score),
                    )
                    for box, label, score in zip(
                        frame_output["boxes"],
                        frame_output["labels"],
                        frame_output["scores"],
                    )
                ]
                results.append(detections)
        return results

    def lineage(self) -> dict:
        """What the detect stage records about this run (see ``preprocess_objects``).

        The checkpoint, its Hugging Face snapshot and the detection settings,
        so two ``meta.json`` files say whether they came from the same detector.
        """
        return {
            "checkpoint": self.model_name,
            "hf_snapshot": getattr(self.model.config, "_commit_hash", None),
            "score_threshold": self.score_threshold,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
        }
