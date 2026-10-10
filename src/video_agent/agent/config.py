"""Configuration of the inference layer: tool behavior and loop control.

Defaults reproduce the reference implementation, so an out-of-the-box run is
VideoAgent; every behavior that can be varied (enforcing the caption window,
recovering from parse errors) is a field here rather than a code change, and
reaches the run record through ``to_dict``.
"""

from dataclasses import asdict, dataclass
from typing import Literal

DEFAULT_ALIGNER_MODEL = "timm/PE-Core-L-14-336"
"""The project's video-text aligner, and the one place that decides it.

PE-Core L replaces the original ViCLIP for the reasons given in Section 5.1
of the monograph (it localizes recipe steps better than the other aligners
compared). The other aligners stay in the registry; X-CLIP uses the
unsuffixed artifact name (``adapters.aligner.DEFAULT_ALIGNER``).

Spelled here, next to the weights it is combined with, because a run's
aligner is part of the configuration a ``config_id`` hashes: the
preprocessing stage and the inference scripts all default to this constant,
so the embeddings that are written are the embeddings that are read.
"""


@dataclass(frozen=True)
class ToolConfig:
    """Parameters of the tools the agent calls.

    The ensemble weights and the top-k come from the reference's
    ``segment_localization``, the threshold and prompt template from its
    object database, and the 6-second visual-question-answering window from
    one 2-second segment on each side of the target.
    """

    visual_weight: float = 18.0
    textual_weight: float = 11.0
    # Which checkpoint the visual channel is: it decides both the tower that
    # embeds a step description and the embeddings file the memory is loaded
    # from, so two runs that differ here are not comparable.
    aligner_model: str = DEFAULT_ALIGNER_MODEL
    # Which object memory the sub-agent queries: None is the unsuffixed
    # files of the default detector (RT-DETRv2), a slug names another detector's
    # memory (``memory.store.object_memory_file``). The detector decides
    # what names exist to be asked about, so two runs that differ here are
    # not comparable either.
    object_memory_slug: str | None = None
    localization_top_k: int = 5
    caption_window_max: int = 15
    # The reference advertised a 15-caption cap in the tool description but
    # never enforced it; enabling this clamps the window instead.
    enforce_caption_window: bool = False
    vqa_context_segments: int = 1  # segments on each side of the target
    vqa_sample_fps: float = 5.0
    vqa_describe_question: str = "what is the video about?"
    # The reference asked its two backends differently: Video-LLaVA got the
    # description and the question as two calls over the same clip, gpt-4v a
    # single prompt whose reply was returned as it came.
    vqa_style: Literal["two_questions", "single"] = "two_questions"
    object_query_template: str = "a photo of a {}."
    # The reference's threshold (0.26, calibrated for CLIP) returns no object
    # on the SigLIP 2 similarity scale, whose (query, object) cosines are much
    # lower, so 0.10 is used.
    object_similarity_threshold: float = 0.10
    sql_read_only: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReActConfig:
    """Control of the ReAct loop and of how much of it survives an error.

    ``max_iterations`` and ``stop`` are LangChain's defaults as the reference
    used them. The three tolerance flags are additions of this
    reimplementation: the reference crashed on a parse error, on an
    unparseable tool argument and on a tool exception, and the caller
    swallowed the crash, which loses the whole episode.

    The procedural task runs with two different values, set from the command
    line: ``--max-iterations 100``, because a recipe has up to 25 steps and
    the agent needs several tool calls per step before it can answer for all
    of them, and ``--max-tokens 16384`` per orchestrator turn (the scripts'
    default, ``scripts/_wiring.py``), which leaves room for a final answer
    that covers every step while capping a turn that never ends.
    """

    max_iterations: int = 15
    stop: tuple[str, ...] = ("\nObservation",)
    parse_errors_as_observations: bool = True
    tool_errors_as_observations: bool = True
    lenient_tuple_parsing: bool = False
    observation_padding: bool = True
    max_tokens: int | None = None

    def to_dict(self) -> dict:
        # ``stop`` becomes a list so the dict survives a JSON round trip
        # unchanged (``json`` writes tuples as arrays and reads them back).
        data = asdict(self)
        data["stop"] = list(self.stop)
        return data
