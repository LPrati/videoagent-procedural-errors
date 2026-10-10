"""The tools the agent calls, and the observations they return.

One toolkit per video, holding the two memories and the models the tools
need. Every returned string reproduces the reference's wording, including
the segment-count preamble the agent relies on to stay inside the grid: the
strings are part of the prompt the model sees, so changing them is changing
the method.

Which tools exist is configurable: no ``vqa`` drops visual question
answering, no object memory (or no sub-agent to query it) drops
``object_memory_querying``. Whether a tool *answers* is configurable too: with
``memory_available`` false every tool returns ``MEMORY_UNAVAILABLE`` without
touching a model, a file or the video, which is the empty-memory control
(see ``memory.ablation``).
"""

from typing import Sequence

from video_agent.agent.config import ToolConfig
from video_agent.agent.runner import (
    AGENT_STOPPED_MESSAGE,
    AgentRunner,
    ParamSpec,
    ToolResult,
    ToolSpec,
)
from video_agent.agent.vqa import answer_about_segment
from video_agent.media import Video
from video_agent.memory.ablation import SegmentPermutation
from video_agent.memory.object_db import ObjectDatabase, retrieve_candidate_objects
from video_agent.memory.retrieval import localize_segments
from video_agent.memory.types import ObjectMemory, Segment, TemporalMemory
from video_agent.models.interfaces import (
    TextAlignedImageEmbedder,
    TextEmbedder,
    VideoQA,
    VideoTextAligner,
)

MAIN_TOOL_NAMES: tuple[str, ...] = (
    "caption_retrieval",
    "segment_localization",
    "visual_question_answering",
    "object_memory_querying",
)
"""The orchestrator's tool names, in the order ``main_tools`` lists them.

The full toolkit, not the subset a configuration left: a report that reads a batch of
transcripts needs the fixed set of rows to print, including the tools a run
never called. ``test_tools`` keeps it equal to what ``main_tools`` builds.
"""

MEMORY_UNAVAILABLE = "MEMORY UNAVAILABLE: no memory exists for this recording"
"""The only observation the tools return when the memory is withheld.

The empty-memory control (``memory.ablation``): the agent keeps the recipe,
the task graph, the question and the same four tools, and every call answers this. It
is one fixed sentence, identical for every tool and every argument, so a
transcript shows the agent being told there is nothing to read rather than
being told four different stories about why.
"""

OPEN_VOCABULARY_UNAVAILABLE = "Open-vocabulary retrieval is not available."
"""Returned when no text-aligned crop embedder was built for this video.

The tool stays listed either way: the reference always offered it, and the
sub-agent's prompt names it, so hiding it would change that prompt.
"""


class ToolKit:
    """The tools of one video's agent, bound to its memories and models."""

    def __init__(
        self,
        *,
        video: Video,
        temporal: TemporalMemory,
        text_embedder: TextEmbedder | None,
        aligner: VideoTextAligner | None,
        objects: ObjectMemory | None = None,
        object_text_embedder: TextAlignedImageEmbedder | None = None,
        object_text_columns: slice | None = None,
        vqa: VideoQA | None = None,
        sub_runner: AgentRunner | None = None,
        sub_prompt: str | None = None,
        config: ToolConfig = ToolConfig(),
        memory_available: bool = True,
        permutation: SegmentPermutation | None = None,
        video_segments: Sequence[Segment] | None = None,
    ) -> None:
        self.video = video
        self.temporal = temporal
        self.text_embedder = text_embedder
        self.aligner = aligner
        self.objects = objects
        self.object_text_embedder = object_text_embedder
        self.object_text_columns = object_text_columns
        self.vqa = vqa
        self.sub_runner = sub_runner
        self.sub_prompt = sub_prompt
        self.config = config
        # Three attributes of the memory this toolkit was handed rather than
        # of the tools themselves (see ``memory.ablation``): whether it holds
        # anything, whether its segment names are a permutation of the
        # video's, and — when they are — the grid the video is still cut on.
        self.memory_available = memory_available
        self.permutation = permutation
        self.video_segments = (
            list(temporal.segments) if video_segments is None else list(video_segments)
        )
        if permutation is not None and permutation.n_segments != len(temporal.segments):
            raise ValueError(
                f"permutation of {permutation.n_segments} segments cannot name "
                f"a memory of {len(temporal.segments)}"
            )
        self.n_segments = len(temporal.segments)
        self.database = (
            None
            if objects is None or not memory_available
            else ObjectDatabase(
                objects, self.n_segments, read_only=config.sql_read_only
            )
        )

    @property
    def _preamble(self) -> str:
        """The segment-count prefix both memory tools open with (reference)."""
        return (
            f"There are {self.n_segments} segments in total, "
            f"ranging from 0 to {self.n_segments - 1}. "
        )

    def caption_retrieval(self, start: int, end: int) -> str:
        """Return the captions of segments ``start..end``, both included."""
        if not self.memory_available:
            return MEMORY_UNAVAILABLE
        start = max(start, 0)
        end = min(end, self.n_segments - 1)
        if self.config.enforce_caption_window:
            # The description promises at most ``caption_window_max`` captions;
            # the reference never enforced it, so this is opt-in.
            end = min(end, start + self.config.caption_window_max - 1)
        if start > end:
            return self._preamble + "Invalid start and end segment IDs!"
        captions = {i: self.temporal.captions[i] for i in range(start, end + 1)}
        return self._preamble + str(captions)

    def segment_localization(self, description: str) -> str:
        """Return the segments whose caption and video best match ``description``."""
        if not self.memory_available:
            return MEMORY_UNAVAILABLE
        text_query = self.text_embedder.embed([description])[0]
        visual_query = self.aligner.encode_text([description])[0]
        ranked = localize_segments(
            self.temporal,
            text_query=text_query,
            visual_query=visual_query,
            visual_weight=self.config.visual_weight,
            textual_weight=self.config.textual_weight,
            k=self.config.localization_top_k,
        )
        candidates = {index: self.temporal.captions[index] for index, _ in ranked}
        return (
            self._preamble + f"The most relevant segments are: {candidates}."
        )

    def _clip_of(self, segment_id: int) -> int:
        """Which segment of the *video* the memory calls ``segment_id``."""
        if self.permutation is None:
            return segment_id
        return self.permutation.perm[segment_id]

    def visual_question_answering(self, question: str, segment_id: int) -> str:
        """Answer ``question`` about the window around ``segment_id``.

        Under a shuffled memory the tool plays the clip that memory calls
        ``segment_id``, which is the video's own segment ``perm[segment_id]``:
        the agent has to watch what the caption it just read describes, or the
        control would be comparing a shuffled memory against an unshuffled eye.
        The window around it is the video's — one segment on each side in real
        time — because a window assembled out of the shuffled neighbours would
        not be a clip of anything.
        """
        if not self.memory_available:
            return MEMORY_UNAVAILABLE
        if self.vqa is None:
            raise ValueError("visual_question_answering needs a VideoQA model")
        if segment_id not in range(self.n_segments):
            return f"Segment ID {segment_id} not in range 0-{self.n_segments - 1}."
        return answer_about_segment(
            self.video,
            self.video_segments,
            self._clip_of(segment_id),
            question,
            self.vqa,
            context=self.config.vqa_context_segments,
            sample_fps=self.config.vqa_sample_fps,
            describe_question=self.config.vqa_describe_question,
            style=self.config.vqa_style,
        )

    def object_memory_querying(self, question: str) -> ToolResult:
        """Ask the object-memory sub-agent ``question`` and return its answer.

        A whole nested episode stands behind this one observation, so it
        rides along in the result and ends up under the parent's tool call.
        """
        if not self.memory_available:
            # No episode at all: the sub-agent would have nothing to query,
            # and a nested transcript of it being told so is noise.
            return ToolResult(observation=MEMORY_UNAVAILABLE)
        if self.sub_runner is None or self.sub_prompt is None:
            raise ValueError(
                "object_memory_querying needs a sub-agent runner and prompt"
            )
        result = self.sub_runner.run(
            question, self.sub_tools(), prompt_template=self.sub_prompt
        )
        observation = (
            AGENT_STOPPED_MESSAGE if result.answer is None else result.answer
        )
        return ToolResult(observation=observation, sub_transcript=result.transcript)

    def database_querying(self, program: str) -> str:
        """Run the sub-agent's SQL against the object database."""
        if not self.memory_available:
            return MEMORY_UNAVAILABLE
        if self.database is None:
            raise ValueError("database_querying needs an object memory")
        return self.database.query(program)

    def open_vocabulary_object_retrieval(self, description: str) -> str:
        """Return the ids of objects whose crops match ``description``."""
        if not self.memory_available:
            return MEMORY_UNAVAILABLE
        if self.objects is None:
            raise ValueError("open_vocabulary_object_retrieval needs an object memory")
        if self.object_text_embedder is None or self.object_text_columns is None:
            return OPEN_VOCABULARY_UNAVAILABLE
        return str(
            retrieve_candidate_objects(
                self.objects,
                description,
                self.object_text_embedder,
                text_columns=self.object_text_columns,
                template=self.config.object_query_template,
                threshold=self.config.object_similarity_threshold,
            )
        )

    def main_tools(self) -> list[ToolSpec]:
        """The orchestrator's tools, in the order the reference listed them."""
        window = self.config.caption_window_max
        context = self.config.vqa_context_segments
        tools = [
            ToolSpec(
                name="caption_retrieval",
                description=(
                    "given an input tuple (start_segment_ID, end_segment_ID), this "
                    "tool will retrieve all the captions between the two segments, "
                    f"{window} captions at most. end_segment_ID < start_segment_ID "
                    f"+ {window}."
                ),
                params=(
                    ParamSpec(name="start_segment_ID", kind="int"),
                    ParamSpec(name="end_segment_ID", kind="int"),
                ),
                input_name="input_tuple",
                fn=self.caption_retrieval,
            ),
            ToolSpec(
                name="segment_localization",
                description=(
                    "Given a textual description, this tool will return the top-"
                    f"{self.config.localization_top_k} candidate segments that are "
                    "most relevant to the description."
                ),
                params=(ParamSpec(name="description", kind="str"),),
                input_name="description",
                fn=self.segment_localization,
            ),
        ]
        if self.vqa is not None:
            tools.append(
                ToolSpec(
                    name="visual_question_answering",
                    description=(
                        "Given an input tuple (question, segment_ID), this tool will "
                        "focus on the video segments starting from "
                        f"segment_ID-{context} to segment_ID+{context}. It will "
                        "return the description of "
                        "the video segment and the answer to the question based on "
                        "the segment."
                    ),
                    params=(
                        ParamSpec(name="question", kind="str"),
                        ParamSpec(name="segment_ID", kind="int"),
                    ),
                    input_name="input_tuple",
                    fn=self.visual_question_answering,
                )
            )
        if (
            self.objects is not None
            and self.sub_runner is not None
            and self.sub_prompt is not None
        ):
            tools.append(
                ToolSpec(
                    name="object_memory_querying",
                    description=(
                        "Given a question about open-vocabulary objects such as 'how "
                        "many people are there in the video?' or 'In which segments "
                        "did the brown dog appear?', this tool will give the answer "
                        "based on the object memory."
                    ),
                    params=(ParamSpec(name="question", kind="str"),),
                    input_name="question",
                    fn=self.object_memory_querying,
                )
            )
        return tools

    def sub_tools(self) -> list[ToolSpec]:
        """The object-memory sub-agent's tools."""
        return [
            ToolSpec(
                name="database_querying",
                description=(
                    "given a MySQL program, this tool will query the database and "
                    "return the results."
                ),
                params=(ParamSpec(name="program", kind="str"),),
                input_name="program",
                fn=self.database_querying,
            ),
            ToolSpec(
                name="open_vocabulary_object_retrieval",
                description=(
                    "given an open-vocabulary description of an object or a person "
                    "(frying pan, person in red clothes e.g.), this tool will return "
                    "the possible candidate object IDs that satisfy the description."
                ),
                params=(ParamSpec(name="description", kind="str"),),
                input_name="description",
                fn=self.open_vocabulary_object_retrieval,
            ),
        ]

    def close(self) -> None:
        """Release the object database's connection."""
        if self.database is not None:
            self.database.close()

    def __enter__(self) -> "ToolKit":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
