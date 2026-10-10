# Architecture

The system follows VideoAgent (Fan et al., 2024): a memory-construction phase
that turns a video into a temporal memory and an object memory, and an
inference phase in which a ReAct orchestrator answers a question by calling
four tools over those memories. All model roles are filled by open models,
and the three language roles (orchestrator, captioner, visual question
answering) are played by one model, Qwen3.5-9B, served by vLLM.

```
video ──► 2-second segments ──┬─► captions (Qwen3.5-9B) ──► caption embeddings (Qwen3-Embedding-4B)
                              ├─► visual embeddings (PE-Core L)
                              └─► detection (RT-DETRv2) ─► tracking (ByteTrack) ─► re-identification
                                                                (SigLIP 2 + DINOv2) ─► object memory (SQLite)

question ──► ReAct orchestrator (Qwen3.5-9B) ◄──► tools:
               caption_retrieval            captions of a range of segments
               segment_localization         top segments for a text query: 18·visual + 11·caption similarity
               visual_question_answering    question about a segment, answered from its frames
               object_memory_querying       sub-agent over the object memory
          ──► Final Answer: one JSON object with a classification per recipe step
```

## Package layout (`src/video_agent/`)

| Module | Responsibility |
|---|---|
| `media.py` | video access, frame sampling, segment grid |
| `models/` | model interfaces and shared types (usage counts, detections) |
| `adapters/` | concrete models: captioners (local Qwen or served via an OpenAI-compatible API), chat clients, visual aligners (PE-Core L by default; X-CLIP, ViCLIP and SigLIP 2 as alternatives), text embedder, detector (RT-DETRv2), tracker, image embedders, VQA |
| `memory/` | building and loading the memories: temporal memory, object memory and its database, re-identification, retrieval, on-disk artifacts, and the memory controls (`ablation.py`: empty and shuffled memory) |
| `agent/` | the ReAct loop, the four tools, the run configuration, transcripts and the prompt templates |
| `task/` | the procedural task: annotations and task graphs, the question, the parsing of the answer into per-step verdicts, and run records |
| `eval/` | matching verdicts to annotated steps, metrics, the computed order, bootstrap intervals, constant baselines, reports, and the synthetic-memory cases |
| `lineage.py` | the code version and library versions written into every output |

## Memory construction

Videos are cut into 2-second segments, as in the original. Each segment is
captioned from a clip of 10 frames at 5 fps, sent to the served model as one
MP4. With `--context-segments 1` the clip also covers the previous and next
segments and the instruction asks about the middle two seconds. Captions are
generated once per instruction and stored side by side
(`captions.<variant>.json`), so the same memory can be read with different
caption sets.

The object memory follows the original pipeline: detections every few frames,
tracks from ByteTrack, and tracks merged into objects by appearance
similarity (SigLIP 2 and DINOv2 features). Objects are stored in an SQLite
database that the object-memory sub-agent queries with SQL.

## Inference

The ReAct loop, the tool descriptions and the system prompt are the
original ones (`agent/prompts/prompt.txt`, `multiple_choice_prompt.txt`,
`database_query_prompt.txt`). For the procedural task, `task_prompt.txt`
changes only the final-answer line, and the user question
(`agent/prompts/task_question.txt`) gives the recipe name, the number of
segments, the steps, the direct prerequisites of each step and the definition
of an order error, and asks for a JSON object with, per step:

- *execution*: performed, not performed, or unsure; with the inclusive
  segment range where the step was seen when performed;
- *order* (performed steps only): ok, violation (naming the other step), or
  unsure.

The episode ends when the model writes a line starting with `Final Answer:`.
The loop is capped at 100 iterations (the original used LangChain's default
of 15) and each orchestrator turn at 16,384 tokens. Decoding is greedy with
the model's thinking mode disabled.

## Single-call systems

`scripts/run_single_call.py` asks the same model the same question in one
request, without tools:

- **captions in a single call**: every caption of the temporal memory, one
  line per segment (`[segment, start-end s] caption`); the model localizes
  steps by segment number;
- **whole video in a single call**: the recording resampled to 1 frame per
  second at 448×256 (about 56 visual tokens per second), within a budget of
  56,000 visual tokens; longer recordings are sampled at a lower rate. The
  model answers in seconds, which are mapped to the 2-second grid.

Both write run records in the same format as the agent, so the same
evaluation scores all systems.

## Evaluation

Each annotated step instance is matched to the verdict of its step.
Indeterminate answers (a step missing from the answer, or a recording without
a final answer) count as "no error", and their share is reported as
*unclassified*. Instances of an optional step that was not performed are
excluded.

Order errors are read in two ways from the same answer:

- **judged order**: the order classification written by the model;
- **computed order**: the model's classification is discarded, and a step is
  marked as a violation when one of its prerequisites, direct or indirect,
  starts after the step's localized interval ends; both steps of the pair are
  marked. Only performed steps with a localized interval take part, so a
  missing prerequisite is a missing step, not an order error. This is the
  rule of the CaptainCook4D annotations, inferred from them
  (`scripts/validate_order_rule.py`).

Reports include precision, recall and F1 per category, the unclassified
share, hit@IoU≥0.5 of the localized intervals, EDA (segment-level error
detection accuracy) and the cost per recording.
