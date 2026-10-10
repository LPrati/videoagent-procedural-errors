# VideoAgent for procedural error detection

An open reimplementation of [VideoAgent](https://arxiv.org/abs/2403.11481)
(Fan et al., 2024), a memory-augmented multimodal agent for video
understanding, and its evaluation on procedural error detection in the
[CaptainCook4D](https://captaincook4d.github.io/captain-cook/) dataset
(Peddi et al., 2024). Every model runs locally on a single 32 GB GPU, except
Claude Opus 5, which is used through a hosted API in one comparison.

This repository contains the code that reproduces the main results
(Tables 5.1–5.5 and the EgoSchema validation) of the monograph:

> Leonardo Giovanni Prati. *Agentes de vídeo para verificação de conformidade
> procedimental: avaliação do VideoAgent na detecção de erros procedimentais*
> (Video agents for procedural compliance verification: evaluating VideoAgent
> on procedural error detection). Monografia (MBA em Inteligência Artificial e
> Big Data), Instituto de Ciências Matemáticas e de Computação, Universidade de
> São Paulo, São Carlos, 2026. Advisor: Prof. Dr. Kelton A. P. da Costa.

The monograph is written in Portuguese; this repository and its
documentation are in English. See [`CITATION.cff`](CITATION.cff) to cite it.

## What is here

- **The agent** (`src/video_agent/`): the two phases of VideoAgent — memory
  construction (a temporal memory of 2-second segment captions with visual
  and text embeddings, and an object memory built from detection, tracking
  and re-identification) and inference (a ReAct orchestrator with the four
  original tools: caption retrieval, segment localization, visual question
  answering and object-memory querying). The original models are replaced by
  open ones:

  | Role | Original | This reimplementation |
  |---|---|---|
  | Orchestrator, captioner, visual QA | GPT-4, LaViLa, Video-LLaVA | Qwen3.5-9B (one served model) |
  | Visual segment embedding | ViCLIP | PE-Core L |
  | Caption embedding | text-embedding-3-large | Qwen3-Embedding-4B |
  | Object detection / tracking | RT-DETR / ByteTrack | RT-DETRv2 / ByteTrack |
  | Re-identification | CLIP + DINOv2 | SigLIP 2 + DINOv2 |

- **The task adaptation**: for each recording the system receives the video,
  the recipe steps and the task graph, and answers, per step, whether it was
  performed (with the segments where it was seen) and whether its order was
  violated. Only the user question changes; the ReAct loop, the tools and the
  system prompt are the original ones.
- **Two single-call comparison systems** with the same model and question:
  all captions of the temporal memory in one request, and the whole video in
  one request (`scripts/run_single_call.py`).
- **The evaluation** (`scripts/evaluate.py`): precision, recall and F1 per
  error category (order errors and missing steps), the share of unclassified
  steps, hit@IoU≥0.5, EDA and cost. Order errors are read two ways from the
  same answer: the *judged order* (the order classification the model wrote)
  and the *computed order* (a program applies the annotation's order rule to
  the step intervals the model localized).
- **Controls**: empty and shuffled memory (`run_task.py --memory`), a
  synthetic memory built from the annotations
  (`scripts/synthetic_harness.py`, `scripts/score_synthetic.py`), and the
  validation of the order rule against the annotations
  (`scripts/validate_order_rule.py`).
- **EgoSchema**: the reimplementation evaluated on the 500 public questions
  of the original paper's benchmark (`scripts/run_egoschema.py`,
  `scripts/evaluate_egoschema.py`).
- **Prompts**: the agent's prompts are package data
  (`src/video_agent/agent/prompts/`); the caption instructions and the
  single-call questions are in [`prompts/`](prompts/).
- **Data splits** ([`data_splits/`](data_splits/)): recording identifiers
  only.

See [`docs/architecture.md`](docs/architecture.md) for how the pieces fit
together and [`docs/reproduction.md`](docs/reproduction.md) for every
command.

## Not covered

The following parts of the monograph are not in this repository:

- the script that drew the data splits: the lists in `data_splits/` are
  fixed and versioned, and the rule is written in each file;
- the selection and calibration probes of Section 5.1 (comparison of
  captioners and visual embedders, recalibration of the localization weights
  and of the object-memory threshold);
- the LLM judge of caption quality (Figure 5.3, Appendix D) and its
  calibration against a second model and a human annotator;
- the analyses of Section 5.6.2 (tool usage, interval widths, localization by
  thirds of the video) and the verification-routine variant of the question;
- the dedicated measurement of the memory-construction cost (the caption
  stage records its own time and tokens in each `meta.json`).

## Requirements

- Linux with one NVIDIA GPU of 32 GB (the results were produced on an
  RTX 5090). The served model owns the GPU during inference; the embedders
  then run on the CPU.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- [vLLM](https://docs.vllm.ai/) in a separate environment, serving
  `Qwen/Qwen3.5-9B` (see [`serving/vllm/README.md`](serving/vllm/README.md)).
  The results were produced with vLLM `0.28.1rc1.dev416+ga11dfcff8` (nightly)
  and the Qwen3.5-9B snapshot `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

## Installation

```shell
git clone https://github.com/LPrati/videoagent-procedural-errors
cd videoagent-procedural-errors
uv sync                               # core package and tests
uv run pytest                         # no GPU and no data needed
uv sync --extra agent --extra models  # everything needed to build memories and run the agent
```

`--extra agent` installs the OpenAI-compatible client used to talk to vLLM;
`--extra models` installs PyTorch, transformers, open_clip and the tracking
libraries used to build the memories.

The defaults of the scripts are the published configuration: the orchestrator,
captioner and visual question answering go to the local vLLM server
(`--chat vllm`, `--captioner api`, `--vqa api`) with `Qwen/Qwen3.5-9B`.
Choosing a hosted backend (`--chat openrouter`, `--base-url` of a provider,
`run_single_call.py --provider openai-compatible`) sends requests to a paid
API.

## Data

No video, annotation or derived artifact is redistributed here. Obtain them
from the official sources:

- **CaptainCook4D annotations**: clone
  [`CaptainCook4D/annotations`](https://github.com/CaptainCook4D/annotations)
  into `data/captaincook4d/annotations` (the results used commit `a8a920a`):

  ```shell
  git clone https://github.com/CaptainCook4D/annotations data/captaincook4d/annotations
  git -C data/captaincook4d/annotations checkout a8a920a
  ```

  Every script also accepts `--annotations-dir`. With the annotations in
  place, `uv run pytest` also runs the tests that read them (or point
  `CC4D_ANNOTATIONS_DIR` at another copy).
- **CaptainCook4D videos**: the GoPro recordings at 360p, from the official
  [downloader](https://github.com/CaptainCook4D/downloader), in one directory
  with names of the form `<recording_id>_360p.mp4` (for example
  `1_28_360p.mp4`). One recording of the development split, `12_6`, has no
  GoPro video in the official release and is not evaluated.
- **EgoSchema**: `questions.json` and `subset_answers.json` from
  [`egoschema/EgoSchema`](https://github.com/egoschema/EgoSchema); the 500
  clips of the public subset can be fetched with
  `scripts/download_egoschema.py`.

`data/`, memories, runs and reports are git-ignored.

### Splits

The 220 CaptainCook4D recordings with at least one annotated error were split
by recipe into a development subset and a test subset (seed 20260903); the
rule is written in each file. All choices were made on the development
subset; the test subset was run once.

| File | Recordings | Use |
|---|---|---|
| `dev.json` | 60 with errors + 24 normal | the development subset as drawn |
| `dev_83.json` | 59 + 24 | the evaluated development subset (`dev.json` without `12_6`) |
| `test.json` | 160 + 48 | the test subset |
| `reserve_normal.json` | 92 normal | not evaluated |
| `frontier_subset.json` | 7 + 3 (from dev) | runs with Claude Opus 5 |
| `short_subset.json` | 7 + 3 (from dev) | repeated runs of the single-call systems |

## Reproducing the results

The full sequence of commands is in
[`docs/reproduction.md`](docs/reproduction.md). In short: build the memories
(`scripts/preprocess_temporal.py`, `scripts/preprocess_objects.py`), serve
Qwen3.5-9B, run each system over a split (`scripts/run_task.py`,
`scripts/run_single_call.py`), score every run in both order readings
(`scripts/evaluate.py`) and print the table column
(`scripts/summarize_reports.py`).

| Result in the monograph | Command (see `docs/reproduction.md`) | Expected |
|---|---|---|
| Order rule vs. annotated order errors (Sec. 4.3.2) | `validate_order_rule.py --split data_splits/dev_83.json` (and `test.json`) | F1 0.92 (dev), 0.90 (test); deterministic |
| Table 5.1, agent with action captions (dev, mean of 3 runs) | `run_task.py --caption-variant action --seed 20260903` (also `…04`, `…05`) | computed order F1 35.0, judged order F1 16.7, missing steps F1 16.0, 581k input tokens per recording |
| Table 5.1, agent with abstention-with-context captions (dev) | `run_task.py --caption-variant abst_ctx --seed 20260903` | computed 44.5, judged 23.0, missing 24.1, 443k tokens |
| Table 5.1, captions in a single call (dev) | `run_single_call.py --arm captions` | computed 54.4, judged 18.2, missing 15.4, 18k tokens |
| Table 5.1, whole video in a single call (dev) | `run_single_call.py --arm video` | computed 60.8, judged 8.1, missing 17.2, 47k tokens |
| Table 5.2, the same four systems (test, one run each) | the same commands with `--split data_splits/test.json` | computed order F1 37.5 / 37.6 / 52.7 / 60.4 |
| Table 5.3, memory controls (dev) | `run_task.py --memory shuffled` / `--memory empty` | computed order F1 18.4 (shuffled) and 0.0 (empty); unclassified 41.8 and 81.2 |
| Table 5.4, synthetic memory (5 normal dev recordings × 7 cases) | `synthetic_harness.py`, then `score_synthetic.py` | 70, 67, 60, 60, 54 and 70 of 72 steps; η from 0.00 to 0.59 |
| Table 5.5, Claude Opus 5 on 10 recordings | `run_task.py --vqa-base-url …` / `run_single_call.py --provider openai-compatible` on `frontier_subset.json` | captions in a single call: computed order F1 84.4, judged 82.4 (decoding of Opus not controllable; approximate) |
| EgoSchema, 500 public questions (Sec. 5.1) | `run_egoschema.py`, `evaluate_egoschema.py` | accuracy 63.2% (6 invalid); 63.8% in a second run |
| Cost (Sec. 5.5) | inference: `evaluate.py` reports; memory: `meta.json` of each recording | see `docs/reproduction.md` |

Values are percentages unless noted. The agent is not deterministic even
with greedy decoding (the server batches requests, and a different token at
one iteration changes the rest of the episode): three runs of the same
configuration differed by up to 6.0 points of computed-order F1 and 13.1
points of missing-step F1, so expect agent numbers to vary by about that
much. The single-call systems gave identical answers when repeated one
request at a time.

## Citation

If you use this code, please cite the monograph (see
[`CITATION.cff`](CITATION.cff)) and the works it builds on:

- Y. Fan, X. Ma, R. Wu, Y. Du, J. Li, Z. Gao, Q. Li. *VideoAgent: A
  Memory-augmented Multimodal Agent for Video Understanding.* ECCV 2024.
  [arXiv:2403.11481](https://arxiv.org/abs/2403.11481),
  [code](https://github.com/YueFan1014/VideoAgent).
- R. Peddi et al. *CaptainCook4D: A Dataset for Understanding Errors in
  Procedural Activities.* NeurIPS 2024 Datasets and Benchmarks Track.
  [arXiv:2312.14556](https://arxiv.org/abs/2312.14556).

## Third-party material

The ReAct prompt templates `prompt.txt`, `multiple_choice_prompt.txt` and
`database_query_prompt.txt` in `src/video_agent/agent/prompts/`, and the
visual question prompt in `src/video_agent/adapters/vqa_openai.py`, are
reproduced from the [VideoAgent repository](https://github.com/YueFan1014/VideoAgent),
distributed under the Apache License 2.0. `task_prompt.txt` is `prompt.txt`
with the final-answer line changed for this task.

## License

This code is released under the [Apache License 2.0](LICENSE).
