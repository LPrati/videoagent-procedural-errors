# Reproducing the results

Every command runs from the repository root with `uv run python …`. The
placeholders are:

| Placeholder | What it is |
|---|---|
| `VIDEOS_DIR` | the CaptainCook4D GoPro videos at 360p, named `<recording_id>_360p.mp4` |
| `PREPROCESS_DIR` | where the memories are written, one directory per video |
| `RUNS_DIR` | where run records are written, one directory per run and per recording |
| `REPORTS_DIR` | where evaluation reports are written |

The annotations are expected in `data/captaincook4d/annotations` (see the
README); pass `--annotations-dir` to use another location. Everything below
was run on one 32 GB GPU. The commands spell out the published settings even
where they equal the script defaults. Runs are resumable: a recording that already has a
record is skipped, and `--retry-errors` re-runs the ones that ended in error.

## 1. Build the memories

The memory is built once per video, for all splits. The steps that need the
GPU for an embedder run with the vLLM server stopped; captioning runs against
the server.

**Visual segment embeddings** (PE-Core L, 2-second segments, 10 frames per
segment), server stopped:

```shell
uv run python scripts/preprocess_temporal.py --videos-dir VIDEOS_DIR --out PREPROCESS_DIR \
    --stages visual_embed --device cuda --clips-per-batch 16
```

**Object memory** (RT-DETRv2 on the 80 COCO classes, ByteTrack, SigLIP 2 and
DINOv2 re-identification), server stopped:

```shell
uv run python scripts/preprocess_objects.py --videos-dir VIDEOS_DIR --out PREPROCESS_DIR --device cuda
```

**Captions**, with the agent server running (`serving/vllm/serve.sh`). The
two caption sets used in the main tables are the *action* instruction (the
closest to the original LaViLa captions) and the *abstention with context*
instruction:

```shell
# action: built-in instruction, 2-second clip
uv run python scripts/preprocess_temporal.py --videos-dir VIDEOS_DIR --out PREPROCESS_DIR \
    --stages caption --captioner api --captioner-model Qwen/Qwen3.5-9B \
    --prompt-name action --workers 16 --caption-transport video

# abstention with context: 6-second clip around the segment
uv run python scripts/preprocess_temporal.py --videos-dir VIDEOS_DIR --out PREPROCESS_DIR \
    --stages caption --captioner api --captioner-model Qwen/Qwen3.5-9B \
    --prompt-file prompts/captions/abstention_context.md --variant-name abst_ctx \
    --context-segments 1 --workers 16 --caption-transport video --clips-per-batch 8
```

The other three instructions of the monograph are the built-in `state` and
`order` (the "phase" instruction) and `prompts/captions/abstention.md`
(`--variant-name abst`). Each caption set is stored as
`captions.<variant>.json` and selected later with `--caption-variant`.

**Caption embeddings** (Qwen3-Embedding-4B), server stopped, once per caption
set:

```shell
uv run python scripts/preprocess_temporal.py --videos-dir VIDEOS_DIR --out PREPROCESS_DIR \
    --stages caption_embed --variant-name action --device cuda
uv run python scripts/preprocess_temporal.py --videos-dir VIDEOS_DIR --out PREPROCESS_DIR \
    --stages caption_embed --variant-name abst_ctx --device cuda
```

The caption stage writes its wall time and token counts to the `meta.json` of
each video; this is the memory-construction cost of the captions. The
visual-embedding and object-memory scripts print the time per video.

## 2. Serve the model

Two server configurations are used (see `serving/vllm/README.md`):

```shell
serving/vllm/serve.sh               # agent and captioning: 128k context
serving/vllm/serve_single_call.sh   # single-call systems: one long video per request
```

## 3. Run the systems

### Agent (Tables 5.1 and 5.2)

```shell
uv run python scripts/run_task.py --split data_splits/dev_83.json \
    --artifacts-root PREPROCESS_DIR --videos-dir VIDEOS_DIR --out RUNS_DIR \
    --run-name agent-abst_ctx --chat vllm --model Qwen/Qwen3.5-9B --vqa api --device cpu \
    --max-iterations 100 --caption-variant abst_ctx --seed 20260903
```

- Agent with action captions (the configuration closest to the original):
  `--caption-variant action`, three runs with `--seed 20260903`, `20260904`
  and `20260905` (`--run-name agent-action-s1`, `-s2`, `-s3`). The dev table
  reports the mean of the three.
- Test subset: the same commands with `--split data_splits/test.json`, one
  run each (seed 20260903).
- The default per-turn cap of the orchestrator is 16,384 tokens
  (`--max-tokens`). `--shard I/N` splits a run over several processes; the
  results were produced with two processes per run against one server.

### Captions in a single call

```shell
uv run python scripts/run_single_call.py --split data_splits/dev_83.json --arm captions \
    --caption-variant abst_ctx --artifacts-root PREPROCESS_DIR --videos-dir VIDEOS_DIR \
    --out RUNS_DIR --run-name single-call-captions \
    --max-tokens 8192 --temperature 0.0 --seed 20260919
```

### Whole video in a single call

```shell
uv run python scripts/run_single_call.py --split data_splits/dev_83.json --arm video \
    --fps 1 --width 448 --height 256 --visual-token-budget 56000 \
    --artifacts-root PREPROCESS_DIR --videos-dir VIDEOS_DIR \
    --out RUNS_DIR --run-name single-call-video \
    --max-tokens 8192 --temperature 0.0 --seed 20260919
```

Recordings longer than the visual-token budget allows at 1 frame per second
are sampled at a proportionally lower rate (28 of the 83 dev recordings).

### Memory controls (Table 5.3)

The agent with abstention-with-context captions and the memory withheld or
shuffled (one permutation per recording):

```shell
uv run python scripts/run_task.py … --caption-variant abst_ctx --seed 20260903 \
    --memory shuffled --run-name control-shuffled
uv run python scripts/run_task.py … --caption-variant abst_ctx --seed 20260903 \
    --memory empty --run-name control-empty
```

(`…` stands for the split, paths and model flags of the agent command above.)
For the shuffled memory, the computed order maps every localized interval
back to the real time of the segments the agent cited. The ratio η is the
recall with the shuffled memory over the recall with the ordered memory.

## 4. Evaluate

Each run is scored twice, once per order reading:

```shell
uv run python scripts/evaluate.py --runs RUNS_DIR/agent-abst_ctx \
    --split data_splits/dev_83.json --out REPORTS_DIR/agent-abst_ctx/llm
uv run python scripts/evaluate.py --runs RUNS_DIR/agent-abst_ctx \
    --split data_splits/dev_83.json --out REPORTS_DIR/agent-abst_ctx/anchors \
    --order-from anchors --order-rule transitive --order-edges both --order-margin end
```

(The `--order-rule`, `--order-edges` and `--order-margin` values shown are the
defaults.) Then print the column of Table 5.1 or 5.2:

```shell
uv run python scripts/summarize_reports.py \
    --judged REPORTS_DIR/agent-abst_ctx/llm --computed REPORTS_DIR/agent-abst_ctx/anchors
```

For the three runs with action captions, pass the three `llm` directories to
`--judged` and the three `anchors` directories to `--computed`; the values are
averaged.

Expected values (percentages; tokens are input tokens per recording, in
thousands):

| Metric | Agent, action captions | Agent, abstention with context | Captions, single call | Whole video, single call |
|---|---|---|---|---|
| **Development subset (83)** | mean of 3 runs | | | |
| Judged order (P / R / F1) | 27.3 / 12.1 / 16.7 | 36.8 / 16.7 / 23.0 | 44.4 / 11.4 / 18.2 | 18.3 / 5.2 / 8.1 |
| Computed order (P / R / F1) | 41.2 / 30.6 / 35.0 | 46.8 / 42.4 / 44.5 | 48.2 / 62.4 / 54.4 | 56.6 / 65.7 / 60.8 |
| Missing steps (P / R / F1) | 12.6 / 22.4 / 16.0 | 18.7 / 34.1 / 24.1 | 11.6 / 23.2 / 15.4 | 17.3 / 17.1 / 17.2 |
| Unclassified | 44.2 | 27.3 | 18.4 | 7.7 |
| hit@IoU≥0.5 | 18.5 | 28.8 | 34.3 | 36.0 |
| EDA (judged / computed) | 66.6 / 68.2 | 66.5 / 67.9 | 68.2 / 70.0 | 67.5 / 71.6 |
| Tokens (thousands) | 581 | 443 | 18 | 47 |
| **Test subset (208)** | one run | | | |
| Judged order (P / R / F1) | 30.1 / 12.3 / 17.5 | 39.7 / 15.9 / 22.7 | 38.7 / 7.0 / 11.9 | 23.7 / 9.4 / 13.5 |
| Computed order (P / R / F1) | 52.6 / 29.1 / 37.5 | 48.8 / 30.7 / 37.6 | 49.0 / 57.0 / 52.7 | 56.8 / 64.4 / 60.4 |
| Missing steps (P / R / F1) | 11.3 / 23.8 / 15.4 | 11.0 / 18.8 / 13.9 | 23.4 / 35.1 / 28.1 | 10.7 / 14.9 / 12.4 |
| Unclassified | 47.4 | 39.2 | 15.2 | 9.2 |
| hit@IoU≥0.5 | 18.8 | 29.6 | 35.6 | 32.8 |
| EDA (judged / computed) | 63.0 / 64.7 | 63.9 / 64.7 | 63.5 / 66.3 | 63.0 / 67.1 |
| Tokens (thousands) | 545 | 622 | 17 | 47 |

Memory controls (development subset, agent with abstention-with-context
captions, one run each):

| Memory | Judged order (P / R / F1) | Computed order (P / R / F1) | Missing steps (P / R / F1) | Unclassified |
|---|---|---|---|---|
| Ordered | 36.8 / 16.7 / 23.0 | 46.8 / 42.4 / 44.5 | 18.7 / 34.1 / 24.1 | 27.3 |
| Shuffled | 14.3 / 12.9 / 13.5 | 21.4 / 16.2 / 18.4 | 7.7 / 12.2 / 9.4 | 41.8 |
| Empty | — / 0.0 / — | 0.0 / 0.0 / 0.0 | — / 0.0 / — | 81.2 |

The single-call systems are close to deterministic and should reproduce
these values closely; the agent varies between runs (see the README).

The reports also contain the cost per recording (`cost` block of
`metrics.json`: model calls, tool calls, input and output tokens, wall time).

## 5. Order rule against the annotations

The computed order applies a rule inferred from the annotations. Applied to
the annotated intervals themselves, it reproduces the annotated order errors
with F1 0.92 on the development subset and 0.90 on the test subset:

```shell
uv run python scripts/validate_order_rule.py --split data_splits/dev_83.json
uv run python scripts/validate_order_rule.py --split data_splits/test.json
```

## 6. Synthetic memory (Table 5.4)

A memory built from the annotations of five normal development recordings,
with seven cases each (unaltered, one step removed, two linked steps swapped,
two unlinked steps swapped, removal and swap together, a repeated step, and
the whole memory shuffled). No video and no vision model is involved:

```shell
uv run python scripts/synthetic_harness.py --from-split data_splits/dev.json \
    --subset normal --limit 5 --out RUNS_DIR --run-name synthetic \
    --chat vllm --model Qwen/Qwen3.5-9B --max-iterations 100 --max-tokens 0 --seed 20260903
uv run python scripts/score_synthetic.py --runs RUNS_DIR/synthetic --out REPORTS_DIR/synthetic
```

`score_synthetic.py` scores the order column with the order definition of the
task question. Expected: 70, 67, 60, 60, 54 and 70 of 72 steps with the
expected classification in the six scored cases, and η between 0.00 and 0.59
(mean 0.31) for the shuffled memory.

## 7. Claude Opus 5 on 10 recordings (Table 5.5)

`data_splits/frontier_subset.json` lists the 10 recordings. The captions in a
single call with Claude Opus 5 use the same script with a hosted
OpenAI-compatible endpoint:

```shell
export OPENAI_API_KEY=...   # key of the endpoint
uv run python scripts/run_single_call.py --split data_splits/frontier_subset.json \
    --arm captions --caption-variant abst_ctx --provider openai-compatible \
    --base-url ENDPOINT_URL --model claude-opus-5 \
    --artifacts-root PREPROCESS_DIR --videos-dir VIDEOS_DIR \
    --out RUNS_DIR --run-name single-call-captions-opus --max-tokens 8192
```

The published values are the mean of three runs. For the agent, the
orchestrator is pointed at the same kind of endpoint while the visual
questions stay with Qwen3.5-9B on the local server, as in the published run:

```shell
export OPENROUTER_API_KEY=...   # key of the orchestrator endpoint
uv run python scripts/run_task.py --split data_splits/frontier_subset.json \
    --artifacts-root PREPROCESS_DIR --videos-dir VIDEOS_DIR --out RUNS_DIR \
    --run-name agent-opus --chat openrouter --base-url ENDPOINT_URL --model claude-opus-5 \
    --vqa api --vqa-base-url http://localhost:8000/v1 --device cpu \
    --max-iterations 100 --caption-variant abst_ctx --seed 20260903
```

The captions, memory, tools and question are the same as for Qwen3.5-9B.
In the published runs the decoding parameters of Claude Opus 5 could not be
controlled (the model ran with its own reasoning), so these numbers are not
expected to be reproduced exactly. Evaluate with
`--split data_splits/frontier_subset.json`.

| System | Judged order (P / R / F1) | Computed order (P / R / F1) | Missing steps (P / R / F1) | Unclassified |
|---|---|---|---|---|
| Agent, Qwen3.5-9B | 69.2 / 14.5 / 24.0 | 82.9 / 46.8 / 59.8 | 37.5 / 42.9 / 40.0 | 26.3 |
| Agent, Claude Opus 5 | 77.5 / 50.0 / 60.8 | 79.6 / 62.9 / 70.3 | 77.8 / 25.0 / 37.8 | 4.7 |
| Captions in a single call, Qwen3.5-9B | 63.2 / 19.4 / 29.6 | 73.9 / 82.3 / 77.9 | 37.5 / 21.4 / 27.3 | 12.1 |
| Captions in a single call, Claude Opus 5 | 85.5 / 79.6 / 82.4 | 84.5 / 84.4 / 84.4 | 84.4 / 45.2 / 58.9 | 5.6 |

## 8. EgoSchema (Section 5.1)

The 500 questions of the public subset, with the original multiple-choice
prompt and the original limit of 15 iterations. The memory is built exactly
as for CaptainCook4D, except for the caption instruction, which marks the
camera wearer with the `#C` prefix (`prompts/captions/ego.md`):

```shell
uv run --with gdown python scripts/download_egoschema.py --questions EGOSCHEMA/questions.json \
    --answers EGOSCHEMA/subset_answers.json --out EGO_VIDEOS
uv run python scripts/preprocess_objects.py --videos-dir EGO_VIDEOS --out EGO_PREPROCESS --device cuda
uv run python scripts/preprocess_temporal.py --videos-dir EGO_VIDEOS --out EGO_PREPROCESS \
    --stages visual_embed --device cuda --clips-per-batch 16
uv run python scripts/preprocess_temporal.py --videos-dir EGO_VIDEOS --out EGO_PREPROCESS \
    --stages caption --captioner api --captioner-model Qwen/Qwen3.5-9B \
    --prompt-file prompts/captions/ego.md --variant-name ego --workers 16
uv run python scripts/preprocess_temporal.py --videos-dir EGO_VIDEOS --out EGO_PREPROCESS \
    --stages caption_embed --variant-name ego --device cuda

uv run python scripts/run_egoschema.py --questions EGOSCHEMA/questions.json \
    --answers EGOSCHEMA/subset_answers.json --videos-dir EGO_VIDEOS \
    --artifacts-root EGO_PREPROCESS --out RUNS_DIR --run-name ego \
    --chat vllm --model Qwen/Qwen3.5-9B --vqa api --device cpu \
    --caption-variant ego --max-iterations 15
uv run python scripts/evaluate_egoschema.py --run-dir RUNS_DIR/ego \
    --answers EGOSCHEMA/subset_answers.json --out REPORTS_DIR/ego
```

Invalid answers (no option, or the iteration limit reached) count as wrong.
Expected accuracy: 63.2% with 6 invalid answers, and 63.8% in a second run
(VideoAgent reports 62.8% on the same subset).
