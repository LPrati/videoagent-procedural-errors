# Serving Qwen3.5-9B with vLLM

One checkpoint, `Qwen/Qwen3.5-9B` (BF16), plays the three language roles of
the system: the ReAct orchestrator (`--chat vllm`), the visual question
answering tool (`--vqa api`) and the segment captioner (`--captioner api`).
vLLM serves it as an OpenAI-compatible endpoint at
`http://localhost:8000/v1`, which is what every script talks to by default
(`--base-url` points elsewhere).

## Install

vLLM goes in an environment of its own, separate from the project's: it
brings its own PyTorch and transformers versions. Qwen3.5 needs a recent
build; the results were produced with the nightly
`0.28.1rc1.dev416+ga11dfcff8`:

```shell
uv venv ~/venvs/vllm
source ~/venvs/vllm/bin/activate
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

Record the installed version (`uv pip show vllm`) next to your runs.

## Start

From the vLLM environment:

```shell
serving/vllm/serve.sh               # agent runs and captioning
serving/vllm/serve_single_call.sh   # the two single-call systems
serving/vllm/serve.sh --dry-run     # print the resolved command, start nothing
```

`serve.sh` resolves to:

```shell
vllm serve Qwen/Qwen3.5-9B --served-model-name Qwen/Qwen3.5-9B \
    --host 127.0.0.1 --port 8000 \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image": 16, "video": 1}' \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --max-model-len 131072 --gpu-memory-utilization 0.92 --max-num-seqs 16
```

`serve_single_call.sh` changes the context to 77,824 tokens, at most 2
concurrent sequences, one video and no image per request, a video loader that
keeps every frame it is sent, a video processor that accepts up to 64,000
visual tokens per item, and `--gpu-memory-utilization 0.90`.

Every setting is an environment variable with a default (`VLLM_MODEL`,
`VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_MAX_NUM_SEQS`,
`VLLM_LIMIT_MM`, `VLLM_MEDIA_IO`, `VLLM_HOST`, `VLLM_PORT`, `VLLM_DTYPE`,
`VLLM_QUANTIZATION`, `VLLM_EXTRA_ARGS`). If the server does not start for
lack of KV cache, lower `VLLM_MAX_MODEL_LEN`.

Check that it is up:

```shell
curl http://localhost:8000/v1/models
```

If the server is started with `--api-key`, export the same value as
`VLLM_API_KEY` before running the scripts.

## Request settings

- **Thinking off.** Qwen3.5 reasons by default, and a `<think>` block breaks
  the ReAct format (the `\nObservation` stop sequence and the `Final Answer:`
  line). Every request sends
  `chat_template_kwargs: {"enable_thinking": false}`; `--reasoning-parser
  qwen3` keeps any leaked reasoning out of the message content.
- **Frames as sent.** Video requests carry
  `mm_processor_kwargs: {"do_sample_frames": false}`, so the processor uses
  exactly the frames the client sampled (10 frames at 5 fps for a caption
  clip; the planned rate for the whole-video system), and the server keeps
  them all on the way in (`--media-io-kwargs`, `num_frames: -1`).
- **Clips as video.** Caption clips are sent as one MP4 each
  (`--caption-transport video`, the default), which costs several times fewer
  prompt tokens than one image per frame.
- **GPU sharing.** The served model owns the GPU while it runs: run the
  agent's embedders with `--device cpu`, and stop the server to compute
  embeddings and object memories on the GPU.
- **FlashInfer sampler.** `serve.sh` exports
  `VLLM_USE_FLASHINFER_SAMPLER=0`, because that sampler is compiled at start
  time and needs a CUDA toolkit; set it to 1 on a machine that has one.

Even with greedy decoding, batched serving is not bit-reproducible: the
composition of a batch can change a token, and in the agent one different
token can change the rest of an episode.
