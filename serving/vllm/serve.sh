#!/usr/bin/env bash
# Serve a Qwen3.5 checkpoint as an OpenAI-compatible endpoint.
#
# One served checkpoint plays every model role of the system: orchestrator
# (--chat vllm), VQA (--vqa api) and captioner (--captioner api). Every knob
# is an environment variable with a default, so a run is one line and its
# settings are still visible in the printed command.
#
# The defaults are the configuration of the agent runs: Qwen/Qwen3.5-9B in
# BF16 with a 128k context on one 32 GB GPU. The single-call systems use a
# different configuration, see serve_single_call.sh.
#
# Run it inside the vLLM environment (see README.md):
#     ./serve.sh
#     VLLM_MAX_MODEL_LEN=65536 ./serve.sh     # smaller KV cache
#     ./serve.sh --dry-run                    # print the command, start nothing
#
set -euo pipefail

# --dry-run resolves everything and prints the command without starting a
# server, so the resolution can be read (and tested) on a machine with no GPU
# and no vLLM.
dry_run=0
if [ "${1:-}" = "--dry-run" ]; then
  dry_run=1
  shift
fi

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-9B}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-$VLLM_MODEL}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
# Defaults with nested braces are written through a variable: bash counts
# braces inside ${VAR:-...}, and an inline JSON default would leave a stray
# closing brace in the value whenever the variable is set.
default_limit_mm='{"image": 16, "video": 1}'
VLLM_LIMIT_MM="${VLLM_LIMIT_MM:-$default_limit_mm}"
# How the server's media loader reads a video: num_frames -1 keeps every
# frame of the data-URL MP4 a clip is sent as, instead of subsampling it on
# the way in. Which frames the model then sees is a per-request decision
# (mm_processor_kwargs.do_sample_frames, see scripts/_wiring.py); this only
# makes sure the loader hands the processor the whole clip to decide over.
default_media_io='{"video": {"num_frames": -1}}'
VLLM_MEDIA_IO="${VLLM_MEDIA_IO:-$default_media_io}"
VLLM_DTYPE="${VLLM_DTYPE:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
# Not a flag: vLLM reads this one from the environment. FlashInfer's sampling
# kernel is JIT-compiled at startup and needs nvcc; without a CUDA toolkit the
# server fails with "Could not find nvcc". Off by default, which makes vLLM
# sample through its PyTorch path instead; on a machine with the toolkit,
# VLLM_USE_FLASHINFER_SAMPLER=1 enables the kernel.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Optional: a quantization method for a quantized checkpoint. The published
# results served Qwen3.5-9B in BF16, with no --quantization flag.
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"

cmd=(
  vllm serve "$VLLM_MODEL"
  --served-model-name "$VLLM_SERVED_MODEL_NAME"
  --host "$VLLM_HOST"
  --port "$VLLM_PORT"
)
if [ -n "$VLLM_QUANTIZATION" ]; then
  cmd+=(--quantization "$VLLM_QUANTIZATION")
fi
cmd+=(
  # Thinking is turned off per request, but if a <think> block ever leaks
  # the parser keeps it out of message.content, where the ReAct loop reads.
  --reasoning-parser qwen3
  # Default is one image per prompt and a caption clip carries ten frames.
  --limit-mm-per-prompt "$VLLM_LIMIT_MM"
  --media-io-kwargs "$VLLM_MEDIA_IO"
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
)
# No speculative decoding (MTP) and no --enable-auto-tool-choice or
# --tool-call-parser: the agent drives a textual ReAct loop, not native tool
# calls.

if [ -n "$VLLM_DTYPE" ]; then
  cmd+=(--dtype "$VLLM_DTYPE")
fi
# Appended verbatim, word-split on purpose: this is where a
# --tensor-parallel-size, a --media-io-kwargs or an --hf-overrides goes.
# shellcheck disable=SC2206
cmd+=(${VLLM_EXTRA_ARGS})

# Settings that are not argv get a line of their own, so that --dry-run shows
# the whole of what the server is started with.
echo "# env: VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
if [ -n "$VLLM_QUANTIZATION" ]; then
  echo "# quantization: $VLLM_QUANTIZATION"
else
  echo "# quantization: none (checkpoint served unquantized)"
fi
# Printed shell-quoted, so the line can be copied and run as it stands.
echo "+ ${cmd[*]@Q}"
if [ "$dry_run" -eq 1 ]; then
  exit 0
fi
exec "${cmd[@]}"
