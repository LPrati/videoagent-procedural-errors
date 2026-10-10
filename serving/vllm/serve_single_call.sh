#!/usr/bin/env bash
# Serve Qwen3.5-9B for the two single-call systems (scripts/run_single_call.py).
#
# The whole-video system sends one long video per request, up to 56,000
# visual tokens, so the server is started with a different configuration
# from the agent's (serve.sh defaults):
#   * a 77,824-token context and at most 2 concurrent sequences;
#   * one video and no image per request;
#   * the video loader keeps every frame of the MP4 it receives (the frames
#     are sampled by run_single_call.py, at most 4 per second);
#   * the video processor accepts up to 64,000 visual tokens per item
#     (longest_edge = 64,000 x 2,048 pixels); the requests stay at 56,000;
#   * --gpu-memory-utilization 0.90, which leaves headroom for the peak
#     memory of the longest videos on a 32 GB GPU.
# The captions system is text only and runs against the same server.
#
# Usage, inside the vLLM environment:
#     ./serve_single_call.sh
#     ./serve_single_call.sh --dry-run
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
visual_cap=64000

export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-9B}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-77824}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
default_limit_mm='{"image": 0, "video": 1}'
export VLLM_LIMIT_MM="${VLLM_LIMIT_MM:-$default_limit_mm}"
default_media_io='{"video": {"num_frames": -1, "fps": 4, "max_frames": 8192}}'
export VLLM_MEDIA_IO="${VLLM_MEDIA_IO:-$default_media_io}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---mm-processor-kwargs {\"size\":{\"longest_edge\":$((visual_cap * 2048)),\"shortest_edge\":4096}}}"

exec "$here/serve.sh" "$@"
