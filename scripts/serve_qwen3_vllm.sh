#!/usr/bin/env bash
# Serve Qwen3-8B with vLLM on a 32 GB cloud GPU.
#
# The client disables Qwen3's thinking trace through
# chat_template_kwargs={"enable_thinking": false}; this server only exposes the
# OpenAI-compatible endpoint.
#
# Usage:
#   bash scripts/serve_qwen3_vllm.sh
#   PORT=8000 GPU_MEMORY_UTILIZATION=0.88 bash scripts/serve_qwen3_vllm.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed. Run: pip install -r requirements-cloud.txt" >&2
  exit 1
fi

echo "Serving ${MODEL} as ${SERVED_MODEL_NAME} on ${HOST}:${PORT}"
exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --disable-log-requests
