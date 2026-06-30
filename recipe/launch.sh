#!/usr/bin/env bash
# MiMo-V2.5 Omni TP=2 / 1M / MTP1 / NVFP4-KV — launch (run on the HEAD node, after `source env.sh`).
# Order: start Ray workers first (run-worker.sh), then the Ray head (run-head.sh), then this launch from the head.
# REQUIRES the patched vLLM container/mod stack (see README Credits) — stock vLLM will reject NVFP4 KV / OOM.
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH in env.sh}"
: "${SERVED_MODEL_NAME:=MiMo-V2.5-NVFP4}"
# These honor env.sh (so the 500K OOM-fallback below actually takes effect). Defaults = the validated 1M config.
: "${LOAD_FORMAT:=safetensors}"
: "${MAX_MODEL_LEN:=1000000}"
: "${MAX_NUM_BATCHED_TOKENS:=4096}"
: "${MAX_NUM_SEQS:=4}"
: "${GPU_MEMORY_UTILIZATION:=0.84}"
: "${MTP_SPEC_TOKENS:=1}"
: "${ENFORCE_EAGER:=1}"
: "${DEFAULT_TEMPERATURE:=0}"
: "${DEFAULT_TOP_P:=0.95}"
: "${REPETITION_PENALTY:=1.08}"
: "${TENSOR_PARALLEL_SIZE:=2}"
: "${PIPELINE_PARALLEL_SIZE:=1}"
: "${USE_LOCAL_ARGMAX_REDUCTION:=0}"
: "${BLOCK_SIZE:=64}"
if [ -n "${HEAD_ROCE_IP:-}" ]; then
  VLLM_HOST_IP="${HEAD_ROCE_IP}"
else
  : "${VLLM_HOST_IP:=}"
fi
if [ -z "${VLLM_HOST_IP:-}" ]; then
  echo "ERROR: set VLLM_HOST_IP or HEAD_ROCE_IP to the head node RoCE/cluster IP" >&2
  exit 2
fi
export VLLM_HOST_IP
EAGER_FLAG=""; [ "${ENFORCE_EAGER}" = "1" ] && EAGER_FLAG="--enforce-eager"
SPECULATIVE_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_SPEC_TOKENS}}"
if [ "${USE_LOCAL_ARGMAX_REDUCTION}" = "1" ]; then
  SPECULATIVE_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_SPEC_TOKENS},\"use_local_argmax_reduction\":true}"
fi
GENERATION_CONFIG="{\"temperature\":${DEFAULT_TEMPERATURE},\"top_p\":${DEFAULT_TOP_P},\"repetition_penalty\":${REPETITION_PENALTY}}"

# OOM fallback: prove 500K first, then climb — export before running this script, e.g.
#   MAX_MODEL_LEN=500000 MAX_NUM_SEQS=2 GPU_MEMORY_UTILIZATION=0.82 bash launch.sh
# Debug-only video isolate: change --limit-mm-per-prompt below to '{"image":4,"video":0,"audio":1}'.
vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --dtype auto \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
  --distributed-executor-backend ray \
  --load-format "${LOAD_FORMAT}" \
  --hf-overrides '{"architectures":["MiMoV2OmniForCausalLM"]}' \
  --limit-mm-per-prompt '{"image":4,"video":1,"audio":1}' \
  --mm-encoder-tp-mode data \
  --attention-backend triton_attn_diffkv \
  --kv-cache-dtype nvfp4 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --block-size "${BLOCK_SIZE}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --no-async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser mimo \
  --reasoning-parser mimo \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --generation-config vllm \
  --override-generation-config "${GENERATION_CONFIG}" \
  --speculative-config "${SPECULATIVE_CONFIG}" \
  ${EAGER_FLAG} \
  --host 0.0.0.0 \
  --port 8000

# NOTE on thinking: --default-chat-template-kwargs '{"enable_thinking":false}' serves with reasoning OFF
# by default (snappier, tool-friendly). To benchmark / serve thinking ON, drop that flag (or set true)
# and relaunch — on this build the per-request enable_thinking override does NOT flip a forced-off default.

# Smoke test (from any node):
#   curl http://<head-ip>:8000/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"model":"MiMo-V2.5-NVFP4","messages":[{"role":"user","content":"Reply exactly: OK 1M MTP1"}],
#          "max_tokens":16,"temperature":0,"repetition_penalty":1.08,
#          "chat_template_kwargs":{"enable_thinking":false}}'
#   Expect: "OK 1M MTP1"
