#!/usr/bin/env bash
# MiMo-V2.5 Omni TP=2 / 1M / MTP1 / NVFP4-KV — launch (run on the HEAD node, after `source env.sh`).
# Order: start Ray workers first (run-worker.sh), then the Ray head (run-head.sh), then this launch from the head.
# REQUIRES the patched vLLM container/mod stack (see README Credits) — stock vLLM will reject NVFP4 KV / OOM.
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH in env.sh}"
: "${SERVED_MODEL_NAME:=MiMo-V2.5-NVFP4}"

vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --dtype auto \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend ray \
  --load-format safetensors \
  --hf-overrides '{"architectures":["MiMoV2OmniForCausalLM"]}' \
  --limit-mm-per-prompt '{"image":4,"video":1,"audio":1}' \
  --mm-encoder-tp-mode data \
  --attention-backend triton_attn_diffkv \
  --kv-cache-dtype nvfp4 \
  --gpu-memory-utilization 0.84 \
  --max-model-len 1000000 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 4 \
  --block-size 32 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --no-async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser mimo \
  --reasoning-parser mimo \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000

# NOTE on thinking: --default-chat-template-kwargs '{"enable_thinking":false}' serves with reasoning OFF
# by default (snappier, tool-friendly). To benchmark / serve thinking ON, drop that flag (or set true)
# and relaunch — on this build the per-request enable_thinking override does NOT flip a forced-off default.

# Smoke test (from any node):
#   curl http://<head-ip>:8000/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"model":"MiMo-V2.5-NVFP4","messages":[{"role":"user","content":"Reply exactly: OK 1M MTP1"}],
#          "max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
#   Expect: "OK 1M MTP1"
