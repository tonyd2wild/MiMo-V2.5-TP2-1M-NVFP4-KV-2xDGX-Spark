# Bluey/Reddie Deployment Notes

This is the deploy target for the 2-node MiMo V2.5 NVFP4-KV stack.

## Current External Check

From the MacBook Pro on 2026-06-30, the visible endpoint check showed:

```text
100.92.77.51:8000 -> DeepSeek/Step labels, not MiMo
100.113.138.96:8000 -> no direct vLLM API response
```

So MiMo V2.5 was not externally visible on the Bluey/Reddie pair at that time.
SSH required a fresh Tailscale SSH auth check before the hosts could be
inspected directly.

## Serving Profile To Bring Up First

Keep the current best 1M/C8 profile:

```bash
MAX_MODEL_LEN=1000000
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=2048
BLOCK_SIZE=64
GPU_MEMORY_UTILIZATION=0.84
MTP_SPEC_TOKENS=1
VLLM_MIMO_MTP1_GREEDY_FAST=1
ENFORCE_EAGER=1
```

Keep the DeepSeek-learned harness safety rules:

```bash
--generation-config vllm
--override-generation-config '{"temperature":0,"top_p":0.95,"repetition_penalty":1.08}'
--default-chat-template-kwargs '{"enable_thinking":false}'
```

This is not a drop-in DeepSeek DSpark transplant. MiMo uses vLLM's
`method:"mtp"` path plus the MiMo-specific MTP and DiffKV patches in this repo.
The closest speedup analogue to DSpark is improving MiMo's MTP proposer,
metadata, and target-token fast paths.

## Bring-Up Order

On both nodes:

```bash
bash recipe/run-container.sh
bash recipe/apply-mods.sh vllm_mimo_tp2
```

Then start Ray:

```bash
# Head
source recipe/env.sh
export HEAD_ROCE_IP=<bluey-roce-ip>
bash recipe/run-head.sh

# Worker
source recipe/env.sh
export HEAD_ROCE_IP=<bluey-roce-ip>
export WORKER_ROCE_IP=<reddie-roce-ip>
bash recipe/run-worker.sh
```

Launch vLLM from the head after `ray status` shows two GPUs:

```bash
source recipe/env.sh
export HEAD_ROCE_IP=<bluey-roce-ip>
export MODEL_PATH=/root/.cache/huggingface/hub/models--lukealonso--MiMo-V2.5-NVFP4/snapshots/a147dd04d6cf861e43b2d783dcde23b53ab7ee68
export SERVED_MODEL_NAME=MiMo-V2.5-NVFP4
bash recipe/launch.sh
```

## Validation Gate

Direct vLLM first:

```bash
curl -fsS http://<bluey-tailscale-ip>:8000/v1/models
```

Then:

```bash
MIMO_BASE_URL=http://<bluey-tailscale-ip>:8000/v1 \
CONCURRENCY=1,2,4,6,8 \
python3 scripts/agent_sanity_bench.py
```

Every concurrency level should show `bad_outputs: 0`.

Only after the direct endpoint is clean should Hermes/OpenClaw agents point at
the endpoint.

## Next Speed Work

The existing benchmarks show C8 aggregate speed is already strong, but C1 decode
stays around low-30 tok/s. The next useful code experiments are:

1. Sweep `VLLM_WMMA_NSPLIT=64,128,256,512` at the current 1M/C8 profile.
2. If flat, build a MiMo-MTP-aware proposer or metadata-cache path inside the
   existing `method:"mtp"` flow.
3. Treat DeepSeek DSpark as an architecture reference, not a copy/paste patch,
   because DSpark's DeepSeek-specific proposer assumes DeepSeek V4 internals.

