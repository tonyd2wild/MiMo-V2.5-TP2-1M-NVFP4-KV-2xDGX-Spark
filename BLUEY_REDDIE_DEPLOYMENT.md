# Bluey/Reddie Deployment Notes

This is the deploy target for the 2-node MiMo V2.5 NVFP4-KV stack.

## Current External Check

From the MacBook Pro after the Reddie/Bluey MiMo relaunch, the visible endpoint
check shows:

```text
100.92.77.51:8000 -> MiMo-V2.5-NVFP4, max_model_len=65536
100.113.138.96:8000 -> worker node, no public vLLM API expected
```

That is the stable 65K/C8 recovery profile, not the full 1M/C8 profile. Use it
for direct endpoint and harness stability checks while C1 speed work continues.

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

Latest stable-lane refresh from the MacBook Pro:

| concurrency | success | aggregate tok/s | bad outputs |
|---:|---:|---:|---:|
| 1 | 1/1 | 23.05 | 0 |
| 2 | 2/2 | 36.79 | 0 |
| 4 | 4/4 | 53.41 | 0 |
| 6 | 6/6 | 74.28 | 0 |

Only after the direct endpoint is clean should Hermes/OpenClaw agents point at
the endpoint.

## Next Speed Work

The existing benchmarks show C8 aggregate speed is already strong, but C1 decode
stays around low-30 tok/s. The next useful code experiments are:

1. Build a MiMo-MTP-aware proposer or metadata-cache path inside the
   existing `method:"mtp"` flow.
2. Treat DeepSeek DSpark as an architecture reference, not a copy/paste patch,
   because DSpark's DeepSeek-specific proposer assumes DeepSeek V4 internals.
3. Keep BS128/SWA WMMA work in the experimental harness lane unless a trace
   first proves the live MiMo MTP path uses the accelerated kernel and does not
   regress C1 speed.

Do not repeat the lower-NSPLIT lane as a first move: `VLLM_WMMA_NSPLIT=256` was
already slower than the default 512 split on the stable 65K MTP1 profile.
