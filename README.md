# MiMo-V2.5 Omni · TP=2 · **1M context** · NVFP4 KV on 2× DGX Spark

> 🔀 This is the **2-Spark (TP=2)** build. Running **3 Sparks**? → [MiMo-V2.5-TP3-NVFP4-KV-3xDGX-Spark](https://github.com/tonyd2wild/MiMo-V2.5-TP3-NVFP4-KV-3xDGX-Spark)

Running [`lukealonso/MiMo-V2.5-NVFP4`](https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4) (Omni: text + image + video + audio) tensor-parallel across **two NVIDIA DGX Spark (GB10)** boxes, with **4-bit `nvfp4` KV cache** + MTP speculative decoding — serving a **1,000,000-token** context with a **2.17M-token KV pool**.

This is the **2-node** sibling of the 3-node build ([MiMo-V2.5-TP3-NVFP4-KV-3xDGX-Spark](https://github.com/tonyd2wild/MiMo-V2.5-TP3-NVFP4-KV-3xDGX-Spark)). Two Sparks instead of three — so it pairs cleanly with another 2-node model on the same fleet (e.g. DeepSeek-V4 TP=2 on the other pair).

> ⚠️ **This repo is the launch config + env + the patch mods — but it runs on top of a specific patched vLLM dev build, not stock vLLM.** The 6 required mods are now **vendored in [`recipe/mods/`](recipe/mods)** (apply them with [`recipe/apply-mods.sh`](recipe/apply-mods.sh) — see [Runtime stack used](#runtime-stack-used)). Stock `pip install vllm` will still **reject NVFP4 KV or OOM** — the mods patch a vLLM dev build (`0.21.1rc1.dev85+gd87ee1893`), they are not a stock-vLLM plugin. The `nvfp4` KV path depends on the DiffKV attention backend in `recipe/mods/nvfp4-kv-diffkv/`. See [Avoiding OOM](#avoiding-oom--full-reproduction) before you run it.

---

## Results at a glance

| | Value |
|---|---|
| Model | `lukealonso/MiMo-V2.5-NVFP4` (Omni) |
| Nodes | 2× DGX Spark / GB10 |
| Parallelism | TP=2, PP=1 |
| **Max context / request** | **1,000,000** (verified, benched) |
| KV cache dtype | **`nvfp4`** (4-bit) |
| GPU KV pool | **2,171,757 tokens** |
| Speculative decoding | MTP, `num_speculative_tokens=1` |
| Loader | `safetensors` |
| GPU mem util | 0.84 |
| `max_num_seqs` | 8 in the latest concurrency checkpoint |
| Single-stream decode speed | **~31.9–32.1 tok/s** |
| C8 short-request aggregate | **184.1 tok/s** |

Experimental speed-tuning notes, including MTP/scheduler dead ends and CUDA graph
tradeoffs, are in [`benchmarks/speed-tuning-notes-20260629.md`](benchmarks/speed-tuning-notes-20260629.md).

### Latest checkpoint: 1M + C8 + basic Omni

Current best single-stream checkpoint:

| Output tokens | server tok/s | acceptance |
|---:|---:|---:|
| 512 | 32.13 | 0.928 |
| 1024 | 31.87 | 0.927 |
| 2048 | 31.89 | 0.926 |

Current C8 relaunch:

```text
MAX_MODEL_LEN=1000000
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=2048
BLOCK_SIZE=64
MTP_SPEC_TOKENS=1
VLLM_MIMO_MTP1_GREEDY_FAST=1
```

Boot log:

```text
GPU KV cache size: 2,171,757 tokens
Maximum concurrency for 1,000,000 tokens per request: 2.17x
```

Static short-request concurrency:

| concurrency | aggregate tok/s | aggregate-derived tok/s per stream | acceptance |
|---:|---:|---:|---:|
| 2 | 60.2 | 30.1 | 0.829 |
| 4 | 94.7 | 23.7 | 0.837 |
| 6 | 141.8 | 23.6 | 0.832 |
| 8 | 184.1 | 23.0 | 0.867 |

Important: C8 here means eight short or moderate-context requests on a
1M-capable server. It does **not** mean eight simultaneous full-1M requests; the
2.17M-token KV pool fits roughly two full-1M requests by memory.

Basic Omni smoke validation passed through the live OpenAI-compatible API:

| modality | result |
|---|---|
| Image | correctly identified red/blue and center circle |
| Audio | correctly identified a generated sine tone as a simple tone |
| Video | correctly identified red/blue and a blue center square |

Checkpoint evidence:

- [`benchmarks/mimo-v25-concurrency-c2-c8-checkpoint-20260629.md`](benchmarks/mimo-v25-concurrency-c2-c8-checkpoint-20260629.md)
- [`benchmarks/mimo-v25-omni-validation-20260629.md`](benchmarks/mimo-v25-omni-validation-20260629.md)
- [`benchmarks/mimo_omni_validation_results_20260629.json`](benchmarks/mimo_omni_validation_results_20260629.json)

### 69-scenario tool-eval (2Wild model-eval harness)

| Run | quality | pass / partial / fail | deployability | tok/s |
|---|---|---|---|---|
| **Thinking OFF** ⭐ | **97.8** | 66 / 3 / 0 | 96.7 | ~30 |
| **Thinking ON** | 90.6 | 61 / 3 / 5 | 89.4 | ~30 raw* |

*Thinking-ON's *effective* throughput is ~2× slower end-to-end from the extra reasoning tokens.

**Quality is identical at 500K and 1M context (97.8 both)** — the 1M ceiling costs nothing on quality. Raw results in [`benchmarks/`](benchmarks/) (thinking-off @500K, thinking-off @1M, thinking-on).

**Verdict: for tool/agent work, thinking-OFF wins outright** — higher quality (97.8 vs 90.6), zero failures (vs 5), ~2× lower latency, ~35% fewer tokens. Thinking-ON over-reasons itself into mistakes on tool tasks. Run agents **thinking-OFF**; reserve thinking-ON for open-ended reasoning.

---

## Context + concurrency: how the shared KV pool works

The **2.17M-token KV pool is shared** across all in-flight requests. `max_model_len` caps any *single* request; `max_num_seqs` caps how many run at once; the pool is the real budget.

- **One deep request → up to the full 1M tokens** (~2.17× — fits ~2 full 1M requests).
- **Many moderate requests → high concurrency.** e.g. **4 agents × 100K = 400K** = ~20% of the pool → all 4 run in parallel with lots of headroom (~20 concurrent 100K agents before KV bites).
- The only thing you can't do: **4 agents all at a full 1M simultaneously** (that'd need 4M) → vLLM just queues the overflow until room frees.

**Tuning `max_num_seqs`:** set it to **2** if your workload is single huge (500K–1M) requests; set it to **4+** for multi-agent / many-moderate-context work. The latest checkpoint boots with `max_num_seqs=8`, `gpu-memory-utilization=0.84`, and a 2.17M-token KV pool.

## The config that works

Key flags (full env + launch in [`recipe/`](recipe/)):

```bash
--tensor-parallel-size 2 --pipeline-parallel-size 1 \
--kv-cache-dtype nvfp4 --attention-backend triton_attn_diffkv \
--max-model-len 1000000 --max-num-seqs 8 \
--gpu-memory-utilization 0.84 \
--speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
--load-format safetensors --enforce-eager \
--hf-overrides '{"architectures":["MiMoV2OmniForCausalLM"]}' \
--tool-call-parser mimo --reasoning-parser mimo
```

`recipe/launch.sh` defaults to `TENSOR_PARALLEL_SIZE=2` and
`PIPELINE_PARALLEL_SIZE=1`. Override those only for topology diagnostics, e.g.
`TENSOR_PARALLEL_SIZE=1 PIPELINE_PARALLEL_SIZE=2`, then compare against the
validated TP=2 result before publishing it as a serving config.

## Reproduce from scratch (start here)

> The recipe files (`launch.sh`, `run-head.sh`, etc.) all run **inside a container** — so the very first step is getting that container onto **both** Sparks. This is the step people miss.

### Prerequisites
- **2× NVIDIA DGX Spark (GB10)**, each with its ConnectX RoCE NIC, on a direct-cable or switched RoCE link between them.
- Docker with GPU access (`--gpus all` / NVIDIA container toolkit) on both.
- The model weights cached on **both** nodes (TP=2 loads shards on each).
- NCCL 2.30u1-class; the validated host build was CUDA 13.2 (arm64/sbsa); driver 580.x.

### 1. Model weights — on BOTH nodes
```bash
hf download lukealonso/MiMo-V2.5-NVFP4 \
  --revision a147dd04d6cf861e43b2d783dcde23b53ab7ee68
# export HF_TOKEN=... first if the repo is gated. Lands in ~/.cache/huggingface (mounted into the container).
```

### 2. The container — on BOTH nodes (this is the missing piece)
The patched vLLM **dev-build** image is **published** — `run-container.sh` defaults to it and Docker auto-pulls on first run:
```
ghcr.io/tonyd2wild/mimo-v2.5-tp2-1m-nvfp4kv:20260620   (~20 GB; arm64/GB10)
```
Start the container on **EACH** Spark (head AND worker — the worker is not remote-driven; Ray spawns the TP rank inside the worker's own container):
```bash
bash recipe/run-container.sh        # pulls + runs the published image
# (override with IMAGE=<your-own-image> if you built your own)
```
It runs: `docker run -d --gpus all --network host --ipc host --shm-size 16g --device /dev/infiniband --ulimit memlock=-1 -v ~/.cache/huggingface:/root/.cache/huggingface ... sleep infinity`.

What the image is: a vLLM **dev build** (`0.21.1rc1.dev85+gd87ee1893`) compiled for GB10 (`TORCH_CUDA_ARCH_LIST=12.1a`) on a **CUDA 13.2 arm64/sbsa** base + `torch 2.11.0` + NCCL + `ray`/`fastsafetensors`. The mods are **NOT baked in** — they're applied at runtime in step 3. (It's a from-wheels build; no Dockerfile is shipped, so use the published image rather than rebuilding.)

### 3. Apply the mods — on BOTH nodes, after the container is up
```bash
bash recipe/apply-mods.sh vllm_mimo_tp2
```
⚠️ `fix-mimo-v2-vllm` curls vLLM **PR #41797** at apply-time → the container needs **outbound GitHub access** during this step (vendor `41797.diff` locally if your nodes are network-restricted).

### 4. Bring up Ray + vLLM (the runbook)
```bash
# find your RoCE IP + HCA on each node first:  ibdev2netdev ; ip -4 addr show <iface>
# export NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME/NCCL_IB_HCA if they differ from the DGX Spark defaults in env.sh
# HEAD container:    source recipe/env.sh && export HEAD_ROCE_IP=<head> && bash recipe/run-head.sh
# WORKER container:  source recipe/env.sh && export HEAD_ROCE_IP=<head> WORKER_ROCE_IP=<worker> && bash recipe/run-worker.sh
# HEAD: wait for 2 GPUs →  until ray status 2>/dev/null | grep -qE '2\.0/2\.0 GPU'; do sleep 2; done
# HEAD: launch vLLM    →  source recipe/env.sh && export HEAD_ROCE_IP=<head> && bash recipe/launch.sh
```
Containers come up worker-then-head; Ray comes up head-then-worker; vLLM launches only on the head once Ray sees 2 GPUs. `launch.sh` honors `env.sh`, so the 500K→1M fallback below works by just exporting the vars before step 4.

## Hard-won lessons (what failed first)

1. **GPU mem util is tight.** 500K barely fit at 0.80 (needed 0.82); the full **1M + `max_num_seqs=4`** wants **0.84** so a 1M request still fits alongside the extra seq slots.
2. **MTP1 beats MTP2 here.** At 96K, MTP2 gave the same quick-decode speed as MTP1 but cut the KV pool nearly in half. Use **MTP1**.
3. **`safetensors`, not InstantTensor.** The MTP + NVFP4-KV path wedged under InstantTensor on the drafter load. Safetensors is slower to load but stable.
4. **Worker-first Ray startup**, then head, then launch vLLM from the head.

## Avoiding OOM / full reproduction

The 1M headline is real (live: `GPU KV cache size: 1,970,104 tokens`, `1.97x` concurrency), but at `gpu-memory-utilization 0.84` the setup is **not forgiving**. Most OOM reports are NOT the headline flags — they're a different, incomplete environment. To reproduce without OOM:

1. **Use the patched container/mod stack** (non-optional): `nvfp4-kv-diffkv`, `fix-mimo-v2-vllm`, `fix-modelopt-mixed-mxfp8`, `ray-keep-node-nccl-hca`, `fix-prometheus-instrumentator-router`, `drop-caches`. Stock vLLM will fail/OOM.
2. **Cap the Ray plasma object store on EVERY node** — `--object-store-memory=1073741824`. Uncapped Ray reserves a huge store and steals unified memory → OOM on weight load/profile. Use the included [`recipe/run-head.sh`](recipe/run-head.sh) + [`recipe/run-worker.sh`](recipe/run-worker.sh).
3. **Worker-first clean start:** stop old containers/Ray → start worker container → start head container → apply mods on both → `run-head.sh` (head Ray) → `run-worker.sh` (worker joins) → wait until `ray status` shows **2 GPUs** → `source env.sh && bash launch.sh`.
4. **Set the memory env vars** (in `env.sh`): `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `RAY_memory_monitor_refresh_ms=0` are the two that most prevent the OOM report.
5. **No stale processes** — kill any leftover vLLM/Ray containers first; they hold unified memory.

**If 1M still OOMs — boot the 500K fallback first, then climb:**
```bash
# Step 1: prove 500K serves
MAX_MODEL_LEN=500000   MAX_NUM_SEQS=2   GPU_MEMORY_UTILIZATION=0.82
# Step 2: only after 500K is live, go 1M
MAX_MODEL_LEN=1000000  MAX_NUM_SEQS=4   GPU_MEMORY_UTILIZATION=0.84
```

### Check Ray didn't bind a link-local interface (most common real crash)
If startup logs show a worker like `RayWorkerWrapper ... ip=169.254.x.x`, Ray bound to a **link-local** interface instead of your RoCE/cluster IP — this crashes/OOMs. Fix: export the host IP before starting Ray (the included `run-head.sh`/`run-worker.sh` already do this):
```bash
export VLLM_HOST_IP=<this-node-roce-ip>
ray start --node-ip-address=<this-node-roce-ip> ...
```

### If it dies/vanishes during profiling — check OOM-kill + swap
```bash
docker inspect <container> --format '{{.State.OOMKilled}}'
dmesg -T | grep -Ei 'killed|oom|out of memory' | tail -n 80
free -h     # if swap is ~100% used and free RAM is tiny, the node is thrashing
```
Clean-start both nodes, kill stale Ray/vLLM, and confirm the Ray object store is capped at `1073741824` on EVERY node.

### Debug-only: isolate video-encoder profiling
If the process dies right after `Encoder cache will be initialized ... profiled with 1 video items ...`, the video max-feature profile may be pushing a low-headroom node over. Boot once with video disabled to isolate:
```bash
--limit-mm-per-prompt '{"image":4,"video":0,"audio":1}'
```
If that boots, text/image/audio is fine and it's specifically video profiling memory. (Debug step — not the full Omni recipe.)

## Repro checklist

1. Worker-first Ray; `safetensors`; Omni arch; `--kv-cache-dtype nvfp4` + `triton_attn_diffkv`; MTP1.
2. `max_model_len=1000000`, `max_num_seqs=8`, `gpu-memory-utilization=0.84`.
3. Confirm the startup log shows a `GPU KV cache size` ≥ 1,000,000 tokens (1M fits).
4. Smoke (`"Reply exactly: OK"`) → then run the 69-eval.

## Files

```
.
├── README.md
├── recipe/{env.sh, launch.sh, run-head.sh, run-worker.sh}
│   ├── run-container.sh                  # docker run the patched container — RUN ON BOTH NODES first
│   ├── apply-mods.sh                     # docker-cp + run each mod into the container (both nodes)
│   └── mods/                             # the 6 patch mods (VENDORED — apply before launch)
│       ├── drop-caches/run.sh
│       ├── ray-keep-node-nccl-hca/run.sh
│       ├── fix-prometheus-instrumentator-router/run.sh
│       ├── fix-mimo-v2-vllm/run.sh       # fetches vLLM PR #41797 + MiMo-V2 fixes
│       ├── fix-modelopt-mixed-mxfp8/run.sh
│       └── nvfp4-kv-diffkv/{run.sh, triton_attn_diffkv.py, triton_unified_attention_diffkv.py, wmma_decode.py}
└── benchmarks/
    ├── thinking-off-1M-69eval.{md,json}   # 97.8 quality @ 1M ctx
    ├── thinking-off-69eval.{md,json}      # 97.8 quality @ 500K ctx
    ├── thinking-on-69eval.{md,json}       # 90.6 quality (thinking on)
    ├── speed-tuning-notes-20260629.md
    ├── mimo-v25-concurrency-c2-c8-checkpoint-20260629.md
    ├── mimo-v25-omni-validation-20260629.md
    └── mimo_omni_validation_results_20260629.json
```

The 6 patch **mods are vendored in [`recipe/mods/`](recipe/mods)** and applied at runtime via [`recipe/apply-mods.sh`](recipe/apply-mods.sh) (see [Runtime stack used](#runtime-stack-used) for what each does + licensing).

## Runtime stack used

This repo documents the launch config + reproducibility notes, but the successful 1M run did **not** use stock vLLM.

**What the stack actually IS (positive ID):** vLLM **`0.21.1rc1.dev85+gd87ee1893`** (commit `d87ee1893`, ~2026-05-18, CUDA 13.2 / `cu132`; torch 2.11.0+cu130) — a **dev build, NOT a released pip wheel** — **+ the 6 local-patch mods below + Ray** for the 2-node split. That's the whole runtime. Because it's a dev build plus patches, **stock `pip install vllm` will NOT have** the NVFP4-KV / DiffKV / MiMoV2Omni code paths and will reject `--kv-cache-dtype nvfp4`, `--attention-backend triton_attn_diffkv`, and the `MiMoV2OmniForCausalLM` arch override.

**Container:** a custom vLLM **dev-build** image (local tag `vllm-mimo-omni-mtp2-1m-audio-exp:20260620`, ~19.9 GB; CUDA 13.2 arm64/sbsa + torch 2.11.0 + the vLLM dev wheel + NCCL + ray/fastsafetensors). The mods are **NOT baked in** — they're applied at runtime by `apply-mods.sh` (so the image is just the dev-build base). **Published + public:** `docker pull ghcr.io/tonyd2wild/mimo-v2.5-tp2-1m-nvfp4kv:20260620` (run-container.sh defaults to it). (The `mtp2` in the local tag was historical — the final recipe is **MTP1**.)

**Lineage:** NOT launched through `eugr/spark-vllm-docker`, and NOT a direct run of `HeNryous/mimo-spark-optimized`. Those informed the Spark/RoCE + TP=2 debugging only.

**Ray is REQUIRED** for this 2-node TP=2. vLLM's `mp`/multiprocessing executor is **single-host only** — cross-node tensor-parallel must use `--distributed-executor-backend ray`. Running it without Ray was not attempted and is not expected to work for a 2-box split (a single-node multiproc backend can't span two physical Sparks).

**Required mods** — these are **local patch scripts (each a `run.sh` applied into the container), NOT published packages**. They patch the dev-build vLLM/deps at container start:
- `nvfp4-kv-diffkv` — enables `--kv-cache-dtype nvfp4` + `--attention-backend triton_attn_diffkv` (the core unlock)
- `fix-mimo-v2-vllm` — adds the `MiMoV2OmniForCausalLM` architecture
- `fix-modelopt-mixed-mxfp8` — ModelOpt mixed-precision fix
- `ray-keep-node-nccl-hca` — adds `NCCL_IB_HCA` to Ray's non-carry-over env so each node keeps its own RoCE HCA (Ray won't copy the head's to workers)
- `fix-prometheus-instrumentator-router` — `prometheus_fastapi_instrumentator` router compat patch (else vLLM startup throws)
- `drop-caches` — page-cache drop before load
A container missing these is **not** the same runtime and may reject NVFP4 KV, OOM, or freeze during MTP drafter/profiling.

**The mods are vendored in [`recipe/mods/`](recipe/mods).** Apply them after the container is up, on **both** nodes:
```bash
bash recipe/apply-mods.sh <container_name>   # docker-cp's each mod in + runs its run.sh
```
Licensing of the vendored files (each carries its own SPDX header):
- `nvfp4-kv-diffkv/triton_attn_diffkv.py` + `triton_unified_attention_diffkv.py` — **Apache-2.0**, derived from the vLLM project (slim DiffKV forks; headers preserved).
- `nvfp4-kv-diffkv/wmma_decode.py` — **original work** (Apache-2.0, © 2026 LaNarde "Tony" DeAngelo / 2Wild): a custom WMMA tensor-core flash-decode kernel, ~2.3× faster than the Triton path. **Optional** (gated by `VLLM_WMMA_DECODE=1`) — the result reproduces without it on the Triton path.
- the four `run.sh` patch scripts (`drop-caches`, `ray-keep-node-nccl-hca`, `fix-prometheus-instrumentator-router`, `fix-modelopt-mixed-mxfp8`) + `fix-mimo-v2-vllm` (which fetches vLLM [PR #41797](https://github.com/vllm-project/vllm/pull/41797)) — MIT, per this repo's LICENSE.

**Launch method** (manual worker/head): clean-stop old containers+Ray → start worker container → start head container → apply mods on both → start Ray with object-store capped to 1GiB on every node → force Ray/vLLM host IPs to the RoCE/static IPs → confirm Ray sees 2 GPUs → launch vLLM on head. Use `recipe/run-worker.sh`, `recipe/run-head.sh`, `recipe/launch.sh` (don't treat `launch.sh` alone as the full recipe — it assumes the patched container + Ray cluster + mods are already correct).

**RoCE interface:** Ray and vLLM must bind to the intended RoCE/static cluster IPs — **not** `169.254.x.x` and not a management fallback (see [Avoiding OOM](#avoiding-oom--full-reproduction)).

## Claims we can / can't make

**Safe:** boots on 2× DGX Spark at TP=2 / MTP1 / NVFP4 KV; serves `max_model_len=1000000` (verified + benched at 97.8 quality); 2.17M-token KV pool; C8 short-request concurrency reaches 184.1 tok/s aggregate; basic image/audio/video inputs pass through the live OpenAI-compatible API; MTP1 > MTP2; safetensors is the stable loader; thinking-OFF beats thinking-ON for tool/agent tasks.

**Not yet:** not production-stable for many simultaneous *full-1M* agents (the pool holds ~2 full 1M requests); no production-grade multimodal claims without a broader modality eval suite.

## Credits

- **drowzeys ("Keys")** — origin of the NVFP4 KV-cache wiring this build’s nvfp4 KV path descends from ([Keys---Full-GLM-5.2-Quantrio…](https://github.com/drowzeys/Keys---Full-GLM-5.2-Quantrio-INT4-INT8-mixed-8bit-Attention-on-4-x-DGX-Spark-GB10-Cluster)).

Explicit so inspiration + upstream work are credited cleanly.

**Model & serving stack**
- [`lukealonso/MiMo-V2.5-NVFP4`](https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4) — the MiMo V2.5 NVFP4 checkpoint (the run pinned snapshot `a147dd04d6cf861e43b2d783dcde23b53ab7ee68`).
- [vLLM](https://github.com/vllm-project/vllm) — OpenAI-compatible serving, TP, speculative decoding, multimodal, Ray executor.
- [Ray](https://github.com/ray-project/ray) — multi-node placement + cluster execution.
- NVIDIA **NCCL** (RoCE transport for TP), **DGX Spark / GB10** + ConnectX networking, **FlashInfer / ModelOpt** kernels + NVFP4 quant paths.

**Community references & debugging signals**
- [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) (esp. [PR #251](https://github.com/eugr/spark-vllm-docker/pull/251) / `a3refaat`) — DGX Spark ConnectX/RoCE quirks + direct-cable networking + base mods. (Informed our debugging — but this recipe is NOT an eugr launch; see "Runtime stack used".)
- [`HeNryous/mimo-spark-optimized`](https://github.com/HeNryous/mimo-spark-optimized) — a comparison point for TP=2 Spark work (no code copied from it).
- NVIDIA Developer Forums DGX Spark / GB10 community posts; Mashie's link-behavior / cold power-drain clues; Renek & MiaAI_Lab notes on MiMo-on-Spark memory pressure + loader difficulty.
- Eval methodology adapted from `tool-eval-bench v2.0.1` by `wolttam`.

**Local integration & experiment work**
- **Tony / `tonyd2wild`** — hardware, fleet orchestration, public writeups, testing goals, repo publication.
- **Kai** — handoff docs, operational coordination, eval-board integration, independent verification.
- **Codex** — iterative TP=2 / NVFP4-KV / MTP testing, launcher/config edits, startup diagnosis, the source report.

**Local patches/mods:** `ray-keep-node-nccl-hca` · `fix-prometheus-instrumentator-router` · `fix-mimo-v2-vllm` · `fix-modelopt-mixed-mxfp8` · `nvfp4-kv-diffkv` · `drop-caches` (keep visible in any tree; document temporary vs upstreamed vs local-only).

## License

MIT (covers this repo's recipe docs + config only — not the upstream mods). See [LICENSE](LICENSE).

*Validated on 2× DGX Spark, 2026-06. 1M context boots + benches at 97.8 quality; pool holds ~2 full-1M requests (~2.17×), or ~20 concurrent 100K-context agents.*
