# MiMo V2.5 NVFP4-KV Concurrency Checkpoint - 2026-06-29

## Result

We relaunched the MiMo V2.5 NVFP4-KV 1M server with concurrency enabled and ran a static concurrency sweep at **2, 4, 6, and 8 simultaneous streams**.

The server is still configured for **1M max context**, but this sweep used short normal benchmark prompts, not eight full 1M-token prompts.

## Live Config

```bash
MODEL_PATH=/root/.cache/huggingface/hub/models--lukealonso--MiMo-V2.5-NVFP4/snapshots/a147dd04d6cf861e43b2d783dcde23b53ab7ee68
SERVED_MODEL_NAME=MiMo-V2.5-NVFP4
HEAD_ROCE_IP=192.168.192.3
VLLM_HOST_IP=192.168.192.3
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=1
MAX_MODEL_LEN=1000000
GPU_MEMORY_UTILIZATION=0.84
MTP_SPEC_TOKENS=1
USE_LOCAL_ARGMAX_REDUCTION=0
VLLM_MIMO_MTP1_GREEDY_FAST=1
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=2048
BLOCK_SIZE=64
ENFORCE_EAGER=1
VLLM_WMMA_QLEN_CAP=3
VLLM_WMMA_DECODE=1
NCCL_SOCKET_IFNAME=enp1s0f0np0
GLOO_SOCKET_IFNAME=enp1s0f0np0
NCCL_SOCKET_FAMILY=AF_INET
NCCL_IB_HCA=rocep1s0f0
NCCL_NET_GDR_LEVEL=LOC
CUDA_DEVICE_MAX_CONNECTIONS=1
```

Launch log confirms:

```text
max_model_len: 1000000
max_num_seqs: 8
max_num_batched_tokens: 2048
block_size: 64
kv_cache_dtype: nvfp4
speculative_config: {"method":"mtp","num_speculative_tokens":1}
```

## KV Pool

From the C8 relaunch log:

```text
Available KV cache memory: 8.25 GiB
GPU KV cache size: 2,171,757 tokens
Maximum concurrency for 1,000,000 tokens per request: 2.17x
```

Important interpretation:

- The server can expose `MAX_NUM_SEQS=8`.
- The total KV pool is about **2.17M tokens**.
- That means it can fit roughly **2 full 1M-token requests** by KV capacity.
- C4/C6/C8 are valid for shorter or moderate-context requests whose combined KV stays under the pool.
- C8 does **not** mean eight simultaneous 1M-context requests.

## Benchmark Method

Benchmark script:

```bash
python3 /tmp/mimo_bench_concurrent.py \
  http://127.0.0.1:8000 \
  2,4,6,8 \
  MiMo-V2.5-NVFP4
```

Details:

- Static simultaneous-request benchmark.
- `max_tokens=256`.
- Server-side aggregate tok/s from `/metrics` deltas.
- Acceptance from:
  - `vllm:spec_decode_num_accepted_tokens_total`
  - `vllm:spec_decode_num_draft_tokens_total`
- Each concurrency level ran twice; table below uses the best aggregate result.

## Results

| Concurrency | Best aggregate tok/s | Aggregate-derived tok/s per stream | Acceptance on best run |
|---:|---:|---:|---:|
| 2 | 60.2 | 30.1 | 0.829 |
| 4 | 94.7 | 23.7 | 0.837 |
| 6 | 141.8 | 23.6 | 0.832 |
| 8 | 184.1 | 23.0 | 0.867 |

Raw run output:

```text
=== concurrency 2 ===
  concurrency=2: per-stream decode [12.6, 13.9] tok/s | server-agg 44.9 tok/s | acceptance 0.835
  concurrency=2: per-stream decode [17.2, 17.2] tok/s | server-agg 60.2 tok/s | acceptance 0.829

=== concurrency 4 ===
  concurrency=4: per-stream decode [13.3, 13.7, 13.7, 13.7] tok/s | server-agg 94.7 tok/s | acceptance 0.837
  concurrency=4: per-stream decode [13.2, 13.6, 13.6, 13.6] tok/s | server-agg 94.4 tok/s | acceptance 0.837

=== concurrency 6 ===
  concurrency=6: per-stream decode [12.7, 12.8, 12.8, 12.8, 12.8, 12.8] tok/s | server-agg 137.0 tok/s | acceptance 0.921
  concurrency=6: per-stream decode [13.4, 13.7, 13.4, 13.7, 13.7, 13.4] tok/s | server-agg 141.8 tok/s | acceptance 0.832

=== concurrency 8 ===
  concurrency=8: per-stream decode [12.6, 12.8, 12.8, 12.8, 12.8, 12.8, 12.8, 12.8] tok/s | server-agg 177.2 tok/s | acceptance 0.842
  concurrency=8: per-stream decode [13.0, 12.7, 12.7, 13.0, 13.0, 13.0, 13.0, 13.0] tok/s | server-agg 184.1 tok/s | acceptance 0.867
```

## Health After Test

The server remained up after the sweep:

```text
/v1/models: READY
vllm:num_requests_running: 0
vllm:num_requests_waiting: 0
```

Final cumulative metrics after the run:

```text
vllm:generation_tokens_total: 10752
vllm:spec_decode_num_draft_tokens_total: 5792
vllm:spec_decode_num_accepted_tokens_total: 4941
```

Overall post-launch acceptance across warmup plus sweep:

```text
4941 / 5792 = 0.853
```

## Takeaway

This checkpoint is a real concurrency win for the MiMo 1M NVFP4-KV server:

- `MAX_NUM_SEQS=8` boots.
- 1M max context remains enabled.
- KV pool remains about **2.17M tokens**.
- C8 short-request serving reaches about **184 tok/s aggregate**.
- Acceptance stays healthy, roughly **0.83-0.87** on the best C2/C4/C6/C8 runs.

The remaining caveat is KV capacity: full 1M-token requests are limited to about two concurrent requests by the 2.17M-token pool. Higher concurrency is for shorter or moderate-context workloads.
