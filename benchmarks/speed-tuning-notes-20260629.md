# Speed tuning notes - 2026-06-29

These are experimental single-stream and concurrency checks from the 2x DGX Spark
TP=2 1M NVFP4-KV runtime. They are included to make the current speed ceiling and
dead ends reproducible.

## Baseline runtime

- Model: `lukealonso/MiMo-V2.5-NVFP4`
- Runtime: patched vLLM `0.21.1rc1.dev85+gd87ee1893`
- Nodes: 2x DGX Spark, TP=2, Ray executor
- Context: `MAX_MODEL_LEN=1000000`
- KV: `--kv-cache-dtype nvfp4`, `--attention-backend triton_attn_diffkv`
- Spec decode: MTP, `MTP_SPEC_TOKENS=1`
- Default launch: `MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=4096`,
  `GPU_MEMORY_UTILIZATION=0.84`, `ENFORCE_EAGER=1`

## Verified measurements

### Eager baseline

Long single-stream runs:

| max tokens | server tok/s | acceptance |
|---:|---:|---:|
| 512 | 29.60 | 0.825 |
| 1024 | 28.89 | 0.785 |

Concurrency at 256 tokens/request:

| concurrency | aggregate tok/s | per-stream tok/s | acceptance |
|---:|---:|---:|---:|
| 1 | 29.54 | 29.54 | 0.835 |
| 2 | 42.49 | 21.24 | 0.821 |
| 4 | 67.17 | 16.79 | 0.875 |

### `MAX_NUM_BATCHED_TOKENS=8192`

This removed the scheduler warning but did not improve throughput.

| test | server tok/s / aggregate | acceptance |
|---|---:|---:|
| C1, 256 tokens | 26.77 | 0.656 |
| 512-token long run | 24.53 | 0.721 |
| 1024-token long run | 28.77 | 0.752 |
| C2 aggregate | 41.20 | 0.762 |
| C4 aggregate | 67.28 | 0.779 |

Conclusion: the 4096 scheduled-token warning is not the single-stream bottleneck.
The 8192 setting also reduced KV headroom.

### CUDA graph experiment

Config:

```bash
ENFORCE_EAGER=0
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=4096
MTP_SPEC_TOKENS=1
```

Boot succeeded and captured graphs, but the speed lift was small and KV headroom
fell materially.

| max tokens | server tok/s | acceptance |
|---:|---:|---:|
| 512 | 32.47 | 0.855 |
| 1024 | 30.31 | 0.746 |

Startup evidence:

- Graph capture finished successfully.
- Available KV cache memory was 4.75 GiB on the limiting rank and 6.94 GiB on the
  other rank.
- Maximum concurrency for a 1,000,000-token request dropped to 1.51x.

Conclusion: CUDA graph mode is a valid experiment and gives a small C1 lift, but
it is not a 45-55 tok/s solution and it costs 1M-context concurrency headroom.

### Lean single-stream CUDA graph experiment

Config:

```bash
ENFORCE_EAGER=0
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=4096
MTP_SPEC_TOKENS=1
VLLM_WMMA_MAX_BATCH=16
```

This improved graph-mode KV headroom versus `MAX_NUM_SEQS=4`, but did not improve
C1 speed.

| max tokens | server tok/s | acceptance |
|---:|---:|---:|
| 512 | 28.61 | 0.914 |
| 1024 | 28.95 | 0.743 |

Startup evidence:

- Available KV cache memory was 5.20 GiB on the limiting rank and 8.84 GiB on the
  other rank.
- Maximum concurrency for a 1,000,000-token request was 1.65x.
- Graph capture succeeded.

Conclusion: reducing `max_num_seqs` and WMMA scratch improves graph-mode KV
headroom a bit, but it does not improve single-stream decode speed.

## MTP conclusion

MTP token count alone is not enough. MTP1 remains the safest default:

- MTP2 did not improve quick decode and cut KV pool materially.
- Acceptance is already healthy enough that the remaining ceiling appears to be
  decode/proposer/runtime overhead, not simply draft quality.

### Experimental MTP3 q_len=4 WMMA path

The default WMMA path was originally hard-capped at `q_len <= 3`. Since MTP3 uses
`q_len=4`, the repo now has an experimental env gate:

```bash
VLLM_WMMA_QLEN_CAP=4
```

Correctness-first compare run:

```bash
MTP_SPEC_TOKENS=3
VLLM_WMMA_QLEN_CAP=4
VLLM_WMMA_DECODE=0
VLLM_WMMA_COMPARE=1
VLLM_WMMA_AUTHCOMPARE=0
```

Result: Triton-authoritative output with WMMA comparison logging produced
`rel=0.0005..0.0019` for `q=(4, 32, 192)`, `mq=4`, so the q_len=4 WMMA math
looked numerically sane in this decode test.

Authoritative speed run:

```bash
MTP_SPEC_TOKENS=3
VLLM_WMMA_QLEN_CAP=4
VLLM_WMMA_DECODE=1
```

| max tokens | server tok/s | acceptance |
|---:|---:|---:|
| 512 | 24.34 | 0.409 |
| 1024 | 36.34 | 0.691 |
| 2048 | 33.50 | 0.608 |

Trace confirmed the full-attention q_len=4 WMMA path became active:
`KERNEL_ACTIVE total_q=4 kvh=2 NSPLIT=512`.

Conclusion: q_len=4 WMMA is the best speed progress so far, peaking at
`36.34 tok/s` and sustaining `33.50 tok/s`, but MTP3 acceptance is inconsistent
and this still does not reach the 45-55 tok/s target.

## Spark-side / distributed runtime checks

### Explicit RoCE socket pin, MTP1

Config:

```bash
MTP_SPEC_TOKENS=1
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=4096
ENFORCE_EAGER=1
NCCL_SOCKET_IFNAME=enp1s0f0np0
GLOO_SOCKET_IFNAME=enp1s0f0np0
NCCL_SOCKET_FAMILY=AF_INET
NCCL_IB_HCA=rocep1s0f0
NCCL_NET_GDR_LEVEL=LOC
```

Startup evidence:

- Ray head and worker bound to `192.168.192.3` / `192.168.192.4`.
- vLLM copied `NCCL_SOCKET_IFNAME`, `NCCL_SOCKET_FAMILY`, and other NCCL
  settings to Ray workers, while preserving node-local `NCCL_IB_HCA` via the
  non-carry-over config.
- TP ranks initialized across the two nodes with NCCL.
- Available KV cache memory was 10.39 GiB on one rank and 7.19 GiB on the other.
- GPU KV cache size was 2,285,451 tokens, or 2.29x concurrency for a 1M-token
  request.

Single-stream measurements:

| max tokens | server tok/s | acceptance |
|---:|---:|---:|
| 512 | 28.52 | 0.732 |
| 1024 | 28.70 | 0.743 |

Conclusion: explicit RoCE socket pinning is now part of the reproducible runbook,
but it did not improve C1 decode speed versus baseline. The remaining C1 ceiling
is therefore not explained by an obvious stale `10.0.0.x` socket-interface bind.

### GDR SYS / GDR read, MTP1

Config was the same as the explicit RoCE socket-pin run, except:

```bash
NCCL_NET_GDR_LEVEL=SYS
NCCL_NET_GDR_READ=1
```

Startup evidence:

- vLLM copied `NCCL_NET_GDR_READ` to workers.
- Both ranks reported `NCCL_NET_GDR_LEVEL` changed from `LOC` to `SYS`.
- Available KV cache memory was 8.57 GiB on one rank and 6.39 GiB on the other.
- GPU KV cache size fell to 2,030,939 tokens, or 2.03x concurrency for a
  1M-token request.

Single-stream measurements:

| max tokens | server tok/s | acceptance |
|---:|---:|---:|
| 512 | 28.82 | 0.759 |
| 1024 | 28.99 | 0.753 |

Conclusion: `SYS`/GDR-read produced only a tiny speed change versus `LOC` and
reduced KV headroom. Keep `NCCL_NET_GDR_LEVEL=LOC` as the default unless a later
transport test shows otherwise.

### Ray compiled-DAG NCCL channel, MTP1

Config was the explicit RoCE socket-pin baseline plus:

```bash
VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0
VLLM_USE_RAY_SPMD_WORKER=1
VLLM_USE_RAY_COMPILED_DAG=1
VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=nccl
VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=1
CUDA_DEVICE_MAX_CONNECTIONS=1
```

First attempt failed on the first generation request with:

```text
ValueError: cupy is not installed but required since
VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE is set to 'nccl'
```

Installed `cupy-cuda12x==14.1.1` inside both Spark containers and relaunched the
same config. Startup confirmed `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` was
overwritten from `0` to `1` on both Ray workers.

Startup evidence:

- Available KV cache memory was 8.62 GiB on the head rank and 6.02 GiB on the
  worker rank.
- GPU KV cache size was 1,915,101 tokens, or 1.92x concurrency for a 1M-token
  request.
- The server reached `/v1/models` successfully and the first generation request
  no longer hit the missing-CuPy failure.

Single-stream measurements:

| max tokens | server tok/s | acceptance |
|---:|---:|---:|
| 512 | 28.87 | 0.744 |
| 1024 | 30.11 | 0.807 |

Conclusion: Ray compiled-DAG with NCCL transport now works after installing
CuPy, but it does not materially improve C1 decode speed versus baseline and it
reduces KV headroom versus the best `LOC` run. This makes the current bottleneck
look less like a simple Ray transport flag and more like the MiMo/MTP decode
execution path or the TP topology itself.

### PP=2 / TP=1 topology isolate

Config was the explicit RoCE socket-pin baseline with:

```bash
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=2
MAX_NUM_SEQS=2
MTP_SPEC_TOKENS=1
VLLM_USE_RAY_COMPILED_DAG=0
VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=0
```

Result: launch failed before model load:

```text
NotImplementedError: Pipeline parallelism is not supported for this model.
Supported models implement the `SupportsPP` interface.
```

Conclusion: PP=2 / TP=1 cannot be used as a quick topology isolate for MiMo on
this vLLM build. The model class would need explicit pipeline-parallel support
before this path can test whether avoiding cross-node TP collectives helps C1.

### MTP1 local argmax reduction

Patch:

- Added `MiMoV2MultiTokenPredictor.get_top_tokens(...)`.
- Added `MiMoV2MTP.get_top_tokens(...)`.
- Launched with:

```bash
USE_LOCAL_ARGMAX_REDUCTION=1
MTP_SPEC_TOKENS=1
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=1
```

Startup evidence:

- Speculative config included `"use_local_argmax_reduction": true`.
- Both Ray workers logged:
  `Using local argmax reduction for draft token generation (communication: O(2*tp_size) vs O(vocab_size)).`
- Available KV cache memory was 8.28 GiB on the head rank and 6.63 GiB on the
  worker rank.
- GPU KV cache size was 2,106,384 tokens, or 2.11x concurrency for a 1M-token
  request.

Single-stream measurements:

| max tokens | server tok/s | acceptance |
|---:|---:|---:|
| 512 | 28.27 | 0.738 |
| 1024 | 28.15 | 0.741 |

Conclusion: the local-argmax fast path works and reduces draft-token
communication, but it does not improve C1 throughput on this MiMo setup. The
remaining low-30 tok/s ceiling is therefore not dominated by full-vocab draft
logit all-gather.

### MTP1 greedy target top-token fast path

Patch:

- Added target-side `get_top_tokens(...)` helpers for `MiMoV2FlashForCausalLM`
  and `MiMoV2OmniForCausalLM`.
- Added an env-gated `VLLM_MIMO_MTP1_GREEDY_FAST=1` path in
  `gpu_model_runner.py` for plain greedy MTP1 requests. The guard requires:
  MTP1, all requests greedy, no logprobs, no penalties, no bad words, no
  allowed-token mask, no logits processors, no prompt logprobs, and no active
  thinking-budget state.
- When the guard passes, the target model computes top-token ids instead of
  full target logits, then builds the MTP1 rejection output directly.

Config:

```bash
MTP_SPEC_TOKENS=1
VLLM_MIMO_MTP1_GREEDY_FAST=1
USE_LOCAL_ARGMAX_REDUCTION=0
MAX_MODEL_LEN=1000000
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=4096
ENFORCE_EAGER=1
```

Startup / validation evidence:

- Patch applied and `py_compile` passed on both Sparks.
- A greedy smoke request returned `OK FAST`.
- First attempt failed with `NameError: name 'os' is not defined` in the guard;
  the recipe now patches the missing import and the retry passed.

Single-stream measurements:

| max tokens | server tok/s | acceptance |
|---:|---:|---:|
| 512 | 30.22 | 0.858 |
| 1024 | 31.77 | 0.943 |
| 1024 repeat | 30.87 | 0.910 |
| 2048 | 31.31 | 0.931 |

Conclusion: the greedy target top-token bypass is functional and gives a small
lift over the ~29-30 tok/s MTP1 baseline, but it does not reach the 45-55 tok/s
target. Full-vocab target logits are therefore part of the overhead, but not the
main single-stream bottleneck.

### C1 isolation shape plus greedy target top-token fast path

Config:

```bash
MTP_SPEC_TOKENS=1
VLLM_MIMO_MTP1_GREEDY_FAST=1
USE_LOCAL_ARGMAX_REDUCTION=0
MAX_MODEL_LEN=1000000
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=2048
BLOCK_SIZE=32
ENFORCE_EAGER=1
```

Startup evidence:

- vLLM launched with `max_num_seqs=1`, `max_num_batched_tokens=2048`,
  `block_size=32`, 1M context, eager mode, TP=2.
- The fast-path env was copied to Ray workers.

Single-stream measurements:

| max tokens | server tok/s | acceptance |
|---:|---:|---:|
| 512 | 31.02 | 0.943 |
| 1024 | 30.95 | 0.956 |
| 2048 | 31.26 | 0.956 |

Conclusion: isolating the server for single-stream scheduling did not improve
throughput beyond the low-31 tok/s range. Scheduler/concurrency reservation is
therefore not the main C1 bottleneck for MiMo on this TP=2 1M setup.

### Block-size 64 plus C1 isolation and greedy target top-token fast path

Config:

```bash
MTP_SPEC_TOKENS=1
VLLM_MIMO_MTP1_GREEDY_FAST=1
USE_LOCAL_ARGMAX_REDUCTION=0
MAX_MODEL_LEN=1000000
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=2048
BLOCK_SIZE=64
ENFORCE_EAGER=1
```

Startup evidence:

- vLLM launched with `block_size=64`, `max_num_seqs=1`,
  `max_num_batched_tokens=2048`, 1M context, eager mode, TP=2.
- The server reached `/v1/models` successfully.

Single-stream measurements:

| max tokens | server tok/s | acceptance |
|---:|---:|---:|
| 512 | 32.13 | 0.928 |
| 1024 | 31.87 | 0.927 |
| 2048 | 31.89 | 0.926 |

Conclusion: block size 64 gives a tiny clean lift over block size 32 C1
isolation, but still lands in the low-32 tok/s range. Block-table/page shape is
not enough to reach the 45-55 tok/s target.

### MTP3 q_len=4 plus local argmax

Config:

```bash
MTP_SPEC_TOKENS=3
VLLM_WMMA_QLEN_CAP=4
VLLM_WMMA_DECODE=1
USE_LOCAL_ARGMAX_REDUCTION=1
MAX_NUM_SEQS=1
ENFORCE_EAGER=1
```

Startup evidence:

- Speculative config included `num_speculative_tokens=3` and
  `use_local_argmax_reduction=true`.
- Both Ray workers logged local argmax reduction active.
- GPU KV cache size was 2,124,248 tokens, or 2.12x concurrency for a 1M-token
  request.
- WMMA trace showed q_len=4 activation:
  `KERNEL_ACTIVE total_q=4 kvh=2 NSPLIT=512`.
- The trace also showed repeated fallback for layers with sliding-window/sinks,
  e.g. `sinks=True`, so this is a mixed WMMA/Triton run rather than full WMMA
  coverage.

Single-stream measurements:

| max tokens | server tok/s | acceptance |
|---:|---:|---:|
| 512 | 25.89 | 0.409 |
| 1024 | 28.28 | 0.473 |
| 2048 | 31.49 | 0.570 |

Conclusion: adding local argmax to MTP3/q_len=4 did not improve the earlier
MTP3/q_len=4 result and may be worse at 512/1024. Keep local argmax as a working
optional patch for MTP1 communication hygiene, but do not use it as the
performance answer for MTP3.

### MTP3 q_len=4 with graph capture

1M config:

```bash
MAX_MODEL_LEN=1000000
MTP_SPEC_TOKENS=3
VLLM_WMMA_QLEN_CAP=4
USE_LOCAL_ARGMAX_REDUCTION=0
ENFORCE_EAGER=0
```

Result: graph/compile launch failed before serving. Torch compile completed, but
KV memory collapsed:

```text
Available KV cache memory: 0.36 GiB
ValueError: To serve at least one request with the model's max seq len (1000000),
3.15 GiB KV cache is needed ... estimated maximum model length is 77440.
```

Reduced-context probe:

```bash
MAX_MODEL_LEN=200000
GPU_MEMORY_UTILIZATION=0.90
MTP_SPEC_TOKENS=3
VLLM_WMMA_QLEN_CAP=4
ENFORCE_EAGER=0
```

Startup evidence:

- vLLM used `CompilationMode.VLLM_COMPILE` with cudagraph capture sizes
  `[1, 2, 4, 8]`.
- Both ranks precompiled the WMMA decode kernel at startup.
- Torch compile took about 22 seconds.
- GPU KV cache size was 1,549,551 tokens, or 7.75x concurrency for a 200K-token
  request.

Single-stream measurements:

| max tokens | server tok/s | acceptance |
|---:|---:|---:|
| 512 | 25.52 | 0.366 |
| 1024 | 28.58 | 0.450 |
| 2048 | 32.40 | 0.565 |

Conclusion: graph capture is not a path to 45-55 tok/s here. At 1M it fails the
KV budget, and at 200K it boots but performs roughly like eager MTP3/q_len=4 with
poor short-run acceptance.

## Next useful experiments

1. Decide whether to implement MiMo `SupportsPP` plumbing, or retire PP as too
   large for this tuning pass.
2. Sweep `VLLM_WMMA_NSPLIT=64,128,256,512` at representative context lengths.
3. If the NSPLIT sweep is also flat, the next real code path is a
   MiMo-MTP-aware proposer wrapper or metadata-cache path inside vLLM's existing
   `method:"mtp"` flow.

Do not treat aggregate batching wins as single-stream speed wins. The target
for this tuning track is C1 decode speed.
