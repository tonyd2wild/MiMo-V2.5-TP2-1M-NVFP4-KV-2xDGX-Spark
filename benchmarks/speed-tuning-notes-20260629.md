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

### 2026-06-30 BS128 WMMA live experiment

The dormant BS128 WMMA decode path was tested in two phases:

- Isolated harness: passed against a PyTorch reference on Bluey with
  `shape=[2,32,192]`, `block_size=128`, `seq_len=180`, `max_abs=0.015426`,
  `mean_abs=0.001595`, `ok=True`.
- Live serving patch: booted successfully on Bluey/Reddie, activated the custom
  full-attention WMMA decode kernel, and produced clean deterministic output.

Live activation evidence:

```text
KERNEL_ACTIVE total_q=2 kvh=2 NSPLIT=512
```

The remaining rejects were the sliding-window/sink path, which the WMMA gate
still intentionally refuses:

```text
REJECT=shape q=(2, 32, 192) kvh=4 ... bs=64 ... win=127 sinks=True
```

However, C1 speed regressed instead of improving:

| test | completion tokens | wall time | client tok/s | quality |
|---|---:|---:|---:|---|
| 1024-token request | 698 | 33.019s | 21.14 | clean, no CJK drift |
| warmed 512-token request | 512 | 20.729s | 24.70 | clean, no CJK drift |

The live runtime was rolled back to the original production
`wmma_decode.py` hash on both nodes:

```text
eabd1fcaa585b5f051b7793acdb28d2e9acfe8f342668c57525b382e353f9cfd
```

Conclusion: BS128 WMMA is correct enough to keep as an experimental harness, but
it is not the current speed fix. The active kernel used `NSPLIT=512`, and the
extra split overhead appears to outweigh the benefit for observed MiMo C1 decode
traffic. Future speed work should look beyond "enable BS128 full-attention WMMA"
alone.

### 2026-06-30 restored 65K no-loop baseline check

After rolling back the live BS128 patch, Bluey/Reddie was restored to the stable
65K/C8 profile:

```bash
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=2048
GPU_MEMORY_UTILIZATION=0.84
MTP_SPEC_TOKENS=1
VLLM_MIMO_MTP1_GREEDY_FAST=1
USE_LOCAL_ARGMAX_REDUCTION=0
ENFORCE_EAGER=1
BLOCK_SIZE=64
```

Runtime evidence:

```text
/v1/models: MiMo-V2.5-NVFP4, max_model_len=65536
GPU KV cache size: 2,452,973 tokens
Maximum concurrency for 65,536 tokens per request: 37.43x
Smoke response: OK RESTORED BASELINE
```

Bounded direct endpoint checks, with no agent harness and no background loop:

| concurrency | completion tokens | client aggregate tok/s | bad outputs |
|---:|---:|---:|---:|
| 1 | 256 | 23.13 | 0 |
| 2 | 512 | 39.50 | 0 |

Metrics after the check showed the endpoint idle:

```text
vllm:num_requests_running 0
vllm:num_requests_waiting 0
```

Server-side generation windows during the same run still reached about
28-34 tok/s, with MTP acceptance around 79-87% once warm. This confirms the
endpoint is stable and clean, but the C1 decode ceiling remains the open speed
problem.

### 2026-06-30 live recovery checkpoint

After an attempted `MAX_MODEL_LEN=1000000`, `MAX_NUM_SEQS=1` C1-isolation
relaunch wedged during model load at checkpoint shard `18/37`, Bluey/Reddie was
restored to the stable 65K/C8 profile:

```bash
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=2048
GPU_MEMORY_UTILIZATION=0.84
MTP_SPEC_TOKENS=1
VLLM_MIMO_MTP1_GREEDY_FAST=1
VLLM_WMMA_DECODE=1
```

Live endpoint evidence:

```text
/v1/models: MiMo-V2.5-NVFP4, max_model_len=65536
GPU KV cache size: 2,416,341 tokens
Maximum concurrency for 65,536 tokens per request: 36.87x
Smoke response: OK LIVE STILL STABLE
```

Single-stream 1024-token check:

| source | tok/s | notes |
|---|---:|---|
| client wall-clock | 26.87 | `max_tokens=1024`, `ignore_eos=true` |
| server log windows | 27.3-28.2 | acceptance ~0.69-0.74 |

Static deterministic sanity sweep (`MAX_TOKENS=256`, direct OpenAI API, checks
for CJK drift/repetition/tool/XML leakage):

| concurrency | success | client aggregate tok/s | bad outputs | best server window |
|---:|---:|---:|---:|---:|
| 1 | 1/1 | 23.17 | 0 | 24.5 |
| 2 | 2/2 | 36.52 | 0 | 34.8 |
| 4 | 4/4 | 56.45 | 0 | 66.0 |
| 6 | 6/6 | 70.95 | 0 | 85.3 |

Conclusion: this profile is stable and clean, but it is not the 60 tok/s
single-upstream target. It is a safe fallback while speed work continues.

### Penalty vs greedy-fast eligibility

The MiMo MTP1 greedy fast path is installed and workers inherit
`VLLM_MIMO_MTP1_GREEDY_FAST=1`, but the guard requires
`sampling_metadata.no_penalties`. Therefore `repetition_penalty=1.08` protects
stability for clients that omit sampling settings, but likely disables the
fastest greedy path for those requests.

Direct C1 A/B on the live 65K profile (`max_tokens=512`, `temperature=0`,
`top_p=1.0`, `ignore_eos=true`):

| repetition penalty | client tok/s | quality check |
|---:|---:|---|
| 1.00 | 29.05 | no CJK, no repeated-char loop |
| 1.08 | 25.74 | no CJK, no repeated-char loop |
| 1.00 | 27.77 | no CJK, no repeated-char loop |

Conclusion: disabling the penalty for controlled greedy requests gives a modest
lift, but it is not enough to reach the target by itself. Keep the server-side
`1.08` fallback for safety unless a harness sends explicit per-request
`repetition_penalty=1.0` for trusted speed tests.

### Live WMMA decode trace

The active runtime has `VLLM_WMMA_DECODE=1` and both ranks precompile the custom
WMMA decode kernel, but the observed MiMo decode shapes are not handled by the
current production gate:

```text
/tmp/wmma_trace.log:
KERNEL_ACTIVE: 0
REJECT: 24

REJECT=shape q=(4, 32, 192) kvh=2 hqk=192 hv=128 bs=128 ... mq=2 win=-1 sinks=False
REJECT=shape q=(4, 32, 192) kvh=4 hqk=192 hv=128 bs=64  ... mq=2 win=127 sinks=True
```

The first shape is the full-attention path missing WMMA because
`wmma_decode.py` only accepts block sizes `32` and `64`. The second is the
sliding-window/sinks path, which is intentionally rejected by the feature gate.

Conclusion: "WMMA enabled" does not currently mean "WMMA is accelerating the
observed MiMo MTP1 decode calls." The closest MiMo-native speed lead remains
the dormant BS128 WMMA experiment, but it must pass compare-mode correctness in
an isolated harness before it is safe for serving.

### DSpark-comparable 200K/C16 lane

DeepSeek DSpark's strongest public concurrency lane uses shorter context and
higher `max_num_seqs` than MiMo's 1M/C8 recipe. To test whether MiMo benefits
from the same serving shape, Bluey/Reddie was relaunched as:

```bash
MAX_MODEL_LEN=200000
MAX_NUM_SEQS=16
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.84
MTP_SPEC_TOKENS=1
VLLM_MIMO_MTP1_GREEDY_FAST=1
VLLM_WMMA_DECODE=1
ENFORCE_EAGER=1
```

Boot evidence:

```text
/v1/models: MiMo-V2.5-NVFP4, max_model_len=200000
GPU KV cache size: 1,642,200 tokens
Maximum concurrency for 200,000 tokens per request: 8.21x
Smoke response: OK 200K C16
```

Direct deterministic sweep (`MAX_TOKENS=256`, checks for CJK
drift/repetition/tool/XML leakage):

| concurrency | success | client aggregate tok/s | per-request mean tok/s | bad outputs | best server window |
|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 22.91 | 22.92 | 0 | 20.3 |
| 4 | 4/4 | 52.38 | 13.25 | 0 | 54.4 |
| 8 | 8/8 | 79.02 | 10.10 | 0 | 96.2 |
| 16 | 16/16 | 106.13 | 6.79 | 0 | 149.4 |

At C16 the server briefly showed `Running: 11 reqs, Waiting: 5 reqs`, then
settled into `Running: 16 reqs, Waiting: 0 reqs`. Acceptance stayed around
0.65-0.74 during the sweep.

WMMA trace still showed no active custom decode kernel:

```text
REJECT=shape q=(2, 32, 192) kvh=2 ... bs=128 mq=2 win=-1 sinks=False
REJECT=shape q=(2, 32, 192) kvh=4 ... bs=64  mq=2 win=127 sinks=True
```

Conclusion: the DSpark-style `200K/C16` shape is clean and bootable for MiMo,
but it is not a convincing speed win. It does not approach DeepSeek DSpark's C16
aggregate and it still hits the same WMMA rejection pattern. After this test the
live endpoint was restored to the safer 65K/C8 profile:

```text
/v1/models: max_model_len=65536
GPU KV cache size: 2,528,716 tokens
Maximum concurrency for 65,536 tokens per request: 38.59x
Smoke response: OK RESTORED 65K
```

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

### Post-checkpoint concurrency speedup levers

These tests were run after the 1M/C8 checkpoint, using the same 2x DGX Spark
TP=2 runtime and comparing against the winning eager C8 profile:

```bash
MAX_MODEL_LEN=1000000
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=2048
BLOCK_SIZE=64
MTP_SPEC_TOKENS=1
VLLM_MIMO_MTP1_GREEDY_FAST=1
ENFORCE_EAGER=1
```

Winning reference:

| concurrency | aggregate tok/s | aggregate-derived tok/s per stream | acceptance |
|---:|---:|---:|---:|
| 2 | 60.2 | 30.1 | 0.829 |
| 4 | 94.7 | 23.7 | 0.837 |
| 6 | 141.8 | 23.6 | 0.832 |
| 8 | 184.1 | 23.0 | 0.867 |

#### `ENFORCE_EAGER=0` at 1M/C8

Config:

```bash
MAX_MODEL_LEN=1000000
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=2048
BLOCK_SIZE=64
MTP_SPEC_TOKENS=1
ENFORCE_EAGER=0
```

Result: failed before serving. Graph compile/capture began, but KV allocation
collapsed below what is needed for 1M context.

Failure evidence:

```text
Compiling a graph for compile range (1, 2048) takes 11.47 s
Profiling CUDA graph memory: PIECEWISE=6 (largest=32), FULL=4 (largest=16)
Available KV cache memory: 0.24 GiB
Available KV cache memory: -1.42 GiB
ValueError: To serve at least one request with the model's max seq len
(1000000), 3.09 GiB KV cache is needed, which is larger than the available
KV cache memory (0.24 GiB). Estimated maximum model length is 57344.
```

Retried with:

```bash
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
```

This also failed before serving:

```text
Available KV cache memory: 0.75 GiB
Available KV cache memory: -1.3 GiB
Estimated maximum model length is 225280.
```

Conclusion: CUDA graph mode is not a valid 1M/C8 speedup lever for this profile.
It cannot preserve enough KV headroom to serve the 1M max length.

#### `MTP_SPEC_TOKENS=2` at 1M/C8

Config:

```bash
MAX_MODEL_LEN=1000000
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=2048
BLOCK_SIZE=64
MTP_SPEC_TOKENS=2
VLLM_MIMO_MTP1_GREEDY_FAST=0
ENFORCE_EAGER=1
```

Startup evidence:

```text
GPU KV cache size: 2,197,421 tokens
Maximum concurrency for 1,000,000 tokens per request: 2.20x
Available KV cache memory: 6.78 GiB
```

Benchmark:

| concurrency | best aggregate tok/s | aggregate-derived tok/s per stream | acceptance |
|---:|---:|---:|---:|
| 2 | 48.0 | 24.0 | 0.682 |
| 4 | 72.3 | 18.1 | 0.628 |
| 6 | 95.0 | 15.8 | 0.597 |
| 8 | 107.5 | 13.4 | 0.626 |

Conclusion: MTP2 is a clear non-winner. It slightly increased the reported KV
pool, but aggregate throughput and acceptance both fell sharply versus MTP1.

#### `MAX_NUM_BATCHED_TOKENS=4096` at 1M/C8

Config:

```bash
MAX_MODEL_LEN=1000000
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=4096
BLOCK_SIZE=64
MTP_SPEC_TOKENS=1
VLLM_MIMO_MTP1_GREEDY_FAST=1
ENFORCE_EAGER=1
```

Startup evidence:

```text
Available KV cache memory: 9.1 GiB
Available KV cache memory: 7.2 GiB
GPU KV cache size: 2,288,537 tokens
Maximum concurrency for 1,000,000 tokens per request: 2.29x
```

Benchmark, including C1:

| concurrency | best aggregate tok/s | aggregate-derived tok/s per stream | acceptance |
|---:|---:|---:|---:|
| 1 | 30.7 | 30.7 | 0.821 |
| 2 | 50.2 | 25.1 | 0.861 |
| 4 | 74.4 | 18.6 | 0.900 |
| 6 | 97.2 | 16.2 | 0.864 |
| 8 | 111.2 | 13.9 | 0.871 |

Conclusion: `MAX_NUM_BATCHED_TOKENS=4096` improves reported KV pool to 2.29M
tokens, but it is not a throughput speedup. Keep `2048` as the serving default
for the C8 speed checkpoint.

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

### 65K MTP1 `VLLM_WMMA_NSPLIT=256`

65K config, matching the stable smoke profile:

```bash
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=2048
BLOCK_SIZE=64
GPU_MEMORY_UTILIZATION=0.84
MTP_SPEC_TOKENS=1
VLLM_MIMO_MTP1_GREEDY_FAST=1
ENFORCE_EAGER=1
VLLM_WMMA_NSPLIT=256
```

Startup evidence:

- Model served successfully at `max_model_len=65536`.
- KV profile reported 38.98x maximum concurrency for a 65,536-token request on
  this launch.
- Trace still showed WMMA rejects for the MTP-shaped `q=(2, 32, 192)` decode
  calls, so this did not improve the critical MTP1 path.

Client-side direct endpoint benchmark (`MAX_TOKENS=160`):

| concurrency | default NSPLIT 512 agg tok/s | NSPLIT 256 agg tok/s | bad outputs |
|---:|---:|---:|---:|
| 1 | 25.71 | 14.21 | 0 |
| 2 | 39.92 | 30.97 | 0 |
| 4 | 63.87 | 53.06 | 0 |
| 6 | 82.73 | 70.08 | 0 |

Conclusion: `VLLM_WMMA_NSPLIT=256` is stable but slower than the default 512
split on the 65K MTP1 profile. Do not use it as the serving default. The live
Reddie/Bluey endpoint was restored to the default 512 split after this test and
smoked with `OK RESTORED 512`.

### 65K live C1 fixed-token check

After rejecting the NSPLIT=256 run and restoring the stable 65K profile, a
fixed-token C1 request was run against the live Bluey/Reddie endpoint:

```json
{
  "max_tokens": 1024,
  "temperature": 0,
  "repetition_penalty": 1.08,
  "ignore_eos": true
}
```

Result:

| profile | max tokens | completion tokens | client tok/s | finish |
|---|---:|---:|---:|---|
| 65K live, `MAX_NUM_SEQS=8` | 1024 | 1024 | 27.88 | length |

The endpoint remained stable and clean, but this is not a C1 speed win. The
current better recorded C1 checkpoint is still the block-size-64 / greedy-fast
1M profile at about 31.87-31.89 tok/s.

### BS128 WMMA full-attention investigation

The WMMA trace explains one real missed speed path on the current MiMo export:

```text
REJECT=shape q=(2, 32, 192) kvh=2 hqk=192 hv=128 bs=128 ... win=-1 sinks=False
```

That call is the full-attention MTP-shaped path. It matches the kernel's
per-rank head grouping (`32 q heads / 2 kv heads = G=16`) but not the current
kernel's accepted block sizes (`32` and `64` only).

A local/live compare attempt added a BS128 template dispatch and the required
`log2(128)=7` block-table shift, then launched with:

```bash
VLLM_WMMA_DECODE=0
VLLM_WMMA_COMPARE=1
VLLM_WMMA_QLEN_CAP=3
```

This was intentionally non-authoritative: Triton would have remained the served
output while WMMA wrote to a temp buffer for relative-error logging. The attempt
did not reach KV/profile or `Application startup complete`; it stalled after the
second model load before any compare log was produced. The live endpoint was
restored to the original kernel file and stable 65K launch.

Conclusion: BS128 WMMA is a valid lead, but not a serving patch yet. Do not land
or enable it until it compiles and compares in an isolated harness first.

## Next useful experiments

1. Treat BS128 WMMA as an experimental harness only. The offline harness passed,
   and a live authoritative patch activated cleanly, but C1 regressed to
   21-25 tok/s. Do not enable it by default.
2. Investigate the still-rejected sliding-window/sink attention path:
   `q=(2,32,192)`, `kvh=4`, `bs=64`, `win=127`, `sinks=True`. This path remains
   Triton-only and appears on every MiMo MTP1 request.
3. Decide whether to implement MiMo `SupportsPP` plumbing, or retire PP as too
   large for this tuning pass. PP=2/TP=1 currently fails before model load.
4. Skip lower `VLLM_WMMA_NSPLIT` values unless a trace first proves the MTP path
   is using WMMA; 256 was already worse than the 512 default at 65K.
5. If WMMA coverage stays mixed, the next real code path is a MiMo-MTP-aware
   proposer wrapper or metadata-cache path inside vLLM's existing `method:"mtp"`
   flow.

Do not treat aggregate batching wins as single-stream speed wins. The target
for this tuning track is C1 decode speed.
