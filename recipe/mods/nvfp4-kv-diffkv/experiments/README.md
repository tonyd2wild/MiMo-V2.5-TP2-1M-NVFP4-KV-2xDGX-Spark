# NVFP4 DiffKV Experiments

This directory holds inactive investigation patches. Files here are not applied
by the normal recipe.

## bs128-wmma.patch

Purpose: investigate full-attention MTP decode calls that currently miss the
custom WMMA path:

```text
REJECT=shape q=(2, 32, 192) kvh=2 hqk=192 hv=128 bs=128 ... win=-1 sinks=False
```

Status: not production-ready. The isolated BS128 harness passed, and a later
live authoritative test activated the BS128 WMMA kernel with clean output, but
C1 throughput regressed to the 21-25 tok/s range. Keep this patch as an
investigation artifact only; do not enable it in serving by default.

Required gate before promotion:

1. Run the offline harness without touching the installed package:
   ```bash
   python3 recipe/mods/nvfp4-kv-diffkv/experiments/bs128_wmma_harness.py
   ```
   From a host shell, without copying this repo into the container:
   ```bash
   docker exec -i vllm_mimo_tp2 python3 - < \
     recipe/mods/nvfp4-kv-diffkv/experiments/bs128_wmma_harness.py
   ```
2. Apply the patch only in a throwaway container or branch.
3. Launch with Triton authoritative:
   `VLLM_WMMA_DECODE=0 VLLM_WMMA_COMPARE=1 VLLM_WMMA_QLEN_CAP=3`.
4. Confirm startup reaches `Application startup complete`.
5. Confirm `/tmp/wmma_cmp.log` contains low-error `bs=128 kvh=2` lines.
6. Only then test authoritative WMMA and C1 tok/s.

## bs128_wmma_harness.py

Offline correctness harness for the live MiMo shape:

```text
q=(2, 32, 192), kvh=2, bs=128, mq=2, win=-1, sinks=False
```

The script copies the active `wmma_decode.py` to `/tmp`, applies the BS128 patch
to that temporary copy, compiles it under a unique extension name, and compares
against a PyTorch reference decoder for synthetic packed DiffKV cache rows.
It does not modify the running vLLM install or the production recipe.

Live Bluey container check, 2026-06-30:

```text
shape=[2, 32, 192], block_size=128, seq_len=180,
max_abs=0.015426158905029297, mean_abs=0.0015948378713801503,
calls=1, ok=True
```

Live serving follow-up, 2026-06-30:

```text
KERNEL_ACTIVE total_q=2 kvh=2 NSPLIT=512
1024-token request: 21.14 tok/s, clean output
warmed 512-token request: 24.70 tok/s, clean output
```

Conclusion: correctness alone was not enough. The BS128 full-attention path
adds overhead in the observed serving shape and should stay off unless a future
kernel redesign changes the speed result.

## swa_sink_reference_harness.py

Offline correctness harness for the other repeated live MiMo MTP1 reject:

```text
q=(2, 32, 192), kvh=4, bs=64, mq=2, win=127, sinks=True
```

This is the sliding-window/sink path. It differs from the full-attention BS128
shape in two important ways:

- `kvh=4`, so the query/KV group is `32 / 4 = 8`, not the WMMA kernel's current
  full-attention `G=16`.
- `sinks=True` seeds the online softmax with a per-query-head virtual sink
  score. The sink contributes to the denominator but has zero value
  contribution.

The script runs the installed Triton DiffKV implementation against a PyTorch
reference that includes the sink denominator and `window_left + 1` sliding
window semantics. It does not modify the running vLLM install.

Live Bluey container check, 2026-06-30:

```text
shape=[2, 32, 192], kv_heads=4, group=8, block_size=64,
seq_len=180, window_left=127, sliding_window=128, sinks=True,
max_abs=0.015810489654541016, mean_abs=0.0017646412597969174,
ok=True
```

Conclusion: the current Triton path is a verified reference for the rejected
sinks/window shape. Any faster kernel for this path should pass this harness
before being tested in serving.

## swa_sink_wmma_harness.py

Standalone WMMA prototype for the same sliding-window/sink shape:

```text
q=(2, 32, 192), kvh=4, group=8, bs=64, sliding_window=128, sinks=True
```

This harness does not patch vLLM. It compiles a separate CUDA extension with a
`G=8` WMMA kernel, includes sink-softmax semantics, applies the sliding-window
lower bound, and compares directly against a PyTorch reference.

Live Bluey container check, 2026-06-30:

```text
shape=[2, 32, 192], kv_heads=4, group=8, block_size=64,
seq_len=180, sliding_window=128, sinks=True, nsplit=8,
max_abs=0.015726089477539062, mean_abs=0.001700876047834754,
max_rel=0.6331995129585266, mean_rel=0.002963173436000943,
ok=True
```

Conclusion: the missing SWA/sink WMMA shape is feasible offline. This is still
not a serving patch: it uses a standalone extension and `NSPLIT=8`. The next
safe promotion step is a non-authoritative in-engine compare path where Triton
continues to serve output while this kernel logs relative error and timing for
live MiMo requests.
