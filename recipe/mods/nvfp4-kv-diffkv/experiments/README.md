# NVFP4 DiffKV Experiments

This directory holds inactive investigation patches. Files here are not applied
by the normal recipe.

## bs128-wmma.patch

Purpose: investigate full-attention MTP decode calls that currently miss the
custom WMMA path:

```text
REJECT=shape q=(2, 32, 192) kvh=2 hqk=192 hv=128 bs=128 ... win=-1 sinks=False
```

Status: not production-ready. A live compare-mode attempt with
`VLLM_WMMA_DECODE=0` and `VLLM_WMMA_COMPARE=1` stalled after the second model
load before KV/profile and produced no compare log. Do not enable this patch in
serving until it compiles and passes relative-error comparison in an isolated
harness.

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
