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

1. Apply the patch only in a throwaway container or branch.
2. Launch with Triton authoritative:
   `VLLM_WMMA_DECODE=0 VLLM_WMMA_COMPARE=1 VLLM_WMMA_QLEN_CAP=3`.
3. Confirm startup reaches `Application startup complete`.
4. Confirm `/tmp/wmma_cmp.log` contains low-error `bs=128 kvh=2` lines.
5. Only then test authoritative WMMA and C1 tok/s.
