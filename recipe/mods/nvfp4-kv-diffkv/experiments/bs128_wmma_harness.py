#!/usr/bin/env python3
"""Offline BS128 WMMA correctness harness for MiMo DiffKV.

This script does not patch the installed vLLM package. It copies the active
wmma_decode.py to /tmp, applies the dormant BS128 experiment changes to that
temporary copy, compiles the temp extension under a unique name, then compares
the WMMA result against a pure PyTorch reference for the live MiMo shape:

    q=(2, 32, 192), num_kv_heads=2, block_size=128, max_seqlen_q=2

Run inside a GB10/vLLM container:

    python3 recipe/mods/nvfp4-kv-diffkv/experiments/bs128_wmma_harness.py

or from the host without copying it into the container:

    docker exec -i vllm_mimo_tp2 python3 - < \
      recipe/mods/nvfp4-kv-diffkv/experiments/bs128_wmma_harness.py
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path

import torch


DEFAULT_SOURCE = (
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/"
    "wmma_decode.py"
)
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def patch_source(text: str) -> str:
    replacements = [
        (
            'name="wmma_decode_diffkv"',
            'name="wmma_decode_diffkv_bs128_harness"',
        ),
        (
            "const int BSL=(BS==32)?5:(BS==16)?4:6;",
            "const int BSL=(BS==16)?4:(BS==32)?5:(BS==64)?6:7;",
        ),
        (
            "cudaFuncSetAttribute((const void*)&wmma_dec_b<192,128,16,64,180>, "
            "cudaFuncAttributePreferredSharedMemoryCarveout,100); once=true; }",
            "cudaFuncSetAttribute((const void*)&wmma_dec_b<192,128,16,64,180>, "
            "cudaFuncAttributePreferredSharedMemoryCarveout,100);\n"
            "             cudaFuncSetAttribute((const void*)&wmma_dec_b<192,128,16,128,180>, "
            "cudaFuncAttributePreferredSharedMemoryCarveout,100); once=true; }",
        ),
        (
            "if(BS==64)\n"
            "    wmma_dec_b<192,128,16,64,180><<<g,32,0,stream>>>(qp,cp,bp,sp,pm_p,pl_p,pa_p,NQH,NKVH,NSPLIT,maxblk,nblk,scale);\n"
            "  else\n"
            "    wmma_dec_b<192,128,16,32,180><<<g,32,0,stream>>>(qp,cp,bp,sp,pm_p,pl_p,pa_p,NQH,NKVH,NSPLIT,maxblk,nblk,scale);",
            "if(BS==128)\n"
            "    wmma_dec_b<192,128,16,128,180><<<g,32,0,stream>>>(qp,cp,bp,sp,pm_p,pl_p,pa_p,NQH,NKVH,NSPLIT,maxblk,nblk,scale);\n"
            "  else if(BS==64)\n"
            "    wmma_dec_b<192,128,16,64,180><<<g,32,0,stream>>>(qp,cp,bp,sp,pm_p,pl_p,pa_p,NQH,NKVH,NSPLIT,maxblk,nblk,scale);\n"
            "  else\n"
            "    wmma_dec_b<192,128,16,32,180><<<g,32,0,stream>>>(qp,cp,bp,sp,pm_p,pl_p,pa_p,NQH,NKVH,NSPLIT,maxblk,nblk,scale);",
        ),
        (
            "block_size not in (32, 64)",
            "block_size not in (32, 64, 128)",
        ),
        (
            "# G=16 invariant; full-attn uses BS=64, SWA BS=32",
            "# G=16 invariant; full-attn may use BS=128, SWA uses smaller blocks",
        ),
    ]
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new, 1)
        elif new not in text:
            raise RuntimeError(f"BS128 patch anchor not found: {old[:80]!r}")
    return text


def import_temp_module(source: Path):
    temp = Path("/tmp/wmma_decode_bs128_harness.py")
    temp.write_text(patch_source(source.read_text()))
    spec = importlib.util.spec_from_file_location("wmma_decode_bs128_harness", temp)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import temp module {temp}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fp8_byte(value: float, device: torch.device) -> int:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required for scale-byte generation")
    t = torch.tensor([value], device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    return int(t.view(torch.uint8).item())


def fp8_decode(bytes_u8: torch.Tensor) -> torch.Tensor:
    return bytes_u8.view(torch.float8_e4m3fn).to(torch.float32)


def pack_e2m1(nibbles: torch.Tensor) -> torch.Tensor:
    even = nibbles[..., 0::2]
    odd = nibbles[..., 1::2]
    return (even | (odd << 4)).to(torch.uint8)


def fill_cache(cache: torch.Tensor, *, seed: int) -> None:
    gen = torch.Generator(device=cache.device).manual_seed(seed)
    blocks, block_size, kv_heads, stride = cache.shape
    assert stride == 180
    scale_one = fp8_byte(1.0, cache.device)
    cache.zero_()
    k_nibbles = torch.randint(
        0, 16, (blocks, block_size, kv_heads, 192),
        device=cache.device, dtype=torch.uint8, generator=gen,
    )
    v_nibbles = torch.randint(
        0, 16, (blocks, block_size, kv_heads, 128),
        device=cache.device, dtype=torch.uint8, generator=gen,
    )
    cache[..., 0:96] = pack_e2m1(k_nibbles)
    cache[..., 96:108] = scale_one
    cache[..., 108:172] = pack_e2m1(v_nibbles)
    cache[..., 172:180] = scale_one


def dequant_row(row: torch.Tensor, dims: int, packed_offset: int, scale_offset: int) -> torch.Tensor:
    packed = row[packed_offset:packed_offset + dims // 2]
    lo = packed & 0x0F
    hi = packed >> 4
    nibbles = torch.empty((dims,), device=row.device, dtype=torch.long)
    nibbles[0::2] = lo.long()
    nibbles[1::2] = hi.long()
    vals = E2M1.to(row.device)[nibbles]
    scales = fp8_decode(row[scale_offset:scale_offset + dims // 16])
    return vals * scales.repeat_interleave(16)


def reference_decode(
    q: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    total_q, n_q_heads, hqk = q.shape
    _, block_size, n_kv_heads, _ = cache.shape
    group = n_q_heads // n_kv_heads
    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    rows = torch.arange(total_q, device=q.device, dtype=torch.long)
    seq_idx = torch.bucketize(rows, cu_seqlens_q[1:].long(), right=True)
    token_in_seq = rows - cu_seqlens_q[seq_idx].long()
    trunc_lens = seqused.long()[seq_idx] - q_lens.long()[seq_idx] + 1 + token_in_seq
    out = torch.empty((total_q, n_q_heads, 128), device=q.device, dtype=torch.float32)

    for row_i in range(total_q):
        seq = int(seq_idx[row_i].item())
        length = int(trunc_lens[row_i].item())
        keys = torch.empty((length, n_kv_heads, hqk), device=q.device, dtype=torch.float32)
        vals = torch.empty((length, n_kv_heads, 128), device=q.device, dtype=torch.float32)
        for pos in range(length):
            phys = int(block_table[seq, pos // block_size].item())
            off = pos & (block_size - 1)
            for kh in range(n_kv_heads):
                packed_row = cache[phys, off, kh]
                keys[pos, kh] = dequant_row(packed_row, 192, 0, 96)
                vals[pos, kh] = dequant_row(packed_row, 128, 108, 172)
        for qh in range(n_q_heads):
            kh = qh // group
            scores = (keys[:, kh] * q[row_i, qh].float()).sum(dim=-1) * softmax_scale
            probs = torch.softmax(scores, dim=0)
            out[row_i, qh] = probs @ vals[:, kh]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--seq-len", type=int, default=180)
    parser.add_argument("--q-len", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    source = Path(args.source)
    module = import_temp_module(source)
    os.environ.setdefault("VLLM_WMMA_DECODE", "1")
    os.environ.setdefault("VLLM_WMMA_QLEN_CAP", str(args.q_len))

    torch.manual_seed(args.seed)
    n_q_heads = args.kv_heads * 16
    nblocks = math.ceil(args.seq_len / args.block_size)
    q = torch.randn(
        (args.q_len, n_q_heads, 192),
        device=device, dtype=torch.bfloat16,
    )
    cache = torch.empty(
        (nblocks, args.block_size, args.kv_heads, 180),
        device=device, dtype=torch.uint8,
    )
    fill_cache(cache, seed=args.seed + 1)
    block_table = torch.arange(nblocks, device=device, dtype=torch.int32).view(1, nblocks)
    seqused = torch.tensor([args.seq_len], device=device, dtype=torch.int32)
    cu = torch.tensor([0, args.q_len], device=device, dtype=torch.int32)
    out = torch.empty((args.q_len, n_q_heads, 128), device=device, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(192)

    handled = module.try_wmma_decode(
        q, cache, out, seqused, block_table, scale,
        args.kv_heads, 192, 128, args.block_size,
        None, 0.0, -1, cu, args.q_len, force=True,
    )
    if not handled:
        raise RuntimeError("temporary BS128 WMMA module rejected the harness shape")
    torch.cuda.synchronize()
    ref = reference_decode(q, cache, block_table, seqused, cu, scale)
    got = out.float()
    abs_err = (got - ref).abs()
    rel_err = abs_err / ref.abs().clamp_min(1e-3)
    stats = {
        "shape": list(q.shape),
        "block_size": args.block_size,
        "seq_len": args.seq_len,
        "max_abs": float(abs_err.max().item()),
        "mean_abs": float(abs_err.mean().item()),
        "max_rel": float(rel_err.max().item()),
        "mean_rel": float(rel_err.mean().item()),
        "calls": int(getattr(module, "_CALLS", -1)),
        "ok": bool(torch.allclose(got, ref, rtol=args.rtol, atol=args.atol)),
    }
    print(stats)
    if not stats["ok"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
