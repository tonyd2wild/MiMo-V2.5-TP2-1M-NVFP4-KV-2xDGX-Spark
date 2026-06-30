#!/usr/bin/env python3
"""Offline reference harness for MiMo's sliding-window/sink DiffKV path.

The live MiMo NVFP4 service repeatedly rejects this shape from the optional
WMMA decode path:

    q=(2, 32, 192), kvh=4, block_size=64, window_left=127, sinks=True

That path is currently served by the Triton DiffKV kernel. This harness builds
the same packed NVFP4 cache shape, runs the installed Triton implementation, and
compares it with a pure PyTorch reference that includes both sliding-window
masking and sink-token softmax behavior.

Run inside the vLLM container:

    python3 recipe/mods/nvfp4-kv-diffkv/experiments/swa_sink_reference_harness.py

or from the host without copying it into the container:

    docker exec -i vllm_mimo_tp2 python3 - < \
      recipe/mods/nvfp4-kv-diffkv/experiments/swa_sink_reference_harness.py
"""

from __future__ import annotations

import argparse
import math

import torch

from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)

E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def fp8_byte(value: float, device: torch.device) -> int:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required")
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
        0,
        16,
        (blocks, block_size, kv_heads, 192),
        device=cache.device,
        dtype=torch.uint8,
        generator=gen,
    )
    v_nibbles = torch.randint(
        0,
        16,
        (blocks, block_size, kv_heads, 128),
        device=cache.device,
        dtype=torch.uint8,
        generator=gen,
    )
    cache[..., 0:96] = pack_e2m1(k_nibbles)
    cache[..., 96:108] = scale_one
    cache[..., 108:172] = pack_e2m1(v_nibbles)
    cache[..., 172:180] = scale_one


def dequant_row(
    row: torch.Tensor,
    dims: int,
    packed_offset: int,
    scale_offset: int,
) -> torch.Tensor:
    packed = row[packed_offset : packed_offset + dims // 2]
    lo = packed & 0x0F
    hi = packed >> 4
    nibbles = torch.empty((dims,), device=row.device, dtype=torch.long)
    nibbles[0::2] = lo.long()
    nibbles[1::2] = hi.long()
    vals = E2M1.to(row.device)[nibbles]
    scales = fp8_decode(row[scale_offset : scale_offset + dims // 16])
    return vals * scales.repeat_interleave(16)


def reference_swa_sink(
    q: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    sinks: torch.Tensor,
    softmax_scale: float,
    sliding_window: int,
) -> torch.Tensor:
    total_q, n_q_heads, hqk = q.shape
    _, block_size, n_kv_heads, _ = cache.shape
    group = n_q_heads // n_kv_heads
    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    rows = torch.arange(total_q, device=q.device, dtype=torch.long)
    seq_idx = torch.bucketize(rows, cu_seqlens_q[1:].long(), right=True)
    token_in_seq = rows - cu_seqlens_q[seq_idx].long()
    context_len = seqused.long()[seq_idx] - q_lens.long()[seq_idx]
    query_abs = context_len + token_in_seq
    trunc_lens = query_abs + 1
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

        key_positions = torch.arange(length, device=q.device)
        window_mask = (query_abs[row_i] - key_positions) < sliding_window
        for qh in range(n_q_heads):
            kh = qh // group
            scores = (keys[:, kh] * q[row_i, qh].float()).sum(dim=-1) * softmax_scale
            scores = scores.masked_fill(~window_mask, float("-inf"))

            # vLLM sinks seed online softmax M with the sink score and L with 1.
            # That is equivalent to adding one virtual key with score=sink and
            # value=0, so it contributes to the denominator but not numerator.
            scores_with_sink = torch.cat([sinks[qh].float().view(1), scores])
            probs = torch.softmax(scores_with_sink, dim=0)
            out[row_i, qh] = probs[1:] @ vals[:, kh]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=180)
    parser.add_argument("--q-len", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--window-left", type=int, default=127)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    n_q_heads = 32
    assert n_q_heads % args.kv_heads == 0
    nblocks = math.ceil(args.seq_len / args.block_size)
    q = torch.randn((args.q_len, n_q_heads, 192), device=device, dtype=torch.bfloat16)
    cache = torch.empty(
        (nblocks, args.block_size, args.kv_heads, 180),
        device=device,
        dtype=torch.uint8,
    )
    fill_cache(cache, seed=args.seed + 1)
    block_table = torch.arange(nblocks, device=device, dtype=torch.int32).view(1, nblocks)
    seqused = torch.tensor([args.seq_len], device=device, dtype=torch.int32)
    cu = torch.tensor([0, args.q_len], device=device, dtype=torch.int32)
    sinks = torch.randn((n_q_heads,), device=device, dtype=torch.float32) * 0.1
    out = torch.empty((args.q_len, n_q_heads, 128), device=device, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(192)
    sliding_window = args.window_left + 1

    unified_attention_diffkv(
        q=q,
        k=cache,
        v=cache,
        out=out,
        cu_seqlens_q=cu,
        seqused_k=seqused,
        softmax_scale=scale,
        causal=True,
        window_size=(args.window_left, -1),
        block_table=block_table,
        softcap=0.0,
        max_seqlen_q=args.q_len,
        sinks=sinks,
        nvfp4_packed=True,
        scale_cache=cache.view(torch.float8_e4m3fn),
        e2m1_lut=E2M1.to(device),
        head_size_v_override=128,
    )
    torch.cuda.synchronize()
    ref = reference_swa_sink(q, cache, block_table, seqused, cu, sinks, scale, sliding_window)
    got = out.float()
    abs_err = (got - ref).abs()
    rel_err = abs_err / ref.abs().clamp_min(1e-3)
    stats = {
        "shape": list(q.shape),
        "kv_heads": args.kv_heads,
        "group": n_q_heads // args.kv_heads,
        "block_size": args.block_size,
        "seq_len": args.seq_len,
        "window_left": args.window_left,
        "sliding_window": sliding_window,
        "sinks": True,
        "max_abs": float(abs_err.max().item()),
        "mean_abs": float(abs_err.mean().item()),
        "max_rel": float(rel_err.max().item()),
        "mean_rel": float(rel_err.mean().item()),
        "ok": bool(torch.allclose(got, ref, rtol=args.rtol, atol=args.atol)),
    }
    print(stats)
    if not stats["ok"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
