#!/usr/bin/env python3
"""Offline WMMA prototype for MiMo's sliding-window/sink DiffKV shape.

This is not a serving patch. It compiles a standalone CUDA extension for the
live rejected shape:

    q=(2, 32, 192), kvh=4, group=8, block_size=64, window_left=127, sinks=True

and compares it against a PyTorch reference. The goal is to prove or disprove
that the rejected SWA/sink path can be handled by a G=8 WMMA kernel before any
live vLLM package is touched.
"""

from __future__ import annotations

import argparse
import math

import torch
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)

E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

CUDA_SRC = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
using namespace nvcuda;

__device__ __forceinline__ float e2m1(unsigned int n) {
  const float L[16] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                       -0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
  return L[n & 15];
}

__device__ __forceinline__ float fp8d(unsigned char b) {
  __nv_fp8_e4m3 v;
  *reinterpret_cast<unsigned char*>(&v) = b;
  return (float)v;
}

#define MT 16
#define NT 16

template<int Hk, int Hv, int G, int BS, int SB>
__global__ void wmma_swa_sink_b(
    const __nv_bfloat16* __restrict__ q,
    const unsigned char* __restrict__ cache,
    const int* __restrict__ bt,
    const int* __restrict__ seqused,
    const float* __restrict__ sinks,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ pa,
    int NQH, int NKVH, int NSPLIT, int maxblk, int nblk,
    int sliding_window, float scale) {
  int kh = blockIdx.x;
  int sp = blockIdx.y;
  int seq = blockIdx.z;
  int lane = threadIdx.x;
  int L = seqused[seq];
  const int Kfb = Hk / 2, Ksb = Hk / 16, Vfb = Hv / 2;
  const int KF = 0, KS = Kfb, VF = Kfb + Ksb, VS = Kfb + Ksb + Vfb;
  const int* bt_s = bt + (size_t)seq * maxblk;
  const __nv_bfloat16* q_s = q + (size_t)seq * NQH * Hk;

  __shared__ __nv_bfloat16 Qs[MT * Hk];
  __shared__ __nv_bfloat16 KVt[NT * Hk];
  __shared__ float Ssh[MT * NT];
  __shared__ __nv_bfloat16 Psh[MT * NT];
  __shared__ float rm[MT], rl[MT];
  __shared__ float acc[G * Hv];

  for (int i = lane; i < MT * Hk; i += 32) {
    int r = i / Hk, c = i % Hk;
    Qs[i] = (r < G) ? q_s[((size_t)(kh * G + r)) * Hk + c]
                    : __float2bfloat16(0.f);
  }
  for (int r = lane; r < MT; r += 32) {
    if (r < G && sp == 0) {
      rm[r] = sinks[kh * G + r];
      rl[r] = 1.f;
    } else {
      rm[r] = -1e30f;
      rl[r] = 0.f;
    }
  }
  for (int i = lane; i < G * Hv; i += 32) acc[i] = 0.f;
  __syncthreads();

  wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> qf[Hk / 16];
  #pragma unroll
  for (int kc = 0; kc < Hk / 16; kc++) {
    wmma::load_matrix_sync(qf[kc], Qs + kc * 16, Hk);
  }

  int hb = seq * NQH;
  if (L <= 0) {
    for (int r = lane; r < G; r += 32) {
      size_t idx = (size_t)((hb + kh * G + r) * NSPLIT + sp);
      pm[idx] = (sp == 0) ? sinks[kh * G + r] : -1e30f;
      pl[idx] = (sp == 0) ? 1.f : 0.f;
    }
    for (int i = lane; i < G * Hv; i += 32) {
      int r = i / Hv, d = i % Hv;
      size_t idx = (size_t)((hb + kh * G + r) * NSPLIT + sp);
      pa[idx * Hv + d] = 0.f;
    }
    return;
  }

  int first_allowed = L - sliding_window;
  if (first_allowed < 0) first_allowed = 0;
  int per = ((L + NSPLIT - 1) / NSPLIT + NT - 1) / NT * NT;
  int j0 = sp * per;
  int j1 = min(L, j0 + per);
  if (j0 < first_allowed) j0 = first_allowed;
  const int BSL = 6;  // block_size=64

  for (int jt = j0; jt < j1; jt += NT) {
    int nv = min(NT, j1 - jt);
    const int KW = Hk / 8;
    for (int i = lane; i < NT * KW; i += 32) {
      int n = i / KW, w = i % KW;
      int base = n * Hk + 8 * w;
      if (n < nv) {
        int p = jt + n;
        int phys = bt_s[p >> BSL];
        const unsigned char* rb = cache + ((size_t)((size_t)phys * BS + (p & (BS - 1))) * NKVH + kh) * SB;
        unsigned int pk = *reinterpret_cast<const unsigned int*>(rb + KF + 4 * w);
        float s = fp8d((rb + KS)[w >> 1]);
        #pragma unroll
        for (int t = 0; t < 8; t++) KVt[base + t] = __float2bfloat16(e2m1(pk >> (4 * t)) * s);
      } else {
        #pragma unroll
        for (int t = 0; t < 8; t++) KVt[base + t] = __float2bfloat16(0.f);
      }
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator,16,16,16,float> cf;
    wmma::fill_fragment(cf, 0.f);
    #pragma unroll
    for (int kc = 0; kc < Hk; kc += 16) {
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bfr;
      wmma::load_matrix_sync(bfr, KVt + kc, Hk);
      wmma::mma_sync(cf, qf[kc / 16], bfr, cf);
    }
    wmma::store_matrix_sync(Ssh, cf, NT, wmma::mem_row_major);
    __syncthreads();

    for (int r = lane; r < MT; r += 32) {
      float mr = rm[r], mloc = -1e30f;
      for (int n = 0; n < nv; n++) {
        float s = Ssh[r * NT + n] * scale;
        Ssh[r * NT + n] = s;
        mloc = fmaxf(mloc, s);
      }
      float mnew = fmaxf(mr, mloc);
      float corr = __expf(mr - mnew);
      float lsum = rl[r] * corr;
      for (int n = 0; n < nv; n++) {
        float p = __expf(Ssh[r * NT + n] - mnew);
        Psh[r * NT + n] = __float2bfloat16(p);
        lsum += p;
      }
      for (int n = nv; n < NT; n++) Psh[r * NT + n] = __float2bfloat16(0.f);
      if (r < G) {
        for (int d = 0; d < Hv; d++) acc[r * Hv + d] *= corr;
      }
      rm[r] = mnew;
      rl[r] = lsum;
    }
    __syncthreads();

    const int VW = Hv / 8;
    for (int i = lane; i < NT * VW; i += 32) {
      int n = i / VW, w = i % VW;
      int base = n * Hv + 8 * w;
      if (n < nv) {
        int p = jt + n;
        int phys = bt_s[p >> BSL];
        const unsigned char* rb = cache + ((size_t)((size_t)phys * BS + (p & (BS - 1))) * NKVH + kh) * SB;
        unsigned int pv = *reinterpret_cast<const unsigned int*>(rb + VF + 4 * w);
        float s = fp8d((rb + VS)[w >> 1]);
        #pragma unroll
        for (int t = 0; t < 8; t++) KVt[base + t] = __float2bfloat16(e2m1(pv >> (4 * t)) * s);
      } else {
        #pragma unroll
        for (int t = 0; t < 8; t++) KVt[base + t] = __float2bfloat16(0.f);
      }
    }
    __syncthreads();

    for (int dc = 0; dc < Hv; dc += 16) {
      wmma::fragment<wmma::accumulator,16,16,16,float> af2;
      wmma::fill_fragment(af2, 0.f);
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> pa_;
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> vb_;
      wmma::load_matrix_sync(pa_, Psh, NT);
      wmma::load_matrix_sync(vb_, KVt + dc, Hv);
      wmma::mma_sync(af2, pa_, vb_, af2);
      wmma::store_matrix_sync(Ssh, af2, 16, wmma::mem_row_major);
      __syncthreads();
      for (int i = lane; i < G * 16; i += 32) {
        int r = i / 16, d = i % 16;
        acc[r * Hv + dc + d] += Ssh[i];
      }
      __syncthreads();
    }
  }

  for (int r = lane; r < G; r += 32) {
    size_t idx = (size_t)((hb + kh * G + r) * NSPLIT + sp);
    pm[idx] = rm[r];
    pl[idx] = rl[r];
  }
  for (int i = lane; i < G * Hv; i += 32) {
    int r = i / Hv, d = i % Hv;
    size_t idx = (size_t)((hb + kh * G + r) * NSPLIT + sp);
    pa[idx * Hv + d] = acc[r * Hv + d];
  }
}

template<int Hv>
__global__ void fa_reduce(const float* pm, const float* pl, const float* pa,
                          __nv_bfloat16* o, int NSPLIT) {
  int h = blockIdx.x, lane = threadIdx.x;
  const int VD = Hv / 32;
  float m = -1e30f, l = 0.f, a[VD];
  #pragma unroll
  for (int i = 0; i < VD; i++) a[i] = 0.f;
  for (int s = 0; s < NSPLIT; s++) {
    size_t idx = (size_t)h * NSPLIT + s;
    float ms = pm[idx];
    float mn = fmaxf(m, ms), c1 = __expf(m - mn), c2 = __expf(ms - mn);
    #pragma unroll
    for (int i = 0; i < VD; i++) a[i] = a[i] * c1 + pa[idx * Hv + lane * VD + i] * c2;
    l = l * c1 + pl[idx] * c2;
    m = mn;
  }
  float inv = (l > 0.f) ? 1.f / l : 0.f;
  #pragma unroll
  for (int i = 0; i < VD; i++) o[(size_t)h * Hv + lane * VD + i] = __float2bfloat16(a[i] * inv);
}

void run(torch::Tensor q, torch::Tensor cache, torch::Tensor bt,
         torch::Tensor seqused, torch::Tensor sinks, torch::Tensor out,
         int NSPLIT, int sliding_window, float scale) {
  int rows = q.size(0) * q.size(1);
  int num_seqs = q.size(0), NQH = q.size(1), NKVH = cache.size(2);
  int maxblk = bt.size(1);
  auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
  auto pm = torch::empty({rows, NSPLIT}, fopt);
  auto pl = torch::empty({rows, NSPLIT}, fopt);
  auto pa = torch::empty({rows, NSPLIT, 128}, fopt);
  dim3 g(NKVH, NSPLIT, num_seqs);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto qp = (const __nv_bfloat16*)q.data_ptr();
  auto cp = cache.data_ptr<unsigned char>();
  auto bp = bt.data_ptr<int>();
  auto sp = seqused.data_ptr<int>();
  auto sk = sinks.data_ptr<float>();
  wmma_swa_sink_b<192,128,8,64,180><<<g,32,0,stream>>>(
      qp, cp, bp, sp, sk, pm.data_ptr<float>(), pl.data_ptr<float>(),
      pa.data_ptr<float>(), NQH, NKVH, NSPLIT, maxblk, cache.size(0),
      sliding_window, scale);
  fa_reduce<128><<<rows,32,0,stream>>>(pm.data_ptr<float>(), pl.data_ptr<float>(),
                                       pa.data_ptr<float>(),
                                       (__nv_bfloat16*)out.data_ptr(), NSPLIT);
}
'''


def compile_module():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="wmma_swa_sink_harness",
        cpp_sources=(
            "void run(torch::Tensor, torch::Tensor, torch::Tensor, "
            "torch::Tensor, torch::Tensor, torch::Tensor, int, int, float);"
        ),
        cuda_sources=CUDA_SRC,
        functions=["run"],
        verbose=False,
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_121,code=sm_121",
            "--use_fast_math",
        ],
    )


def fp8_byte(value: float, device: torch.device) -> int:
    t = torch.tensor([value], device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    return int(t.view(torch.uint8).item())


def fp8_decode(bytes_u8: torch.Tensor) -> torch.Tensor:
    return bytes_u8.view(torch.float8_e4m3fn).to(torch.float32)


def pack_e2m1(nibbles: torch.Tensor) -> torch.Tensor:
    return (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8)


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
    sinks: torch.Tensor,
    softmax_scale: float,
    sliding_window: int,
) -> torch.Tensor:
    total_q, n_q_heads, hqk = q.shape
    _, block_size, n_kv_heads, _ = cache.shape
    group = n_q_heads // n_kv_heads
    out = torch.empty((total_q, n_q_heads, 128), device=q.device, dtype=torch.float32)

    for row_i in range(total_q):
        length = int(seqused[row_i].item())
        first_allowed = max(0, length - sliding_window)
        keys = torch.empty((length, n_kv_heads, hqk), device=q.device, dtype=torch.float32)
        vals = torch.empty((length, n_kv_heads, 128), device=q.device, dtype=torch.float32)
        for pos in range(length):
            phys = int(block_table[row_i, pos // block_size].item())
            off = pos & (block_size - 1)
            for kh in range(n_kv_heads):
                packed_row = cache[phys, off, kh]
                keys[pos, kh] = dequant_row(packed_row, 192, 0, 96)
                vals[pos, kh] = dequant_row(packed_row, 128, 108, 172)
        allowed = torch.zeros((length,), device=q.device, dtype=torch.bool)
        allowed[first_allowed:length] = True
        for qh in range(n_q_heads):
            kh = qh // group
            scores = (keys[:, kh] * q[row_i, qh].float()).sum(dim=-1) * softmax_scale
            scores = scores.masked_fill(~allowed, float("-inf"))
            scores_with_sink = torch.cat([sinks[qh].float().view(1), scores])
            probs = torch.softmax(scores_with_sink, dim=0)
            out[row_i, qh] = probs[1:] @ vals[:, kh]
    return out


def cuda_time_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=180)
    parser.add_argument("--q-len", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--nsplit", type=int, default=8)
    parser.add_argument("--sliding-window", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--bench-iters", type=int, default=0)
    parser.add_argument("--bench-warmup", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.block_size != 64 or args.kv_heads != 4:
        raise RuntimeError("this prototype only targets bs=64, kv_heads=4")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    module = compile_module()

    n_q_heads = 32
    nblocks = math.ceil(args.seq_len / args.block_size)
    q = torch.randn((args.q_len, n_q_heads, 192), device=device, dtype=torch.bfloat16)
    cache = torch.empty(
        (nblocks, args.block_size, args.kv_heads, 180),
        device=device, dtype=torch.uint8,
    )
    fill_cache(cache, seed=args.seed + 1)
    block_table_one = torch.arange(nblocks, device=device, dtype=torch.int32).view(1, nblocks)
    block_table = block_table_one.repeat(args.q_len, 1).contiguous()
    full_seq_lens = torch.arange(
        args.seq_len - args.q_len + 1,
        args.seq_len + 1,
        device=device,
        dtype=torch.int32,
    )
    sinks = torch.randn((n_q_heads,), device=device, dtype=torch.float32) * 0.1
    out = torch.empty((args.q_len, n_q_heads, 128), device=device, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(192)
    cu = torch.tensor([0, args.q_len], device=device, dtype=torch.int32)
    seqused_one = torch.tensor([args.seq_len], device=device, dtype=torch.int32)

    module.run(
        q.contiguous(),
        cache.contiguous(),
        block_table,
        full_seq_lens.contiguous(),
        sinks.contiguous(),
        out.view(args.q_len * n_q_heads, 128),
        int(args.nsplit),
        int(args.sliding_window),
        float(scale),
    )
    torch.cuda.synchronize()

    ref = reference_swa_sink(
        q, cache, block_table, full_seq_lens, sinks, scale, args.sliding_window
    )
    got = out.float()
    abs_err = (got - ref).abs()
    rel_err = abs_err / ref.abs().clamp_min(1e-3)
    stats = {
        "shape": list(q.shape),
        "kv_heads": args.kv_heads,
        "group": n_q_heads // args.kv_heads,
        "block_size": args.block_size,
        "seq_len": args.seq_len,
        "sliding_window": args.sliding_window,
        "sinks": True,
        "nsplit": args.nsplit,
        "max_abs": float(abs_err.max().item()),
        "mean_abs": float(abs_err.mean().item()),
        "max_rel": float(rel_err.max().item()),
        "mean_rel": float(rel_err.mean().item()),
        "ok": bool(torch.allclose(got, ref, rtol=args.rtol, atol=args.atol)),
    }
    if args.bench_iters > 0:
        out_wmma = torch.empty_like(out)
        out_triton = torch.empty_like(out)
        lut = E2M1.to(device)

        def run_wmma() -> None:
            module.run(
                q.contiguous(),
                cache.contiguous(),
                block_table,
                full_seq_lens.contiguous(),
                sinks.contiguous(),
                out_wmma.view(args.q_len * n_q_heads, 128),
                int(args.nsplit),
                int(args.sliding_window),
                float(scale),
            )

        def run_triton() -> None:
            unified_attention_diffkv(
                q=q,
                k=cache,
                v=cache,
                out=out_triton,
                cu_seqlens_q=cu,
                seqused_k=seqused_one,
                softmax_scale=scale,
                causal=True,
                window_size=(args.sliding_window - 1, -1),
                block_table=block_table_one,
                softcap=0.0,
                max_seqlen_q=args.q_len,
                sinks=sinks,
                nvfp4_packed=True,
                scale_cache=cache.view(torch.float8_e4m3fn),
                e2m1_lut=lut,
                head_size_v_override=128,
            )

        triton_ms = cuda_time_ms(
            run_triton,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
        )
        wmma_ms = cuda_time_ms(
            run_wmma,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
        )
        torch.cuda.synchronize()
        diff = (out_wmma.float() - out_triton.float()).abs()
        stats["bench"] = {
            "iters": args.bench_iters,
            "warmup": args.bench_warmup,
            "triton_ms": triton_ms,
            "wmma_ms": wmma_ms,
            "speedup_vs_triton": triton_ms / wmma_ms if wmma_ms else None,
            "wmma_vs_triton_max_abs": float(diff.max().item()),
            "wmma_vs_triton_mean_abs": float(diff.mean().item()),
        }
    print(stats)
    if not stats["ok"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
