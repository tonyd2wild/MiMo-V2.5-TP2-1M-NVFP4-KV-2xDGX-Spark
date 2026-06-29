# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LaNarde "Tony" DeAngelo (github.com/tonyd2wild) — 2Wild
#
# Original work: a custom WMMA (tensor-core) flash-decode kernel for the nvfp4 packed
# DiffKV cache, authored for the MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark recipe. Optional
# speed path (gated by VLLM_WMMA_DECODE=1) — the Triton DiffKV path produces the same
# result without it. Released open-source; attribution appreciated.
#
# EXPERIMENTAL: custom WMMA (tensor-core) flash-decode for nvfp4 packed DiffKV cache.
# Targets the long-single-context FULL-attention decode regime where the Triton
# nvfp4 path is attention-bound. Measured 2.3x faster than Triton-nvfp4 on the
# paged layout, correct (rel-err 0.0026 vs Triton). Gated by VLLM_WMMA_DECODE=1.
# Falls back (returns False) for any unsupported shape / SWA / sinks / softcap /
# prefill, so the engine stays correct.
import os
import torch

_M = None
_OK = None  # tri-state: None=untried, True=compiled, False=failed (stop retrying)
_CALLS = 0  # invocation counter for verification
_DBG = 0    # one-time gate-decision diagnostics

# MiMo-V2.5 per-rank shapes under TP=2 (global 64 q-heads / 4 kv-heads -> per rank 32 / 2; G=16 invariant).
# The kernel is templated on G=16 (= num_queries_per_kv); NKVH is a runtime arg (grid.x).
_HK, _HV, _BS, _SB = 192, 128, 32, 180
_G = 16

# --- cudagraph-capturable config -------------------------------------------------
# NSPLIT is FIXED to a process-constant so the kernel grid (grid.y == NSPLIT) is
# static per batch size -> a CUDA graph can capture the launch. The old code chose
# NSPLIT = min(512, max(8, (ctx+255)//256)) per call (varied with context length and
# required a GPU->CPU sync to read max-ctx). The kernel already handles "too many"
# splits for free: empty per-split ranges (j0>=j1) are no-ops and L<=0 writes neutral
# partials, so a fixed-large NSPLIT is numerically IDENTICAL to the longest-context
# case, just with more empty (skipped) blocks for short contexts.
#   Default 512 == the historical ceiling of the old formula (512*256 = 131072 tokens
#   of context granularity), so we never under-split a long context. Override via
#   VLLM_WMMA_NSPLIT (e.g. lower it to trade a little long-ctx parallelism for fewer
#   empty launches on short batches).
_NSPLIT_MAX = max(8, min(512, int(os.environ.get("VLLM_WMMA_NSPLIT", "512"))))
# Generous per-rank decode batch cap for the static partial-stat scratch. The C++
# side allocates pm/pl/pa ONCE sized for max_batch*NQH rows and reuses them, so no
# allocation happens inside a CUDA graph capture. Grows automatically (in eager
# warmup) if a batch ever exceeds this; never shrinks. NQH/Hv/G are TP=2 per-rank
# invariants (32 / 128 / 16).
_MAX_BATCH = max(1, int(os.environ.get("VLLM_WMMA_MAX_BATCH", "64")))

_CUDA = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
using namespace nvcuda;
__device__ __forceinline__ float e2m1(unsigned int n){
  const float L[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
  return L[n&15]; }
__device__ __forceinline__ float fp8d(unsigned char b){ __nv_fp8_e4m3 v; *reinterpret_cast<unsigned char*>(&v)=b; return (float)v; }
#define MT 16
#define NT 16
template<int Hk,int Hv,int G,int BS,int SB>
__global__ void wmma_dec_b(const __nv_bfloat16* __restrict__ q,
    const unsigned char* __restrict__ cache,const int* __restrict__ bt,const int* __restrict__ seqused,
    float* __restrict__ pm,float* __restrict__ pl,float* __restrict__ pa,
    int NQH,int NKVH,int NSPLIT,int maxblk,int nblk,float scale){
  int kh=blockIdx.x, sp=blockIdx.y, seq=blockIdx.z, lane=threadIdx.x;
  int L=seqused[seq];
  const int Kfb=Hk/2,Ksb=Hk/16,Vfb=Hv/2;
  const int KF=0, KS=Kfb, VF=Kfb+Ksb, VS=Kfb+Ksb+Vfb;
  const int* bt_s = bt + (size_t)seq*maxblk;
  const __nv_bfloat16* q_s = q + (size_t)seq*NQH*Hk;
  __shared__ __nv_bfloat16 Qs[MT*Hk];
  __shared__ __nv_bfloat16 KVt[NT*Hk];
  __shared__ float Ssh[MT*NT];
  __shared__ __nv_bfloat16 Psh[MT*NT];
  for(int i=lane;i<MT*Hk;i+=32){ int r=i/Hk,c=i%Hk;
    Qs[i]=(r<G)? q_s[((size_t)(kh*G+r))*Hk+c] : __float2bfloat16(0.f); }
  __shared__ float rm[MT], rl[MT];
  __shared__ float acc[G*Hv];
  for(int i=lane;i<MT;i+=32){ rm[i]=-1e30f; rl[i]=0.f; }
  for(int i=lane;i<G*Hv;i+=32) acc[i]=0.f;
  __syncthreads();
  wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> qf[Hk/16];
  #pragma unroll
  for(int kc=0;kc<Hk/16;kc++) wmma::load_matrix_sync(qf[kc], Qs+kc*16, Hk);
  if(L<=0){ // empty seq: write neutral partials
    int hb=(seq*NQH);
    for(int r=lane;r<G;r+=32){ size_t idx=(size_t)((hb+kh*G+r)*NSPLIT+sp); pm[idx]=-1e30f; pl[idx]=0.f; }
    for(int i=lane;i<G*Hv;i+=32){ int r=i/Hv,d=i%Hv; size_t idx=(size_t)((hb+kh*G+r)*NSPLIT+sp); pa[idx*Hv+d]=0.f; }
    return; }
  int per=((L+NSPLIT-1)/NSPLIT + NT-1)/NT*NT; int j0=sp*per; int j1=min(L,j0+per);
  const int BSL=(BS==32)?5:(BS==16)?4:6;
  for(int jt=j0; jt<j1; jt+=NT){
    int nv=min(NT,j1-jt);
    const int KW=Hk/8;
    for(int i=lane;i<NT*KW;i+=32){ int n=i/KW,w=i%KW; int base=n*Hk+8*w;
      if(n<nv){ int p=jt+n; int phys=bt_s[p>>BSL];
        const unsigned char* rb=cache+((size_t)((size_t)phys*BS+(p&(BS-1)))*NKVH+kh)*SB;
        unsigned int pk=*reinterpret_cast<const unsigned int*>(rb+KF+4*w); float s=fp8d((rb+KS)[w>>1]);
        #pragma unroll
        for(int t2=0;t2<8;t2++) KVt[base+t2]=__float2bfloat16(e2m1(pk>>(4*t2))*s);
      } else { for(int t2=0;t2<8;t2++) KVt[base+t2]=__float2bfloat16(0.f); } }
    __syncthreads();
    wmma::fragment<wmma::accumulator,16,16,16,float> cf; wmma::fill_fragment(cf,0.f);
    #pragma unroll
    for(int kc=0;kc<Hk;kc+=16){
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bfr;
      wmma::load_matrix_sync(bfr, KVt+kc, Hk);
      wmma::mma_sync(cf, qf[kc/16], bfr, cf); }
    wmma::store_matrix_sync(Ssh, cf, NT, wmma::mem_row_major);
    __syncthreads();
    for(int r=lane;r<MT;r+=32){
      float mr=rm[r], mloc=-1e30f;
      for(int n=0;n<nv;n++){ float s=Ssh[r*NT+n]*scale; Ssh[r*NT+n]=s; mloc=fmaxf(mloc,s); }
      float mnew=fmaxf(mr,mloc); float corr=__expf(mr-mnew); float lsum=rl[r]*corr;
      for(int n=0;n<nv;n++){ float p=__expf(Ssh[r*NT+n]-mnew); Psh[r*NT+n]=__float2bfloat16(p); lsum+=p; }
      for(int n=nv;n<NT;n++) Psh[r*NT+n]=__float2bfloat16(0.f);
      if(r<G) for(int d=0;d<Hv;d++) acc[r*Hv+d]*=corr;
      rm[r]=mnew; rl[r]=lsum; }
    __syncthreads();
    const int VW=Hv/8;
    for(int i=lane;i<NT*VW;i+=32){ int n=i/VW,w=i%VW; int base=n*Hv+8*w;
      if(n<nv){ int p=jt+n; int phys=bt_s[p>>BSL];
        const unsigned char* rb=cache+((size_t)((size_t)phys*BS+(p&(BS-1)))*NKVH+kh)*SB;
        unsigned int pv=*reinterpret_cast<const unsigned int*>(rb+VF+4*w); float s=fp8d((rb+VS)[w>>1]);
        #pragma unroll
        for(int t2=0;t2<8;t2++) KVt[base+t2]=__float2bfloat16(e2m1(pv>>(4*t2))*s);
      } else { for(int t2=0;t2<8;t2++) KVt[base+t2]=__float2bfloat16(0.f); } }
    __syncthreads();
    for(int dc=0; dc<Hv; dc+=16){
      wmma::fragment<wmma::accumulator,16,16,16,float> af2; wmma::fill_fragment(af2,0.f);
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> pa_;
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> vb_;
      wmma::load_matrix_sync(pa_, Psh, NT);
      wmma::load_matrix_sync(vb_, KVt+dc, Hv);
      wmma::mma_sync(af2, pa_, vb_, af2);
      wmma::store_matrix_sync(Ssh, af2, 16, wmma::mem_row_major);
      __syncthreads();
      for(int i=lane;i<G*16;i+=32){ int r=i/16,d=i%16; acc[r*Hv+dc+d]+=Ssh[i]; }
      __syncthreads(); } }
  int hb=(seq*NQH);
  for(int r=lane;r<G;r+=32){ size_t idx=(size_t)((hb+kh*G+r)*NSPLIT+sp); pm[idx]=rm[r]; pl[idx]=rl[r]; }
  for(int i=lane;i<G*Hv;i+=32){ int r=i/Hv,d=i%Hv; size_t idx=(size_t)((hb+kh*G+r)*NSPLIT+sp); pa[idx*Hv+d]=acc[r*Hv+d]; }
}
template<int Hv>
__global__ void fa_reduce(const float* pm,const float* pl,const float* pa,__nv_bfloat16* o,int NSPLIT){
  int h=blockIdx.x, lane=threadIdx.x; const int VD=Hv/32; float m=-1e30f,l=0.f,a[VD];
  #pragma unroll
  for(int i=0;i<VD;i++) a[i]=0.f;
  for(int s=0;s<NSPLIT;s++){ size_t idx=(size_t)h*NSPLIT+s; float ms=pm[idx];
    float mn=fmaxf(m,ms),c1=__expf(m-mn),c2=__expf(ms-mn);
    #pragma unroll
    for(int i=0;i<VD;i++) a[i]=a[i]*c1+pa[idx*Hv+lane*VD+i]*c2;
    l=l*c1+pl[idx]*c2; m=mn; }
  float inv = (l>0.f)? 1.f/l : 0.f;
  #pragma unroll
  for(int i=0;i<VD;i++) o[(size_t)h*Hv+lane*VD+i]=__float2bfloat16(a[i]*inv);
}
// Static partial-stat scratch (pm/pl/pa), allocated ONCE per device and reused.
// Sizing is driven by the FIXED NSPLIT (== NSPLIT_MAX) and a generous row budget
// (max_batch*NQH). Reused across calls so no torch::empty happens inside a CUDA
// graph capture. The buffers only ever grow (during eager warmup), never inside
// a capture, because num_seqs<=max_batch and NSPLIT is constant per process.
static torch::Tensor g_pm, g_pl, g_pa;   // [maxrows,NSPLIT], [maxrows,NSPLIT], [maxrows,NSPLIT,Hv]
static long g_cap_rows = 0, g_cap_nsplit = 0;
static void* g_cap_dev = nullptr;
static void ensure_scratch(long want_rows,int NSPLIT,int Hv,const torch::TensorOptions& fopt,void* dev){
  if(g_pm.defined() && want_rows<=g_cap_rows && NSPLIT==g_cap_nsplit && dev==g_cap_dev) return;
  if(want_rows < g_cap_rows) want_rows = g_cap_rows;   // never shrink
  if(want_rows < 1) want_rows = 1;
  g_pm = torch::empty({want_rows,NSPLIT},fopt);
  g_pl = torch::empty({want_rows,NSPLIT},fopt);
  g_pa = torch::empty({want_rows,NSPLIT,Hv},fopt);
  g_cap_rows = want_rows; g_cap_nsplit = NSPLIT; g_cap_dev = dev;
}
// out: caller-provided [num_seqs*NQH, Hv] bf16 buffer (row-major flat view of
// [total_q,NQH,Hv]); fa_reduce writes the final result directly into it.
// min_rows: a row floor (= max_batch*NQH) so the static scratch is pre-grown to its
// ceiling on the FIRST call, guaranteeing no torch::empty ever runs during capture.
void run(torch::Tensor q,torch::Tensor cache,torch::Tensor bt,torch::Tensor seqused,torch::Tensor out,int NKVH,int NSPLIT,float scale,int BS,int min_rows){
  int num_seqs=q.size(0),NQH=q.size(1); const int Hv=128; int maxblk=bt.size(1); int nblk=cache.size(0);
  int rows=num_seqs*NQH;
  long want_rows = (rows>min_rows)? (long)rows : (long)min_rows;
  auto fopt=torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
  ensure_scratch(want_rows,NSPLIT,Hv,fopt,q.device().has_index()? (void*)(intptr_t)q.device().index() : (void*)0);
  // Slice the static scratch to exactly this call's row count (a view, no alloc).
  // The kernels index rows as r*NSPLIT(+...) so the leading [rows,...] slice is a
  // contiguous prefix of the static buffer.
  float* pm_p = g_pm.data_ptr<float>();
  float* pl_p = g_pl.data_ptr<float>();
  float* pa_p = g_pa.data_ptr<float>();
  dim3 g(NKVH,NSPLIT,num_seqs);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static bool once=false;
  if(!once){ cudaFuncSetAttribute((const void*)&wmma_dec_b<192,128,16,32,180>, cudaFuncAttributePreferredSharedMemoryCarveout,100);
             cudaFuncSetAttribute((const void*)&wmma_dec_b<192,128,16,64,180>, cudaFuncAttributePreferredSharedMemoryCarveout,100); once=true; }
  auto qp=(const __nv_bfloat16*)q.data_ptr(); auto cp=cache.data_ptr<unsigned char>(); auto bp=bt.data_ptr<int>(); auto sp=seqused.data_ptr<int>();
  if(BS==64)
    wmma_dec_b<192,128,16,64,180><<<g,32,0,stream>>>(qp,cp,bp,sp,pm_p,pl_p,pa_p,NQH,NKVH,NSPLIT,maxblk,nblk,scale);
  else
    wmma_dec_b<192,128,16,32,180><<<g,32,0,stream>>>(qp,cp,bp,sp,pm_p,pl_p,pa_p,NQH,NKVH,NSPLIT,maxblk,nblk,scale);
  fa_reduce<128><<<rows,32,0,stream>>>(pm_p,pl_p,pa_p,(__nv_bfloat16*)out.data_ptr(),NSPLIT);
}
'''


def _compile():
    global _M, _OK
    if _OK is not None:
        return _OK
    try:
        from torch.utils.cpp_extension import load_inline
        _M = load_inline(
            name="wmma_decode_diffkv",
            cpp_sources="void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int,int,float,int,int);",
            cuda_sources=_CUDA, functions=["run"], verbose=False,
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_121,code=sm_121", "--use_fast_math"],
        )
        _OK = True
    except Exception as e:  # pragma: no cover
        import sys
        print(f"[wmma_decode] compile FAILED, falling back to Triton: {e}", file=sys.stderr)
        _OK = False
    return _OK


# Default remains the validated MTP2 envelope (q_len = num_spec_tokens + 1 = 3).
# Raising this is experimental: first run with VLLM_WMMA_DECODE=0 and
# VLLM_WMMA_COMPARE=1 so Triton stays authoritative while WMMA logs rel-err.
_QLEN_CAP = max(1, int(os.environ.get("VLLM_WMMA_QLEN_CAP", "3")))

def try_wmma_decode(q, k_cache, out, seqused_k, block_table, softmax_scale,
                    num_kv_heads, head_size_qk, head_size_v, block_size, sinks, softcap,
                    window_left, cu_seqlens_q, max_seqlen_q, force=False):
    """Return True if the WMMA kernel handled this call (out written); else False to fall back.

    Handles decode (q_len=1) AND MTP/speculative decode (q_len = num_spec+1) via per-query-token
    expansion: each query token t of a sequence attends to the contiguous causal prefix ending at
    its position, i.e. truncated seq_len = seqused_k - q_len + 1 + t. The underlying M=8 kernel is
    unchanged; only the work-item batch is expanded (validated q_len=3, rel-err ~quant-noise).
    """
    global _DBG
    def _dbg(reason):
        global _DBG
        if _DBG < 12:
            _DBG += 1
            try:
                with open("/tmp/wmma_trace.log", "a") as _f:
                    _f.write(f"REJECT={reason} q={tuple(q.shape)} kvh={num_kv_heads} hqk={head_size_qk} "
                             f"hv={head_size_v} bs={block_size} cache_last={k_cache.shape[-1] if hasattr(k_cache,'shape') else type(k_cache).__name__} "
                             f"mq={max_seqlen_q} win={window_left} sinks={sinks is not None} "
                             f"softcap={softcap} dt={q.dtype} cu={cu_seqlens_q is not None}\n")
            except Exception:
                pass
    if not force and os.environ.get("VLLM_WMMA_DECODE", "1") == "0":   # default ON; only explicit "0" disables
        return False
    if (head_size_qk != _HK or head_size_v != _HV or block_size not in (32, 64)
            or k_cache.shape[-1] != _SB or q.shape[1] != num_kv_heads * _G):  # G=16 invariant; full-attn uses BS=64, SWA BS=32
        _dbg("shape"); return False
    if sinks is not None or softcap not in (0.0, None) or window_left >= 0:
        _dbg("feature(sink/softcap/window)"); return False
    if q.dtype != torch.bfloat16 or cu_seqlens_q is None:
        _dbg("dtype/cu_none"); return False
    if max_seqlen_q is None or max_seqlen_q > _QLEN_CAP:   # prefill -> Triton
        _dbg("max_seqlen_q"); return False
    if not _compile():
        _dbg("compile"); return False
    # Prefill detection (seqused <= q_len  =>  first prefill chunk, no prior context).
    # This needs a GPU->CPU sync (seqused_k.min().item()), which is ILLEGAL during a
    # CUDA graph capture. It is also UNNECESSARY there: vLLM only ever captures pure
    # DECODE batches (uniform q_len, all sequences have prior context), so a prefill
    # chunk can never reach this code while capturing. So: skip the sync (and the
    # check) iff we are currently capturing; keep it on the eager path where real
    # prefill chunks can appear. NSPLIT is now fixed (_NSPLIT_MAX) and no longer needs
    # the seqused_max sync at all.
    if not torch.cuda.is_current_stream_capturing():
        _lo = int(seqused_k.min().item())   # eager-only sync; capture path skips it
        if _lo <= int(max_seqlen_q):
            _dbg("prefill(seqused<=mq)"); return False
    dev = q.device
    total_q = q.shape[0]
    cu = cu_seqlens_q.to(torch.int64)
    num_seqs = cu.shape[0] - 1
    q_lens = cu[1:] - cu[:-1]                          # [num_seqs]; max == max_seqlen_q (already capped), no sync needed
    # map each query-token row -> its sequence and within-seq offset t
    rows = torch.arange(total_q, device=dev, dtype=torch.int64)
    seq_idx = torch.bucketize(rows, cu[1:], right=True)   # token row -> seq index
    seq_idx = seq_idx.clamp_(max=num_seqs - 1)
    t = rows - cu[seq_idx]
    su_full = seqused_k.to(torch.int64)[seq_idx]
    su = (su_full - q_lens[seq_idx] + 1 + t).to(torch.int32).contiguous()   # truncated per-token seq_len
    bt = block_table.to(torch.int32)[seq_idx].contiguous()                   # broadcast block_table per token
    NSPLIT = _NSPLIT_MAX   # FIXED -> static grid (grid.y) -> cudagraph-capturable
    cache_u8 = k_cache.reshape(k_cache.shape[0], block_size, num_kv_heads, _SB)
    # Write the final result DIRECTLY into the caller's `out` buffer (no extra alloc,
    # no copy). `out` is [total_q, NQH, Hv]; its flat row-major layout matches the
    # [num_seqs*NQH, Hv] view the reduce kernel writes (num_seqs == total_q here, one
    # work-row per (token,head)). When `out` is contiguous (the cudagraph/captured
    # case: vLLM's attention output is a persistent contiguous buffer) `reshape`
    # returns a VIEW, so the kernel writes land in `out` with zero extra work. For a
    # rare non-contiguous eager `out`, reshape would copy; guard against silently
    # dropping writes by routing through a contiguous temp + copy_ (eager-only path,
    # never inside a capture, so the temp alloc is safe).
    min_rows = _MAX_BATCH * q.shape[1]   # pre-grow static scratch to ceiling (no capture-time alloc)
    if out.is_contiguous():
        out_flat = out.view(total_q * q.shape[1], _HV)
        _M.run(q.contiguous(), cache_u8.contiguous() if not cache_u8.is_contiguous() else cache_u8,
               bt, su, out_flat, num_kv_heads, NSPLIT, float(softmax_scale), int(block_size), int(min_rows))
    else:
        out_flat = torch.empty((total_q * q.shape[1], _HV), dtype=out.dtype, device=dev)
        _M.run(q.contiguous(), cache_u8.contiguous() if not cache_u8.is_contiguous() else cache_u8,
               bt, su, out_flat, num_kv_heads, NSPLIT, float(softmax_scale), int(block_size), int(min_rows))
        out.copy_(out_flat.view(total_q, q.shape[1], _HV))
    global _CALLS
    _CALLS += 1
    if _CALLS == 1:   # one-time confirmation the kernel is live (file is reliable; worker stderr is not)
        try:
            with open("/tmp/wmma_trace.log", "a") as _f:
                _f.write(f"KERNEL_ACTIVE total_q={total_q} kvh={num_kv_heads} NSPLIT={NSPLIT}\n")
        except Exception:
            pass
    return True
