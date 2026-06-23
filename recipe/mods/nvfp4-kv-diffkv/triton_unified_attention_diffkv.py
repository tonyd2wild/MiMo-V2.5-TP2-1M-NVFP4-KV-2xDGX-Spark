# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton unified attention with different K/V head dimensions (DiffKV).

This is a slimmed fork of ``triton_unified_attention.py`` for models like
MiMo-V2 where the V tensor's head dimension differs from K's.  The KV cache
is the same packed layout used by ``FlashAttentionDiffKVBackend``:

    kv_cache: [num_blocks, block_size, num_kv_heads, head_size_qk + head_size_v]

We slice ``key_cache = kv_cache[..., :head_size_qk]`` and
``value_cache = kv_cache[..., head_size_qk:]`` on the host, so the kernel
takes two cache pointers but with two distinct head sizes.

Both 2D and 3D launches are supported:
  - 2D: one program per (q-block, kv-head); tile-loop walks the full KV
    sequence; final output written directly.  Used for prefill and large
    decode batches.
  - 3D: one program per (q-block, kv-head, segm); each program covers a
    KV slice and writes per-segment partials (max/expsum/output).  A
    follow-up ``kernel_reduce_segments_diffkv`` combines them.  Selected
    for decode-only batches whose 2D grid would under-fill the GPU.
"""

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_alibi_to_score,
    apply_softcap,
    cdiv_fn,
    compute_kv_seq_mask,
    compute_tile_loop_bounds,
    find_seq_idx,
    init_softmax_M,
    resolve_seq_and_query_len,
    softmax_step,
    store_segm_reduce_scalars,
)

logger = init_logger(__name__)


@triton.jit
def kernel_unified_attention_diffkv(
    # Output destinations.  In 2D mode we write the final result into
    # ``output_ptr``; in 3D mode we write per-segment partials into
    # ``segm_*`` and ``output_ptr`` is unused (callers may pass any
    # non-null pointer).
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    key_cache_ptr,  # view of packed cache: [..., :head_size_qk]
    value_cache_ptr,  # view of packed cache: [..., head_size_qk:hqk+hv]
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    scale,
    softcap,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,  # == HEAD_SIZE_QK
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,  # == HEAD_SIZE_V
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE_QK: tl.constexpr,
    HEAD_SIZE_QK_PADDED: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    # Strides for both cache views (they share the same packed buffer, so
    # dims 0/1/2 strides match; only the per-head extent differs).
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    # ``IS_3D`` toggles between 2D layout (one program walks the full KV
    # sequence) and 3D layout (split-KV / FlashDecoding-style: per-segm
    # programs write partials, finalized by ``kernel_reduce_segments_diffkv``).
    IS_3D: tl.constexpr,
    # EXPERIMENTAL nvfp4 inline-dequant: read packed 4-bit KV directly + dequant
    # in registers (no bf16 scratch). key/value_cache_ptr = packed uint8 cache;
    # scale_cache_ptr = same buffer viewed as fp8 (block scales); lut_ptr = e2m1.
    IS_NVFP4: tl.constexpr = False,
    scale_cache_ptr=None,
    lut_ptr=None,
    K_FP4_OFF: tl.constexpr = 0,
    K_BSF_OFF: tl.constexpr = 0,
    V_FP4_OFF: tl.constexpr = 0,
    V_BSF_OFF: tl.constexpr = 0,
    word_cache_ptr=None,
    stride_w0: tl.int64 = 0,
    stride_w1: tl.int64 = 0,
    stride_w2: tl.int64 = 0,
    stride_w3: tl.constexpr = 1,
    K_WORD_OFF: tl.constexpr = 0,
    V_WORD_OFF: tl.constexpr = 0,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_d_qk = tl.arange(0, HEAD_SIZE_QK_PADDED)
    offs_d_v = tl.arange(0, HEAD_SIZE_V_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d_qk[None, :]
    )

    dim_mask_qk = tl.where(offs_d_qk < HEAD_SIZE_QK, 1, 0).to(tl.int1)
    dim_mask_v = tl.where(offs_d_v < HEAD_SIZE_V, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_QK_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask_qk[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # acc : (BLOCK_M, HEAD_SIZE_V_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_V_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        False,  # USE_MM_PREFIX
        IS_3D,
    )

    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        slot_in_block = seq_offset % BLOCK_SIZE
        if IS_NVFP4:
            # K : uint32-vectorized — load 8 nibbles/word (32-bit coalesced),
            # unpack via broadcast-shift, reshape [HQKp//8,8,TILE]->[HQKp,TILE].
            offs_wk = tl.arange(0, HEAD_SIZE_QK_PADDED // 8)
            wkmask = offs_wk < (HEAD_SIZE_QK // 8)
            kw_off = (
                physical_block_idx[None, :] * stride_w0
                + kv_head_idx * stride_w2
                + (K_WORD_OFF + offs_wk)[:, None] * stride_w3
                + slot_in_block[None, :] * stride_w1
            )
            kw = tl.load(word_cache_ptr + kw_off,
                         mask=wkmask[:, None] & tile_mask[None, :], other=0).to(tl.uint32)
            shk = (4 * tl.arange(0, 8)).to(tl.uint32)
            k_nib = ((kw[:, None, :] >> shk[None, :, None]) & 0xF).reshape(
                HEAD_SIZE_QK_PADDED, TILE_SIZE)
            k_e2m1 = tl.load(lut_ptr + k_nib)
            k_sf_off = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + (K_BSF_OFF + offs_d_qk // 16)[:, None] * stride_k_cache_3
                + slot_in_block[None, :] * stride_k_cache_1
            )
            k_sf = tl.load(
                scale_cache_ptr + k_sf_off,
                mask=dim_mask_qk[:, None] & tile_mask[None, :], other=0.0,
            ).to(tl.float32)
            K = (k_e2m1 * k_sf).to(Q.dtype)
            # V : (TILE_SIZE, HEAD_SIZE_V_PADDED)
            offs_wv = tl.arange(0, HEAD_SIZE_V_PADDED // 8)
            wvmask = offs_wv < (HEAD_SIZE_V // 8)
            vw_off = (
                physical_block_idx[:, None] * stride_w0
                + kv_head_idx * stride_w2
                + (V_WORD_OFF + offs_wv)[None, :] * stride_w3
                + slot_in_block[:, None] * stride_w1
            )
            vw = tl.load(word_cache_ptr + vw_off,
                         mask=wvmask[None, :] & tile_mask[:, None], other=0).to(tl.uint32)
            shv = (4 * tl.arange(0, 8)).to(tl.uint32)
            v_nib = ((vw[:, :, None] >> shv[None, None, :]) & 0xF).reshape(
                TILE_SIZE, HEAD_SIZE_V_PADDED)
            v_e2m1 = tl.load(lut_ptr + v_nib)  # LUT gather
            v_sf_off = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + (V_BSF_OFF + offs_d_v // 16)[None, :] * stride_v_cache_3
                + slot_in_block[:, None] * stride_v_cache_1
            )
            v_sf = tl.load(
                scale_cache_ptr + v_sf_off,
                mask=dim_mask_v[None, :] & tile_mask[:, None], other=0.0,
            ).to(tl.float32)
            V = (v_e2m1 * v_sf).to(Q.dtype)
        else:
            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d_v[None, :] * stride_v_cache_3
                + slot_in_block[:, None] * stride_v_cache_1
            )
            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d_qk[:, None] * stride_k_cache_3
                + slot_in_block[None, :] * stride_k_cache_1
            )
            # K : (HEAD_SIZE_QK_PADDED, TILE_SIZE)
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask_qk[:, None] & tile_mask[None, :],
                other=0.0,
            )
            K = K_load.to(Q.dtype)
            # V : (TILE_SIZE, HEAD_SIZE_V_PADDED)
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask_v[None, :] & tile_mask[:, None],
                other=0.0,
            )
            V = V_load.to(Q.dtype)

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            None,  # mm_prefix_range_ptr
            SLIDING_WINDOW,
            False,  # USE_MM_PREFIX
            0,  # MAX_MM_RANGES
        )

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )
        acc += tl.dot(P.to(V.dtype), V)

    # ---- Epilogue --------------------------------------------------------
    if IS_3D:
        # Store per-segment partials; finalized by reduce_segments_diffkv.
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
            + segm_idx * HEAD_SIZE_V_PADDED
            + tl.arange(0, HEAD_SIZE_V_PADDED)[None, :]
        )
        tl.store(
            segm_output_ptr + segm_output_offset,
            acc,
            mask=dim_mask_v[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        output_offset = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
            + offs_d_v[None, :]
        )
        tl.store(
            output_ptr + output_offset,
            acc,
            mask=dim_mask_v[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )


@triton.jit
def kernel_reduce_segments_diffkv(
    output_ptr,  # [num_tokens, num_query_heads, head_size_v]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size_v]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,
    num_query_heads: tl.constexpr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,  # == HEAD_SIZE_V
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    """Combine per-segment partials into the final softmax output.

    Mirrors ``reduce_segments`` from triton_unified_attention.py but
    indexes V's head size (``HEAD_SIZE_V``) instead of the shared one.
    """
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_V_PADDED) < HEAD_SIZE_V, 1, 0).to(
        tl.int1
    )

    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_V_PADDED
        + tl.arange(0, HEAD_SIZE_V_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_V_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def unified_attention_diffkv(
    q,  # [num_tokens, num_query_heads, head_size_qk]
    k,  # view: [num_blocks, block_size, num_kv_heads, head_size_qk]
    v,  # view: [num_blocks, block_size, num_kv_heads, head_size_v]
    out,  # [num_tokens, num_query_heads, head_size_v]
    cu_seqlens_q,
    seqused_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    max_seqlen_q: int = 1,
    alibi_slopes=None,
    sinks=None,
    use_alibi_sqrt=False,
    # 3D / split-KV softmax buffers.  When all four are provided and the
    # batch is decode-only with few sequences, the 3D path is taken.
    seq_threshold_3D: int | None = None,
    num_par_softmax_segments: int | None = None,
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
    # EXPERIMENTAL nvfp4 inline-dequant: k and v are the SAME packed uint8 cache
    # (last dim = packed bytes). scale_cache = same buffer viewed as fp8.
    nvfp4_packed: bool = False,
    scale_cache: torch.Tensor | None = None,
    e2m1_lut: torch.Tensor | None = None,
    head_size_v_override: int | None = None,
):
    assert causal, "Only causal attention is supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_alibi_slopes = alibi_slopes is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size_qk = q.shape[2]
    head_size_v = head_size_v_override if nvfp4_packed else v.shape[3]
    # packed nvfp4 region offsets within the last (byte) dim:
    #   [k_fp4 hqk//2 | k_bsf hqk//16 | v_fp4 hv//2 | v_bsf hv//16]
    _k_fp4_off = 0
    _k_bsf_off = head_size_qk // 2
    _v_fp4_off = head_size_qk // 2 + head_size_qk // 16
    _v_bsf_off = _v_fp4_off + head_size_v // 2
    _scale_ptr = scale_cache if nvfp4_packed else k
    _lut_ptr = e2m1_lut if nvfp4_packed else k
    _word_cache = k.view(torch.int32) if nvfp4_packed else k
    _sw0 = _word_cache.stride(0) if nvfp4_packed else 0
    _sw1 = _word_cache.stride(1) if nvfp4_packed else 0
    _sw2 = _word_cache.stride(2) if nvfp4_packed else 0
    _sw3 = _word_cache.stride(3) if nvfp4_packed else 1
    _k_word_off = _k_fp4_off // 4
    _v_word_off = _v_fp4_off // 4

    # EXPERIMENTAL: custom WMMA tensor-core flash-decode for the nvfp4 FULL-attention
    # decode case (2.3x faster than this Triton path on long context). Gated by
    # VLLM_WMMA_DECODE=1; returns False (falls through) for SWA/sinks/softcap/prefill
    # /unsupported shapes, and any exception falls back to Triton below.
    if nvfp4_packed and max_seqlen_q <= 3:   # decode + MTP (q_len<=num_spec+1); larger -> Triton prefill
        try:
            from vllm.v1.attention.ops.wmma_decode import try_wmma_decode
            if try_wmma_decode(q, k, out, seqused_k, block_table, softmax_scale,
                               num_kv_heads, head_size_qk, head_size_v, block_size,
                               sinks, softcap, window_size[0], cu_seqlens_q, max_seqlen_q):
                return
        except Exception as _wmma_e:  # pragma: no cover
            logger.warning_once(f"[wmma_decode] fell back to Triton: {_wmma_e}")

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Decide between 2D and 3D launch.  Mirrors the standard launcher:
    # 3D requires preallocated softmax buffers, decode-only batches, and
    # a small number of sequences (otherwise 2D already saturates the SM).
    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
    )

    # Tile size: 32 for prefill-class kernels.  Decode (small Q) prefers
    # smaller tiles to expose more parallelism along the KV dim.
    tile_size = 32 if not use_3d else (16 if q.element_size() >= 2 else 32)
    if nvfp4_packed:
        tile_size = min(tile_size, 16)  # tile32 measured SLOWER (less occupancy)

    grid: tuple[Any, ...]
    if use_3d:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        segm_output_ptr = softmax_segm_output
        segm_max_ptr = softmax_segm_max
        segm_expsum_ptr = softmax_segm_expsum
        num_segments = num_par_softmax_segments
    else:
        grid = (total_num_q_blocks, num_kv_heads)
        # 2D never touches the segm tensors but Triton wants a non-null
        # pointer; reuse ``out``.
        segm_output_ptr = out
        segm_max_ptr = out
        segm_expsum_ptr = out
        num_segments = 1

    kernel_unified_attention_diffkv[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        scale=softmax_scale,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE_QK=head_size_qk,
        HEAD_SIZE_QK_PADDED=triton.next_power_of_2(head_size_qk),
        HEAD_SIZE_V=head_size_v,
        HEAD_SIZE_V_PADDED=triton.next_power_of_2(head_size_v),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=sliding_window_val,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        IS_3D=use_3d,
        IS_NVFP4=nvfp4_packed,
        scale_cache_ptr=_scale_ptr,
        lut_ptr=_lut_ptr,
        K_FP4_OFF=_k_fp4_off,
        K_BSF_OFF=_k_bsf_off,
        V_FP4_OFF=_v_fp4_off,
        V_BSF_OFF=_v_bsf_off,
        word_cache_ptr=_word_cache,
        stride_w0=_sw0,
        stride_w1=_sw1,
        stride_w2=_sw2,
        stride_w3=_sw3,
        K_WORD_OFF=_k_word_off,
        V_WORD_OFF=_v_word_off,
    )

    if use_3d:
        kernel_reduce_segments_diffkv[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            TILE_SIZE=tile_size,
            HEAD_SIZE_V=head_size_v,
            HEAD_SIZE_V_PADDED=triton.next_power_of_2(head_size_v),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
        )

    # EXPERIMENTAL in-engine correctness check: run WMMA kernel into a temp and compare
    # to the Triton result `out` (which is now filled). Logs rel-err to a file. Engine
    # output (out) is UNCHANGED — Triton stays authoritative during comparison.
    import os as _oscmp
    if _oscmp.environ.get("VLLM_WMMA_COMPARE", "0") == "1" and nvfp4_packed and max_seqlen_q <= 3 \
            and window_size[0] < 0 and sinks is None and (softcap == 0.0):
        try:
            from vllm.v1.attention.ops.wmma_decode import try_wmma_decode as _twd
            import torch as _tcmp
            _tmp = _tcmp.empty_like(out)
            if _twd(q, k, _tmp, seqused_k, block_table, softmax_scale, num_kv_heads,
                    head_size_qk, head_size_v, block_size, sinks, softcap,
                    window_size[0], cu_seqlens_q, max_seqlen_q, force=True):
                _rel = ((_tmp.float() - out.float()).norm() / (out.float().norm() + 1e-9)).item()
                with open("/tmp/wmma_cmp.log", "a") as _fc:
                    _fc.write(f"rel={_rel:.4f} q={tuple(q.shape)} bs={block_size} kvh={num_kv_heads} maxL={int(seqused_k.max())} mq={max_seqlen_q}\n")
                if _oscmp.environ.get("VLLM_WMMA_AUTHCOMPARE", "0") == "1":
                    out.copy_(_tmp)   # make WMMA authoritative (my kernel drives trajectory) while logging per-step rel
        except Exception as _ecmp:
            try:
                with open("/tmp/wmma_cmp.log", "a") as _fc:
                    _fc.write(f"CMP_ERR {type(_ecmp).__name__}: {_ecmp}\n")
            except Exception:
                pass
