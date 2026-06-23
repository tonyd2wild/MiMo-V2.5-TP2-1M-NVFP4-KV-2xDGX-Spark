# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton attention backend with different K/V head dimensions (DiffKV).

The KV cache layout is identical to ``FlashAttentionDiffKVBackend`` — K
and V are packed along the last dim:

    [num_blocks, block_size, num_kv_heads, head_size_qk + head_size_v]

so existing helpers (``triton_reshape_and_cache_flash_diffkv``) are reused.
"""

from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# EXPERIMENTAL nvfp4 KV cache (DiffKV) — store + dequant helpers.
# Packed slot layout per (block, slot, head), uint8:
#   [ k_fp4: hqk//2 | k_bsf: hqk//16 (fp8) | v_fp4: hv//2 | v_bsf: hv//16 (fp8) ]
# Global scale fixed at 1.0 (per-group-16 fp8 block scales carry magnitude).
# Validated standalone: K/V recon rel-err ~0.095, attn-output ~0.12.
# ---------------------------------------------------------------------------
import os as _os
_VLLM_NVFP4_INLINE = _os.environ.get("VLLM_NVFP4_INLINE", "0") == "1"
logger.info("[nvfp4-kv-diffkv] inline-dequant kernel %s (VLLM_NVFP4_INLINE=%s)",
            "ENABLED" if _VLLM_NVFP4_INLINE else "disabled (scratch path)",
            _os.environ.get("VLLM_NVFP4_INLINE"))
_NVFP4_LUT = {}

def _nvfp4_lut(device):
    import torch
    g = _NVFP4_LUT.get(device)
    if g is None:
        g = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                          -0., -.5, -1, -1.5, -2, -3, -4, -6],
                         device=device, dtype=torch.float32)
        _NVFP4_LUT[device] = g
    return g

_NVFP4_KV_GS = {}

def _nvfp4_gs(device):
    g = _NVFP4_KV_GS.get(device)
    if g is None:
        import torch as _t
        g = _t.tensor(1.0, device=device, dtype=_t.float32)
        _NVFP4_KV_GS[device] = g
    return g

import triton as _triton
import triton.language as _tl

@_triton.jit
def _nvfp4_scatter_kernel(packed_ptr, slot_ptr, cache_ptr, NH, BS, SB,
                          s_pt, s_ph, s_cb, s_cs, s_ch,
                          BLK: _tl.constexpr):
    # grid (T*NH,): write packed[t,h,:SB] -> cache[slot//BS, slot%BS, h, :SB]
    # In-kernel slot<0 skip (no host sync) + static shapes => cudagraph-capturable.
    pid = _tl.program_id(0)
    t = pid // NH
    h = pid % NH
    slot = _tl.load(slot_ptr + t)
    if slot < 0:
        return
    blk = slot // BS
    off = slot % BS
    o = _tl.arange(0, BLK)
    m = o < SB
    src = _tl.load(packed_ptr + t * s_pt + h * s_ph + o, mask=m, other=0)
    dst = cache_ptr + blk * s_cb + off * s_cs + h * s_ch + o
    _tl.store(dst, src, mask=m)

def _nvfp4_store_diffkv(key, value, kv_cache, slot_mapping):
    # Capturable nvfp4 store: quantize ALL tokens (static), then masked scatter.
    # flashinfer nvfp4_kv_quantize is cudagraph-safe; the OLD host-side filter
    # (bool(torch.all)+boolean-mask indexing) was the only capture-breaker.
    import torch
    from flashinfer import nvfp4_kv_quantize
    T, NH, Hk = key.shape
    Hv = value.shape[2]
    BS = kv_cache.shape[1]
    kfb, kbb, vfb, vbb = Hk // 2, Hk // 16, Hv // 2, Hv // 16
    SB = kfb + kbb + vfb + vbb
    gs = _nvfp4_gs(key.device)
    kf, kbsf = nvfp4_kv_quantize(key.reshape(-1, Hk).contiguous(), gs)
    vf, vbsf = nvfp4_kv_quantize(value.reshape(-1, Hv).contiguous(), gs)
    packed = torch.cat([kf.view(T, NH, kfb), kbsf.view(T, NH, kbb),
                        vf.view(T, NH, vfb), vbsf.view(T, NH, vbb)],
                       dim=2).contiguous()  # [T, NH, SB] uint8
    c = kv_cache
    BLK = _triton.next_power_of_2(SB)
    _nvfp4_scatter_kernel[(T * NH,)](
        packed, slot_mapping, c, NH, BS, SB,
        packed.stride(0), packed.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        BLK=BLK,
    )

def _nvfp4_dequant_active_blocks(kv_cache, block_table, Hk, Hv, out_dtype):
    """Dequant ONLY the physical blocks referenced by block_table into a small
    bf16 scratch (sized to active blocks, not the full pool), and return a
    remapped block_table that indexes the scratch. Avoids full-cache alloc."""
    import torch
    from flashinfer import nvfp4_kv_dequantize
    nblk, BS, NH, _ = kv_cache.shape
    kfb, kbb, vfb, vbb = Hk // 2, Hk // 16, Hv // 2, Hv // 16
    gs = _nvfp4_gs(kv_cache.device)
    bt = block_table
    active = torch.unique(bt[bt >= 0]).to(torch.long)
    nact = int(active.numel())
    if nact == 0:
        scratch = torch.zeros(1, BS, NH, Hk + Hv, dtype=out_dtype, device=kv_cache.device)
        return scratch, torch.zeros_like(bt)
    blk = kv_cache[active]  # [nact, BS, NH, slot_bytes]
    kf = blk[..., 0:kfb].reshape(-1, kfb).contiguous()
    kbsf = blk[..., kfb:kfb + kbb].reshape(-1, kbb).contiguous()
    vf = blk[..., kfb + kbb:kfb + kbb + vfb].reshape(-1, vfb).contiguous()
    vbsf = blk[..., kfb + kbb + vfb:].reshape(-1, vbb).contiguous()
    Kd = nvfp4_kv_dequantize(kf, kbsf, gs, output_dtype=out_dtype).reshape(nact, BS, NH, Hk)
    Vd = nvfp4_kv_dequantize(vf, vbsf, gs, output_dtype=out_dtype).reshape(nact, BS, NH, Hv)
    scratch = torch.empty(nact, BS, NH, Hk + Hv, dtype=out_dtype, device=kv_cache.device)
    scratch[:, :, :, :Hk] = Kd
    scratch[:, :, :, Hk:] = Vd
    # remap: physical block id -> scratch index (active order); padding (-1) -> 0
    remap = torch.zeros(nblk, dtype=bt.dtype, device=bt.device)
    remap[active] = torch.arange(nact, dtype=bt.dtype, device=bt.device)
    remapped_bt = torch.where(bt >= 0, remap[bt.clamp(min=0)], bt)
    return scratch, remapped_bt


class TritonAttentionDiffKVMetadataBuilder(TritonAttentionMetadataBuilder):
    """Override the parent's softmax buffer last-dim to head_size_v.

    The parent allocates ``softmax_segm_output`` with last-dim sized to
    ``next_power_of_2(head_size)`` (== Q/K head size).  For DiffKV the
    accumulator and per-segment partial outputs are V-shaped, so we
    re-allocate with ``next_power_of_2(head_size_v)`` instead.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        head_size_v = TritonAttentionDiffKVBackend.head_size_v
        head_size_v_padded = next_power_of_2(head_size_v)
        self.softmax_segm_output = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
                head_size_v_padded,
            ),
            dtype=torch.float32,
            device=device,
        )


class TritonAttentionDiffKVBackend(TritonAttentionBackend):
    # V head dim — set per layer via ``set_head_size_v`` before instantiation.
    head_size_v: int = 128

    # No FP8 / int8 KV cache for the DiffKV path yet; require fp16/bf16/fp32.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "nvfp4",  # EXPERIMENTAL_DIFFKV_NVFP4_SHAPE — store/decode kernels still TODO
    ]

    @classmethod
    def set_head_size_v(cls, head_size_v: int) -> None:
        cls.head_size_v = head_size_v

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN_DIFFKV"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionDiffKVImpl"]:
        return TritonAttentionDiffKVImpl

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionDiffKVMetadataBuilder"]:
        return TritonAttentionDiffKVMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # EXPERIMENTAL_DIFFKV_NVFP4_SHAPE: when nvfp4 requested, return packed
        # shape: per-head (head_size//2 + head_size//16) bytes = fp4 data + fp8
        # block scales. Total dim = sum over K and V sides.
        if cache_dtype_str == "nvfp4":
            from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim
            packed_k = nvfp4_kv_cache_full_dim(head_size)
            packed_v = nvfp4_kv_cache_full_dim(TritonAttentionDiffKVBackend.head_size_v)
            return (num_blocks, block_size, num_kv_heads, packed_k + packed_v)
        return (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size + TritonAttentionDiffKVBackend.head_size_v,
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, block_size,
            #  num_kv_heads, head_size + head_size_v)
            return (1, 0, 2, 3, 4)
        elif cache_layout == "NHD":
            return (0, 1, 2, 3)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers,
            #  block_size, head_size + head_size_v)
            return (1, 3, 0, 2, 4)
        elif cache_layout == "HND":
            return (0, 2, 1, 3)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # DiffKV K head sizes (e.g. 192 for MiMo-V2.5) need to be allowed.
        return head_size >= 32


class TritonAttentionDiffKVImpl(TritonAttentionImpl):
    """Triton attention impl for the DiffKV packed KV cache layout."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # EXPERIMENTAL_ALLOW_DIFFKV_QUANT_KV: kv-cache-dtype quantization
        # was conservatively blocked here but the underlying store kernel
        # (triton_reshape_and_cache_flash_diffkv) already accepts dtype +
        # scales. Allow it to fall through; revert if output becomes garbage.
        if False and is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not yet support quantized "
                f"KV cache (got kv_cache_dtype={self.kv_cache_dtype!r})."
            )
        if self._is_per_token_head_quant:
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not support per-token-head "
                "quantization."
            )
        if self.chunk_lookback > -1:
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not support chunked "
                "attention with lookback."
            )

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        # EXPERIMENTAL nvfp4 KV: quantize+pack into the packed cache layout.
        # Detect via cache shape (packed last-dim < hqk+hv), NOT the dtype
        # string: self.kv_cache_dtype can be a stale "auto" while the cache is
        # actually allocated packed (cache_config resolved to nvfp4 later).
        if kv_cache.shape[-1] != key.shape[2] + value.shape[2]:
            _nvfp4_store_diffkv(key, value, kv_cache, slot_mapping)
            return
        # Cache is packed [..., head_size_qk + head_size_v]; the diffkv
        # reshape kernel writes K to [..., :head_size_qk] and V to
        # [..., head_size_qk:hqk+hv].
        triton_reshape_and_cache_flash_diffkv(
            key,
            value,
            kv_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        # The fused rope+cache path assumes the standard 2-tensor layout.
        return False

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Shapes:
            query:    [num_tokens, num_heads, head_size_qk]
            key:      [num_tokens, num_kv_heads, head_size_qk]
            value:    [num_tokens, num_kv_heads, head_size_v]
            kv_cache: [num_blocks, block_size, num_kv_heads,
                       head_size_qk + head_size_v]
            output:   [num_tokens, num_heads, head_size_v]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported for "
                "TritonAttentionDiffKVImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.use_cascade is False, (
            "Cascade attention not supported for TritonAttentionDiffKVImpl"
        )
        assert self.attn_type not in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ), "Encoder attention not supported for TritonAttentionDiffKVImpl"

        num_actual_tokens = attn_metadata.num_actual_tokens
        head_size_qk = self.head_size
        head_size_v = TritonAttentionDiffKVBackend.head_size_v

        # EXPERIMENTAL nvfp4 KV: cache is packed 4-bit (last-dim < hqk+hv).
        nvfp4_block_table = attn_metadata.block_table
        is_packed = kv_cache.shape[-1] != head_size_qk + head_size_v
        nvfp4_kwargs = {}
        if is_packed and _VLLM_NVFP4_INLINE:
            # INLINE: read packed fp4 directly in the fused kernel + dequant in
            # registers (no bf16 scratch). Passes the packed cache as K and V,
            # plus its fp8-view for block scales and the e2m1 LUT.
            import torch as _t
            key_cache = kv_cache
            value_cache = kv_cache
            nvfp4_kwargs = dict(
                nvfp4_packed=True,
                scale_cache=kv_cache.view(_t.float8_e4m3fn),
                e2m1_lut=_nvfp4_lut(kv_cache.device),
                head_size_v_override=head_size_v,
            )
        elif is_packed:
            # SCRATCH (correct-first): dequant active blocks -> bf16 + remap.
            kv_cache, nvfp4_block_table = _nvfp4_dequant_active_blocks(
                kv_cache, attn_metadata.block_table,
                head_size_qk, head_size_v, query.dtype,
            )
            key_cache = kv_cache[..., :head_size_qk]
            value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]
        else:
            key_cache = kv_cache[..., :head_size_qk]
            value_cache = kv_cache[..., head_size_qk : head_size_qk + head_size_v]

        unified_attention_diffkv(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            seqused_k=attn_metadata.seq_lens,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=nvfp4_block_table,
            softcap=self.logits_soft_cap,
            sinks=self.sinks,
            max_seqlen_q=attn_metadata.max_query_len,
            seq_threshold_3D=attn_metadata.seq_threshold_3D,
            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
            softmax_segm_output=attn_metadata.softmax_segm_output,
            softmax_segm_max=attn_metadata.softmax_segm_max,
            softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
            **nvfp4_kwargs,
        )
        return output


# Pre-compile the experimental WMMA decode kernel at startup (module import = backend
# registration, before any forward). Avoids the ~60s inline load_inline() stall on the
# FIRST decode forward, which under TP desyncs the ranks -> NCCL watchdog -> EngineDead.
if _os.environ.get("VLLM_WMMA_DECODE", "1") != "0":
    try:
        from vllm.v1.attention.ops.wmma_decode import _compile as _wmma_precompile
        if _wmma_precompile():
            logger.info("[wmma_decode] pre-compiled WMMA decode kernel at startup")
        else:
            logger.warning("[wmma_decode] pre-compile returned False (kernel disabled)")
    except Exception as _wmma_pe:
        logger.warning("[wmma_decode] pre-compile failed: %s", _wmma_pe)
