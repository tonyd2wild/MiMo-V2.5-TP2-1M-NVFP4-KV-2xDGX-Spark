#!/bin/bash
set -euo pipefail

SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
PR41797_URL="https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/41797.diff"

cd "$SITE_PACKAGES"

echo "[fix-mimo-v2-vllm] Applying MiMo V2.5 vLLM fixes"

# EXPERIMENTAL: enable quantized KV cache on TRITON_ATTN_DIFFKV.
# The DIFFKV impl raises NotImplementedError on any quantized kv_cache_dtype,
# but the underlying store kernel (triton_reshape_and_cache_flash_diffkv)
# already accepts dtype + k_scale/v_scale params. The check is defensive,
# not technical. Without it, KV cache halves (fp8) or quarters (nvfp4),
# unlocking larger KV pool → more concurrency or context.
# Revert by removing the patch line and restoring the NotImplementedError.
DIFFKV="vllm/v1/attention/backends/triton_attn_diffkv.py"
if [ -f "$DIFFKV" ] && grep -q 'does not yet support quantized' "$DIFFKV" \
   && ! grep -q 'EXPERIMENTAL_ALLOW_DIFFKV_QUANT_KV' "$DIFFKV"; then
    python3 - <<'PY'
import re, pathlib
p = pathlib.Path("vllm/v1/attention/backends/triton_attn_diffkv.py")
src = p.read_text()
old = """        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not yet support quantized "
                f"KV cache (got kv_cache_dtype={self.kv_cache_dtype!r})."
            )"""
new = """        # EXPERIMENTAL_ALLOW_DIFFKV_QUANT_KV: kv-cache-dtype quantization
        # was conservatively blocked here but the underlying store kernel
        # (triton_reshape_and_cache_flash_diffkv) already accepts dtype +
        # scales. Allow it to fall through; revert if output becomes garbage.
        if False and is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not yet support quantized "
                f"KV cache (got kv_cache_dtype={self.kv_cache_dtype!r})."
            )"""
if old in src:
    p.write_text(src.replace(old, new))
    print('[fix-mimo-v2-vllm] patched DIFFKV: removed NotImplementedError on quantized KV (EXPERIMENTAL_ALLOW_DIFFKV_QUANT_KV)')
PY
fi

# EXPERIMENTAL Phase 1 Items 1+2: declare nvfp4 + return packed shape.
# vLLM's kv_cache_interface already has DIFFKV-NVFP4 plumbing
# (FullAttentionSpec.real_page_size_bytes nvfp4 branch with head_size_v).
# This patch adds the missing backend declaration so vLLM accepts the dtype.
if [ -f "$DIFFKV" ] && ! grep -q 'EXPERIMENTAL_DIFFKV_NVFP4_SHAPE' "$DIFFKV"; then
    python3 - <<'PY'
import pathlib
p = pathlib.Path("vllm/v1/attention/backends/triton_attn_diffkv.py")
src = p.read_text()

# Item 1: add nvfp4 to supported list
old1 = '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]'''
new1 = '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "nvfp4",  # EXPERIMENTAL_DIFFKV_NVFP4_SHAPE — store/decode kernels still TODO
    ]'''
if old1 in src:
    src = src.replace(old1, new1)

# Item 2: get_kv_cache_shape returns packed shape when nvfp4
old2 = '''    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size + TritonAttentionDiffKVBackend.head_size_v,
        )'''
new2 = '''    @staticmethod
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
        )'''
if old2 in src:
    src = src.replace(old2, new2)

p.write_text(src)
print('[fix-mimo-v2-vllm] patched DIFFKV: declared nvfp4 + packed shape (EXPERIMENTAL_DIFFKV_NVFP4_SHAPE — store/decode kernels TODO)')
PY
fi

# Apply vllm-project/vllm PR #42969 — Qwen3XMLToolParser duplicate close fix.
# The "mimo" tool_call_parser is registered as alias to Qwen3XMLToolParser
# in tool_parsers/__init__.py; without this patch, streaming function close
# emits duplicate '}' / leaves stale current_function_name causing the client
# to reject the call as "Model generated invalid tool call".
QXML="vllm/tool_parsers/qwen3xml_tool_parser.py"
if [ -f "$QXML" ] && grep -q "self.current_function_open = False$" "$QXML" \
   && ! grep -q "self.current_function_name = None  # PR42969" "$QXML"; then
    sed -i 's|^\(            self.current_function_open = False\)$|\1\n            self.current_function_name = None  # PR42969|' "$QXML"
    echo "[fix-mimo-v2-vllm] patched Qwen3XMLToolParser (PR #42969 — clear current_function_name on function end)"
fi

# CyberTen forum note: vLLM's multimodal audio path uses soundfile + librosa,
# not torchcodec. Keep this harmless for text-only runs and necessary for audio.
# PyAV is the fallback decoder vLLM tries when soundfile cannot read the file;
# without it the placeholder module raises "Please install vllm[audio]".
python3 - <<'PY' || uv pip install --quiet soundfile librosa av
import soundfile  # noqa: F401
import librosa  # noqa: F401
import av  # noqa: F401
PY

# Some current vLLM builds have the MiMo V2 model class but not a HF config
# registry entry for model_type=mimo_v2. Transformers then tries to fetch a
# nonexistent remote configuration_mimo_v2.py from the NVFP4 repo and aborts
# before vLLM can select its local model implementation.
echo "[fix-mimo-v2-vllm] Installing local MiMoV2Config registration if needed"
cat > "$SITE_PACKAGES/vllm/transformers_utils/configs/mimo_v2.py" <<'PY'
# SPDX-License-Identifier: Apache-2.0
from transformers import PretrainedConfig


class MimoV2Config(PretrainedConfig):
    model_type = "mimo_v2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
PY
python3 - <<'PY'
from pathlib import Path
site = Path('/usr/local/lib/python3.12/dist-packages')
init = site / 'vllm/transformers_utils/configs/__init__.py'
text = init.read_text()
if '"MimoV2Config": "vllm.transformers_utils.configs.mimo_v2"' not in text:
    text = text.replace(
        '    "MiDashengLMConfig": "vllm.transformers_utils.configs.midashenglm",\n',
        '    "MiDashengLMConfig": "vllm.transformers_utils.configs.midashenglm",\n'
        '    "MimoV2Config": "vllm.transformers_utils.configs.mimo_v2",\n',
    )
if '    "MimoV2Config",\n' not in text:
    text = text.replace(
        '    "MiDashengLMConfig",\n',
        '    "MiDashengLMConfig",\n'
        '    "MimoV2Config",\n',
    )
init.write_text(text)

cfg = site / 'vllm/transformers_utils/config.py'
text = cfg.read_text()
if 'mimo_v2="MimoV2Config"' not in text:
    text = text.replace(
        '    midashenglm="MiDashengLMConfig",\n',
        '    midashenglm="MiDashengLMConfig",\n'
        '    mimo_v2="MimoV2Config",\n',
    )
cfg.write_text(text)
PY

# Respect an explicit text-only architecture override. Current vLLM's MiMoV2
# arch convertor unconditionally rewrites any config containing vision_config to
# MiMoV2OmniForCausalLM, even when the launch passes
# --hf-overrides '{"architectures":["MiMoV2ForCausalLM"]}'. That makes text-only
# bring-up load the vision/audio modules and currently fails on missing merger
# bias weights in this NVFP4 export.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/model_arch_config_convertor.py')
text = path.read_text()
old = '''class MimoV2ModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def __init__(self, hf_config: PretrainedConfig, hf_text_config: PretrainedConfig):
        if getattr(hf_config, "vision_config", None):
            hf_config.architectures = ["MiMoV2OmniForCausalLM"]
        super().__init__(hf_config, hf_text_config)
        _strip_mimo_v2_attention_chunk_size(hf_config, hf_text_config)
'''
new = '''class MimoV2ModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def __init__(self, hf_config: PretrainedConfig, hf_text_config: PretrainedConfig):
        # Preserve explicit text-only override for MiMo-V2.5 NVFP4 bring-up.
        # The checkpoint config includes vision/audio sections, but text-only
        # serving should use MiMoV2ForCausalLM when requested via hf_overrides.
        if getattr(hf_config, "vision_config", None) and getattr(
            hf_config, "architectures", None
        ) != ["MiMoV2ForCausalLM"]:
            hf_config.architectures = ["MiMoV2OmniForCausalLM"]
        super().__init__(hf_config, hf_text_config)
        _strip_mimo_v2_attention_chunk_size(hf_config, hf_text_config)
'''
if old in text:
    path.write_text(text.replace(old, new, 1))
    print('[fix-mimo-v2-vllm] patched MimoV2 arch convertor to preserve text-only override')
elif 'Preserve explicit text-only override for MiMo-V2.5 NVFP4 bring-up' in text:
    print('[fix-mimo-v2-vllm] MimoV2 arch convertor already patched')
else:
    raise SystemExit('[fix-mimo-v2-vllm] ERROR: MimoV2 arch convertor pattern not found')
PY

# Text-only MiMoV2ForCausalLM should ignore multimodal and MTP weights present
# in the Omni/NVFP4 checkpoint. AutoWeightsLoader otherwise treats e.g.
# visual.* as an unknown module and aborts.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py')
text = path.read_text()
old = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
'''
new = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def text_only_weights():
            skip_prefixes = (
                "visual.",
                "audio_encoder.",
                "speech_embeddings.",
                "model.mtp.",
            )
            for name, tensor in weights:
                if name.startswith(skip_prefixes):
                    continue
                yield name, tensor

        loader = AutoWeightsLoader(self)
        return loader.load_weights(text_only_weights())
'''
if old in text:
    path.write_text(text.replace(old, new, 1))
    print('[fix-mimo-v2-vllm] patched MiMoV2ForCausalLM.load_weights to skip non-text checkpoint tensors')
elif 'def text_only_weights():' in text and 'speech_embeddings.' in text:
    print('[fix-mimo-v2-vllm] MiMoV2 text-only weight filter already patched')
else:
    raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2 load_weights pattern not found')
PY

# Omni/multimodal fixes for this checkpoint:
# - The reference MiMo vision merger uses biased Linear layers and the NVFP4
#   checkpoint contains visual.merger.mlp.{0,2}.bias.  vLLM's local MiMo copy
#   had these as bias=False, causing unknown/missing bias handling and an
#   architecture mismatch.
# - Target Omni model loading should skip MTP weights; the MTP drafter loads
#   those separately.
# - The top-level Omni class is SupportsQuant, so it is the class that mutates
#   ModelOptMixedPrecisionConfig.  Without a packed_modules_mapping on Omni,
#   nested language_model.model.layers.*.mlp.gate_up_proj does not resolve the
#   checkpoint's separate gate_proj/up_proj quantization entries and is treated
#   as unquantized, corrupting the target text path even for text-only requests.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py')
text = path.read_text()
orig = text
old = '''class MiMoV2OmniForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant):
    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
'''
new = '''class MiMoV2OmniForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant):
    # Ensure ModelOpt mixed-precision resolves fused language/MTP modules after
    # the Omni hf_to_vllm prefix mapper rewrites model.* -> language_model.model.*.
    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2Omni class anchor not found')
    text = text.replace(old, new, 1)
text = text.replace(
'''            ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.0",
''',
'''            ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.0",
''',
1)
text = text.replace(
'''            RowParallelLinear(
                self.hidden_size,
                d_model,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.2",
''',
'''            RowParallelLinear(
                self.hidden_size,
                d_model,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.2",
''',
1)
old = '''        loader = AutoWeightsLoader(self, skip_prefixes=["audio_tokenizer."])
        auto_loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
'''
new = '''        loader = AutoWeightsLoader(
            self,
            skip_prefixes=[
                # PATCH: do NOT skip audio_tokenizer — required at runtime when
                # audio inputs are sent. Recipe-default skipped it which causes
                # "audio_tokenizer is not loaded" RuntimeError in
                # mimo_audio.py:1353 on first audio_url request.
                # After hf_to_vllm_mapper, checkpoint model.mtp.* becomes
                # language_model.model.mtp.*.  The target model should ignore
                # it; MiMoV2MTP/OmniMTP loads these weights separately.
                "language_model.model.mtp.",
            ],
        )
        auto_loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
'''
if new not in text:
    if old not in text:
        print('[fix-mimo-v2-vllm] MiMoV2Omni load_weights skip pattern not found; skipping audio/MTP skip patch')
    else:
        text = text.replace(old, new, 1)
omni_mtp_skip_old = '''                # After hf_to_vllm_mapper, checkpoint model.mtp.* becomes
                # language_model.model.mtp.*.  The target model should ignore
                # it; MiMoV2MTP/OmniMTP loads these weights separately.
                "language_model.model.mtp.",
'''
omni_mtp_skip_new = '''                # Skip raw and mapped MTP tensors on the target model; the
                # MiMoV2MTP/OmniMTP draft model loads these weights separately.
                "model.mtp.",
                "language_model.model.mtp.",
'''
if omni_mtp_skip_old in text:
    text = text.replace(omni_mtp_skip_old, omni_mtp_skip_new, 1)
    print('[fix-mimo-v2-vllm] patched MiMoV2Omni target loader to skip raw model.mtp tensors')
elif omni_mtp_skip_new in text:
    print('[fix-mimo-v2-vllm] MiMoV2Omni target raw/mapped MTP skip already patched')
if text != orig:
    path.write_text(text)
    print('[fix-mimo-v2-vllm] patched MiMoV2Omni vision merger bias and MTP skip')
else:
    print('[fix-mimo-v2-vllm] MiMoV2Omni multimodal fixes already patched')
PY

# MiMo-V2.5-NVFP4/chimera stores fused qkv_proj tensors as canonical
# [Q_all][K_all][V_all] (see checkpoint metadata: qkv-deinterleaved).  The
# upstream MiMoV2 loader has a Pro-format shortcut that blindly chunks the
# fused row dimension by TP rank; that is shape-correct but semantically wrong
# for this checkpoint because K/V slots receive Q rows.  Let QKVParallelLinear's
# native fused-QKV loader split Q/K/V and their MXFP8 scale rows correctly.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py')
text = path.read_text()
old = '''            # Support fused qkv_proj checkpoint (Pro format)
            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    loaded_weight = loaded_weight.chunk(tp_size, dim=0)[tp_rank]
                    default_weight_loader(param, loaded_weight)
                continue
'''
new = '''            # MiMo-V2.5-NVFP4/chimera stores fused qkv_proj tensors as
            # canonical [Q_all][K_all][V_all] (checkpoint metadata says
            # qkv-deinterleaved), not the native FP8 Pro TP-prepacked layout
            # [Q0 K0 V0][Q1 K1 V1]...
            #
            # A blind chunk(tp_size, dim=0) is shape-correct for TP=2 but
            # semantically corrupts K/V rows and the row-aligned MXFP8
            # weight_scale_inv tensors.  Use QKVParallelLinear.weight_loader so
            # each rank receives [Q_rank][K_rank][V_rank].
            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                continue
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2 qkv_proj loader shortcut pattern not found')
    path.write_text(text.replace(old, new, 1))
    print('[fix-mimo-v2-vllm] patched MiMoV2 fused qkv_proj loader to use QKVParallelLinear.weight_loader')
else:
    print('[fix-mimo-v2-vllm] MiMoV2 fused qkv_proj loader already patched')
PY

# Apply the same qkv-deinterleaved handling to the MiMo-V2 MTP draft model.
# The NVFP4 checkpoint includes MTP qkv_proj tensors in the same canonical
# [Q_all][K_all][V_all] layout.  Also keep duplicate parameter aliases when
# building params_dict so `.weight_scale_inv` checkpoint tensors load into the
# MXFP8 `weight_scale` alias registered by fix-modelopt-mixed-mxfp8.
#
# Critical Omni+MTP quant fix: MiMoV2OmniForCausalLM's hf_to_vllm_mapper rewrites
# checkpoint quantized_layers from model.* to language_model.model.* so the Omni
# target layers quantize correctly.  That same global QuantizationConfig is then
# reused for the MTP drafter, whose prefixes are still model.mtp.layers.*.  Mirror
# language_model.model.mtp.* quant metadata back to model.mtp.* when the draft
# class is initialized, otherwise OmniMTP layers silently become unquantized.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_mtp.py')
text = path.read_text()
orig = text
old = '''from .utils import _merge_multimodal_embeddings, maybe_prefix
'''
new = '''from .utils import WeightsMapper, _merge_multimodal_embeddings, maybe_prefix
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2MTP utils import pattern not found')
    text = text.replace(old, new, 1)

old = '''class MiMoV2MTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
'''
new = '''class MiMoV2MTP(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Undo the Omni target mapper for MTP draft quant metadata only.
            "language_model.model.mtp.": "model.mtp.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2MTP class anchor not found')
    text = text.replace(old, new, 1)

old = '''        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
'''
new = '''        # Keep duplicate aliases such as MXFP8 `weight_scale` /
        # `weight_scale_inv`; the checkpoint uses the latter.
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2MTP params_dict pattern not found')
    text = text.replace(old, new, 1)

old = '''            # Support fused qkv_proj checkpoint (Pro format).
            # The checkpoint is stored pre-sharded for TP=8 as
            # [Q_rank0, K_rank0, V_rank0, Q_rank1, ...], so splitting along
            # dim 0 with chunk(tp_size) gives each rank its Q+K+V slice for
            # both the FP8 weight and the block weight_scale_inv. This matches
            # how the main model loads the same layout.
            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    loaded_weight = loaded_weight.chunk(tp_size, dim=0)[tp_rank]
                    default_weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                continue
'''
new = '''            # MiMo-V2.5-NVFP4/chimera MTP qkv_proj tensors are canonical
            # [Q_all][K_all][V_all], not TP-prepacked Pro layout.  Use
            # QKVParallelLinear.weight_loader for Q/K/V-aware TP slicing of
            # both FP8 weights and row-aligned MXFP8 weight_scale_inv tensors.
            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                continue
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2MTP qkv_proj loader shortcut pattern not found')
    text = text.replace(old, new, 1)

if text != orig:
    path.write_text(text)
    print('[fix-mimo-v2-vllm] patched MiMoV2MTP quant mapping and qkv-deinterleaved MXFP8 loader')
else:
    print('[fix-mimo-v2-vllm] MiMoV2MTP quant mapping and loader already patched')
PY

# Expose the local-argmax fast path for MiMo MTP. vLLM's proposer can avoid a
# full-vocab TP all-gather when `use_local_argmax_reduction` is enabled, but only
# if the draft model implements get_top_tokens().
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_mtp.py')
text = path.read_text()
orig = text

old = '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.logits_processor(lm_head, hidden_states)
'''
new = '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.logits_processor(lm_head, hidden_states)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(lm_head, hidden_states)
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2MultiTokenPredictor compute_logits anchor not found')
    text = text.replace(old, new, 1)

old = '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, self.lm_head, spec_step_idx)
'''
new = '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, self.lm_head, spec_step_idx)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.get_top_tokens(hidden_states, self.lm_head, spec_step_idx)
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2MTP compute_logits anchor not found')
    text = text.replace(old, new, 1)

if text != orig:
    path.write_text(text)
    print('[fix-mimo-v2-vllm] patched MiMoV2MTP local argmax get_top_tokens')
else:
    print('[fix-mimo-v2-vllm] MiMoV2MTP local argmax get_top_tokens already patched')

import ast
tree = ast.parse(path.read_text(), filename=str(path))
class_node = next(
    (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'MiMoV2MTP'
    ),
    None,
)
if class_node is None or not any(
    isinstance(node, ast.FunctionDef) and node.name == 'get_top_tokens'
    for node in class_node.body
):
    raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2MTP.get_top_tokens validation failed')
print('[fix-mimo-v2-vllm] validated MiMoV2MTP.get_top_tokens')
PY

# Expose target-side top-token helpers.  This lets a guarded greedy MTP1 path
# avoid materializing full-vocab target logits when all the request needs is
# target argmax for rejection sampling.
python3 - <<'PY'
from pathlib import Path

patches = [
    (
        Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py'),
        '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
''',
        '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
''',
        'MiMoV2FlashForCausalLM',
    ),
    (
        Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py'),
        '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)
''',
        '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.language_model.get_top_tokens(hidden_states)
''',
        'MiMoV2OmniForCausalLM',
    ),
]

for path, old, new, class_name in patches:
    text = path.read_text()
    if new not in text:
        if old not in text:
            raise SystemExit(
                f'[fix-mimo-v2-vllm] ERROR: {class_name}.compute_logits anchor not found'
            )
        path.write_text(text.replace(old, new, 1))
        print(f'[fix-mimo-v2-vllm] patched {class_name}.get_top_tokens')
    else:
        print(f'[fix-mimo-v2-vllm] {class_name}.get_top_tokens already patched')

import ast
for path, _, _, class_name in patches:
    tree = ast.parse(path.read_text(), filename=str(path))
    class_node = next(
        (
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    if class_node is None or not any(
        isinstance(node, ast.FunctionDef) and node.name == 'get_top_tokens'
        for node in class_node.body
    ):
        raise SystemExit(
            f'[fix-mimo-v2-vllm] ERROR: {class_name}.get_top_tokens validation failed'
        )
print('[fix-mimo-v2-vllm] validated target get_top_tokens helpers')
PY

# EXPERIMENTAL: greedy MTP1 target top-token fast path.  For plain greedy MTP1
# requests, rejection sampling only needs target argmax for each draft/bonus
# row.  When VLLM_MIMO_MTP1_GREEDY_FAST=1 and the strict guard passes, compute
# top-token ids instead of full target logits, then build the MTP1 sampler
# output directly.  Any feature needing logits falls back to normal vLLM.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
text = path.read_text()
orig = text

if '\nimport os\n' not in text:
    if 'import itertools\n' not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: import anchor not found')
    text = text.replace('import itertools\n', 'import itertools\nimport os\n', 1)

old = '''    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
'''
new = '''    logits: torch.Tensor | None
    greedy_spec_top_token_ids: torch.Tensor | None
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: ExecuteModelState anchor not found')
    text = text.replace(old, new, 1)

old = '''    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
'''
new = '''    def _mimo_mtp1_greedy_fast_trace(
        self,
        status: str,
        reason: str,
    ) -> None:
        if os.environ.get("VLLM_MIMO_MTP1_GREEDY_FAST_TRACE", "0") != "1":
            return
        count = getattr(self, "_mimo_mtp1_greedy_fast_trace_count", 0) + 1
        self._mimo_mtp1_greedy_fast_trace_count = count
        if count > 128 and count % 1024 != 0:
            return
        path = os.environ.get(
            "VLLM_MIMO_MTP1_GREEDY_FAST_TRACE_PATH",
            "/tmp/mimo_mtp1_greedy_fast_trace.log",
        )
        try:
            with open(path, "a", encoding="utf-8") as trace_file:
                trace_file.write(
                    f"{status} reason={reason} count={count} "
                    f"num_spec={getattr(self, 'num_spec_tokens', None)}\\n"
                )
        except Exception:
            pass

    def _mimo_mtp1_greedy_fast_guard(
        self,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> bool:
        if os.environ.get("VLLM_MIMO_MTP1_GREEDY_FAST", "0") != "1":
            return False
        trace_enabled = (
            os.environ.get("VLLM_MIMO_MTP1_GREEDY_FAST_TRACE", "0") == "1"
        )
        if spec_decode_metadata is None or self.num_spec_tokens != 1:
            if trace_enabled:
                self._mimo_mtp1_greedy_fast_trace(
                    "miss", "metadata_or_num_spec"
                )
            return False
        if not hasattr(self.model, "get_top_tokens"):
            if trace_enabled:
                self._mimo_mtp1_greedy_fast_trace("miss", "no_get_top_tokens")
            return False
        if any(num_draft != 1 for num_draft in spec_decode_metadata.num_draft_tokens):
            if trace_enabled:
                self._mimo_mtp1_greedy_fast_trace("miss", "draft_tokens_not_one")
            return False
        sampling_metadata = self.input_batch.sampling_metadata
        logitsprocs = sampling_metadata.logitsprocs
        active_logitsprocs = list(logitsprocs.argmax_invariant)
        for processor in logitsprocs.non_argmax_invariant:
            if (
                processor.__class__.__name__ == "MinTokensLogitsProcessor"
                and not getattr(processor, "min_toks", {})
            ):
                continue
            active_logitsprocs.append(processor)
        has_logitsprocs = bool(active_logitsprocs)
        thinking_holder = sampling_metadata.thinking_budget_state_holder
        has_thinking_budget = (
            thinking_holder is not None
            and thinking_holder.has_tracked_requests()
        )
        checks = (
            ("all_greedy", sampling_metadata.all_greedy),
            ("no_max_logprobs", sampling_metadata.max_num_logprobs is None),
            ("no_logprob_token_ids", not sampling_metadata.logprob_token_ids),
            ("no_penalties", sampling_metadata.no_penalties),
            (
                "no_allowed_token_mask",
                sampling_metadata.allowed_token_ids_mask is None,
            ),
            ("no_bad_words", not sampling_metadata.bad_words_token_ids),
            (
                "no_logits_processors",
                not has_logitsprocs,
            ),
            ("no_prompt_logprobs", not self.num_prompt_logprobs),
            ("no_thinking_budget", not has_thinking_budget),
        )
        for name, ok in checks:
            if not ok:
                if trace_enabled:
                    self._mimo_mtp1_greedy_fast_trace("miss", name)
                return False
        if trace_enabled:
            self._mimo_mtp1_greedy_fast_trace("hit", "eligible")
        return True

    def _sample_mimo_mtp1_greedy_fast(
        self,
        greedy_spec_top_token_ids: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata,
    ) -> SamplerOutput:
        target_token_ids = greedy_spec_top_token_ids.index_select(
            0,
            spec_decode_metadata.target_logits_indices.long(),
        ).long()
        bonus_token_ids = greedy_spec_top_token_ids.index_select(
            0,
            spec_decode_metadata.bonus_logits_indices.long(),
        ).long()
        draft_token_ids = spec_decode_metadata.draft_token_ids.long()
        accepted = draft_token_ids.eq(target_token_ids)

        output_token_ids = torch.full(
            (len(spec_decode_metadata.num_draft_tokens), 2),
            -1,
            dtype=torch.int32,
            device=draft_token_ids.device,
        )
        output_token_ids[:, 0] = torch.where(
            accepted,
            draft_token_ids,
            target_token_ids,
        ).to(torch.int32)
        output_token_ids[:, 1] = torch.where(
            accepted,
            bonus_token_ids,
            output_token_ids[:, 1].long(),
        ).to(torch.int32)
        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=None,
        )

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        greedy_spec_top_token_ids: torch.Tensor | None = None,
    ) -> SamplerOutput:
'''
old_existing_greedy_fast = '''    def _mimo_mtp1_greedy_fast_guard(
        self,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> bool:
        if os.environ.get("VLLM_MIMO_MTP1_GREEDY_FAST", "0") != "1":
            return False
        if spec_decode_metadata is None or self.num_spec_tokens != 1:
            return False
        if not hasattr(self.model, "get_top_tokens"):
            return False
        if any(num_draft != 1 for num_draft in spec_decode_metadata.num_draft_tokens):
            return False
        sampling_metadata = self.input_batch.sampling_metadata
        logitsprocs = sampling_metadata.logitsprocs
        has_logitsprocs = bool(logitsprocs.argmax_invariant) or bool(
            logitsprocs.non_argmax_invariant
        )
        thinking_holder = sampling_metadata.thinking_budget_state_holder
        has_thinking_budget = (
            thinking_holder is not None
            and thinking_holder.has_tracked_requests()
        )
        return (
            sampling_metadata.all_greedy
            and sampling_metadata.max_num_logprobs is None
            and not sampling_metadata.logprob_token_ids
            and sampling_metadata.no_penalties
            and sampling_metadata.allowed_token_ids_mask is None
            and not sampling_metadata.bad_words_token_ids
            and not has_logitsprocs
            and not self.num_prompt_logprobs
            and not has_thinking_budget
        )

    def _sample_mimo_mtp1_greedy_fast(
        self,
        greedy_spec_top_token_ids: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata,
    ) -> SamplerOutput:
        target_token_ids = greedy_spec_top_token_ids.index_select(
            0,
            spec_decode_metadata.target_logits_indices.long(),
        ).long()
        bonus_token_ids = greedy_spec_top_token_ids.index_select(
            0,
            spec_decode_metadata.bonus_logits_indices.long(),
        ).long()
        draft_token_ids = spec_decode_metadata.draft_token_ids.long()
        accepted = draft_token_ids.eq(target_token_ids)

        output_token_ids = torch.full(
            (len(spec_decode_metadata.num_draft_tokens), 2),
            -1,
            dtype=torch.int32,
            device=draft_token_ids.device,
        )
        output_token_ids[:, 0] = torch.where(
            accepted,
            draft_token_ids,
            target_token_ids,
        ).to(torch.int32)
        output_token_ids[:, 1] = torch.where(
            accepted,
            bonus_token_ids,
            output_token_ids[:, 1].long(),
        ).to(torch.int32)
        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=None,
        )

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        greedy_spec_top_token_ids: torch.Tensor | None = None,
    ) -> SamplerOutput:
'''
if new not in text:
    if old in text:
        text = text.replace(old, new, 1)
    elif old_existing_greedy_fast in text:
        text = text.replace(old_existing_greedy_fast, new, 1)
    elif '    def _mimo_mtp1_greedy_fast_trace(\n' in text:
        start = text.index('    def _mimo_mtp1_greedy_fast_trace(\n')
        sample_anchor = '''    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        greedy_spec_top_token_ids: torch.Tensor | None = None,
    ) -> SamplerOutput:
'''
        end = text.index(sample_anchor, start) + len(sample_anchor)
        text = text[:start] + new + text[end:]
    else:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: _sample anchor not found')

old = '''        self._maybe_observe_dspark_position0_quality(
            spec_decode_metadata,
            logits,
        )
        draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            draft_probs,
            logits,
            sampling_metadata,
        )
        return sampler_output
'''
new = '''        self._maybe_observe_dspark_position0_quality(
            spec_decode_metadata,
            logits,
        )
        if greedy_spec_top_token_ids is not None:
            return self._sample_mimo_mtp1_greedy_fast(
                greedy_spec_top_token_ids,
                spec_decode_metadata,
            )

        draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            draft_probs,
            logits,
            sampling_metadata,
        )
        return sampler_output
'''
if new not in text:
    if old not in text:
        old = '''        draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            draft_probs,
            logits,
            sampling_metadata,
        )
        return sampler_output
'''
        new = '''        if greedy_spec_top_token_ids is not None:
            return self._sample_mimo_mtp1_greedy_fast(
                greedy_spec_top_token_ids,
                spec_decode_metadata,
            )

        draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            draft_probs,
            logits,
            sampling_metadata,
        )
        return sampler_output
'''
        if old not in text:
            raise SystemExit('[fix-mimo-v2-vllm] ERROR: rejection sampler anchor not found')
    text = text.replace(old, new, 1)

old = '''                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
'''
new = '''                sample_hidden_states = hidden_states[logits_indices]
                greedy_spec_top_token_ids = None
                if self._mimo_mtp1_greedy_fast_guard(spec_decode_metadata):
                    greedy_spec_top_token_ids = self.model.get_top_tokens(
                        sample_hidden_states
                    )
                    logits = None
                else:
                    logits = self.model.compute_logits(sample_hidden_states)
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: target logits anchor not found')
    text = text.replace(old, new, 1)

old = '''                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
'''
new = '''                sample_hidden_states = hidden_states[logits_indices]
                greedy_spec_top_token_ids = None
                if not get_pp_group().is_last_rank:
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: PP sample_hidden_states anchor not found')
    text = text.replace(old, new, 1)

old = '''            scheduler_output,
            logits,
            spec_decode_metadata,
'''
new = '''            scheduler_output,
            logits,
            greedy_spec_top_token_ids,
            spec_decode_metadata,
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: ExecuteModelState init anchor not found')
    text = text.replace(old, new, 1)

old = '''            scheduler_output,
            logits,
            spec_decode_metadata,
'''
new = '''            scheduler_output,
            logits,
            greedy_spec_top_token_ids,
            spec_decode_metadata,
'''
if text.count(new) < 2:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: ExecuteModelState unpack anchor not found')
    text = text.replace(old, new, 1)

old = '''        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )
'''
new = '''        # Apply structured output bitmasks if present. Structured output needs
        # full logits, so any speculative top-token fast path falls back here.
        if grammar_output is not None:
            if logits is None:
                logits = self.model.compute_logits(sample_hidden_states)
                greedy_spec_top_token_ids = None
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: grammar fallback anchor not found')
    text = text.replace(old, new, 1)

old = '''            sampler_output = self._sample(logits, spec_decode_metadata)
'''
new = '''            sampler_output = self._sample(
                logits,
                spec_decode_metadata,
                greedy_spec_top_token_ids,
            )
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: _sample call anchor not found')
    text = text.replace(old, new, 1)

if text != orig:
    path.write_text(text)
    print('[fix-mimo-v2-vllm] patched greedy MTP1 target top-token fast path')
else:
    print('[fix-mimo-v2-vllm] greedy MTP1 target top-token fast path already patched')

import ast
ast.parse(path.read_text(), filename=str(path))
if 'VLLM_MIMO_MTP1_GREEDY_FAST' not in path.read_text():
    raise SystemExit('[fix-mimo-v2-vllm] ERROR: greedy fast-path marker missing')
print('[fix-mimo-v2-vllm] validated greedy MTP1 target top-token fast path')
PY

# MiMo-V2.5 Omni checkpoints may keep the raw HF architecture as
# MiMoV2ForCausalLM even though the target model is resolved to
# MiMoV2OmniForCausalLM because vision/audio config is present or because the
# launch passes an Omni hf override.  The speculative draft ModelConfig reloads
# the raw checkpoint config and applies only SpeculativeConfig.hf_config_override;
# without this patch the draft resolves to text-only MiMoV2MTPModel instead of
# the official multimodal MiMoV2OmniMTPModel wrapper.  Select OmniMTP whenever
# the draft checkpoint advertises vision_config.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py')
text = path.read_text()
old = '''        if (arch := hf_config.architectures[0]) in (
            "MiMoV2ForCausalLM",
            "MiMoV2OmniForCausalLM",
        ):
            from vllm.model_executor.models.mimo_v2_mtp import (
                _MIMO_V2_PRO_NUM_MTP_LAYERS,
            )

            mtp_arch_maps = {
                "MiMoV2ForCausalLM": "MiMoV2MTPModel",
                "MiMoV2OmniForCausalLM": "MiMoV2OmniMTPModel",
            }

            hf_config.model_type = "mimo_v2_mtp"
'''
new = '''        if (arch := hf_config.architectures[0]) in (
            "MiMoV2ForCausalLM",
            "MiMoV2OmniForCausalLM",
        ):
            from vllm.model_executor.models.mimo_v2_mtp import (
                _MIMO_V2_PRO_NUM_MTP_LAYERS,
            )

            # The raw HF config for some Omni-capable MiMo-V2.5 exports still
            # says MiMoV2ForCausalLM even though vision/audio config is present.
            # The target may be resolved to MiMoV2OmniForCausalLM, but the MTP
            # draft config only sees the raw checkpoint architecture.  Mirror
            # the official Omni MTP path for such checkpoints.
            if arch == "MiMoV2ForCausalLM" and getattr(
                hf_config, "vision_config", None
            ):
                arch = "MiMoV2OmniForCausalLM"

            mtp_arch_maps = {
                "MiMoV2ForCausalLM": "MiMoV2MTPModel",
                "MiMoV2OmniForCausalLM": "MiMoV2OmniMTPModel",
            }

            hf_config.model_type = "mimo_v2_mtp"
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: SpeculativeConfig MiMoV2 MTP mapping pattern not found')
    path.write_text(text.replace(old, new, 1))
    print('[fix-mimo-v2-vllm] patched SpeculativeConfig to select MiMoV2OmniMTPModel for Omni exports')
else:
    print('[fix-mimo-v2-vllm] SpeculativeConfig Omni MTP mapping already patched')
PY

# PR #41797: add TRITON_ATTN_DIFFKV and make MiMoV2 auto-fallback to it on
# non-FA3 hardware (GB10/sm_121a). Without this, MiMoV2's K/V head-dim split
# forces FlashAttentionDiffKV and fails on DGX Spark.
if python3 - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('vllm.v1.attention.backends.triton_attn_diffkv') else 1)
PY
then
    echo "[fix-mimo-v2-vllm] TRITON_ATTN_DIFFKV already present; skipping PR #41797"
else
    echo "[fix-mimo-v2-vllm] Applying vLLM PR #41797 (TRITON_ATTN_DIFFKV)"
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT
    curl -fsL "$PR41797_URL" -o "$tmpdir/pr41797.diff"

    # The upstream diff contains docs; only apply package files under vllm/.
    python3 - "$tmpdir/pr41797.diff" "$tmpdir/pr41797-vllm-only.diff" <<'PY'
from pathlib import Path
import sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
keep = False
out = []
for line in src.read_text().splitlines(True):
    if line.startswith('diff --git '):
        parts = line.strip().split()
        # format: diff --git a/path b/path
        b_path = parts[3][2:] if len(parts) >= 4 and parts[3].startswith('b/') else ''
        keep = b_path.startswith('vllm/')
    if keep:
        out.append(line)
dst.write_text(''.join(out))
PY

    if git apply --check --unsafe-paths "$tmpdir/pr41797-vllm-only.diff"; then
        git apply --unsafe-paths --whitespace=nowarn "$tmpdir/pr41797-vllm-only.diff"
    elif patch -p1 --dry-run --forward --batch < "$tmpdir/pr41797-vllm-only.diff" >/dev/null 2>&1; then
        patch -p1 --forward --batch < "$tmpdir/pr41797-vllm-only.diff"
    else
        echo "[fix-mimo-v2-vllm] ERROR: PR #41797 is not applicable to this vLLM install" >&2
        echo "[fix-mimo-v2-vllm] Rebuild with --apply-vllm-pr 41797 or update the base image." >&2
        exit 1
    fi
fi

# CyberTen's #41834 minimal fallback for V2 executor + MTP + cudagraph.
# Newer vLLM already contains this; older builds may not. Apply only when the
# exact anchor exists and the snippet is absent.
GPU_RUNNER="$SITE_PACKAGES/vllm/v1/worker/gpu_model_runner.py"
if [ -f "$GPU_RUNNER" ] && ! grep -q "sync_without_prev_positions" "$GPU_RUNNER"; then
    echo "[fix-mimo-v2-vllm] Applying minimal sync_without_prev_positions fallback"
    python3 - "$GPU_RUNNER" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
anchors = [
    "        prev_positions = self.prev_positions.np[: len(num_draft_tokens)]\n",
    "        prev_positions = self.prev_positions.np[:len(num_draft_tokens)]\n",
]
snippet = '''\
        sync_without_prev_positions = (\n            not self.use_async_scheduling and np.all(prev_positions < 0)\n        )\n        if sync_without_prev_positions:\n            if draft_probs.ndim == 2:\n                return draft_probs[:total_num_draft_tokens].contiguous()\n            if draft_probs.shape[0] >= len(num_draft_tokens):\n                prev_positions = np.arange(len(num_draft_tokens))\n            else:\n                packed_probs = []\n                draft_row = 0\n                for num_tokens in num_draft_tokens:\n                    if num_tokens == 0:\n                        continue\n                    if draft_row >= draft_probs.shape[0]:\n                        raise RuntimeError(\n                            "Spec decode metadata references more draft token "\n                            "rows than were recorded by the draft model."\n                        )\n                    packed_probs.append(draft_probs[draft_row, :num_tokens])\n                    draft_row += 1\n                if not packed_probs:\n                    return None\n                return torch.cat(packed_probs, dim=0).contiguous()\n'''
for anchor in anchors:
    if anchor in text:
        path.write_text(text.replace(anchor, anchor + snippet, 1))
        print("[fix-mimo-v2-vllm] patched", path)
        break
else:
    print("[fix-mimo-v2-vllm] anchor not found; skipping #41834 fallback (likely different vLLM version)")
PY
else
    echo "[fix-mimo-v2-vllm] sync_without_prev_positions already present or gpu_model_runner.py missing; skipping"
fi

find "$SITE_PACKAGES/vllm" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

python3 - <<'PY'
import importlib.util
assert importlib.util.find_spec('vllm.v1.attention.backends.triton_attn_diffkv'), 'TRITON_ATTN_DIFFKV not installed'
from vllm.transformers_utils.config import _CONFIG_REGISTRY
assert 'mimo_v2' in _CONFIG_REGISTRY, 'mimo_v2 config registry entry missing'
from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs.mimo_v2 import MimoV2Config
cfg = MimoV2Config(architectures=['MiMoV2ForCausalLM'], vision_config={'enabled': True})
out = SpeculativeConfig.hf_config_override(cfg)
assert out.architectures == ['MiMoV2OmniMTPModel'], out.architectures
from vllm.model_executor.models.mimo_v2_mtp import MiMoV2MTP
mapped = MiMoV2MTP.hf_to_vllm_mapper.apply_dict({
    'language_model.model.mtp.layers.0.self_attn.qkv_proj': {'quant_algo': 'MXFP8'},
    'language_model.model.layers.0.self_attn.qkv_proj': {'quant_algo': 'MXFP8'},
})
assert 'model.mtp.layers.0.self_attn.qkv_proj' in mapped, mapped
assert 'language_model.model.layers.0.self_attn.qkv_proj' in mapped, mapped
assert MiMoV2MTP.packed_modules_mapping['gate_up_proj'] == ['gate_proj', 'up_proj']
print('[fix-mimo-v2-vllm] validation OK')
PY

# ============================================================================
# PR #251 (a3refaat) MiMo reasoning loop-fix — appended 2026-05-29
#   1. fixed chat template (prefill <think> for thinking-on turns)
#   2. MiMo reasoning parser (is_reasoning_end prompt-state fix)
#   3. server-side default thinking_token_budget (forces </think> after budget)
# All anchors/fields verified present in our vLLM build before applying.
# ============================================================================
echo "[fix-mimo-v2-vllm] PR251 loop-fix: chat template + reasoning parser + thinking budget"

cat > /root/mimo_chat_template.jinja <<'MIMO_JINJA_EOF'
{%- if not add_generation_prompt is defined -%}
    {%- set add_generation_prompt = false -%}
{%- endif -%}
{%- if not enable_thinking is defined -%}
    {%- set enable_thinking = true -%}
{%- endif -%}
{%- if not keep_all_reasoning is defined -%}
    {%- set keep_all_reasoning = true -%}
{%- endif -%}
{%- macro render_extra_keys(json_dict, handled_keys) -%}
    {%- if json_dict is mapping %}
        {%- for json_key in json_dict if json_key not in handled_keys %}
            {%- if json_dict[json_key] is mapping or (json_dict[json_key] is sequence and json_dict[json_key] is not string) %}
                {{- '\n<' ~ json_key ~ '>' ~ (json_dict[json_key] | tojson | safe) ~ '</' ~ json_key ~ '>' }}
            {%- else %}
                {{-'\n<' ~ json_key ~ '>' ~ (json_dict[json_key] | string) ~ '</' ~ json_key ~ '>' }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
{%- endmacro -%}
{%- macro render_content(message_content) -%}
    {%- if message_content is string -%}
        {{- message_content -}}
    {%- else -%}
        {%- for content in message_content -%}
            {%- if content['type'] == 'image' or 'image' in content or 'image_url' in content -%}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' -}}
            {%- elif content['type'] == 'audio' or 'audio' in content or 'audio_url' in content -%}
                {{- '<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>' -}}
            {%- elif content['type'] == 'video' or 'video' in content or 'video_url' in content -%}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' -}}
            {%- elif 'text' in content -%}
                {{- content['text'] -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}
{%- if messages[0]["role"] == "system" %}
    {%- set system_message = messages[0]["content"] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
{%- endif %}
{%- set ns = namespace(last_user_index=-1) %}
{%- for m in loop_messages %}
    {%- if m.role == 'user' %}
        {%- set ns.last_user_index = loop.index0 -%}
    {%- endif %}
{%- endfor %}
{%- if not tools is defined %}
    {%- set tools = [] %}
{%- endif %}
{%- if system_message is defined %}
    {{- "<|im_start|>system\n" + render_content(system_message) }}
{%- else %}
    {{- "<|im_start|>system\nYou are MiMo, a helpful AI assistant engineered by Xiaomi." }}
{%- endif %}
{%- if tools is iterable and tools | length > 0 %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou have access to the following functions:\n\n" }}
    {{- "<tools>" }}
    {%- for tool in tools %}
        {%- if tool.function is defined %}
            {%- set tool = tool.function %}
        {%- endif %}
        {{- "\n<function>\n<name>" ~ tool.name ~ "</name>" }}
        {%- if tool.description is defined %}
            {{- '\n<description>' ~ (tool.description | trim) ~ '</description>' }}
        {%- endif %}
        {{- '\n<parameters>' }}
        {%- if tool.parameters is defined and tool.parameters is mapping and tool.parameters.properties is defined and tool.parameters.properties is mapping %}
            {%- for param_name, param_fields in tool.parameters.properties|items %}
                {{- '\n<parameter>' }}
                {{- '\n<name>' ~ param_name ~ '</name>' }}
                {%- if param_fields.type is defined %}
                    {{- '\n<type>' ~ (param_fields.type | string) ~ '</type>' }}
                {%- endif %}
                {%- if param_fields.description is defined %}
                    {{- '\n<description>' ~ (param_fields.description | trim) ~ '</description>' }}
                {%- endif %}
                {%- set handled_keys = ['name', 'type', 'description'] %}
                {{- render_extra_keys(param_fields, handled_keys) }}
                {{- '\n</parameter>' }}
            {%- endfor %}
        {%- endif %}
        {%- set handled_keys = ['type', 'properties'] %}
        {{- render_extra_keys(tool.parameters, handled_keys) }}
        {{- '\n</parameters>' }}
        {%- set handled_keys = ['type', 'name', 'description', 'parameters'] %}
        {{- render_extra_keys(tool, handled_keys) }}
        {{- '\n</function>' }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nFor each function call, output the function name and arguments in the following format:\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>value_1</parameter>\n<parameter=example_parameter_2>This is the value for the second parameter\nthat can span\nmultiple lines</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n- Use the <think>...</think> block for private planning or synthesis.\n- If you need a tool, close </think> first, then output the <tool_call> block immediately after it.\n- If you do not need a tool or already have the needed information, close </think> and answer the user normally.\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags.\n- Do NOT place <tool_call> or <function=...> blocks inside the <think>...</think> reasoning block.\n- The value enclosed between parameter tags is preserved exactly as-is, including newlines and spaces.\n</IMPORTANT>' }}
{%- endif %}
{{- '<|im_end|>' }}
{%- for message in loop_messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = render_content(message.content) %}
    {%- endif %}
    {%- if message.role == "assistant" %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- set reasoning_content = '' %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].split('<think>')[-1] %}
                {%- set content = content.split('</think>')[-1] %}
            {%- endif %}
        {%- endif %}
        {%- if (keep_all_reasoning or loop.index0 > ns.last_user_index) and reasoning_content -%}
            {{- '<|im_start|>' + message.role + '\n<think>' + reasoning_content + '</think>' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls is defined and message.tool_calls is iterable and message.tool_calls | length > 0 %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- if tool_call.arguments is defined %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>' }}
                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                        {{- args_value }}
                        {{- '</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>' }}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + render_content(message.content) + '<|im_end|>' }}
    {%- elif message.role == "system" %}
        {{- '<|im_start|>' + message.role + '\n' + render_content(message.content) + '<|im_end|>' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>tool\n' }}
        {%- endif %}
        {{- '<tool_response>\n' }}
        {{- render_content(message.content) }}
        {{- '\n</tool_response>\n' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>' }}
        {%- elif loop.last %}
            {{- '<|im_end|>' }}
        {%- endif %}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\n' + render_content(message.content) + '<|im_end|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if not enable_thinking -%}
        {{- '<think></think>' -}}
    {%- else -%}
        {{- '<think>' -}}
    {%- endif -%}
{%- endif %}
MIMO_JINJA_EOF
echo "[fix-mimo-v2-vllm] wrote /root/mimo_chat_template.jinja ($(wc -l < /root/mimo_chat_template.jinja) lines)"

cat > /usr/local/lib/python3.12/dist-packages/vllm/reasoning/mimo_reasoning_parser.py <<'MIMO_PARSER_EOF'
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser


class MimoReasoningParser(Qwen3ReasoningParser):
    """MiMo V2.5 reasoning parser.

    MiMo's chat template leaves the assistant generation prompt at
    ``<|im_start|>assistant\n`` when thinking is enabled.  Closed think tags
    can still occur earlier in the prompt (tool instructions say
    ``<think></think>`` and assistant history is rendered with think spans), so
    Qwen3ReasoningParser.is_reasoning_end(prompt_ids) can return True before
    any new assistant tokens have been generated.  That leaks generated
    ``<think>`` text as normal content and confuses streamed tool-call parsing.
    """

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._assistant_generation_prefix_token_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self._assistant_thinking_disabled_suffix_token_ids = tokenizer.encode(
            "<|im_start|>assistant\n<think></think>", add_special_tokens=False
        )

    @staticmethod
    def _endswith(input_ids: Sequence[int], suffix: Sequence[int]) -> bool:
        if not suffix or len(input_ids) < len(suffix):
            return False
        return list(input_ids[-len(suffix) :]) == list(suffix)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # This is the prompt-time check used by vLLM before streaming starts.
        # If generation is about to begin after the bare assistant prefix,
        # reasoning is open even if earlier prompt text contains </think>.
        if self._endswith(input_ids, self._assistant_generation_prefix_token_ids):
            return False

        # Thinking-off MiMo prompts end with an explicit empty reasoning span;
        # in that case output should be routed as content/tool text immediately.
        if self._endswith(
            input_ids, self._assistant_thinking_disabled_suffix_token_ids
        ):
            return True

        return super().is_reasoning_end(input_ids)
MIMO_PARSER_EOF

python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/reasoning/__init__.py')
text = path.read_text()
old = '''    "mimo": (
        "qwen3_reasoning_parser",
        "Qwen3ReasoningParser",
    ),
'''
new = '''    "mimo": (
        "mimo_reasoning_parser",
        "MimoReasoningParser",
    ),
'''
if new in text:
    print('[fix-mimo-v2-vllm] MiMo reasoning parser alias already patched')
elif old in text:
    path.write_text(text.replace(old, new, 1))
    print('[fix-mimo-v2-vllm] patched MiMo reasoning parser alias qwen3->mimo')
else:
    print('[fix-mimo-v2-vllm] WARNING: mimo alias block not found; reasoning parser alias NOT changed')
PY

python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py')
if not path.exists():
    print('[fix-mimo-v2-vllm] chat completion serving.py not present; skipping default thinking budget patch')
    raise SystemExit
text = path.read_text()
orig = text
if 'import os\n' not in text.split('import time\n', 1)[0] + 'import time\n':
    text = text.replace('import json\nimport time\n', 'import json\nimport os\nimport time\n', 1)
old = '''            sampling_params: SamplingParams | BeamSearchParams
            if request.use_beam_search:
'''
new = '''            default_thinking_token_budget = os.environ.get(
                "MIMO_DEFAULT_THINKING_TOKEN_BUDGET", ""
            )
            if (
                default_thinking_token_budget
                and request.thinking_token_budget is None
                and request.include_reasoning
                and reasoning_parser is not None
            ):
                try:
                    parsed_default_thinking_token_budget = int(
                        default_thinking_token_budget
                    )
                except ValueError as exc:
                    raise ValueError(
                        "MIMO_DEFAULT_THINKING_TOKEN_BUDGET must be a "
                        "non-negative integer"
                    ) from exc
                if parsed_default_thinking_token_budget < 0:
                    raise ValueError(
                        "MIMO_DEFAULT_THINKING_TOKEN_BUDGET must be a "
                        "non-negative integer"
                    )
                request.thinking_token_budget = parsed_default_thinking_token_budget

            sampling_params: SamplingParams | BeamSearchParams
            if request.use_beam_search:
'''
if new not in text:
    if old not in text:
        print('[fix-mimo-v2-vllm] WARNING: serving.py anchor not found; thinking-budget default NOT applied (non-fatal)')
        raise SystemExit
    text = text.replace(old, new, 1)
if text != orig:
    path.write_text(text)
    print('[fix-mimo-v2-vllm] patched OpenAI chat serving default MiMo thinking budget')
else:
    print('[fix-mimo-v2-vllm] OpenAI chat serving default MiMo thinking budget already patched')
PY
