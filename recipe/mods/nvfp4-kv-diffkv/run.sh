#!/bin/bash
set -euo pipefail
SITE="/usr/local/lib/python3.12/dist-packages"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[nvfp4-kv-diffkv] installing nvfp4 DiffKV store+decode backend"
cp "$HERE/triton_attn_diffkv.py" "$SITE/vllm/v1/attention/backends/triton_attn_diffkv.py"
cp "$HERE/triton_unified_attention_diffkv.py" "$SITE/vllm/v1/attention/ops/triton_unified_attention_diffkv.py"
cp "$HERE/wmma_decode.py" "$SITE/vllm/v1/attention/ops/wmma_decode.py"
python3 -c "import ast; ast.parse(open('$SITE/vllm/v1/attention/backends/triton_attn_diffkv.py').read()); print('[nvfp4-kv-diffkv] backend syntax OK')"
python3 - <<'PYEOF'
import pathlib
p=pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py")
s=p.read_text()
anchor="        quant_mode = get_kv_quant_mode(self.kv_cache_dtype)"
fix=(anchor
     + "\n        if getattr(vllm_config.cache_config, 'cache_dtype', None) == 'nvfp4':"
     + "\n            quant_mode = get_kv_quant_mode('nvfp4')  # nvfp4-kv-diffkv: live dtype not stale self"
     + "\n            import torch as _t_nv; self.kv_cache_torch_dtype = _t_nv.uint8  # nvfp4 packed uint8 cache")
assert anchor in s, "anchor not found"
if "nvfp4-kv-diffkv: live dtype" not in s:
    s=s.replace(anchor, fix); p.write_text(s); print("[nvfp4-kv-diffkv] patched get_kv_cache_spec: quant_mode + uint8 dtype")
else:
    print("[nvfp4-kv-diffkv] attention.py already patched")
PYEOF
echo "[nvfp4-kv-diffkv] done"
