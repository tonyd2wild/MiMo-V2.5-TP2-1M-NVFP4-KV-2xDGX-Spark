#!/usr/bin/env bash
set -euo pipefail

echo "[ray-keep-node-nccl-hca] Preventing vLLM Ray from copying head NCCL_IB_HCA to workers"

mkdir -p /root/.config/vllm
python3 - <<'PY'
import json
from pathlib import Path

path = Path("/root/.config/vllm/ray_non_carry_over_env_vars.json")
try:
    existing = json.loads(path.read_text()) if path.exists() else []
except json.JSONDecodeError:
    existing = []

items = set(existing if isinstance(existing, list) else [])
items.add("NCCL_IB_HCA")
path.write_text(json.dumps(sorted(items)) + "\n")
print(f"[ray-keep-node-nccl-hca] {path}: {sorted(items)}")
PY
