#!/usr/bin/env bash
# Start the Ray HEAD (the node that will run vLLM). Caps the Ray plasma object store —
# uncapped Ray reserves a large object store and steals unified memory => OOM at 1M.
# Run on the HEAD node, inside the patched vLLM container, AFTER `source env.sh`.
set -euo pipefail
: "${RAY_PORT:=6379}"
: "${HEAD_ROCE_IP:?set HEAD_ROCE_IP (this node's RoCE/cluster IP)}"

ray stop --force || true
ray start \
  --head \
  --port="${RAY_PORT}" \
  --node-ip-address="${HEAD_ROCE_IP}" \
  --dashboard-host=0.0.0.0 \
  --num-gpus=1 \
  --object-store-memory=1073741824   # 1 GiB cap — do NOT omit

echo "Ray head up on ${HEAD_ROCE_IP}:${RAY_PORT}."
echo "Now start the worker (run-worker.sh on the other node), wait for 2 GPUs:"
echo "  until ray status 2>/dev/null | grep -qE '2\\.0/2\\.0 GPU|2\\.0 GPU'; do sleep 2; done"
echo "Then on THIS node: source env.sh && bash launch.sh"
