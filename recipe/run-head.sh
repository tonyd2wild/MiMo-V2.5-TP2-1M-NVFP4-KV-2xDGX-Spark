#!/usr/bin/env bash
# Start the Ray HEAD (the node that will run vLLM). Caps the Ray plasma object store —
# uncapped Ray reserves a large object store and steals unified memory => OOM at 1M.
# Run on the HEAD node, inside the patched vLLM container, AFTER `source env.sh`.
set -euo pipefail
: "${RAY_PORT:=6379}"
if [ -z "${HEAD_ROCE_IP:-}" ]; then
  echo "ERROR: set HEAD_ROCE_IP to this node RoCE/cluster IP" >&2
  exit 2
fi

# Pin the host IP so Ray/vLLM don't bind a link-local 169.254.x.x interface (a known OOM/crash cause).
export VLLM_HOST_IP="${HEAD_ROCE_IP}"

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
echo "Then on THIS node: source env.sh && source your cluster exports && bash launch.sh"
