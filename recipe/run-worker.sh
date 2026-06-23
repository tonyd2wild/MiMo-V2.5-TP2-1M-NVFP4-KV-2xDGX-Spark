#!/usr/bin/env bash
# Join the Ray cluster as a WORKER. Same object-store cap as the head (critical for OOM).
# Run on the WORKER node, inside the patched vLLM container, AFTER `source env.sh`.
# Bring the worker CONTAINER up first (before the head), but start the head's Ray first,
# then run this — the worker joins the head's address.
set -euo pipefail
: "${RAY_PORT:=6379}"
: "${HEAD_ROCE_IP:?set HEAD_ROCE_IP (the head node's RoCE/cluster IP)}"
: "${WORKER_ROCE_IP:?set WORKER_ROCE_IP (this node's RoCE/cluster IP)}"

ray stop --force || true
ray start \
  --address="${HEAD_ROCE_IP}:${RAY_PORT}" \
  --node-ip-address="${WORKER_ROCE_IP}" \
  --num-gpus=1 \
  --object-store-memory=1073741824   # 1 GiB cap — do NOT omit

echo "Ray worker joined ${HEAD_ROCE_IP}:${RAY_PORT}. Confirm 2 GPUs on the head via 'ray status'."
