#!/usr/bin/env bash
# Start the patched vLLM container. RUN THIS ON BOTH SPARKS (head AND worker).
# Each node runs its OWN container from the same image — the worker is NOT driven
# remotely; Ray launches the TP worker/rank inside the worker's own container.
# After this on each node: `bash recipe/apply-mods.sh "$CONTAINER"`, then Ray + vLLM.
#
# IMAGE = the patched vLLM DEV-build image. Defaults to the public published image
#   (docker auto-pulls it on first run). The mods are NOT baked in — they're applied
#   at runtime via apply-mods.sh after this container is up.
set -euo pipefail
: "${IMAGE:=ghcr.io/tonyd2wild/mimo-v2.5-tp2-1m-nvfp4kv:20260620}"
: "${CONTAINER:=vllm_mimo_tp2}"
: "${HF_CACHE:=$HOME/.cache/huggingface}"      # model weights live here; mounted into the container
: "${RECIPE_DIR:=$PWD}"                         # this repo's recipe/ dir → reachable in-container

docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 16g \
  --device /dev/infiniband:/dev/infiniband \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$RECIPE_DIR:/workspace/recipe" \
  "$IMAGE" sleep infinity

echo "Container '$CONTAINER' up on $(hostname)."
echo "Next on THIS node:  bash recipe/apply-mods.sh $CONTAINER"
echo "Do the SAME (run-container.sh + apply-mods.sh) on the OTHER Spark before starting Ray."
