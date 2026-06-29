#!/usr/bin/env bash
# MiMo-V2.5 Omni TP=2 / 1M / MTP1 / NVFP4-KV — environment
# Apply on BOTH nodes (worker + head) inside the vLLM container.
#
# vLLM BUILD: 0.21.1rc1.dev85+gd87ee1893 (commit d87ee1893, ~2026-05-18, CUDA 13.2/cu132; torch 2.11.0+cu130) —
#   a DEV build, NOT a released pip wheel — PLUS the 6 patch mods (see README "Runtime stack
#   used"). Stock `pip install vllm` will reject --kv-cache-dtype nvfp4 / triton_attn_diffkv /
#   the MiMoV2OmniForCausalLM arch. Ray is REQUIRED (mp executor is single-host; can't span 2 boxes).

# --- core serving shape ---
export LOAD_FORMAT=safetensors          # NOT instanttensor (it wedges the MTP+NVFP4-KV 2nd load)
export MAX_MODEL_LEN=1000000
export MAX_NUM_BATCHED_TOKENS=4096
export MAX_NUM_SEQS=4
export GPU_MEMORY_UTILIZATION=0.84      # 0.80 just-barely OOMs at 500K (caps ~474K)
export ENABLE_MTP=1
export MTP_SPEC_TOKENS=1                # MTP1 > MTP2 here (MTP2 halves KV pool, no speed gain)
export ENFORCE_EAGER=1

# --- memory / Ray stability (CRITICAL for avoiding OOM at 1M) ---
# Without these the 0.84 GMU + 1M pool is unforgiving and users hit OOM on load/profile.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # avoids fragmentation OOM
export RAY_memory_monitor_refresh_ms=0                    # stops Ray's 95% monitor false-killing TP0 post-warmup
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0
export VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=0
# Cap the Ray plasma object store on EVERY node (see run-head.sh / run-worker.sh):
#   ray start ... --object-store-memory=1073741824   # 1 GiB; uncapped Ray steals unified mem -> OOM

# --- NVFP4 weights + KV + WMMA decode ---
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_FLASHINFER_MOE_BACKEND=throughput
export VLLM_NVFP4_INLINE=1
export VLLM_WMMA_DECODE=1
export VLLM_WMMA_INSPECT=0
export VLLM_WMMA_COMPARE=0
export VLLM_WMMA_AUTHCOMPARE=0

# --- RoCE / NCCL (direct-cable 2-node) ---
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0}"
export NCCL_IB_GID_INDEX=3
export NCCL_IB_MERGE_NICS=0
export NCCL_IB_SUBNET_AWARE_ROUTING=1
export NCCL_CROSS_NIC=1
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_NET_GDR_LEVEL=LOC

# Set these to your environment:
# export MODEL_PATH=/root/.cache/huggingface/hub/models--lukealonso--MiMo-V2.5-NVFP4/snapshots/a147dd04d6cf861e43b2d783dcde23b53ab7ee68  # pinned revision used for the 1M run
# export SERVED_MODEL_NAME=MiMo-V2.5-NVFP4
