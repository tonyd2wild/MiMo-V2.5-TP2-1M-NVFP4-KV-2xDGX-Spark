#!/usr/bin/env bash
# MiMo-V2.5 Omni TP=2 / 500K / MTP1 / NVFP4-KV — environment
# Apply on BOTH nodes (worker + head) inside the vLLM container.

# --- core serving shape ---
export LOAD_FORMAT=safetensors          # NOT instanttensor (it wedges the MTP+NVFP4-KV 2nd load)
export MAX_MODEL_LEN=500000
export MAX_NUM_BATCHED_TOKENS=4096
export MAX_NUM_SEQS=2
export GPU_MEMORY_UTILIZATION=0.82      # 0.80 just-barely OOMs at 500K (caps ~474K)
export ENABLE_MTP=1
export MTP_SPEC_TOKENS=1                # MTP1 > MTP2 here (MTP2 halves KV pool, no speed gain)
export ENFORCE_EAGER=1

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
export NCCL_IB_GID_INDEX=3
export NCCL_IB_MERGE_NICS=0
export NCCL_IB_SUBNET_AWARE_ROUTING=1
export NCCL_CROSS_NIC=1
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_NET_GDR_LEVEL=LOC

# Set these to your environment:
# export MODEL_PATH=/root/.cache/huggingface/hub/models--lukealonso--MiMo-V2.5-NVFP4/snapshots/<SNAPSHOT_HASH>
# export SERVED_MODEL_NAME=MiMo-V2.5-NVFP4
