# MiMo-V2.5 Omni · TP=2 · 500K context · NVFP4 KV on 2× DGX Spark

Running [`lukealonso/MiMo-V2.5-NVFP4`](https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4) (Omni: text + image + video + audio) tensor-parallel across **two NVIDIA DGX Spark (GB10)** boxes, with **4-bit `nvfp4` KV cache** + MTP speculative decoding — serving a **500,000-token** context with a **~1.066M-token KV pool (≈2.13× full-500K concurrency)**.

This is the **2-node** sibling of our 3-node build ([MiMo-V2.5-TP3-NVFP4-KV-3xDGX-Spark](https://github.com/tonyd2wild/MiMo-V2.5-TP3-NVFP4-KV-3xDGX-Spark)). Two Sparks instead of three — so it pairs cleanly with another 2-node model on the same fleet (e.g. DeepSeek-V4 TP=2 on the other pair).

> ⚠️ Experimental. The `nvfp4` KV path depends on the matching patched attention backend (DiffKV). Without the mod stack, vLLM will reject NVFP4 KV.

---

## Results at a glance

| | Value |
|---|---|
| Model | `lukealonso/MiMo-V2.5-NVFP4` (Omni) |
| Nodes | 2× DGX Spark / GB10 |
| Parallelism | TP=2, PP=1 |
| Max context / request | **500,000** |
| KV cache dtype | **`nvfp4`** (4-bit) |
| GPU KV pool | **1,066,112 tokens** |
| Concurrency @ 500K | **≈2.13×** |
| Speculative decoding | MTP, `num_speculative_tokens=1` |
| Loader | `safetensors` |
| GPU mem util | 0.82 |

### 69-scenario tool-eval (2Wild model-eval harness)

| Run | quality | pass / partial / fail | deployability | median latency | tokens used |
|---|---|---|---|---|---|
| **Thinking OFF** ⭐ | **97.8** | 66 / 3 / 0 | **96.6** | **1.7s** | **87.8K** |
| **Thinking ON** | 90.6 | 61 / 3 / 5 | 89.4 | 3.1s | 118.1K |

**Verdict: for tool/agent work, thinking-OFF wins outright** — higher quality (97.8 vs 90.6), zero failures (vs 5), ~2× lower latency, and ~35% fewer tokens. With thinking ON the model tends to over-reason itself into mistakes on tool tasks (fabricating a tool result, over-refusing a benign request). Run this config **thinking-OFF** for agents/tools; reserve thinking-ON for open-ended reasoning. Both via the `2wild-model-eval` 69-scenario harness; raw results in [`benchmarks/`](benchmarks/).

---

## Why TP=2 (vs the TP=3 build)

- **Frees a node.** A 4-Spark fleet can run TWO independent TP=2 models (this MiMo + e.g. DeepSeek-V4) instead of one TP=3 model + an idle box.
- **Still huge context.** 500K per request with a ~1.066M KV pool (≈2.13× concurrency) — two ~100K agents, or two simultaneous 500K-class requests on paper.
- **Omni preserved** (`MiMoV2OmniForCausalLM`) — text/image/video/audio path stays live.

## The config that works

Key flags (full env + launch in [`recipe/`](recipe/)):

```bash
--tensor-parallel-size 2 --pipeline-parallel-size 1 \
--kv-cache-dtype nvfp4 --attention-backend triton_attn_diffkv \
--max-model-len 500000 --max-num-seqs 2 \
--gpu-memory-utilization 0.82 \
--speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
--load-format safetensors --enforce-eager \
--hf-overrides '{"architectures":["MiMoV2OmniForCausalLM"]}' \
--tool-call-parser mimo --reasoning-parser mimo
```

Startup log should show:
```
GPU KV cache size: 1,066,112 tokens
Maximum concurrency for 500,000 tokens per request: 2.13x
Application startup complete.
```

## Hard-won lessons (what failed first)

1. **`gpu_memory_utilization=0.80` was just barely too low** for 500K — KV init needed 1.64 GiB but only 1.56 GiB was free (capped at ~474K). Bumping to **0.82** boots clean.
2. **MTP1 beats MTP2 here.** At 96K, MTP2 gave the same quick-decode speed as MTP1 but cut the KV pool nearly in half (675K → 419K tokens). Use **MTP1**.
3. **`safetensors`, not InstantTensor.** The MTP + NVFP4-KV path wedged under InstantTensor on the second (drafter) load. Safetensors is slower to load but stable.
4. **Worker-first Ray startup**, then head, then launch vLLM from the head.

## Repro checklist

1. Start from the known-working TP=2 launch shape; worker-first Ray.
2. `safetensors` loader, Omni architecture, `--kv-cache-dtype nvfp4` + `triton_attn_diffkv`.
3. MTP1; `max_model_len=500000`, `max_num_seqs=2`, `gpu_memory_utilization=0.82`.
4. Confirm the `GPU KV cache size: 1,066,112` + `2.13x` startup lines.
5. Smoke (`"Reply exactly: OK 500K MTP1"`) → then run the 69-eval.

## Files

```
.
├── README.md
├── recipe/
│   ├── env.sh              # full environment
│   └── launch.sh           # vllm serve command
└── benchmarks/
    ├── thinking-off-69eval.json   # (added)
    └── thinking-on-69eval.json    # (added)
```

The patched **mods** (`nvfp4-kv-diffkv`, `fix-mimo-v2-vllm`, etc.) are NOT vendored — they carry upstream licenses. See **Credits** and pull from upstream.

## Claims we can / can't make

**Safe:** boots on 2× DGX Spark at TP=2 / MTP1 / NVFP4 KV; serves `max_model_len=500000`; vLLM reports 1,066,112 KV tokens / 2.13× concurrency; MTP1 > MTP2 for this setup; safetensors is the stable loader.

**Not yet:** not a 1M-context-*per-request* recipe; not production-stable for two simultaneous 500K agents until a real concurrency test; no audio/video quality claims without separate modality evals.

## Credits

Explicit so inspiration + upstream work are credited cleanly.

**Model & serving stack**
- [`lukealonso/MiMo-V2.5-NVFP4`](https://huggingface.co/lukealonso) — the MiMo V2.5 NVFP4 checkpoint.
- [vLLM](https://github.com/vllm-project/vllm) — OpenAI-compatible serving, TP, speculative decoding, multimodal, Ray executor.
- [Ray](https://github.com/ray-project/ray) — multi-node placement + cluster execution.
- NVIDIA **NCCL** (RoCE transport for TP), **DGX Spark / GB10** + ConnectX networking, **FlashInfer / ModelOpt** kernels + NVFP4 quant paths.

**Community references & debugging signals**
- [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) — DGX Spark ConnectX/RoCE quirks + direct-cable networking guidance + base mods.
- [`HeNryous/mimo-spark-optimized`](https://github.com/HeNryous/mimo-spark-optimized) — a comparison point for TP=2 Spark work (no code copied from it).
- NVIDIA Developer Forums DGX Spark / GB10 community posts; Mashie's link-behavior / cold power-drain clues; Renek & MiaAI_Lab notes on MiMo-on-Spark memory pressure + loader difficulty.
- Eval methodology adapted from `tool-eval-bench v2.0.1` by `wolttam`.

**Local integration & experiment work**
- **Tony / `tonyd2wild`** — hardware, fleet orchestration, public writeups, testing goals, repo publication.
- **Kai** — handoff docs, operational coordination, eval-board integration, independent verification.
- **Codex** — iterative TP=2 / NVFP4-KV / MTP testing, launcher/config edits, startup diagnosis, the source report.

**Local patches/mods in this experiment** (keep visible in any tree; document temporary vs upstreamed vs local-only):
`ray-keep-node-nccl-hca` · `fix-prometheus-instrumentator-router` · `fix-mimo-v2-vllm` · `fix-modelopt-mixed-mxfp8` · `nvfp4-kv-diffkv` · `drop-caches`

## License

MIT (covers this repo's recipe docs + config only — not the upstream mods). See [LICENSE](LICENSE).

*Validated on 2× DGX Spark, 2026-06. "2.13×" = full-500K-request KV capacity, not a single 1M-token request.*
