# MiMo-V2.5 Omni · TP=2 · **1M context** · NVFP4 KV on 2× DGX Spark

Running [`lukealonso/MiMo-V2.5-NVFP4`](https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4) (Omni: text + image + video + audio) tensor-parallel across **two NVIDIA DGX Spark (GB10)** boxes, with **4-bit `nvfp4` KV cache** + MTP speculative decoding — serving a **1,000,000-token** context with a **~1.97M-token KV pool**.

This is the **2-node** sibling of the 3-node build ([MiMo-V2.5-TP3-NVFP4-KV-3xDGX-Spark](https://github.com/tonyd2wild/MiMo-V2.5-TP3-NVFP4-KV-3xDGX-Spark)). Two Sparks instead of three — so it pairs cleanly with another 2-node model on the same fleet (e.g. DeepSeek-V4 TP=2 on the other pair).

> ⚠️ Experimental. The `nvfp4` KV path depends on the matching patched attention backend (DiffKV). Without the mod stack, vLLM will reject NVFP4 KV.

---

## Results at a glance

| | Value |
|---|---|
| Model | `lukealonso/MiMo-V2.5-NVFP4` (Omni) |
| Nodes | 2× DGX Spark / GB10 |
| Parallelism | TP=2, PP=1 |
| **Max context / request** | **1,000,000** (verified, benched) |
| KV cache dtype | **`nvfp4`** (4-bit) |
| GPU KV pool | **~1.97M tokens** |
| Speculative decoding | MTP, `num_speculative_tokens=1` |
| Loader | `safetensors` |
| GPU mem util | 0.84 |
| `max_num_seqs` | 4 (tune — see below) |
| Decode speed | **~28–30 tok/s** |

### 69-scenario tool-eval (2Wild model-eval harness)

| Run | quality | pass / partial / fail | deployability | tok/s |
|---|---|---|---|---|
| **Thinking OFF** ⭐ | **97.8** | 66 / 3 / 0 | 96.7 | ~30 |
| **Thinking ON** | 90.6 | 61 / 3 / 5 | 89.4 | ~30 raw* |

*Thinking-ON's *effective* throughput is ~2× slower end-to-end from the extra reasoning tokens.

**Quality is identical at 500K and 1M context (97.8 both)** — the 1M ceiling costs nothing on quality. Raw results in [`benchmarks/`](benchmarks/) (thinking-off @500K, thinking-off @1M, thinking-on).

**Verdict: for tool/agent work, thinking-OFF wins outright** — higher quality (97.8 vs 90.6), zero failures (vs 5), ~2× lower latency, ~35% fewer tokens. Thinking-ON over-reasons itself into mistakes on tool tasks. Run agents **thinking-OFF**; reserve thinking-ON for open-ended reasoning.

---

## Context + concurrency: how the shared KV pool works

The **~1.97M-token KV pool is shared** across all in-flight requests. `max_model_len` caps any *single* request; `max_num_seqs` caps how many run at once; the pool is the real budget.

- **One deep request → up to the full 1M tokens** (~1.97× — fits ~2 full 1M requests).
- **Many moderate requests → high concurrency.** e.g. **4 agents × 100K = 400K** = ~20% of the pool → all 4 run in parallel with lots of headroom (~20 concurrent 100K agents before KV bites).
- The only thing you can't do: **4 agents all at a full 1M simultaneously** (that'd need 4M) → vLLM just queues the overflow until room frees.

**Tuning `max_num_seqs`:** set it to **2** if your workload is single huge (500K–1M) requests; set it to **4+** for multi-agent / many-moderate-context work (the config here ships `max_num_seqs=4`, `gpu-memory-utilization=0.84` so a full 1M request still fits alongside the extra seq slots).

## The config that works

Key flags (full env + launch in [`recipe/`](recipe/)):

```bash
--tensor-parallel-size 2 --pipeline-parallel-size 1 \
--kv-cache-dtype nvfp4 --attention-backend triton_attn_diffkv \
--max-model-len 1000000 --max-num-seqs 4 \
--gpu-memory-utilization 0.84 \
--speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
--load-format safetensors --enforce-eager \
--hf-overrides '{"architectures":["MiMoV2OmniForCausalLM"]}' \
--tool-call-parser mimo --reasoning-parser mimo
```

## Hard-won lessons (what failed first)

1. **GPU mem util is tight.** 500K barely fit at 0.80 (needed 0.82); the full **1M + `max_num_seqs=4`** wants **0.84** so a 1M request still fits alongside the extra seq slots.
2. **MTP1 beats MTP2 here.** At 96K, MTP2 gave the same quick-decode speed as MTP1 but cut the KV pool nearly in half. Use **MTP1**.
3. **`safetensors`, not InstantTensor.** The MTP + NVFP4-KV path wedged under InstantTensor on the drafter load. Safetensors is slower to load but stable.
4. **Worker-first Ray startup**, then head, then launch vLLM from the head.

## Repro checklist

1. Worker-first Ray; `safetensors`; Omni arch; `--kv-cache-dtype nvfp4` + `triton_attn_diffkv`; MTP1.
2. `max_model_len=1000000`, `max_num_seqs=4`, `gpu-memory-utilization=0.84`.
3. Confirm the startup log shows a `GPU KV cache size` ≥ 1,000,000 tokens (1M fits).
4. Smoke (`"Reply exactly: OK"`) → then run the 69-eval.

## Files

```
.
├── README.md
├── recipe/{env.sh, launch.sh}
└── benchmarks/
    ├── thinking-off-1M-69eval.{md,json}   # 97.8 quality @ 1M ctx
    ├── thinking-off-69eval.{md,json}      # 97.8 quality @ 500K ctx
    └── thinking-on-69eval.{md,json}       # 90.6 quality (thinking on)
```

The patched **mods** (`nvfp4-kv-diffkv`, `fix-mimo-v2-vllm`, etc.) are NOT vendored — they carry upstream licenses. See **Credits** and pull from upstream.

## Claims we can / can't make

**Safe:** boots on 2× DGX Spark at TP=2 / MTP1 / NVFP4 KV; serves `max_model_len=1000000` (verified + benched at 97.8 quality); ~1.97M-token KV pool; MTP1 > MTP2; safetensors is the stable loader; thinking-OFF beats thinking-ON for tool/agent tasks.

**Not yet:** not production-stable for many simultaneous *full-1M* agents (the pool holds ~2 full 1M requests); no audio/video quality claims without separate modality evals.

## Credits

Explicit so inspiration + upstream work are credited cleanly.

**Model & serving stack**
- [`lukealonso/MiMo-V2.5-NVFP4`](https://huggingface.co/lukealonso) — the MiMo V2.5 NVFP4 checkpoint.
- [vLLM](https://github.com/vllm-project/vllm) — OpenAI-compatible serving, TP, speculative decoding, multimodal, Ray executor.
- [Ray](https://github.com/ray-project/ray) — multi-node placement + cluster execution.
- NVIDIA **NCCL** (RoCE transport for TP), **DGX Spark / GB10** + ConnectX networking, **FlashInfer / ModelOpt** kernels + NVFP4 quant paths.

**Community references & debugging signals**
- [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) (esp. PR #251 / `a3refaat`) — DGX Spark ConnectX/RoCE quirks + direct-cable networking + base mods.
- [`HeNryous/mimo-spark-optimized`](https://github.com/HeNryous/mimo-spark-optimized) — a comparison point for TP=2 Spark work (no code copied from it).
- NVIDIA Developer Forums DGX Spark / GB10 community posts; Mashie's link-behavior / cold power-drain clues; Renek & MiaAI_Lab notes on MiMo-on-Spark memory pressure + loader difficulty.
- Eval methodology adapted from `tool-eval-bench v2.0.1` by `wolttam`.

**Local integration & experiment work**
- **Tony / `tonyd2wild`** — hardware, fleet orchestration, public writeups, testing goals, repo publication.
- **Kai** — handoff docs, operational coordination, eval-board integration, independent verification.
- **Codex** — iterative TP=2 / NVFP4-KV / MTP testing, launcher/config edits, startup diagnosis, the source report.

**Local patches/mods:** `ray-keep-node-nccl-hca` · `fix-prometheus-instrumentator-router` · `fix-mimo-v2-vllm` · `fix-modelopt-mixed-mxfp8` · `nvfp4-kv-diffkv` · `drop-caches` (keep visible in any tree; document temporary vs upstreamed vs local-only).

## License

MIT (covers this repo's recipe docs + config only — not the upstream mods). See [LICENSE](LICENSE).

*Validated on 2× DGX Spark, 2026-06. 1M context boots + benches at 97.8 quality; pool holds ~2 full-1M requests (~1.97×), or ~20 concurrent 100K-context agents.*
