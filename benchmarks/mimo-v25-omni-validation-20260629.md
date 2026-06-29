# MiMo V2.5 Omni Smoke Validation - 2026-06-29

## Verdict

Basic Omni validation passed on the live MiMo V2.5 NVFP4-KV server.

We validated all three media input paths through the OpenAI-compatible `/v1/chat/completions` API:

- Image
- Audio
- Video

This upgrades the claim from "Omni architecture loads" to:

> The MiMo V2.5 Omni runtime accepts and correctly responds to basic image, audio, and video inputs on the live 2x DGX Spark TP=2 NVFP4-KV server.

This is still a smoke test, not a full multimodal eval suite.

## Runtime

Server:

```text
http://127.0.0.1:8000
```

Model:

```text
MiMo-V2.5-NVFP4
```

Current server profile:

```text
MAX_MODEL_LEN=1000000
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=2048
BLOCK_SIZE=64
MTP_SPEC_TOKENS=1
VLLM_MIMO_MTP1_GREEDY_FAST=1
kv_cache_dtype=nvfp4
```

Server health after validation:

```text
/v1/models: READY
vllm:num_requests_running: 0
vllm:num_requests_waiting: 0
```

## Test Assets

Generated locally and sent as base64 data URLs:

| Asset | Description | Size |
|---|---|---:|
| `test_image.png` | Red square image with blue center circle and text | 1.5 KB |
| `test_tone.wav` | 1 second 440 Hz sine tone | 31 KB |
| `test_video.mp4` | Short red video with blue center square | 2.8 KB |

## Results

### Image

Prompt:

```text
Look at the image. What are the two main colors and what shape is in the center? Answer briefly.
```

Result:

```text
HTTP 200
elapsed: 1.51s
prompt_tokens: 115
completion_tokens: 35
```

Model answer:

```text
Based on the image provided:

*   Two main colors: Red and Blue.
*   Shape in the center: A circle (or circle within a circle).
```

Interpretation: passed. The answer correctly identified the colors and the center shape.

### Audio

Prompt:

```text
Listen to the audio. Is it silence, speech, music, or a simple tone? Answer briefly.
```

Result:

```text
HTTP 200
elapsed: 5.65s
prompt_tokens: 57
completion_tokens: 8
```

Model answer:

```text
The audio is a simple tone.
```

Interpretation: passed. The answer correctly recognized the generated sine tone as a simple tone.

### Video

Prompt:

```text
Watch the video. What are the dominant colors and what colored shape appears near the center? Answer briefly.
```

Result:

```text
HTTP 200
elapsed: 1.05s
prompt_tokens: 265
completion_tokens: 17
```

Model answer:

```text
The dominant colors are red and blue. A blue square appears near the center.
```

Interpretation: passed. The answer correctly identified the dominant colors and the center square.

## Raw Result File

JSON output:

```text
outputs/mimo_omni_validation_results_20260629.json
```

## Claim Boundary

Safe claim:

> MiMo V2.5 NVFP4-KV on 2x DGX Spark is 1M-context capable, concurrency-tested to C8 on short requests, and basic Omni image/audio/video inputs pass through the live vLLM OpenAI API.

Do not yet claim:

- production-grade multimodal correctness
- long-video understanding
- multi-image stress behavior
- audio transcription quality
- multimodal concurrency scaling
- tool-use correctness with multimodal prompts

Those need separate evals.
