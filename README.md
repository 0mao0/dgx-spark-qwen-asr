# dgx-spark-qwen-asr

Realtime (streaming) speech recognition recipe for **Qwen3-ASR-0.6B** on
**vLLM 0.32.0**: two engine patches plus one thin WebSocket proxy. It makes
the existing realtime code in vLLM usable as-is, without waiting for upstream
fixes. The official Qwen3-ASR release is offline-only; this is the community
path to microphone-level streaming on a single GPU (tested on NVIDIA GB10).

## What is broken out of the box, and what this repo fixes

vLLM 0.32.0 already exposes a realtime websocket endpoint, but three bugs
make the output unusable. Each is fixed by one component here:

| Bug | Root cause | Fix here |
|---|---|---|
| Later segments repeat earlier text | scheduler appends previous output tokens into the next prompt | `patches/scheduler.py` — port of vLLM PR #56218 (opt-out of session carry) |
| Last ~2 s of every segment is dropped | audio buffer has no overlap/carry | `patches/qwen3_asr_realtime.py` — port of vLLM PR #62487 (PCM16-style sample carry) |
| Every segment starts with `language <L><asr_text>` header text | the realtime serving path never runs `post_process_output` | `proxy/ws_proxy.py` — streaming inline header stripper |

The scheduler patch also makes the policy explicit per model
(`realtime_carries_context`), so non-realtime models keep vLLM default
behaviour.

## Layout

```
patches/                     engine files to bind-mount into the vLLM container
  scheduler.py               vLLM 0.32.0 scheduler.py + PR #56218 port
  qwen3_asr_realtime.py      vLLM 0.32.0 realtime model + PR #62487 port
serve/qwen3-asr-realtime.sh  docker run launcher (no image build needed)
proxy/ws_proxy.py            WebSocket proxy: strips headers, frames like HTTP streaming
examples/client.py           minimal wav-over-WS client
```

## Quick start

```bash
# 1. serve the engine (realtime architecture + patch mounts)
MODEL_DIR=/path/to/Qwen3-ASR-0.6B PORT=18101 ./serve/qwen3-asr-realtime.sh

# 2. run the proxy (strips the language header for clients)
python3 proxy/ws_proxy.py        # ws://127.0.0.1:18103/

# 3. stream a wav
python3 examples/client.py test.wav
```

Without the `--hf-overrides` architecture switch in the serve script, the
engine serves transcription only and the websocket endpoint answers 403.

## Wire protocol

Client side is the standard OpenAI Realtime-style subset:

1. `session.update` → `{"type":"session.update","model":"qwen3-asr"}`
2. `input_audio_buffer.append` with base64 PCM16 mono 16 kHz
3. `input_audio_buffer.commit` every ~2-5 s of audio (starts one segment);
   a final `{"commit","final":true}` ends the stream

Server side, after the proxy: `{"text": <cumulative>, "delta": <piece>,
"final": false}` per partial piece and `{"text": <full>, "final": true}` per
segment. First partial arrives ~0.1 s after the first 5 s of audio on GB10;
5 concurrent streams at 0.14 s p50.

## Notes

- Realtime segment duration is configurable via the `QWEN_RT_SEG` env in
  the serve script (`SEG=2.0 ./serve/qwen3-asr-realtime.sh`, default 5.0).
  Quality vs latency on dictation audio (measured on AISHELL clips):
  5 s = clean baseline; 3 s ≈ clean; 2 s = whole sentences with minor
  boundary slips, first partial ~2.3 s; 1 s = 1–2-character fragments,
  unusable. "Word-at-a-time under ~1 s" is not achievable with this
  model's window decoding — it is a model property, not a patch gap.
- The patches bind-mount into `vllm/vllm-openai:v0.32.0`. They are pinned to
  that image; other vLLM versions need a rebase of `patches/scheduler.py`.
- The two patch files are derived from vLLM sources and carry the same
  Apache-2.0 license header.
- This is not an official vLLM feature and not a fork: it is a working
  configuration + patches, tracked against upstream PRs #62487, #56218.

## References

- [vLLM](https://github.com/vllm-project/vllm) — the serving engine
- [vLLM PR #62487](https://github.com/vllm-project/vllm/pull/62487) — realtime audio carry
- [vLLM PR #56218](https://github.com/vllm-project/vllm/pull/56218) — realtime context policy
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — the model
