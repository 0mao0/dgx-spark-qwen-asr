#!/usr/bin/env bash
# Launch vLLM serving Qwen3-ASR-0.6B with realtime WS support.
# Both patched engine files are bind-mounted read-only into the container,
# so no custom image build is needed.
#
# Usage:
#   MODEL_DIR=/path/to/Qwen3-ASR-0.6B ./qwen3-asr-realtime.sh
#
# Env:
#   MODEL_DIR  local dir with the Qwen3-ASR-0.6B snapshot (default /models/Qwen3-ASR-0.6B)
#   IMAGE      vLLM image (default vllm/vllm-openai:v0.32.0)
#   PORT       host port, loopback only (default 18101)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/models/Qwen3-ASR-0.6B}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.32.0}"
PORT="${PORT:-18101}"

SEG="${SEG:-5.0}"   # realtime segment duration in seconds (2.0 = faster first partial)
docker rm -f qwen3-asr-realtime 2>/dev/null || true
docker run -d --gpus all --name qwen3-asr-realtime --restart unless-stopped \
  -e QWEN_RT_SEG="$SEG" \
  -v "$MODEL_DIR":/models/model:ro \
  -v "$HERE/patches/qwen3_asr_realtime.py":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_asr_realtime.py:ro \
  -v "$HERE/patches/scheduler.py":/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py:ro \
  -p "127.0.0.1:${PORT}:18000" \
  "$IMAGE" \
  --model /models/model --served-model-name qwen3-asr \
  --host 0.0.0.0 --port 18000 \
  --gpu-memory-utilization 0.15 --max-model-len 8192 \
  --hf-overrides '{"architectures":["Qwen3ASRRealtimeGeneration"]}'

echo "qwen3-asr realtime on ws://127.0.0.1:${PORT}/v1/realtime"
echo "start the header-stripping proxy next: ../proxy/ws_proxy.py"
