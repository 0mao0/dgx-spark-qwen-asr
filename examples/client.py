#!/usr/bin/env python3
"""Minimal realtime ASR client: streams a 16 kHz mono PCM16 wav over WS.

Usage:
  python client.py <file.wav> [ws://127.0.0.1:18103/]
The wav is pushed as fast as the socket allows and committed in 5 s segments
(each `commit` starts generation of one segment; the last one is final).
Requires: websockets, soundfile, numpy  (pip install websockets soundfile numpy)
"""
import asyncio
import base64
import json
import sys

import numpy as np
import soundfile as sf
import websockets

SEG_SAMPLES = 5 * 16000  # 5 s segments


async def run(wav_path: str, url: str) -> None:
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import scipy.signal
        wav = scipy.signal.resample_poly(wav, 16000, sr)
    pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

    async with websockets.connect(url, max_size=2**24) as ws:
        await ws.recv()  # session.created
        await ws.send(json.dumps(
            {"type": "session.update", "model": "qwen3-asr"}))
        for i in range(0, len(pcm), SEG_SAMPLES):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[i:i + SEG_SAMPLES].tobytes()).decode(),
            }))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit",
                                  "final": True}))
        while True:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), 30))
            except asyncio.TimeoutError:
                break
            if m.get("type") == "error":
                print("ERROR", m)
                break
            if m.get("final") is True:
                print(m["text"])
                break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    url = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:18103/"
    asyncio.run(run(sys.argv[1], url))
