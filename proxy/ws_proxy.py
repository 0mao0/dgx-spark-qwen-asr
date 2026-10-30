#!/usr/bin/env python3
"""Thin WS proxy: client <-> vLLM realtime ASR.

Strips the per-segment `language <L><asr_text>` header and presents the same
frame semantics as the HTTP streaming endpoint:
  {"text": <cumulative across segments>, "delta": <piece>, "final": false}
  {"text": <cumulative>, "final": true}        at each segment done
Everything else (session.*, errors) passes through.

Env:
  ASR_UPSTREAM   ws url of the realtime server (default ws://127.0.0.1:18101/v1/realtime)
  ASR_PROXY_HOST listen host (default 127.0.0.1)
  ASR_PROXY_PORT listen port (default 18103)
"""
import asyncio
import json
import os
import re
import traceback

import websockets

UPSTREAM = os.environ.get("ASR_UPSTREAM", "ws://127.0.0.1:18101/v1/realtime")
LISTEN_HOST = os.environ.get("ASR_PROXY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("ASR_PROXY_PORT", "18103"))

HDR = re.compile(r"language\s+[^<]*<asr_text>")


class SegFilter:
    """Streaming inline header stripper.

    The realtime protocol emits no per-segment done events, so the
    `language <L><asr_text>` markers can appear anywhere mid-stream.
    Hold the tail while it could still be an incomplete marker, strip
    complete markers anywhere, emit everything else.
    """

    def __init__(self):
        self.pending = ""

    @staticmethod
    def _holds(p):
        if "language" not in p:
            return False
        i = p.rfind("language")
        tail = p[i:]
        return "<asr_text>" not in tail and len(tail) < 60

    def feed(self, piece):
        self.pending += piece
        if HDR.search(self.pending):
            self.pending = HDR.sub("", self.pending)
        if self._holds(self.pending):
            return ""
        out = self.pending
        self.pending = ""
        return out

    def finish(self):
        out = HDR.sub("", self.pending)
        self.pending = ""
        return out


async def handle(client):
    total = []

    async def u2c():
        seg = SegFilter()
        async with websockets.connect(UPSTREAM, max_size=2**24) as up:

            async def c2u():
                async for msg in client:
                    await up.send(msg)

            task = asyncio.create_task(c2u())
            try:
                async for msg in up:
                    try:
                        evt = json.loads(msg)
                    except json.JSONDecodeError:
                        await client.send(msg)
                        continue
                    t = evt.get("type")
                    if t == "transcription.delta":
                        piece = seg.feed(evt.get("delta", ""))
                        if piece:
                            total.append(piece)
                            await client.send(json.dumps(
                                {"text": "".join(total), "delta": piece,
                                 "final": False},
                                ensure_ascii=False))
                    elif t == "transcription.done":
                        rest = seg.finish()
                        if rest:
                            total.append(rest)
                        await client.send(json.dumps(
                            {"text": "".join(total), "final": True},
                            ensure_ascii=False))
                        seg = SegFilter()
                    else:
                        await client.send(msg)
            finally:
                task.cancel()

    try:
        await u2c()
    except asyncio.CancelledError:
        pass
    except Exception:
        traceback.print_exc()


async def main():
    async with websockets.serve(handle, LISTEN_HOST, LISTEN_PORT,
                                max_size=2**24):
        print(f"rt proxy up on ws://{LISTEN_HOST}:{LISTEN_PORT}/", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
