#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["qwen-asr[vllm]==0.0.6", "websockets>=15", "numpy"]
# ///
"""Streaming speech-to-text websocket server backed by Qwen3-ASR (vLLM).
Upstream model and toolkit: https://github.com/QwenLM/Qwen3-ASR

  uv run --script server/qwen3_asr_server.py                       # loopback only, downloads the model on first run
  uv run --script server/qwen3_asr_server.py --host 0.0.0.0 --mic-allow <client-ip>

Wire protocol (one dictation per connection):
  client -> binary frames: raw PCM s16le mono 16 kHz, any frame size
  client -> text frame   : {"type": "audio.done"}  (end of dictation)
  server -> {"type": "transcript.created"}                      after connect
  server -> {"type": "transcript.partial", "text": <full hypothesis so far>,
             "is_final": false, "speech_final": false, "duration": <seconds>}
  server -> {"type": "transcript.done", "text": <final>, "language": ..., "duration": <seconds>}
  server -> {"type": "error", "message": ...}                   then closes

Query option `?mic=server`: instead of receiving audio, the server records ITS OWN default
microphone (parec) for the connection. Refused unless the client IP is in --mic-allow
(loopback is always allowed).

Qwen3-ASR streaming is not causal: every chunk_size_sec it re-decodes ALL audio so
far with a rolling text prefix, so partial.text is the whole hypothesis (it may
rewrite its last few tokens). One GPU model, decode calls are serialized.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import threading
import time
import urllib.parse

import numpy as np
import websockets

SAMPLE_RATE = 16000
MAX_STEP_SEC = 1.5
SERVER_MIC_CMD = ["parec", "--rate=16000", "--channels=1", "--format=s16le", "--raw"]
GPU_LOCK = threading.Lock()
asr = None
ARGS: argparse.Namespace
STOP: asyncio.Event
EXIT_CODE = 0  # non-zero when we stop because the engine died, so a supervisor restarts us


def _feed(state, pcm: np.ndarray) -> None:
    with GPU_LOCK:
        before, t0 = state.chunk_id, time.time()
        asr.streaming_transcribe(pcm, state)
        steps = state.chunk_id - before
        if steps:
            # decode cost grows with total audio; stretch the step so decoding stays faster than real time
            per_step = (time.time() - t0) / steps
            stretch = min(2.0 * per_step, MAX_STEP_SEC)  # one slow decode must not freeze updates
            state.chunk_size_samples = int(max(ARGS.chunk_size_sec, stretch) * SAMPLE_RATE)


def _finish(state) -> None:
    with GPU_LOCK:
        asr.finish_streaming_transcribe(state)


async def gpu(fn, *args) -> None:
    """Run a blocking vLLM call; a failure means the engine is gone, so stop the whole server."""
    try:
        await asyncio.to_thread(fn, *args)
    except Exception as exc:
        print(f"engine call failed ({type(exc).__name__}), shutting down", flush=True)
        fail_and_stop()
        raise


def fail_and_stop() -> None:
    global EXIT_CODE
    EXIT_CODE = 1
    STOP.set()


def engine_dead() -> bool:
    return bool(asr.model.llm_engine.engine_core.resources.engine_dead)


def shutdown_engine() -> None:
    """Terminate the EngineCore child process so no vLLM process or VRAM is left behind."""
    try:
        asr.model.llm_engine.engine_core.shutdown()
    except Exception as exc:
        print(f"engine shutdown: {exc}", flush=True)


async def handle(ws) -> None:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(ws.request.path).query)
    server_mic = query.get("mic", [""])[0] == "server"
    peer = ws.remote_address[0] if ws.remote_address else ""
    if server_mic and peer not in ARGS.mic_allow_set and peer not in ("127.0.0.1", "::1"):
        print(f"refused server-mic request from {peer} (not in --mic-allow)", flush=True)
        await ws.send(json.dumps({"type": "error", "message": "server mic not allowed for this client"}))
        await ws.close(1008, "server mic not allowed")
        return
    state = asr.init_streaming_state(
        unfixed_chunk_num=ARGS.unfixed_chunk_num,
        unfixed_token_num=ARGS.unfixed_token_num,
        chunk_size_sec=ARGS.chunk_size_sec,
    )
    await ws.send(json.dumps({"type": "transcript.created"}))
    received = 0  # samples
    last_chunk = 0
    odd = b""  # carry a stray byte if a frame splits an int16 sample

    def dur() -> float:
        return round(received / SAMPLE_RATE, 2)

    async def ingest(data: bytes) -> None:
        nonlocal received, last_chunk, odd
        buf = odd + data
        n = len(buf) // 2 * 2
        odd = buf[n:]
        if not n:
            return
        pcm = np.frombuffer(buf[:n], dtype=np.int16)
        received += len(pcm)
        t0 = time.time()
        await gpu(_feed, state, pcm)
        if state.chunk_id != last_chunk:
            last_chunk = state.chunk_id
            print(f"chunk {last_chunk:3d} decode {1000 * (time.time() - t0):5.0f} ms  {state.text!r}", flush=True)
            await ws.send(json.dumps({
                "type": "transcript.partial", "text": state.text,
                "is_final": False, "speech_final": False, "duration": dur(),
            }))

    async def mic_pump() -> None:
        """Capture this machine's default mic and feed it as if the client had sent the audio."""
        proc = await asyncio.create_subprocess_exec(
            *SERVER_MIC_CMD, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        print(f"server mic open for {peer}", flush=True)
        try:
            while buf := await proc.stdout.read(SAMPLE_RATE * 2 // 10):
                await ingest(buf)
        finally:
            if proc.returncode is None:
                proc.terminate()
            await proc.wait()

    pump = asyncio.create_task(mic_pump()) if server_mic else None
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                if not server_mic:
                    await ingest(msg)
                continue
            try:
                event = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "audio.done":
                if pump:
                    pump.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await pump
                t0 = time.time()
                await gpu(_finish, state)
                print(f"final       decode {1000 * (time.time() - t0):5.0f} ms  {state.text!r}", flush=True)
                await ws.send(json.dumps({
                    "type": "transcript.done", "text": state.text,
                    "language": state.language, "duration": dur(),
                }))
                return
    finally:
        if pump and not pump.done():
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump


async def watchdog() -> None:
    while not STOP.is_set():
        if engine_dead():
            print("engine process died, shutting down", flush=True)
            fail_and_stop()
        await asyncio.sleep(3)


async def main_async() -> None:
    global STOP
    STOP = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, STOP.set)
    async with websockets.serve(handle, ARGS.host, ARGS.port, max_size=None, max_queue=256, ping_interval=None):
        print(f"listening ws://{ARGS.host}:{ARGS.port}/v1/stt", flush=True)
        dog = asyncio.create_task(watchdog())
        await STOP.wait()
        dog.cancel()
    print("stopping", flush=True)


def main() -> None:
    global asr, ARGS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B", help="Hugging Face id (downloaded on first run) or local directory")
    p.add_argument("--host", default="127.0.0.1", help="bind address; 127.0.0.1 = this machine only")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--mic-allow", default="", help="comma-separated client IPs allowed to use mic=server (this machine's mic); loopback is always allowed")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    p.add_argument("--max-model-len", type=int, default=16384, help="tokens of context = max dictation length (~12 audio tokens/s)")
    p.add_argument("--chunk-size-sec", type=float, default=0.5, help="decode step; smaller = livelier, more GPU work")
    p.add_argument("--unfixed-chunk-num", type=int, default=2)
    p.add_argument("--unfixed-token-num", type=int, default=5)
    ARGS = p.parse_args()
    ARGS.mic_allow_set = {ip.strip() for ip in ARGS.mic_allow.split(",") if ip.strip()}

    from qwen_asr import Qwen3ASRModel

    asr = Qwen3ASRModel.LLM(
        model=ARGS.model,
        gpu_memory_utilization=ARGS.gpu_memory_utilization,
        max_model_len=ARGS.max_model_len,
        max_new_tokens=64,
    )
    # the first decode costs ~3 s (kernel warm-up); pay it now instead of on the first key press
    warm = asr.init_streaming_state(chunk_size_sec=ARGS.chunk_size_sec)
    asr.streaming_transcribe(np.zeros(int(SAMPLE_RATE * ARGS.chunk_size_sec), dtype=np.float32), warm)
    print("model loaded + warm", flush=True)
    try:
        asyncio.run(main_async())
    finally:
        shutdown_engine()
    os._exit(EXIT_CODE)  # a normal exit can hang on vLLM's non-daemon threads


if __name__ == "__main__":
    main()
