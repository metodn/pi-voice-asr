#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi>=0.115", "uvicorn[standard]>=0.30", "httpx>=0.27"]
# ///
"""LFM3-Audio ASR server: same /v1/stt protocol as transcribe_server.py, llama.cpp backend.

Backend is a llama-server running LFM3-Audio (LiquidAI LFM3-Audio-2B GGUF + mmproj),
reached over HTTP. Every decode sends the accumulated tail to the model — the same
offline-decode behaviour the Qwen3-ASR backend of transcribe_server.py has, so the
pi extension UX (partials that may rewrite the tail, pause commits) is identical.

  uv run --script lfm3_asr_server.py --llama-url http://127.0.0.1:8182
  uv run --script lfm3_asr_server.py --port 8765 --language en

Protocol (drop-in for the pi live-stt extension):
  client -> binary frames: raw PCM s16le mono 16 kHz
  client -> text frame   : {"type": "audio.done"}
  server -> {"type": "transcript.created"}
  server -> {"type": "transcript.partial", "text", "is_final": false,
             "speech_final": false, "duration"}
  server -> {"type": "transcript.done", "text", "language", "duration"}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import struct

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

SAMPLE_RATE = 16000

# defaults; run_lfm3_asr.sh / CLI args override via main()
ARGS = argparse.Namespace(
    llama_url="http://127.0.0.1:8182", language=None,
    step_sec=1.2, commit_pause_sec=0.8, min_decode_sec=0.4,
)
_client = httpx.AsyncClient(timeout=120.0)

app = FastAPI(title="lfm3-asr server")


# ---------------------------------------------------------------- WAV + decode

def pcm16_to_wav(pcm: bytes) -> bytes:
    n = len(pcm) // 2
    return b"".join([
        b"RIFF", struct.pack("<I", 36 + n * 2), b"WAVE",
        b"fmt ", struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16),
        b"data", struct.pack("<I", n * 2), pcm,
    ])


LANG_NAMES = {  # ISO code -> name for "Perform ASR in <name>."
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "el": "Greek", "tr": "Turkish", "ar": "Arabic", "he": "Hebrew",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ru": "Russian",
    "cs": "Czech", "uk": "Ukrainian", "hu": "Hungarian", "ro": "Romanian",
}


def rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    n = len(pcm) // 2
    acc = 0
    for i in range(0, n, max(1, n // 800)):  # sample ~800 frames max
        (s,) = struct.unpack_from("<h", pcm, i * 2)
        acc += s * s
    return (acc / max(1, n // max(1, n // 800))) ** 0.5 / 32768.0


async def decode(client: httpx.AsyncClient, pcm: bytes, language: str | None) -> tuple[str, str | None]:
    """PCM16 -> (text, detected_language) via llama-server chat completions."""
    import base64
    lang = LANG_NAMES.get(language, language) if language else None
    system = f"Perform ASR in {lang}." if lang else "Perform ASR."
    payload = {
        "model": "lfm3-audio",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "input_audio",
                 "input_audio": {"data": base64.b64encode(pcm16_to_wav(pcm)).decode()}},
                {"type": "text", "text": "<__media__>"},
            ]},
        ],
        "temperature": 0,
        "max_tokens": 1024,
    }
    r = await client.post(f"{ARGS.llama_url}/v1/chat/completions", json=payload, timeout=120.0)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    # the model sometimes echoes the media placeholder — never show it to the client
    text = re.sub(r"<__media__>", "", text).strip()
    return text, lang


# ---------------------------------------------------------------- /v1/stt WS

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{ARGS.llama_url}/health")
            ok = r.status_code == 200
    except Exception:
        ok = False
    return JSONResponse({"status": "ok" if ok else "degraded", "backend": ARGS.llama_url})


@app.websocket("/v1/stt")
async def stt(ws: WebSocket) -> None:
    language = ws.query_params.get("language") or ARGS.language
    await ws.accept()
    await ws.send_json({"type": "transcript.created"})

    step = ARGS.step_sec            # decode the tail every this many seconds
    pause = ARGS.commit_pause_sec   # freeze the utterance after this much quiet
    frozen: list[str] = []          # committed utterances
    tail = b""                      # audio since the last commit
    tail_text = ""                  # last decode of the tail
    received = 0                    # total samples
    decoded_at = 0                  # tail bytes decoded at the last partial
    quiet_ms = 0
    busy = asyncio.Lock()

    async def run_decode() -> None:
        nonlocal tail_text, decoded_at, quiet_ms
        async with busy:
            if len(tail) < SAMPLE_RATE * 2 * ARGS.min_decode_sec:
                return
            try:
                text, _ = await decode(_client, tail, language)
                if text:
                    tail_text = text
                decoded_at = len(tail)
                await ws.send_json({
                    "type": "transcript.partial",
                    "text": " ".join(t for t in frozen + [tail_text] if t),
                    "is_final": False, "speech_final": False,
                    "duration": round(received / SAMPLE_RATE, 2),
                })
            except Exception as e:  # keep the session alive on backend hiccups
                await ws.send_json({"type": "error", "message": f"decode failed: {e}"})

    loop = asyncio.get_running_loop()

    def schedule_decode() -> None:
        loop.create_task(run_decode())

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return

            if (data := msg.get("bytes")) is not None:
                tail += data
                received += len(data) // 2
                loud = rms(data) > 0.008
                quiet_ms = 0 if loud else quiet_ms + len(data) / 2 / SAMPLE_RATE * 1000

                if len(tail) - decoded_at >= step * SAMPLE_RATE * 2:
                    schedule_decode()
                elif quiet_ms >= pause * 1000 and tail_text and decoded_at == len(tail):
                    # freeze the finished utterance, keep decoding only what follows
                    frozen.append(tail_text)
                    tail_text = ""
                    tail = tail[len(tail):]  # nothing undecoded left
                    quiet_ms = 0
                    await ws.send_json({
                        "type": "transcript.partial",
                        "text": " ".join(t for t in frozen if t),
                        "is_final": False, "speech_final": True,
                        "duration": round(received / SAMPLE_RATE, 2),
                    })

            elif (txt := msg.get("text")) is not None:
                ev = json.loads(txt)
                if ev.get("type") == "audio.done":
                    async with busy:
                        if tail:
                            try:
                                text, _lang = await decode(_client, tail, language)
                                if text:
                                    tail_text = text
                            except Exception as e:
                                await ws.send_json({"type": "error", "message": f"final decode failed: {e}"})
                    final = " ".join(t for t in frozen + [tail_text] if t)
                    await ws.send_json({
                        "type": "transcript.done",
                        "text": final,
                        "language": language or "auto",
                        "duration": round(received / SAMPLE_RATE, 2),
                    })
                    return
    except WebSocketDisconnect:
        return
    finally:
        pass


def main() -> None:
    global ARGS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--llama-url", default=ARGS.llama_url,
                   help="llama-server running LFM3-Audio (run_lfm3_asr.sh starts one)")
    p.add_argument("--language", default=ARGS.language, help="ISO 639-1 code; default auto-detect")
    p.add_argument("--step-sec", type=float, default=ARGS.step_sec, help="partial decode interval")
    p.add_argument("--commit-pause-sec", type=float, default=ARGS.commit_pause_sec,
                   help="silence that freezes an utterance")
    p.add_argument("--min-decode-sec", type=float, default=ARGS.min_decode_sec,
                   help="don't decode tails shorter than this")
    ns = p.parse_args()
    ARGS = argparse.Namespace(**vars(ns))
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="info")


if __name__ == "__main__":
    main()
