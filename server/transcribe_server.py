#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi>=0.115", "uvicorn[standard]>=0.30", "python-multipart>=0.0.9", "numpy"]
# ///
"""Speech-to-text server on transcribe.cpp (ggml / CUDA): HTTP + WebSocket on one port.

  uv run --script transcribe_server.py --model models/Qwen3-ASR-0.6B-Q4_K_M.gguf
  uv run --script transcribe_server.py --host 0.0.0.0 --port 8765 --mic-allow <client-ip>

Drop-in endpoints (API keys / auth headers are accepted and ignored):
  OpenAI       POST /v1/audio/transcriptions   multipart: file, [model], [language], [prompt], [temperature],
                                               [response_format]: json | text | srt | verbose_json | vtt,
                                               [timestamp_granularities[]]: segment | word
               WS   /v1/realtime               minimal Realtime transcription: append base64 pcm16 (24 kHz), commit, get
                                               delta + completed. No server VAD (client must commit), no g711.
               GET  /v1/models, GET /v1/models/{id}, GET /health
  ElevenLabs   POST /v1/speech-to-text         multipart: file, [model_id], [language_code], [timestamps_granularity]
  Deepgram     POST /v1/listen                 raw audio body; query: language, detect_language, utterances, ...
               WS   /v1/listen                 linear16/mulaw/alaw (encoding, sample_rate, channels) or any container;
                                               interim_results, endpointing; Finalize / CloseStream / KeepAlive
  Errors come back in the shape of the API that was called (OpenAI / ElevenLabs / Deepgram).

Qwen3-ASR returns no timestamps. Long audio is therefore cut at pauses into chunks of at most --segment-sec
seconds that are decoded one by one: segment times are the chunk bounds, word times are interpolated inside
a chunk in proportion to word length (approximate). confidence / logprob fields are placeholders.

Own WebSocket /v1/stt (one dictation per connection):
  client -> binary frames: raw PCM s16le mono 16 kHz, any frame size
  client -> text frame   : {"type": "audio.done"}  (end of dictation)
  server -> {"type": "transcript.created"}                      after connect
  server -> {"type": "transcript.partial", "text": <full hypothesis so far>,
             "is_final": false, "speech_final": false, "duration": <seconds>}
  server -> {"type": "transcript.done", "text": <final>, "language": ..., "duration": <seconds>}
  server -> {"type": "error", "message": ...}                   then closes
Query option `?mic=server` records the server's own default microphone (parec); refused unless the
client IP is in --mic-allow (loopback is always allowed).
Pause commits (--commit-pause-sec, on by default): after a pause the utterance so far is decoded one last time and
frozen. Later steps decode only the audio since that pause, so the cost per step stays flat however long the
dictation runs. partial.text is still the full text (frozen part + current utterance). 0 = decode everything
each step, like before.

Qwen3-ASR is an offline model: every step the audio so far is decoded again, so a partial may rewrite its
tail. One GPU session, decodes are serialized.
llama-swap: use `cmd: ... transcribe_server.py --port ${PORT}`, `checkEndpoint: /health`.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

HERE = Path(__file__).resolve().parent
# Dev-tree fallback: use the in-repo bindings and shared build when transcribe_cpp is not installed.
sys.path.insert(0, str(HERE / "transcribe.cpp" / "bindings" / "python" / "src"))
_shared = HERE / "transcribe.cpp" / "build-shared" / "src" / "libtranscribe.so"
if _shared.exists():
    os.environ.setdefault("TRANSCRIBE_LIBRARY", str(_shared))

import transcribe_cpp  # noqa: E402

SAMPLE_RATE = 16000
MIN_SAMPLES = SAMPLE_RATE // 4  # chunks shorter than this are not decoded
SERVER_MIC_CMD = ["parec", "--rate=16000", "--channels=1", "--format=s16le", "--raw"]
GPU_LOCK = threading.Lock()
app = FastAPI(title="transcribe.cpp server")
ARGS: argparse.Namespace
MODEL: transcribe_cpp.Model
SESSION: transcribe_cpp.Session
MODEL_ID = "qwen3-asr"

LANG_NAMES = {  # languages of Qwen3-ASR; OpenAI reports the lowercase English name
    "zh": "chinese", "en": "english", "yue": "cantonese", "ar": "arabic", "de": "german", "fr": "french",
    "es": "spanish", "pt": "portuguese", "id": "indonesian", "it": "italian", "ko": "korean", "ru": "russian",
    "th": "thai", "vi": "vietnamese", "ja": "japanese", "tr": "turkish", "hi": "hindi", "ms": "malay",
    "nl": "dutch", "sv": "swedish", "da": "danish", "fi": "finnish", "pl": "polish", "cs": "czech",
    "fil": "filipino", "fa": "persian", "el": "greek", "ro": "romanian", "hu": "hungarian", "mk": "macedonian",
}
ISO3 = {  # ISO 639-3 (ElevenLabs) -> the codes above
    "zho": "zh", "cmn": "zh", "eng": "en", "ara": "ar", "deu": "de", "fra": "fr", "spa": "es", "por": "pt",
    "ind": "id", "ita": "it", "kor": "ko", "rus": "ru", "tha": "th", "vie": "vi", "jpn": "ja", "tur": "tr",
    "hin": "hi", "msa": "ms", "nld": "nl", "swe": "sv", "dan": "da", "fin": "fi", "pol": "pl", "ces": "cs",
    "fas": "fa", "ell": "el", "ron": "ro", "hun": "hu", "mkd": "mk",
}


def norm_lang(code: str | None) -> str | None:
    """Client language hint (en, en-US, eng, multi, auto, ...) -> model language code; None = auto-detect."""
    if not code or code.lower() in ("auto", "multi", "detect"):
        return None
    c = code.lower().replace("_", "-").split("-")[0]
    return ISO3.get(c, c)


def qbool(value: str | None) -> bool:
    return (value or "").lower() in ("1", "true", "yes", "on")


# ----------------------------------------------------------------------------------------------------------
# errors, each API gets its own error shape
# ----------------------------------------------------------------------------------------------------------

class ApiError(Exception):
    def __init__(self, status: int, message: str, code: str | None = None):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


def error_response(path: str, status: int, message: str, code: str | None = None) -> JSONResponse:
    if path.startswith("/v1/speech-to-text"):  # ElevenLabs
        body = {"detail": {"status": code or "error", "message": message}}
    elif path.startswith("/v1/listen"):  # Deepgram
        body = {"err_code": (code or "ERROR").upper(), "err_msg": message, "request_id": str(uuid.uuid4())}
    else:  # OpenAI
        body = {"error": {"message": message, "type": "invalid_request_error" if status < 500 else "server_error",
                          "param": None, "code": code}}
    return JSONResponse(body, status_code=status)


@app.exception_handler(ApiError)
async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
    return error_response(request.url.path, exc.status, exc.message, exc.code)


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return error_response(request.url.path, exc.status_code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0]
    where = ".".join(str(p) for p in first["loc"] if p != "body")
    return error_response(request.url.path, 400, f"{where}: {first['msg']}" if where else first["msg"], "invalid_request")


def too_large() -> ApiError:
    return ApiError(413, f"Maximum content size limit ({ARGS.max_upload_mb * 2**20}) exceeded", "payload_too_large")


@app.middleware("http")
async def limit_upload(request: Request, call_next):
    """Refuse oversized uploads from Content-Length before the body is parsed / spooled."""
    if request.method == "POST":
        with contextlib.suppress(ValueError):
            if int(request.headers.get("content-length", 0)) > ARGS.max_upload_mb * 2**20:
                exc = too_large()
                return error_response(request.url.path, exc.status, exc.message, exc.code)
    return await call_next(request)


async def read_upload(file: UploadFile) -> bytes:
    buf = bytearray()
    while chunk := await file.read(1 << 20):
        buf += chunk
        if len(buf) > ARGS.max_upload_mb * 2**20:  # chunked upload without Content-Length
            raise too_large()
    return bytes(buf)


async def read_body(request: Request) -> bytes:
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > ARGS.max_upload_mb * 2**20:
            raise too_large()
    return bytes(buf)


# ----------------------------------------------------------------------------------------------------------
# decoding, segmentation, word timing
# ----------------------------------------------------------------------------------------------------------

def transcribe(pcm: np.ndarray, language: str | None = None, prompt: str | None = None):
    """Blocking, serialized decode of float32 16 kHz mono PCM."""
    kwargs = {}
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt
    with GPU_LOCK:
        return SESSION.run(np.ascontiguousarray(pcm, dtype=np.float32), **kwargs)


async def decode_file(data: bytes, *input_args: str) -> np.ndarray:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-v", "error", *input_args, "-i", "pipe:0", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(data)
    if proc.returncode != 0:
        raise ApiError(400, f"could not decode audio: {err.decode(errors='replace').strip()[:300]}", "invalid_audio")
    return np.frombuffer(out, dtype=np.float32)


HOP = SAMPLE_RATE // 50  # 20 ms analysis frames


def voiced_mask(pcm: np.ndarray) -> np.ndarray:
    """Per 20 ms frame: is there speech? Energy gate relative to the loud end of the clip."""
    n = len(pcm) // HOP
    if not n:
        return np.zeros(0, dtype=bool)
    rms = np.sqrt((pcm[: n * HOP].reshape(n, HOP) ** 2).mean(axis=1))
    return rms > max(0.004, 0.06 * float(np.percentile(rms, 99)))


def split_speech(pcm: np.ndarray) -> list[tuple[int, int]]:
    """Sample ranges to decode: speech islands, cut at pauses >= 0.3 s and packed into chunks of at most
    --segment-sec seconds. Audio without any speech yields no ranges."""
    voiced = np.flatnonzero(voiced_mask(pcm))
    if not len(voiced):
        return []
    pad, gap = 5, 15  # frames: 0.1 s of context around speech, pauses shorter than 0.3 s do not split
    breaks = np.flatnonzero(np.diff(voiced) > gap)
    starts = np.r_[voiced[0], voiced[breaks + 1]]
    ends = np.r_[voiced[breaks], voiced[-1]] + 1
    limit = int(ARGS.segment_sec * SAMPLE_RATE)
    out: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = None
    for s, e in zip(starts, ends):
        s, e = max(0, int(s) - pad) * HOP, min(len(pcm), (int(e) + pad) * HOP)
        while e - s > limit:  # one island longer than the cap: hard split
            if cur:
                out.append(cur)
                cur = None
            out.append((s, s + limit))
            s += limit
        if cur and e - cur[0] <= limit:
            cur = (cur[0], e)
        else:
            if cur:
                out.append(cur)
            cur = (s, e)
    if cur:
        out.append(cur)
    return out


def speech_state(pcm: np.ndarray, silence_sec: float) -> tuple[bool, bool]:
    """(any speech in pcm, has it been quiet for the last silence_sec)."""
    voiced = np.flatnonzero(voiced_mask(pcm))
    if not len(voiced):
        return False, False
    return True, len(pcm) // HOP - 1 - int(voiced[-1]) >= silence_sec * 50


_CJK = "぀-ヿ㐀-䶿一-鿿฀-๿"  # kana, han, thai: written without spaces
_TOKEN = re.compile(f"[{_CJK}]|[^\\s{_CJK}]+")
_CJK_CHAR = re.compile(f"[{_CJK}]")


@dataclass
class Word:
    text: str  # as written, with punctuation
    start: float
    end: float

    @property
    def bare(self) -> str:
        return re.sub(r"^\W+|\W+$", "", self.text)


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word]


@dataclass
class Transcript:
    text: str
    language: str  # model code (en, de, ...), "" if unknown
    duration: float
    segments: list[Segment]

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]


def needs_space(a: str, b: str) -> bool:
    return not (_CJK_CHAR.match(a[-1]) or _CJK_CHAR.match(b[0]))


def join_texts(parts: list[str]) -> str:
    out = ""
    for p in parts:
        out += (" " if out and needs_space(out, p) else "") + p
    return out


def words_for(text: str, t0: float, t1: float) -> list[Word]:
    """Spread the words of text over [t0, t1] in proportion to their length (the model gives no word times)."""
    toks: list[str] = []
    prefix = ""
    for m in _TOKEN.finditer(text):
        tok = m.group()
        if not re.search(r"\w", tok):  # lone punctuation sticks to its neighbour
            if toks:
                toks[-1] += tok
            else:
                prefix += tok
        else:
            toks.append(prefix + tok)
            prefix = ""
    if not toks:
        return []
    weights = [max(1, len(Word(t, 0, 0).bare)) for t in toks]
    unit = (t1 - t0) / sum(weights)
    words, at = [], t0
    for tok, w in zip(toks, weights):
        words.append(Word(tok, round(at, 3), round(at + w * unit, 3)))
        at += w * unit
    return words


def transcribe_full(pcm: np.ndarray, language: str | None, prompt: str | None) -> Transcript:
    """Blocking. Decode speech chunk by chunk (handles long audio) and keep per-chunk times."""
    segments: list[Segment] = []
    detected = ""
    for s, e in split_speech(pcm):
        if e - s < MIN_SAMPLES:
            continue
        try:
            result = transcribe(pcm[s:e], language, prompt)
        except transcribe_cpp.InputTooShort:
            continue
        if result.language and not detected:
            detected = result.language
        text = result.text.strip()
        if text:
            t0, t1 = s / SAMPLE_RATE, e / SAMPLE_RATE
            segments.append(Segment(len(segments), round(t0, 3), round(t1, 3), text, words_for(text, t0, t1)))
    return Transcript(join_texts([s.text for s in segments]), language or detected, len(pcm) / SAMPLE_RATE, segments)


async def run_transcription(pcm: np.ndarray, language: str | None, prompt: str | None, tag: str) -> Transcript:
    if not len(pcm):
        raise ApiError(400, "audio is empty", "invalid_audio")
    lang = norm_lang(language)
    t0 = time.time()
    try:
        tr = await asyncio.to_thread(transcribe_full, pcm, lang, prompt)
    except transcribe_cpp.UnsupportedRequest as exc:
        raise ApiError(400, f"unsupported language {language!r} (supported: {', '.join(LANG_NAMES)})", "unsupported_language") from exc
    except transcribe_cpp.TranscribeError as exc:
        raise ApiError(422, str(exc), "transcription_failed") from exc
    print(f"{tag:<5} {tr.duration:6.1f}s audio, {len(tr.segments)} segments, decoded in {1000 * (time.time() - t0):5.0f} ms  {tr.text[:200]!r}", flush=True)
    return tr


# ----------------------------------------------------------------------------------------------------------
# subtitles
# ----------------------------------------------------------------------------------------------------------

def stamp(sec: float, sep: str) -> str:
    ms = round(sec * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def cues(tr: Transcript, max_sec: float = 6.0, max_chars: int = 80) -> list[tuple[float, float, str]]:
    """Subtitle cues built from the word times: short, and broken at sentence ends."""
    out: list[tuple[float, float, str]] = []
    cur: list[Word] = []

    def flush() -> None:
        if cur:
            out.append((cur[0].start, cur[-1].end, join_texts([w.text for w in cur])))
            cur.clear()

    for w in tr.words:
        if cur and (w.end - cur[0].start > max_sec or len(join_texts([x.text for x in cur])) + len(w.text) > max_chars):
            flush()
        cur.append(w)
        if re.search(r"[.!?。！？]$", w.text) and w.end - cur[0].start > 2:
            flush()
    flush()
    return out


def subtitles(tr: Transcript, fmt: str) -> str:
    sep = "," if fmt == "srt" else "."
    blocks = [f"{stamp(a, sep)} --> {stamp(b, sep)}\n{text}" for a, b, text in cues(tr)]
    if fmt == "srt":
        return "".join(f"{i}\n{b}\n\n" for i, b in enumerate(blocks, 1))
    return "WEBVTT\n\n" + "".join(f"{b}\n\n" for b in blocks)


# ----------------------------------------------------------------------------------------------------------
# info endpoints
# ----------------------------------------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID, "backend": str(MODEL.backend)}


def model_card() -> dict:
    return {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "local"}


@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [model_card()]}


@app.get("/v1/models/{model_id}")
def model_info(model_id: str) -> dict:
    return model_card()  # any id resolves to the one loaded model, like the transcription endpoints


# ----------------------------------------------------------------------------------------------------------
# OpenAI: POST /v1/audio/transcriptions
# ----------------------------------------------------------------------------------------------------------

@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(""),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),  # accepted for client compatibility, ignored
    granularities: list[str] = Form([], alias="timestamp_granularities[]"),
    granularities_plain: list[str] = Form([], alias="timestamp_granularities"),
):
    if response_format not in ("json", "text", "srt", "verbose_json", "vtt"):
        raise ApiError(400, f"Invalid value: '{response_format}'. Supported values are: 'json', 'text', 'srt', 'verbose_json', and 'vtt'.", "invalid_value")
    grans = set(granularities) | set(granularities_plain) or {"segment"}
    if not grans <= {"word", "segment"}:
        raise ApiError(400, "timestamp_granularities must be 'word' and/or 'segment'", "invalid_value")
    tr = await run_transcription(await decode_file(await read_upload(file)), language, prompt, "http")
    if response_format == "text":
        return PlainTextResponse(tr.text)
    if response_format in ("srt", "vtt"):
        return PlainTextResponse(subtitles(tr, response_format), media_type="text/vtt" if response_format == "vtt" else "text/plain")
    if response_format == "json":
        return {"text": tr.text}
    body: dict = {"task": "transcribe", "language": LANG_NAMES.get(tr.language, tr.language), "duration": round(tr.duration, 2), "text": tr.text}
    if "word" in grans:
        body["words"] = [{"word": w.bare, "start": w.start, "end": w.end} for w in tr.words]
    if "segment" in grans:
        body["segments"] = [
            {"id": s.id, "seek": 0, "start": s.start, "end": s.end, "text": s.text, "tokens": [], "temperature": 0.0,
             "avg_logprob": 0.0, "compression_ratio": 1.0, "no_speech_prob": 0.0}
            for s in tr.segments
        ]
    return body


# ----------------------------------------------------------------------------------------------------------
# ElevenLabs: POST /v1/speech-to-text
# ----------------------------------------------------------------------------------------------------------

@app.post("/v1/speech-to-text")
async def elevenlabs_speech_to_text(
    file: UploadFile = File(...),
    model_id: str = Form(""),
    language_code: str | None = Form(None),
    timestamps_granularity: str = Form("word"),
):
    tr = await run_transcription(await decode_file(await read_upload(file)), language_code, None, "http")
    words: list[dict] = []
    if timestamps_granularity != "none":
        prev: Word | None = None
        for w in tr.words:
            if prev and needs_space(prev.text, w.text):
                words.append({"text": " ", "start": prev.end, "end": w.start, "type": "spacing"})
            words.append({"text": w.text, "start": w.start, "end": w.end, "type": "word"})
            prev = w
    return {"language_code": tr.language, "language_probability": 1.0, "text": tr.text, "words": words,
            "transcription_id": uuid.uuid4().hex[:22]}


# ----------------------------------------------------------------------------------------------------------
# Deepgram: POST /v1/listen
# ----------------------------------------------------------------------------------------------------------

def dg_words(words: list[Word]) -> list[dict]:
    return [{"word": w.bare.lower(), "start": w.start, "end": w.end, "confidence": 1.0, "punctuated_word": w.text} for w in words]


def dg_model_info() -> dict:
    return {MODEL_ID: {"name": MODEL_ID, "version": "1", "arch": "qwen3-asr"}}


@app.post("/v1/listen")
async def deepgram_listen(request: Request):
    q = request.query_params
    if request.headers.get("content-type", "").startswith("application/json"):
        raise ApiError(400, "url sources are not supported; send the audio bytes as the request body", "UNSUPPORTED")
    data = await read_body(request)
    if not data:
        raise ApiError(400, "Bad Request: empty audio body", "INVALID_AUDIO")
    request_id = str(uuid.uuid4())
    tr = await run_transcription(await decode_file(data), q.get("language"), None, "http")
    channel: dict = {"alternatives": [{"transcript": tr.text, "confidence": 1.0, "words": dg_words(tr.words)}]}
    if qbool(q.get("detect_language")):
        channel |= {"detected_language": tr.language, "language_confidence": 1.0}
    results: dict = {"channels": [channel]}
    if qbool(q.get("utterances")):
        results["utterances"] = [
            {"start": s.start, "end": s.end, "confidence": 1.0, "channel": 0, "transcript": s.text,
             "words": dg_words(s.words), "id": str(uuid.uuid4())}
            for s in tr.segments
        ]
    return {
        "metadata": {"transaction_key": "deprecated", "request_id": request_id, "sha256": "",
                     "created": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                     "duration": round(tr.duration, 3), "channels": 1, "models": [MODEL_ID], "model_info": dg_model_info()},
        "results": results,
    }


# ----------------------------------------------------------------------------------------------------------
# Deepgram: WS /v1/listen
# ----------------------------------------------------------------------------------------------------------

DG_INPUT = {"linear16": ["-f", "s16le"], "mulaw": ["-f", "mulaw"], "alaw": ["-f", "alaw"]}


@app.websocket("/v1/listen")
async def deepgram_stream(ws: WebSocket) -> None:
    q = ws.query_params
    language = norm_lang(q.get("language"))
    interim = qbool(q.get("interim_results"))
    ep = q.get("endpointing", "")
    endpoint_sec = None if ep == "false" else max(0.4, int(ep) / 1000) if ep.isdigit() else 0.7
    encoding = (q.get("encoding") or "").lower()
    try:
        rate, channels = int(q.get("sample_rate", SAMPLE_RATE)), int(q.get("channels", 1))
    except ValueError:
        rate, channels = SAMPLE_RATE, 1
    request_id = str(uuid.uuid4())
    # browsers send the API key as a subprotocol ("token", key) and need one of them echoed back
    protocols = ws.scope.get("subprotocols") or []
    await ws.accept(subprotocol=protocols[0] if protocols else None)

    proc = None
    if not (encoding == "linear16" and rate == SAMPLE_RATE and channels == 1):
        # resample / decode with ffmpeg; an unknown or missing encoding means a container (wav, webm, ogg, ...)
        raw_in = ["-ar", str(rate), "-ac", str(channels)] if encoding in DG_INPUT else []
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", *DG_INPUT.get(encoding, []), *raw_in, "-i", "pipe:0",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )

    chunks: list[np.ndarray] = []  # audio of the utterance in progress (the "tail")
    tail_start = 0  # samples before the tail
    total = 0  # samples received
    decoded_at = 0  # tail length at the last interim decode
    step = ARGS.chunk_size_sec
    odd = b""
    send_lock = asyncio.Lock()

    def tail() -> np.ndarray:
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    async def emit(text: str, t0: float, t1: float, final: bool, speech_final: bool, from_finalize: bool) -> None:
        msg = {
            "type": "Results", "channel_index": [0, 1], "duration": round(t1 - t0, 3), "start": round(t0, 3),
            "is_final": final, "speech_final": speech_final,
            "channel": {"alternatives": [{"transcript": text, "confidence": 1.0, "words": dg_words(words_for(text, t0, t1))}]},
            "metadata": {"request_id": request_id, "model_info": {"name": MODEL_ID, "version": "1", "arch": "qwen3-asr"}, "model_uuid": request_id},
            "from_finalize": from_finalize,
        }
        await ws.send_json(msg)

    async def decode_tail(final: bool, speech_final: bool = False, from_finalize: bool = False) -> None:
        nonlocal chunks, tail_start, decoded_at, step
        async with send_lock:
            pcm = tail()
            text = ""
            if len(pcm) >= MIN_SAMPLES and speech_state(pcm, 0)[0]:
                t0 = time.time()
                try:
                    text = (await asyncio.to_thread(transcribe, pcm, language)).text.strip()
                except transcribe_cpp.InputTooShort:
                    pass
                step = max(ARGS.chunk_size_sec, min(2.0 * (time.time() - t0), ARGS.max_step_sec))
                print(f"dg {'final' if final else 'partial'} {len(pcm) / SAMPLE_RATE:6.2f}s decode {1000 * (time.time() - t0):5.0f} ms  {text!r}", flush=True)
            start = tail_start / SAMPLE_RATE
            end = (tail_start + len(pcm)) / SAMPLE_RATE
            if final:
                tail_start += len(pcm)
                chunks, decoded_at = [], 0
                if not text and not from_finalize:
                    return
            elif not text:
                return
            await emit(text, start, end, final, speech_final, from_finalize)

    async def ingest(data: bytes) -> None:
        nonlocal total, decoded_at, odd, tail_start
        buf = odd + data
        n = len(buf) // 2 * 2
        odd = buf[n:]
        if not n:
            return
        pcm = np.frombuffer(buf[:n], dtype=np.int16).astype(np.float32) / 32768.0
        chunks.append(pcm)
        total += len(pcm)
        have = sum(len(c) for c in chunks)
        if endpoint_sec is not None:
            spoke, quiet = speech_state(tail(), endpoint_sec)
            if spoke and quiet:
                await decode_tail(True, speech_final=True)
                return
            if not spoke and have > 2 * SAMPLE_RATE:  # drop long silence instead of letting the tail grow
                chunks.clear()
                tail_start += have
                decoded_at = 0
                return
        if have >= ARGS.segment_sec * SAMPLE_RATE:  # no pause for a long time: cut anyway
            await decode_tail(True)
        elif interim and have - decoded_at >= step * SAMPLE_RATE:
            decoded_at = have
            await decode_tail(False)

    async def pump_ffmpeg() -> None:
        while buf := await proc.stdout.read(SAMPLE_RATE * 2 // 10):
            await ingest(buf)

    pump = asyncio.create_task(pump_ffmpeg()) if proc else None
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes") is not None:
                if proc:
                    proc.stdin.write(msg["bytes"])
                    await proc.stdin.drain()
                else:
                    await ingest(msg["bytes"])
                continue
            try:
                kind = json.loads(msg.get("text") or "").get("type")
            except (json.JSONDecodeError, AttributeError):
                continue
            if kind == "Finalize":
                await decode_tail(True, speech_final=True, from_finalize=True)
            elif kind == "CloseStream":
                break
        if proc:
            proc.stdin.close()
            await pump
        await decode_tail(True, speech_final=True, from_finalize=True)
        await ws.send_json({"type": "Metadata", "transaction_key": "deprecated", "request_id": request_id, "sha256": "",
                            "created": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                            "duration": round(total / SAMPLE_RATE, 3), "channels": 1})
        await ws.close()
    except WebSocketDisconnect:
        return
    except transcribe_cpp.TranscribeError as exc:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "Error", "description": str(exc), "message": str(exc), "variant": ""})
            await ws.close(1011)
    finally:
        if pump and not pump.done():
            pump.cancel()
        if proc and proc.returncode is None:
            proc.kill()
        if proc:
            with contextlib.suppress(Exception):
                await proc.wait()


# ----------------------------------------------------------------------------------------------------------
# OpenAI Realtime, transcription sessions: WS /v1/realtime  (minimal: the client commits, no server VAD)
# ----------------------------------------------------------------------------------------------------------

@app.websocket("/v1/realtime")
async def openai_realtime(ws: WebSocket) -> None:
    """append base64 pcm16 (session rate, default 24 kHz) -> commit -> committed, one delta with the whole text, completed."""
    protocols = ws.scope.get("subprotocols") or []
    await ws.accept(subprotocol="realtime" if "realtime" in protocols else (protocols[0] if protocols else None))
    sid = "sess_" + uuid.uuid4().hex[:16]
    rate, language = 24000, None
    audio = bytearray()
    prev_item = None

    async def send(kind: str, **fields) -> None:
        await ws.send_json({"event_id": "event_" + uuid.uuid4().hex[:20], "type": kind, **fields})

    def session() -> dict:
        return {"id": sid, "object": "realtime.transcription_session", "type": "transcription",
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": rate},
                                    "transcription": {"model": MODEL_ID, "language": language}, "turn_detection": None}}}

    await send("session.created", session=session())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            try:
                event = json.loads(msg.get("text") or "")
                kind = event["type"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if kind == "input_audio_buffer.append":
                with contextlib.suppress(KeyError, TypeError, ValueError):
                    audio += base64.b64decode(event["audio"])
            elif kind == "input_audio_buffer.clear":
                audio.clear()
                await send("input_audio_buffer.cleared")
            elif kind in ("session.update", "transcription_session.update"):
                inp = (event.get("session") or {}).get("audio", {}).get("input", {})
                rate = int((inp.get("format") or {}).get("rate") or rate)
                language = norm_lang((inp.get("transcription") or {}).get("language")) or language
                await send("session.updated", session=session())
            elif kind == "input_audio_buffer.commit":
                if len(audio) < rate * 2 // 10:  # < 100 ms
                    await send("error", error={"type": "invalid_request_error", "code": "input_audio_buffer_commit_empty",
                                               "message": "Error committing input audio buffer: buffer too small. Expected at least 100ms of audio.",
                                               "param": None, "event_id": event.get("event_id")})
                    continue
                item_id, previous = "item_" + uuid.uuid4().hex[:20], prev_item
                prev_item = item_id
                data, audio = bytes(audio), bytearray()
                await send("input_audio_buffer.committed", previous_item_id=previous, item_id=item_id)
                try:
                    pcm = await decode_file(data, "-f", "s16le", "-ar", str(rate), "-ac", "1")
                    tr = await asyncio.to_thread(transcribe_full, pcm, language, None)
                except (ApiError, transcribe_cpp.TranscribeError) as exc:
                    await send("conversation.item.input_audio_transcription.failed", item_id=item_id, content_index=0,
                               error={"type": "server_error", "code": "transcription_failed", "message": str(getattr(exc, "message", exc)), "param": None})
                    continue
                print(f"rt    {tr.duration:6.2f}s  {tr.text!r}", flush=True)
                if tr.text:
                    await send("conversation.item.input_audio_transcription.delta", item_id=item_id, content_index=0, delta=tr.text)
                await send("conversation.item.input_audio_transcription.completed", item_id=item_id, content_index=0, transcript=tr.text)
    except WebSocketDisconnect:
        return


# ----------------------------------------------------------------------------------------------------------
# own streaming protocol: WS /v1/stt
# ----------------------------------------------------------------------------------------------------------

@app.websocket("/v1/stt")
async def stt(ws: WebSocket) -> None:
    server_mic = ws.query_params.get("mic") == "server"
    language = ws.query_params.get("language") or None
    peer = ws.client.host if ws.client else ""
    if server_mic and peer not in ARGS.mic_allow_set and peer not in ("127.0.0.1", "::1"):
        print(f"refused server-mic request from {peer} (not in --mic-allow)", flush=True)
        await ws.accept()
        await ws.send_json({"type": "error", "message": "server mic not allowed for this client"})
        await ws.close(1008)
        return
    await ws.accept()
    await ws.send_json({"type": "transcript.created"})

    pause = ARGS.commit_pause_sec  # > 0: freeze the utterance after this much quiet, decode only what follows
    committed: list[str] = []  # frozen utterances, never decoded again
    chunks: list[np.ndarray] = []  # audio since the last commit (the "tail")
    tail_len = 0  # samples in the tail
    received = 0  # samples in total
    decoded_at = 0  # tail length at the last partial decode
    step = ARGS.chunk_size_sec
    lang_seen: str | None = None
    odd = b""  # carry a stray byte if a frame splits an int16 sample

    def tail() -> np.ndarray:
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def dur() -> float:
        return round(received / SAMPLE_RATE, 2)

    def full(tail_text: str) -> str:
        return join_texts([*committed, *([tail_text] if tail_text else [])])

    async def send_partial(tail_text: str) -> None:
        await ws.send_json({"type": "transcript.partial", "text": full(tail_text), "is_final": False, "speech_final": False, "duration": dur()})

    async def commit() -> None:
        """Decode the finished utterance one last time, freeze its text, start a fresh tail."""
        nonlocal chunks, tail_len, decoded_at, step, lang_seen
        pcm = tail()
        chunks, tail_len, decoded_at, step = [], 0, 0, ARGS.chunk_size_sec
        if len(pcm) < MIN_SAMPLES:
            return
        t0 = time.time()
        try:
            result = await asyncio.to_thread(transcribe, pcm, language)
        except transcribe_cpp.TranscribeError:
            return
        lang_seen = lang_seen or result.language
        t = result.text.strip()
        print(f"commit  {len(pcm) / SAMPLE_RATE:6.2f}s decode {1000 * (time.time() - t0):5.0f} ms  {t!r}", flush=True)
        if t:
            committed.append(t)
            await send_partial("")

    async def ingest(data: bytes) -> None:
        nonlocal tail_len, received, decoded_at, step, odd, chunks
        buf = odd + data
        n = len(buf) // 2 * 2
        odd = buf[n:]
        if not n:
            return
        pcm = np.frombuffer(buf[:n], dtype=np.int16).astype(np.float32) / 32768.0
        chunks.append(pcm)
        tail_len += len(pcm)
        received += len(pcm)
        if pause > 0:
            spoke, quiet = speech_state(tail(), pause)
            if not spoke:  # nothing to decode; do not let silence pile up in the tail
                if tail_len > 2 * SAMPLE_RATE:
                    chunks, tail_len, decoded_at = [], 0, 0
                return
            if quiet or tail_len >= ARGS.segment_sec * SAMPLE_RATE:  # pause, or no pause for too long
                await commit()
                return
        if tail_len - decoded_at < step * SAMPLE_RATE:
            return
        decoded_at = tail_len
        t0 = time.time()
        try:
            result = await asyncio.to_thread(transcribe, tail(), language)
        except transcribe_cpp.TranscribeError:
            return  # e.g. input still too short; try again on the next step
        cost = time.time() - t0
        # decode cost grows with the tail; stretch the step so decoding stays faster than real time
        step = max(ARGS.chunk_size_sec, min(2.0 * cost, ARGS.max_step_sec))
        t = result.text.strip()
        print(f"partial {dur():6.2f}s decode {1000 * cost:5.0f} ms  {full(t)[-80:]!r}", flush=True)
        await send_partial(t)

    async def mic_pump() -> None:
        proc = await asyncio.create_subprocess_exec(*SERVER_MIC_CMD, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
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
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes") is not None:
                if not server_mic:
                    await ingest(msg["bytes"])
                continue
            try:
                event = json.loads(msg.get("text") or "")
            except json.JSONDecodeError:
                continue
            if event.get("type") != "audio.done":
                continue
            if pump:
                pump.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pump
            final_tail = ""
            pcm = tail()
            if len(pcm) and (pause <= 0 or (len(pcm) >= MIN_SAMPLES and speech_state(pcm, 0)[0])):
                t0 = time.time()
                try:
                    result = await asyncio.to_thread(transcribe, pcm, language)
                    final_tail, lang_seen = result.text.strip(), lang_seen or result.language
                except transcribe_cpp.TranscribeError as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})
                    await ws.close()
                    return
                print(f"final   {len(pcm) / SAMPLE_RATE:6.2f}s decode {1000 * (time.time() - t0):5.0f} ms  {full(final_tail)[-80:]!r}", flush=True)
            await ws.send_json({"type": "transcript.done", "text": full(final_tail), "language": lang_seen, "duration": dur()})
            await ws.close()
            return
    except WebSocketDisconnect:
        return
    finally:
        if pump and not pump.done():
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump


HF_URL = "https://huggingface.co/handy-computer/Qwen3-ASR-{size}-gguf/resolve/main/Qwen3-ASR-{size}-{quant}.gguf"
SIZES = ("0.6B", "1.7B")
QUANTS = ("BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")
DEFAULT_DOWNLOADS = ("0.6B-Q4_K_M", "1.7B-Q8_0", "1.7B-F16")


def download_models(specs: list[str], models_dir: Path) -> None:
    """Fetch GGUFs from huggingface.co/handy-computer into models/. Resumes partial files, skips complete ones."""
    wanted: list[str] = []
    for spec in specs or DEFAULT_DOWNLOADS:
        names = [f"{s}-{q}" for s in SIZES for q in QUANTS] if spec == "all" else \
                [f"{spec}-{q}" for q in QUANTS] if spec in SIZES else [spec]
        for name in names:
            size, _, quant = name.partition("-")
            if size not in SIZES or quant not in QUANTS:
                sys.exit(f"unknown model {name!r}: use all, {', '.join(SIZES)}, or <size>-<quant> with quant in {', '.join(QUANTS)}")
            if name not in wanted:
                wanted.append(name)
    models_dir.mkdir(exist_ok=True)
    for name in wanted:
        size, _, quant = name.partition("-")
        dest = models_dir / f"Qwen3-ASR-{name}.gguf"
        url = HF_URL.format(size=size, quant=quant)
        part = dest.with_name(dest.name + ".part")
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req) as r:
                resumed = r.status == 206
                total = int(r.headers.get("Content-Length", 0)) + (have if resumed else 0)
                if dest.exists() and dest.stat().st_size == total:
                    print(f"have  {dest.name}", flush=True)
                    continue
                done = have if resumed else 0
                t0 = time.time()
                with open(part, "ab" if resumed else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        print(f"\rget   {dest.name}  {done / 1e9:5.2f}/{total / 1e9:.2f} GB  {done / max(total, 1):4.0%}", end="", flush=True)
            part.replace(dest)
            print(f"\rdone  {dest.name}  {total / 1e9:.2f} GB in {time.time() - t0:.0f} s{' ' * 20}", flush=True)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and part.exists():  # .part is already complete
                part.replace(dest)
                print(f"done  {dest.name}", flush=True)
            else:
                sys.exit(f"{name}: {exc} ({url})")
        except urllib.error.URLError as exc:
            sys.exit(f"{name}: {exc.reason} ({url})")


def load_model() -> None:
    global MODEL, SESSION
    MODEL = transcribe_cpp.Model(str(ARGS.model), backend=ARGS.backend)
    SESSION = MODEL.session(n_threads=ARGS.threads)
    # the first decode pays kernel warm-up; do it now instead of on the first request
    with contextlib.suppress(transcribe_cpp.TranscribeError):
        SESSION.run(np.zeros(SAMPLE_RATE, dtype=np.float32))
    print(f"model loaded + warm: {ARGS.model} on {MODEL.backend}", flush=True)


def main() -> None:
    global ARGS, MODEL_ID
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", default="0.6B-Q4_K_M",
                   help="GGUF path, or a short name looked up in models/ (e.g. 1.7B-Q8_0, 1.7B-F16; see --list-models)")
    p.add_argument("--list-models", action="store_true", help="list GGUF files in models/ and exit")
    p.add_argument("--download", nargs="*", metavar="MODEL", default=None,
                   help="download GGUFs into models/ and exit. MODEL: <size>-<quant> (0.6B-Q4_K_M), a size (0.6B, 1.7B = every quant) or all.\n"
                        f"Without MODEL: {', '.join(DEFAULT_DOWNLOADS)}. Quants: {', '.join(QUANTS)}")
    p.add_argument("--model-id", default="", help="name reported by /v1/models (default: GGUF file stem)")
    p.add_argument("--backend", default="auto", help="auto, cpu, cuda, vulkan, ...")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = library default)")
    p.add_argument("--host", default="0.0.0.0", help="bind address; 127.0.0.1 = this machine only")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--mic-allow", default="", help="comma-separated client IPs allowed to use mic=server; loopback is always allowed")
    p.add_argument("--chunk-size-sec", type=float, default=0.5, help="minimum audio between partial decodes")
    p.add_argument("--max-step-sec", type=float, default=3.0, help="upper bound for the adaptive partial step")
    p.add_argument("--commit-pause-sec", type=float, default=0.7,
                   help="/v1/stt: after this much quiet the utterance is frozen and later steps decode only what follows,\n"
                        "so long dictations stay fast. Lower = earlier commits, higher = fewer splits mid-sentence. 0 = off")
    p.add_argument("--max-upload-mb", type=int, default=100, help="largest accepted upload / request body (OpenAI's own limit is 25)")
    p.add_argument("--segment-sec", type=float, default=20.0,
                   help="longest chunk decoded at once: caps segments / subtitle blocks, and the audio /v1/stt\n"
                        "decodes between pause commits (a commit is forced when it is reached)")
    ARGS = p.parse_args()
    models_dir = HERE / "models"
    if ARGS.download is not None:
        download_models(ARGS.download, models_dir)
        return
    if ARGS.list_models:
        for f in sorted(models_dir.glob("*.gguf")):
            print(f"{f.stem.removeprefix('Qwen3-ASR-'):<14} {f.stat().st_size / 1e9:5.2f} GB  {f}")
        return
    if not Path(ARGS.model).exists():
        for cand in (models_dir / ARGS.model, models_dir / f"{ARGS.model}.gguf", models_dir / f"Qwen3-ASR-{ARGS.model}.gguf"):
            if cand.is_file():
                ARGS.model = str(cand)
                break
        else:
            p.error(f"model not found: {ARGS.model} (try --list-models)")
    ARGS.mic_allow_set = {ip.strip() for ip in ARGS.mic_allow.split(",") if ip.strip()}
    MODEL_ID = ARGS.model_id or Path(ARGS.model).stem
    load_model()
    print(f"listening http://{ARGS.host}:{ARGS.port}  (OpenAI /v1/audio/transcriptions, ElevenLabs /v1/speech-to-text, "
          f"Deepgram /v1/listen http+ws, own ws /v1/stt)", flush=True)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning", ws_ping_interval=None, ws_max_size=2**31)


if __name__ == "__main__":
    main()
