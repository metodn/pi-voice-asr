# Set up the STT server

Instructions for an agent (or a person) to stand up the speech-to-text service that `pi-voice-asr` talks to. It runs [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (Qwen team, Apache-2.0) as a GGUF model on [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (ggml, MIT). One file, `server/transcribe_server.py`, plus a locally built `libtranscribe.so`.

## Goal

One port (8765) serving a WebSocket endpoint `/v1/stt` for the pi extension (16 kHz mono PCM in, live text out) and an OpenAI-compatible `POST /v1/audio/transcriptions`.

## Requirements (check first)

1. Linux and a C/C++ toolchain: `cmake`, a compiler, `git`. Optional but recommended: `libopenblas-dev` (speeds up the host-side decoder).
2. For GPU speed: an NVIDIA GPU and the CUDA toolkit (`nvcc` on `PATH`). Reference setup: GTX 1660 SUPER (6 GB), where the 0.6B model decoded about 14 times faster than real time. Only CUDA was tested here. transcribe.cpp also has Vulkan, HIP and CPU backends, see its README, but the commands below are for CUDA.
3. [`uv`](https://docs.astral.sh/uv/) (`uv --version`) and `ffmpeg` (used by the HTTP endpoint to decode uploaded files).
4. Disk for weights: 0.6B Q4_K_M is 590 MB, 1.7B Q8_0 is 2.2 GB, 1.7B F16 is 4.1 GB.
5. Only for server-side microphone capture (`micSource: "server"`): `parec` (Debian/Ubuntu: `sudo apt install pulseaudio-utils`) and a working default microphone.

## Steps

1. Get the repo and enter it:

   ```bash
   git clone https://github.com/metodn/pi-voice-asr && cd pi-voice-asr
   ```

2. Build transcribe.cpp **inside `server/`**. The server looks for `server/transcribe.cpp/build-shared/src/libtranscribe.so` and the Python bindings in `server/transcribe.cpp/bindings/python/src`:

   ```bash
   git clone https://github.com/handy-computer/transcribe.cpp server/transcribe.cpp
   cd server/transcribe.cpp
   cmake -B build-shared -DCMAKE_BUILD_TYPE=Release -DTRANSCRIBE_CUDA=ON -DTRANSCRIBE_BUILD_SHARED=ON \
         -DCMAKE_CUDA_ARCHITECTURES=75
   cmake --build build-shared -j"$(nproc)"
   cd ../..
   ```

   Set `CMAKE_CUDA_ARCHITECTURES` to your GPU's compute capability without the dot: 75 for GTX 16xx and RTX 20xx, 86 for RTX 30xx, 89 for RTX 40xx. `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` prints it. Reference build: transcribe.cpp 0.4.1 (commit `4d7c1cce`), which produces `libtranscribe.so.0.4.1`.

3. Download a model. `--download` fetches GGUFs from Hugging Face (`handy-computer/Qwen3-ASR-<size>-gguf`) into `server/models/` and exits. It resumes interrupted downloads and skips files that are complete:

   ```bash
   server/transcribe_server.py --download 0.6B-Q4_K_M    # one model, 590 MB
   server/transcribe_server.py --download                # no argument: 0.6B-Q4_K_M, 1.7B-Q8_0, 1.7B-F16
   server/transcribe_server.py --download 1.7B all       # a whole size (every quant), or "all" for everything
   ```

   Names are `<size>-<quant>` with size `0.6B` or `1.7B` and quant `BF16`, `F16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`. You can also place a `Qwen3-ASR-<size>-<quant>.gguf` yourself (links in the [model doc](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/qwen3-asr-0.6b.md)).

4. Start the server. `uv` builds its environment from the script header:

   ```bash
   server/transcribe_server.py                       # default model 0.6B-Q4_K_M
   server/transcribe_server.py --model 1.7B-Q8_0     # more accurate, about 2.5 times the cost
   ```

5. Wait for these two lines:

   ```
   model loaded + warm: .../Qwen3-ASR-0.6B-Q4_K_M.gguf on cuda
   listening http://0.0.0.0:8765  (OpenAI /v1/audio/transcriptions, ElevenLabs /v1/speech-to-text, Deepgram /v1/listen http+ws, own ws /v1/stt)
   ```

   The server runs one dummy decode at startup so the first real request is fast.

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `0.6B-Q4_K_M` | GGUF path, or a short name `<size>-<quant>` looked up in `server/models/` (for example `1.7B-Q8_0`, `1.7B-F16`). |
| `--download [MODEL ...]` | off | Download GGUFs into `server/models/` and exit (see step 3). |
| `--list-models` | off | List the GGUF files in `server/models/` and exit. |
| `--model-id` | file stem | Name reported by `/v1/models`. |
| `--backend` | `auto` | `auto`, `cpu`, `cuda`, `vulkan`, ... Use `cuda` to fail loudly instead of falling back. |
| `--threads` | `0` | CPU threads (0 means the library default). |
| `--host` | `0.0.0.0` | Bind address. **The default serves the whole network.** Use `127.0.0.1` for this machine only, and open the port in your firewall otherwise (for example `sudo ufw allow from <client-ip> to any port 8765`). |
| `--port` | `8765` | Port. |
| `--mic-allow` | empty | Comma-separated client IPs allowed to use `?mic=server` (the server records its own mic). Loopback is always allowed. |
| `--chunk-size-sec` | `0.5` | Minimum audio between partial decodes. Smaller means livelier text and more GPU work. |
| `--max-step-sec` | `3.0` | Upper bound for the adaptive partial step. |
| `--commit-pause-sec` | `0.7` | `/v1/stt` pause commits, see below. After this much quiet the utterance is frozen. `0` turns it off. |
| `--segment-sec` | `20` | Longest audio decoded in one piece. Caps segments and subtitle blocks on the file endpoints, and forces a pause commit in `/v1/stt` when speech runs this long without a pause. |
| `--max-upload-mb` | `100` | Largest accepted upload or request body. |

Example for a LAN with one remote client that may use the server's mic:

```bash
server/transcribe_server.py --host 0.0.0.0 --mic-allow 192.0.2.10
```

## Verify

1. Health and model (expected: `{"status":"ok","model":"Qwen3-ASR-0.6B-Q4_K_M","backend":"..."}`):

   ```bash
   curl -s http://127.0.0.1:8765/health
   ```

2. HTTP transcription of any audio file (ffmpeg converts it on the server):

   ```bash
   curl -s http://127.0.0.1:8765/v1/audio/transcriptions -F file=@speech.wav -F response_format=text
   ```

3. The WebSocket answers (expected output: `{'type': 'transcript.created'}`):

   ```bash
   uv run --with websockets python - <<'EOF'
   import asyncio, json, websockets
   async def main():
       async with websockets.connect("ws://127.0.0.1:8765/v1/stt?sample_rate=16000&encoding=pcm") as ws:
           print(json.loads(await ws.recv()))
   asyncio.run(main())
   EOF
   ```

4. Live streaming. Convert any speech file to 16 kHz mono 16-bit WAV (`ffmpeg -i in.mp3 -ar 16000 -ac 1 -sample_fmt s16 speech.wav`), then stream it at real-time speed:

   ```bash
   uv run --with websockets python - speech.wav <<'EOF'
   import asyncio, json, sys, wave, websockets
   async def main():
       pcm = wave.open(sys.argv[1]).readframes(10**9)
       async with websockets.connect("ws://127.0.0.1:8765/v1/stt?sample_rate=16000&encoding=pcm", ping_interval=None) as ws:
           assert json.loads(await ws.recv())["type"] == "transcript.created"
           async def rx():
               async for raw in ws:
                   ev = json.loads(raw)
                   print(ev["type"], ev.get("text", "")[-80:])
                   if ev["type"] == "transcript.done":
                       return
           task = asyncio.create_task(rx())
           for i in range(0, len(pcm), 3200):          # 100 ms per frame
               await ws.send(pcm[i:i + 3200])
               await asyncio.sleep(0.1)
           await ws.send(json.dumps({"type": "audio.done"}))
           await task
   asyncio.run(main())
   EOF
   ```

   Expected: a series of `transcript.partial` lines whose text grows, ending in `transcript.done` with the full text.

## Protocol

One dictation per connection.

| Direction | Message |
| --- | --- |
| client to server | binary frames: raw PCM s16le, mono, 16 kHz, any frame size |
| client to server | text `{"type": "audio.done"}` when finished |
| server to client | `{"type": "transcript.created"}` right after connecting |
| server to client | `{"type": "transcript.partial", "text": <whole hypothesis so far>, "is_final": false, "speech_final": false, "duration": <s>}` while audio arrives |
| server to client | `{"type": "transcript.done", "text": <final>, "language": ..., "duration": <s>}` after `audio.done` |
| server to client | `{"type": "error", "message": ...}` then the connection closes |

`partial.text` is the full transcript so far, not a delta, and the last few words can change between updates.

### Pause commits (long dictations)

Qwen3-ASR is an offline model, so the server decodes the audio again at every step. Decoding everything since the start would make each step slower the longer you talk. With pause commits (on by default) the server instead:

1. keeps the audio since the last pause (the "tail") and decodes only that at each step,
2. after `--commit-pause-sec` of quiet, decodes the tail one last time, freezes its text and starts a new tail,
3. sends `frozen text + current tail` as `partial.text`, so the protocol and the extension are unchanged.

Cost per step then depends on the length of the current utterance, not the whole dictation, and frozen text never changes again. The price: the model no longer sees earlier speech when it decodes a new utterance, which can affect casing, punctuation or names at a boundary, and a long thinking pause splits the text there. Raise `--commit-pause-sec` (for example `1.2`) for fewer splits, or use `0` for the old behaviour. In a noisy room the quiet may never be detected, and `--segment-sec` then forces a commit. Tested with a synthetic 3 minute stream and a fake model: the longest decode went from 187 s of audio to 5 s, and total decoded audio dropped about 29 times. Not yet measured with the real model, so accuracy at the boundaries is unverified.

### Other endpoints

The same server also answers OpenAI (`POST /v1/audio/transcriptions` with `json`, `text`, `srt`, `vtt`, `verbose_json` and word or segment timestamps, plus a minimal `ws /v1/realtime`), ElevenLabs (`POST /v1/speech-to-text`) and Deepgram (`POST` and `ws /v1/listen`) clients, with errors in each API's own shape. Qwen3-ASR gives no timestamps, so segment times are the bounds of the chunks cut at pauses and word times are interpolated. See the docstring at the top of `server/transcribe_server.py`. Query options: `?mic=server` makes the server record its own microphone instead of receiving audio (refused for clients not in `--mic-allow`), and `?language=<code>` fixes the language instead of auto-detecting.

## Stop and clean up

Press Ctrl+C in the server terminal, or send SIGTERM to the process. Check that the GPU is free with `nvidia-smi`.

## Use with llama-swap (untested)

```yaml
cmd: /path/to/pi-voice-asr/server/transcribe_server.py --backend cuda --port ${PORT}
checkEndpoint: /health
```

## Keep it running (untested sketch)

A user systemd unit, saved as `~/.config/systemd/user/pi-voice-stt.service`, then `systemctl --user daemon-reload && systemctl --user enable --now pi-voice-stt`:

```ini
[Unit]
Description=Qwen3-ASR STT server (transcribe.cpp)

[Service]
WorkingDirectory=%h/pi-voice-asr
ExecStart=%h/.local/bin/uv run --script server/transcribe_server.py --backend cuda
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

This unit has not been run by the author. Verify it before relying on it.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `ModuleNotFoundError: transcribe_cpp` or a missing-library error at start | The checkout is not at `server/transcribe.cpp`, or `build-shared/src/libtranscribe.so` was not built. Redo step 2. To use a library elsewhere, set `TRANSCRIBE_LIBRARY=/path/to/libtranscribe.so` (it wins over the default). |
| `model not found: ...` | The file is not in `server/models/`. Run `server/transcribe_server.py --download <model>` or `--list-models`. |
| `unknown model ...` or a download error | Wrong `<size>-<quant>` name, or no network. Valid sizes are `0.6B` and `1.7B`. Quants: `BF16`, `F16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`. Run `--download` again to resume from the `.part` file. |
| Log says `on cpu`, or slow decoding | The library was built without `-DTRANSCRIBE_CUDA=ON`, or `nvcc` was missing at build time. Rebuild, then start with `--backend cuda` to get an error instead of a silent fallback. |
| ggml warns that Turing GPUs lack tensor cores | Harmless. The 0.6B model still ran about 14 times faster than real time on a GTX 1660 SUPER. |
| First request takes about 3 s | Kernel warm-up. The startup dummy decode already pays this. |
| Text lags more the longer you talk | Only with `--commit-pause-sec 0`, or when you never pause: each step re-decodes the audio since the last commit, so cost grows until a pause or `--segment-sec` (20 s) commits it. Keep the default `0.7`. |
| Several clients are slow at once | Decodes are serialized on one GPU session. Three simultaneous clients each finished in about 8 s in a test. |
| `server mic not allowed for this client` | Add the client's IP to `--mic-allow`. Tunneled clients arrive as `127.0.0.1` and are always allowed. |
| Server mic produces empty text | Check the server machine's default microphone and volume (`pactl get-default-source`, `pavucontrol`). Sanity test on the server: `parec --rate=16000 --channels=1 --format=s16le --raw \| head -c 32000 \| wc -c` should print `32000`. |
| HTTP upload returns `could not decode audio` | `ffmpeg` is missing on the server or the file is not valid audio. |
