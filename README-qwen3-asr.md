# Set up the Qwen3-ASR streaming service

Instructions for an agent (or a person) to stand up the speech-to-text service that `pi-voice-asr` talks to. The model and the streaming toolkit come from [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) by the Qwen team (Apache-2.0), with weights on [Hugging Face](https://huggingface.co/Qwen/Qwen3-ASR-0.6B). Minimal on purpose: one file, `server/qwen3_asr_server.py`, plus a GPU.

## Goal

A WebSocket server on port 8765 that accepts 16 kHz mono PCM and streams back text while you talk, using the open-source `Qwen/Qwen3-ASR-0.6B` model through vLLM.

## Requirements (check first)

1. Linux with an NVIDIA GPU. `nvidia-smi` must list it, and the driver must support CUDA 12.8 (R570 or newer), because the installed PyTorch wheels are CUDA 12.8 builds. The server needs about 8 GB of free VRAM (8.3 GB observed with the defaults). The default `--gpu-memory-utilization 0.4` reserves 40 percent of the card, which is enough on a 24 GB card. On a smaller card raise the fraction until fraction times VRAM is at least about 8 GB. With less, vLLM fails with a KV cache error (seen at a 6 GB budget).
2. [`uv`](https://docs.astral.sh/uv/) installed (`uv --version`).
3. About 8 GB of free disk in total (measured: about 6 GB of Python packages plus the 1.9 GB model). uv's cache may keep extra copies.
4. Internet access for the first run (packages and model download from Hugging Face).
5. Only for server-side microphone capture (`micSource: "server"`): `parec` on this machine (Debian/Ubuntu: `sudo apt install pulseaudio-utils`) and a working default microphone.

## Steps

1. Get the repo and enter it:

   ```bash
   git clone https://github.com/metodn/pi-voice-asr && cd pi-voice-asr
   ```

2. Start the server. `uv` builds the environment from the dependency header in the script (pinned `qwen-asr[vllm]==0.0.6`), and vLLM fetches the model on first run:

   ```bash
   uv run --script server/qwen3_asr_server.py
   ```

   The first start downloads several GB and can take many minutes. Later starts take about 25 seconds.

3. Wait for these two lines in the output:

   ```
   model loaded + warm
   listening ws://127.0.0.1:8765/v1/stt
   ```

   The server runs one dummy decode at startup so the first real request is fast.

### Optional: download the model yourself first

Needed for offline machines or to control where weights live:

```bash
uv run --with "huggingface_hub[cli]" hf download Qwen/Qwen3-ASR-0.6B --local-dir models/Qwen3-ASR-0.6B
uv run --script server/qwen3_asr_server.py --model models/Qwen3-ASR-0.6B
```

`Qwen/Qwen3-ASR-1.7B` is more accurate and uses more memory. Pass it the same way. Only the 0.6B model was tested here.

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `Qwen/Qwen3-ASR-0.6B` | Hugging Face id or local directory. |
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` to serve the LAN (then open the port in your firewall, for example `sudo ufw allow from <client-ip> to any port 8765`). |
| `--port` | `8765` | Port. |
| `--mic-allow` | empty | Comma-separated client IPs allowed to use `?mic=server` (the server records its own mic). Loopback is always allowed. |
| `--gpu-memory-utilization` | `0.4` | Fraction of GPU memory vLLM reserves. |
| `--max-model-len` | `16384` | Context tokens, which caps dictation length (about 20 minutes, rough estimate). |
| `--chunk-size-sec` | `0.5` | Update step. Smaller means livelier text and more GPU work. |

Example for a LAN with one remote client that may use the server's mic:

```bash
uv run --script server/qwen3_asr_server.py --host 0.0.0.0 --mic-allow 192.0.2.10
```

## Verify

1. The server answers (expected output: `{'type': 'transcript.created'}`):

   ```bash
   uv run --with websockets python - <<'EOF'
   import asyncio, json, websockets
   async def main():
       async with websockets.connect("ws://127.0.0.1:8765/v1/stt?sample_rate=16000&encoding=pcm") as ws:
           print(json.loads(await ws.recv()))
   asyncio.run(main())
   EOF
   ```

2. It transcribes. Convert any speech file to 16 kHz mono 16-bit WAV (`ffmpeg -i in.mp3 -ar 16000 -ac 1 -sample_fmt s16 speech.wav`), then stream it at real-time speed:

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
| server to client | `{"type": "transcript.partial", "text": <whole hypothesis so far>, "duration": <s>}` while audio arrives |
| server to client | `{"type": "transcript.done", "text": <final>, "language": ..., "duration": <s>}` after `audio.done` |
| server to client | `{"type": "error", "message": ...}` then the connection closes |

`partial.text` is the full transcript so far, not a delta, and the last few words can change between updates. Query option `?mic=server` makes the server record its own microphone instead of receiving audio. It is refused for clients not in `--mic-allow`.

## Stop and clean up

Send SIGTERM to the server process (or press Ctrl+C in its terminal). It shuts down vLLM's engine process and frees the GPU. Check with `nvidia-smi`. If a stray `VLLM::EngineCore` process remains, kill it by PID. Avoid `pkill -f` patterns that can also match your own shell.

## Keep it running (untested sketch)

The server exits with status 1 if the vLLM engine dies, so a supervisor can restart it. A user systemd unit, saved as `~/.config/systemd/user/pi-voice-stt.service`, then `systemctl --user daemon-reload && systemctl --user enable --now pi-voice-stt`:

```ini
[Unit]
Description=Qwen3-ASR streaming STT

[Service]
WorkingDirectory=%h/pi-voice-asr
ExecStart=%h/.local/bin/uv run --script server/qwen3_asr_server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

This unit has not been run by the author. Verify it before relying on it.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `ValueError: To serve at least one request with the models's max seq len (65536) ... KV cache` | vLLM sized memory for the default context. This server already sets `--max-model-len 16384`. If it still fails, raise `--gpu-memory-utilization` or lower `--max-model-len`. |
| `Error retrieving safetensors: ... is not a safetensors repo` in the log | Harmless warning when `--model` is a local directory. |
| First request takes about 3 s | Kernel warm-up. The startup dummy decode already pays this. |
| Text lags more the longer you talk | Each step re-decodes all audio so far, so cost grows with length. The server stretches its step to stay real-time. Start a new dictation to reset. |
| Server printed `engine process died, shutting down` and exited | The vLLM engine process was killed or crashed. During testing something outside this project killed it about every two hours on one machine and the cause was not found. Restart it, or use a supervisor. |
| `server mic not allowed for this client` | Add the client's IP to `--mic-allow`. Note that tunneled clients arrive as `127.0.0.1` and are always allowed. |
| Server mic produces empty text | Check the server machine's default microphone and volume (`pactl get-default-source`, `pavucontrol`). Sanity test on the server: `parec --rate=16000 --channels=1 --format=s16le --raw \| head -c 32000 \| wc -c` should print `32000`. |
