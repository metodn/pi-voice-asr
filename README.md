# pi-voice-asr

Hold **Space** in [pi](https://pi.dev) and talk. The words appear live in the prompt box. Let go and they stay there, ready to edit or send.

Speech recognition runs on your own machine with [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (Qwen3-ASR or Parakeet models) behind a small WebSocket server. Nothing leaves your network. It also works in pi on remote machines over SSH.

## Quick start

**1. Build transcribe.cpp** (on the machine with the GPU; needs `git`, `cmake`, a compiler, [`uv`](https://docs.astral.sh/uv/), `ffmpeg`, and for NVIDIA the CUDA toolkit):

```bash
git clone https://github.com/metodn/pi-voice-asr && cd pi-voice-asr
git clone https://github.com/handy-computer/transcribe.cpp server/transcribe.cpp
cd server/transcribe.cpp
cmake -B build-shared -DCMAKE_BUILD_TYPE=Release -DTRANSCRIBE_BUILD_SHARED=ON \
      -DTRANSCRIBE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-shared -j"$(nproc)"
cd ../..
```

No NVIDIA GPU? Drop the two CUDA flags to run on CPU, or see the transcribe.cpp README for Vulkan and HIP. `native` needs CMake 3.24+; on older versions use your GPU's compute capability without the dot (for example `75`).

**2. Download a model** (pick one) and start the server:

```bash
# Qwen3-ASR 1.7B Q8_0 (2.2 GB)
server/transcribe_server.py --download 1.7B-Q8_0
server/transcribe_server.py --model 1.7B-Q8_0

# or Parakeet TDT 0.6B v3 Q8_0 (740 MB)
curl -L --create-dirs -o server/models/parakeet-tdt-0.6b-v3-Q8_0.gguf \
  https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/main/parakeet-tdt-0.6b-v3-Q8_0.gguf
server/transcribe_server.py --model parakeet-tdt-0.6b-v3-Q8_0
```

Wait for `listening http://0.0.0.0:8765 ...`. More models, flags and fixes: [README-qwen3-asr.md](https://github.com/metodn/pi-voice-asr/blob/main/README-qwen3-asr.md).

**3. Install the extension into pi** (on the machine where you run pi):

```bash
pi install git:github.com/metodn/pi-voice-asr   # from GitHub
pi install npm:@metodn/pi-voice-asr             # or from npm
pi install ./pi-voice-asr                       # or from a local clone (path to this repo)
```

Then **restart pi**. `pi install` registers the package in pi's settings, so it loads in every session. Useful variants:

```bash
pi install -l <source>             # this project only
pi -e ./extensions/live-stt.ts     # try it for one session, no install
pi list                            # show installed packages
pi update --extensions             # update
pi remove <source>                 # uninstall
```

Do not also copy `live-stt.ts` into `~/.pi/agent/extensions/`, or it may load twice.

Inside pi, point the extension at your server if it is not on the same machine, then check:

```
/live-stt url 192.168.1.20
/live-stt doctor
```

`doctor` should answer `live-stt ok: recorder parec, STT server reachable`. Then, on an empty prompt, hold Space and talk.

The pi machine needs a microphone and one of `parec`, `pw-record`, `arecord`, `ffmpeg` (Linux), or use `micSource: "server"` below. Tested on Linux with pi 1.3.0 and 1.2.0.

## Controls

| Action | Result |
| --- | --- |
| Hold Space (empty prompt) | Record while held. Shortly after you release, the text stays in the prompt. |
| Tap Space | Latch recording on. Tap again to stop. |
| Enter while recording | Stop and send. |
| Esc while recording | Cancel and restore the prompt. |
| Space again | While the prompt holds untouched dictation, record more and append. |

Terminals do not report key release, so release is detected when the keyboard's auto-repeat stops. Without auto-repeat, hold behaves like tap-to-toggle.

Commands: `/live-stt status`, `/live-stt doctor`, `/live-stt url <host[:port]>` (change server, saved).

## Configuration

Optional `~/.pi/agent/live-stt.json`, every key has a default (see `examples/`):

| Key | Meaning |
| --- | --- |
| `url` | STT server address. Easier to set with `/live-stt url`. |
| `micSource` | `local` (default): this machine records and streams audio. `server`: the STT server records its own microphone. |
| `trigger` | Key that starts dictation (default `space`). |
| `holdDetectMs` | `800`. Window in which a key repeat means "held". Must exceed your OS key-repeat delay. |
| `releaseMs` | `250`. Held mode ends when no repeat arrives for this long. Raise it on laggy SSH links. |
| `recorder` | Custom command printing raw 16 kHz mono s16le PCM. Default: first of parec, pw-record, arecord, ffmpeg. |

Env overrides: `PI_LIVE_STT_URL`, `PI_LIVE_STT_MIC`, `PI_LIVE_STT_TRIGGER`, `PI_LIVE_STT_REC`.

## Remote machines (SSH)

Pi runs on the remote machine, you sit at the one with the microphone and the GPU.

- **Server records the mic** (simplest, mic and server on the same machine): the remote needs only the extension and `"micSource": "server"`.

  ```bash
  STT_URL=ws://stt-host.example:8765/v1/stt MIC=server scripts/make-remote-bundle.sh
  unzip -o pi-voice-asr-remote.zip -d ~/.pi/agent      # on the remote
  ```

  Reach the server either over the LAN (`--host 0.0.0.0 --mic-allow <remote-ip>`) or an SSH tunnel (`RemoteForward 8765 127.0.0.1:8765`, see `examples/ssh_config.example`).
- **Remote records your mic** (`micSource: "local"`): forward PulseAudio with the `RemoteForward 4713` lines from `examples/ssh_config.example`, install `parec` on the remote (`sudo apt install pulseaudio-utils`), and add the `PULSE_SERVER` line from that file to the remote's `~/.bashrc`. Then `MIC=local scripts/make-remote-bundle.sh`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Space types a space | It only starts dictation on an empty prompt. Clear it first (Ctrl+U). |
| `/live-stt doctor` cannot connect | Server not running or unreachable. Check the server terminal, the URL (`/live-stt url`), your firewall or SSH forward. |
| `no audio from parec in 4s` | Check the default microphone: `parec --rate=16000 --channels=1 --format=s16le --raw \| head -c 32000 \| wc -c` should print `32000`. Pick a device with `pactl list short sources` and `pactl set-default-source <name>`. |
| Recording works, text stays empty | The microphone is silent or very quiet. Check its level in `pavucontrol` or `wpctl status`. |
| Recording stops while you hold Space | Raise `releaseMs` (for example 400). |
| Hold is treated as a tap | Raise `holdDetectMs` above your OS key-repeat delay (`xset q` on X11). |
| `server mic not allowed for this client` | Add the client IP to the server's `--mic-allow`. |

Server-side problems: [README-qwen3-asr.md](https://github.com/metodn/pi-voice-asr/blob/main/README-qwen3-asr.md#troubleshooting).

## Security

- The STT server has **no authentication** and binds `0.0.0.0` by default, so it is reachable on your LAN. Do not expose it to the internet. Use `--host 127.0.0.1` to keep it on one machine.
- The server's microphone can only be opened by IPs in `--mic-allow` (and loopback). Through an SSH `RemoteForward`, clients arrive from loopback and are always trusted.
- The PulseAudio forward uses `auth-anonymous=1` on loopback: any local process on either end can read the forwarded mic.
- Traffic is plain `ws://`. Keep it on a trusted network or inside SSH.

## Limitations

- Hold-to-talk depends on key auto-repeat. Tune `holdDetectMs` and `releaseMs` if it misfires.
- The models are offline: the server re-decodes the current utterance at each step, so the live text can rewrite its last words. After a pause (`--commit-pause-sec`) the utterance is frozen. One GPU session, so concurrent clients queue.
- Tested with English only and CUDA only.
- The extension replaces pi's prompt editor component, so it can conflict with other extensions that do the same.

## Credits

[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) by handy-computer (MIT), [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) by the Qwen team (Apache-2.0), [Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) by NVIDIA, [pi](https://pi.dev) for the agent and extension API, and [pi-voice-stt](https://pi.dev/packages/pi-voice-stt) by cgarrot as the reference for the editor hook. Not affiliated with pi or the Qwen team.

## License

MIT, see [LICENSE](https://github.com/metodn/pi-voice-asr/blob/main/LICENSE).
