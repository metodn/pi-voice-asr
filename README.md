# pi-voice-asr

Hold **Space** in [pi](https://pi.dev) and talk. The words appear live in the prompt box. Let go and they stay there, ready to edit or send.

Speech recognition runs on your own GPU with [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) behind a small WebSocket server. Nothing leaves your network. It also works in pi on remote machines over SSH.

```
micSource "local" (default): the sound card is on the pi machine, the server can be anywhere

  microphone --> pi + extension ---- audio ----> STT server (Qwen3-ASR, your GPU)
                      ^                                  |
                      '------------ live text -----------'

micSource "server": the sound card is on the server machine, pi can be anywhere

  microphone --> STT server (Qwen3-ASR, your GPU) ---- live text ----> pi + extension
```

## What is in the box

| Part | Path | Purpose |
| --- | --- | --- |
| pi extension | `extensions/live-stt.ts` | Hold Space to dictate, live text in the prompt. One file, no dependencies. |
| STT server | `server/qwen3_asr_server.py` | Streaming speech-to-text service. Setup guide: [README-qwen3-asr.md](https://github.com/metodn/pi-voice-asr/blob/main/README-qwen3-asr.md). |
| Remote bundle | `scripts/make-remote-bundle.sh` | Zips the extension and a config for another machine. |
| Examples | `examples/` | Config files and an SSH config snippet. |

## Requirements

| Where | You need |
| --- | --- |
| Machine running pi | [pi](https://pi.dev) (tested with 1.3.0 and 1.2.0). To dictate there: a microphone and one of `parec`, `pw-record`, `arecord`, `ffmpeg` on Linux. Or use `micSource: "server"` and no recorder at all (see Remote machines). |
| Machine running the STT server | Linux with an NVIDIA GPU (about 8 GB of free VRAM), a driver that supports CUDA 12.8 (R570 or newer), [`uv`](https://docs.astral.sh/uv/), and about 8 GB of disk. There is no CPU or Apple Silicon path in this server. |
| Network | The pi machine must reach the server over WebSocket (default port 8765). Same machine, LAN, or an SSH tunnel all work. |

The pi machine and the server machine can be the same computer, or different ones.

Tested on Linux only. On macOS or Windows you can run pi as a client against a Linux server, but you must supply a `recorder` command yourself or use `micSource: "server"`. An untested starting point for macOS: `"recorder": ["ffmpeg", "-loglevel", "error", "-f", "avfoundation", "-i", ":0", "-ar", "16000", "-ac", "1", "-f", "s16le", "-"]`.

## Quick start

1. **Start the server** (needs an NVIDIA GPU). Follow [README-qwen3-asr.md](https://github.com/metodn/pi-voice-asr/blob/main/README-qwen3-asr.md).
2. **Install the extension** on the machine where you run pi:

   ```bash
   pi install npm:@metodn/pi-voice-asr
   # or straight from GitHub
   pi install git:github.com/metodn/pi-voice-asr
   # or from a clone
   pi install /path/to/pi-voice-asr
   # to try it for one session only:
   pi -e ./extensions/live-stt.ts
   ```

3. **Check it**: run `/live-stt doctor` inside pi. Expected: `live-stt ok: recorder parec, STT server reachable`. It checks that a recorder exists and that the server answers. It does not prove the microphone works, see Troubleshooting.
4. **Dictate**: on an empty prompt, hold Space and talk.

By default the extension records on the local machine (`parec`, `pw-record`, `arecord` or `ffmpeg`, first one found) and talks to `ws://127.0.0.1:8765`.

## Controls

| Action | Result |
| --- | --- |
| Hold Space (empty prompt) | Record while held. About 0.25 s after you release, recording stops and the text stays in the prompt. |
| Tap Space | Latch recording on. Tap Space again to stop. |
| Enter while recording | Stop and send to the model. |
| Esc while recording | Cancel and restore the prompt. |
| Space again | While the prompt still holds untouched dictation, Space records more and appends. Once you type anything else, Space types a space as usual. |

Terminals do not report key release, so "release" is detected when the keyboard's auto-repeat stops. Tap versus hold is decided by whether a repeat arrives within `holdDetectMs` after the first press. If auto-repeat is disabled on your system, hold-to-talk behaves like tap-to-toggle.

## Configuration

Optional file `~/.pi/agent/live-stt.json`. Every key has a default.

| Key | Default | Meaning |
| --- | --- | --- |
| `url` | `ws://127.0.0.1:8765/v1/stt?sample_rate=16000&encoding=pcm` | STT server address. |
| `micSource` | `local` | `local`: this machine records and streams audio. `server`: the STT server records its own microphone. |
| `trigger` | `space` | Key that starts dictation (any pi key id). With `space`, it only fires on an empty prompt or untouched dictation. |
| `holdDetectMs` | `800` | Window after the first press in which a key repeat means "held". Must exceed your OS key-repeat delay. |
| `releaseMs` | `250` | Held mode ends when no repeat arrives for this long. Must exceed your key-repeat interval. Raise it on laggy SSH links. |
| `recorder` | auto | Custom recorder command that prints raw 16 kHz mono s16le PCM, e.g. `["parec", "--raw", "--rate=16000", "--channels=1", "--format=s16le"]`. |

Environment overrides: `PI_LIVE_STT_URL`, `PI_LIVE_STT_MIC`, `PI_LIVE_STT_TRIGGER`, and `PI_LIVE_STT_REC` (a shell command that prints PCM, useful for tests). See `examples/` for ready files.

Commands inside pi: `/live-stt doctor` (check setup), `/live-stt status` (show effective config).

## Remote machines (SSH)

Pi runs on the remote machine, but you sit at the one with the microphone and the GPU. There are two ways to get audio to the server.

### Mode A: the server records the mic (simplest)

Use this when the microphone and the STT server are on the same machine, which is also the one you sit at.

- Remote needs only the extension and a config. No recorder, no PulseAudio, no tunnel for audio.
- Build the bundle on any machine with this repo:

  ```bash
  STT_URL=ws://stt-host.example:8765/v1/stt MIC=server scripts/make-remote-bundle.sh
  # then, on the remote:
  unzip -o pi-voice-asr-remote.zip -d ~/.pi/agent     # no unzip? copy the two files by hand
  ```

  The zip holds `extensions/live-stt.ts` and `live-stt.json`, unpacked into `~/.pi/agent/`. Pi loads extensions from `~/.pi/agent/extensions/`.
- The remote must reach the server. Either:
  - **LAN**: start the server with `--host 0.0.0.0 --mic-allow <remote-ip>`. Only listed IPs may open the server's mic. See [Security](#security).
  - **SSH tunnel**: add `RemoteForward 8765 127.0.0.1:8765` for that host (see `examples/ssh_config.example`) and set `STT_URL=ws://127.0.0.1:8765/v1/stt`.

### Mode B: the remote records your mic (`micSource: "local"`)

Use this when the STT server is elsewhere or you want the audio to originate on the remote.

1. On your machine, forward PulseAudio over SSH. `examples/ssh_config.example` has the `RemoteForward 4713` plus a `LocalCommand` that loads `module-native-protocol-tcp` for you. It needs PulseAudio or PipeWire-pulse on your machine.
2. On the remote, install `parec` (Debian/Ubuntu: `sudo apt install pulseaudio-utils`) and point it at the tunnel:

   ```bash
   # ~/.bashrc on the remote
   [ -n "$SSH_CONNECTION" ] && ss -ltn | grep -q 127.0.0.1:4713 && export PULSE_SERVER=tcp:127.0.0.1:4713
   ```

3. Install the extension with `MIC=local scripts/make-remote-bundle.sh` and run `/live-stt doctor` in pi.

If you use `arecord` instead, it must go through the PulseAudio ALSA plugin (`libasound2-plugins` and a `~/.asoundrc` with `pcm.!default pulse`). Prefer `parec`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Space types a space | Space only starts dictation on an empty prompt, or while the prompt holds untouched dictation. Clear the prompt first (Ctrl+U in pi's editor). |
| `/live-stt doctor` says it cannot connect | The server is not running or not reachable. Check the server terminal, the `url` in `~/.pi/agent/live-stt.json`, and for a remote server your firewall or SSH forward. |
| `no audio from parec in 4s` | The recorder started but got no data. Check the default microphone. Sanity test: `parec --rate=16000 --channels=1 --format=s16le --raw \| head -c 32000 \| wc -c` should print `32000` within a second or two. Pick a device with `pactl list short sources` and `pactl set-default-source <name>`. |
| Recording starts, text stays empty | The microphone is silent or very quiet. Check its level in `pavucontrol` or `wpctl status`. Qwen returns nothing for silence. |
| Recording stops while you still hold Space | Auto-repeat gaps are longer than `releaseMs`. Raise it, for example to 400. |
| Hold is treated as a tap | Your key-repeat delay is longer than `holdDetectMs`. Raise it above your OS delay (`xset q` shows it on X11). |
| `server mic not allowed for this client` | Add the client's IP to the server's `--mic-allow`. |
| Text appears late in long dictations | Cost grows with length. Stop and start a new dictation (Space again appends). |

More server-side problems: [README-qwen3-asr.md](https://github.com/metodn/pi-voice-asr/blob/main/README-qwen3-asr.md#troubleshooting).

## Security

- The STT server has **no authentication**. By default it listens on loopback only. Do not expose it to the internet.
- `--host 0.0.0.0` makes it reachable on your LAN. Any reachable client can send audio for transcription (it uses your GPU). Opening the *server's microphone* is refused unless the client IP is in `--mic-allow`.
- Through an SSH `RemoteForward`, clients arrive from loopback and are always trusted. Anyone with a shell on that remote can open the server's mic while the tunnel is up.
- The PulseAudio forward uses `auth-anonymous=1` on loopback. Any local process on either end can read the forwarded mic. Use only on machines you trust.
- Traffic is plain `ws://`. Keep it on a trusted network or inside SSH.

## Limitations

- Hold-to-talk depends on key auto-repeat (see Controls). Tune `holdDetectMs` and `releaseMs` if it misfires.
- Qwen3-ASR re-decodes all audio so far at each step, so cost grows with dictation length. The server stretches its update step to keep up. Context is capped at 16384 tokens (a rough estimate is 20 minutes of audio). After the step-stretching fix it was tested with a 188 s real-time recording and a 90 s stream at 6x speed. An 8 minute dictation failed before that fix and has not been retried.
- Tested with English only. Qwen3-ASR advertises 52 languages.
- The extension replaces pi's prompt editor component, so it can conflict with other extensions that do the same.
- Tested with pi 1.3.0 (Node) and pi 1.2.0 (compiled binary).
- The vLLM engine process was killed by something outside this project on one test machine, roughly every two hours (cause not found). The server notices, exits with a non-zero status and frees the GPU. Run it under a supervisor if you need it always on, see [README-qwen3-asr.md](https://github.com/metodn/pi-voice-asr/blob/main/README-qwen3-asr.md).

## Credits

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) by the Qwen team (Apache-2.0) does the recognition.
- [pi](https://pi.dev) provides the terminal agent and extension API.
- [pi-voice-stt](https://pi.dev/packages/pi-voice-stt) by cgarrot was the reference for the editor hook. It is a batch design (record, then transcribe). This project is original code with live streaming, hold-to-talk and a local server.

This project is not affiliated with pi or the Qwen team.

## License

MIT, see [LICENSE](https://github.com/metodn/pi-voice-asr/blob/main/LICENSE).
