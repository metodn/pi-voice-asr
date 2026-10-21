#!/usr/bin/env bash
# Build dist/pi-voice-asr-remote.zip: the extension plus a ready live-stt.json for a remote machine.
#
#   STT_URL=ws://stt-host.example:8765/v1/stt MIC=server scripts/make-remote-bundle.sh
#
# MIC is "server" (default: the STT server records its own mic) or "local" (the remote records).
# On the remote: unzip -o pi-voice-asr-remote.zip -d ~/.pi/agent
set -euo pipefail
cd "$(dirname "$0")/.."

: "${STT_URL:?set STT_URL, for example ws://stt-host.example:8765/v1/stt}"
MIC="${MIC:-server}"
case "$MIC" in local | server) ;; *) echo "MIC must be local or server" >&2; exit 1 ;; esac

root="$PWD"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/extensions" dist
cp extensions/live-stt.ts "$stage/extensions/"
printf '{\n  "url": "%s?sample_rate=16000&encoding=pcm",\n  "micSource": "%s"\n}\n' "${STT_URL%%\?*}" "$MIC" > "$stage/live-stt.json"
rm -f dist/pi-voice-asr-remote.zip
(cd "$stage" && python3 -m zipfile -c "$root/dist/pi-voice-asr-remote.zip" extensions live-stt.json)
echo "wrote dist/pi-voice-asr-remote.zip"
