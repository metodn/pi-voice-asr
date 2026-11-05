#!/usr/bin/env bash
# Start the ASR server with the LFM3-Audio-2B backend (llama.cpp, local GPU).
# Drop-in replacement for server/transcribe_server.py: same /v1/stt WebSocket
# protocol, so the pi live-stt extension works unchanged.
#
#   LFM3_ROOT=/path/to/lfm3 ./run_lfm3_asr.sh   # adapter :8765, llama-server :8182
#   ./run_lfm3_asr.sh --port 8766               # extra args pass to the adapter
#   ./run_lfm3_asr.sh --language de             # pin a language (default: auto)
#
# Layout:
#   - starts llama-server with LFM3-Audio + mmproj on $LLAMA_PORT (8182) if none
#     is running there, then starts server/lfm3_asr_server.py on $PORT (8765).
#   - model files: GGUF + mmproj from https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF
#     and a llama.cpp build with LFM3-Audio support. Default layout under $LFM3_ROOT:
#       llama.cpp/build/bin/llama-server, models/LFM3-Audio-2B-Q8_0.gguf,
#       models/mmproj-LFM3-Audio-2B-BF16.gguf
#     or point LLAMA_BIN, MODEL, MMPROJ at the files directly.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# --- paths to the LFM3 model (llama.cpp build + GGUFs) ---
LFM3_ROOT="${LFM3_ROOT:-$HOME/lfm3}"
LLAMA_BIN="${LLAMA_BIN:-$LFM3_ROOT/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$LFM3_ROOT/models/LFM3-Audio-2B-Q8_0.gguf}"
MMPROJ="${MMPROJ:-$LFM3_ROOT/models/mmproj-LFM3-Audio-2B-BF16.gguf}"

LLAMA_PORT="${LLAMA_PORT:-8182}"
LLAMA_URL="http://127.0.0.1:$LLAMA_PORT"
PORT="${PORT:-8765}"
NGL="${NGL:-99}"
LOG_LLAMA="${LOG_LLAMA:-server/llama-lfm3.log}"

for f in "$LLAMA_BIN" "$MODEL" "$MMPROJ"; do
    [[ -e "$f" ]] || { echo "ERROR: missing $f (set LFM3_ROOT, or LLAMA_BIN / MODEL / MMPROJ)"; exit 1; }
done

# --- start llama-server if it isn't healthy yet ---
if ! curl -s -m 2 "$LLAMA_URL/health" | grep -q '"ok"'; then
    echo "Starting llama-server (LFM3-Audio) on :$LLAMA_PORT ..."
    export LD_LIBRARY_PATH="$(dirname "$(readlink -f "$LLAMA_BIN")")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    nohup "$LLAMA_BIN" \
        -m "$MODEL" -mm "$MMPROJ" \
        --host 127.0.0.1 --port "$LLAMA_PORT" \
        -ngl "$NGL" --reasoning off \
        > "$LOG_LLAMA" 2>&1 &
    echo "  pid $! (log: $LOG_LLAMA)"
    for i in $(seq 1 120); do
        curl -s -m 2 "$LLAMA_URL/health" | grep -q '"ok"' && break
        sleep 2
    done
    curl -s -m 2 "$LLAMA_URL/health" | grep -q '"ok"' || { echo "ERROR: llama-server not healthy — see $LOG_LLAMA"; exit 1; }
fi
echo "llama-server ready at $LLAMA_URL"

# --- start the /v1/stt adapter (same protocol as transcribe_server.py) ---
exec uv run --script server/lfm3_asr_server.py \
    --llama-url "$LLAMA_URL" \
    --port "$PORT" \
    "$@"
