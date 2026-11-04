#!/usr/bin/env bash
# Run the Sightline backend in the foreground (WSL, ~/vjepa-venv). start_backend.ps1 launches this
# detached through a hidden wsl.exe so WSL stays up. Logs: $SIGHTLINE_LOG_DIR (default F:\models\logs).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${SIGHTLINE_PY:-$HOME/vjepa-venv/bin/python}"
export SIGHTLINE_LOG_DIR="${SIGHTLINE_LOG_DIR:-/mnt/f/models/logs}"
HOST="${SIGHTLINE_HOST:-127.0.0.1}"
PORT="${SIGHTLINE_PORT:-8000}"
PIDFILE="$SIGHTLINE_LOG_DIR/sightline_backend.pid"
OUT="$SIGHTLINE_LOG_DIR/backend_stdout.log"
mkdir -p "$SIGHTLINE_LOG_DIR"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "backend already running (pid $(cat "$PIDFILE"))"; exit 0
fi
# rotate the raw stdout/stderr log (the app log backend.log rotates itself: 10 MB x 5)
if [[ -f "$OUT" && $(stat -c %s "$OUT") -gt 20000000 ]]; then
  for i in 2 1; do [[ -f "$OUT.$i" ]] && mv -f "$OUT.$i" "$OUT.$((i+1))"; done
  mv -f "$OUT" "$OUT.1"
fi
cd "$REPO"
echo "$(date -Iseconds) starting backend on $HOST:$PORT" >> "$OUT"
echo $$ > "$PIDFILE"
# exec keeps the PID (the pidfile points at uvicorn). uvicorn logs warnings only: its INFO lines would
# print WebSocket URLs with their one-time ticket. App logs go to backend.log (no query strings).
exec "$PY" -m uvicorn --factory backend.app:create_app --host "$HOST" --port "$PORT" \
  --log-level warning --no-access-log --proxy-headers --forwarded-allow-ips 127.0.0.1 --timeout-graceful-shutdown 30 \
  --ws-ping-interval 20 --ws-ping-timeout 20 >> "$OUT" 2>&1
