#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK="${SIGHTLINE_WORK_DIR:-$REPO/work}"
LOG_DIR="${SIGHTLINE_LOG_DIR:-$WORK/logs}"
PIDFILE="$LOG_DIR/sightline_backend.pid"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "running pid $(cat "$PIDFILE")"; else echo "stopped"; fi
curl -s -m 5 "http://127.0.0.1:${SIGHTLINE_PORT:-8000}/health" || echo "health: no answer"
echo
