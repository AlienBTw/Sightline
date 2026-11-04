#!/usr/bin/env bash
LOG_DIR="${SIGHTLINE_LOG_DIR:-/mnt/f/models/logs}"
PIDFILE="$LOG_DIR/sightline_backend.pid"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "running pid $(cat "$PIDFILE")"; else echo "stopped"; fi
curl -s -m 5 "http://127.0.0.1:${SIGHTLINE_PORT:-8000}/health" || echo "health: no answer"
echo
