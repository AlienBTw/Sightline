#!/usr/bin/env bash
# Stop the backend gracefully (camera workers finish their current clip), then force after 45 s.
LOG_DIR="${SIGHTLINE_LOG_DIR:-/mnt/f/models/logs}"
PIDFILE="$LOG_DIR/sightline_backend.pid"
[[ -f "$PIDFILE" ]] || { echo "not running (no pidfile)"; exit 0; }
PID="$(cat "$PIDFILE")"
if ! kill -0 "$PID" 2>/dev/null; then echo "not running (stale pidfile)"; rm -f "$PIDFILE"; exit 0; fi
kill -TERM "$PID"
for _ in $(seq 1 45); do kill -0 "$PID" 2>/dev/null || { rm -f "$PIDFILE"; echo "stopped $PID"; exit 0; }; sleep 1; done
kill -KILL "$PID" 2>/dev/null; rm -f "$PIDFILE"; echo "killed $PID"
