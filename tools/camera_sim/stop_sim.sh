#!/usr/bin/env bash
# Stop the camera simulator.  stop_sim.sh            -> everything
#                             stop_sim.sh pub 101    -> one publisher (simulates the camera dropping off)
#                             stop_sim.sh server     -> MediaMTX only (simulates the NVR rebooting)
LOG="${CAMSIM_LOG:-/mnt/f/models/logs/camsim}"
kill_one() { local f="$LOG/$1.pid"; [[ -f "$f" ]] || return 0; local p; p=$(cat "$f")
  if kill -0 "$p" 2>/dev/null; then kill "$p"; for _ in $(seq 25); do kill -0 "$p" 2>/dev/null || break; sleep 0.2; done
    kill -0 "$p" 2>/dev/null && kill -9 "$p"; echo "stopped $1 (pid $p)"; fi; rm -f "$f"; }
case "${1:-all}" in
  pub) shift; for id in "$@"; do kill_one "pub_$id"; done ;;
  server) kill_one mediamtx ;;
  all) for f in "$LOG"/pub_*.pid "$LOG"/latency_pub.pid "$LOG"/onvif.pid; do [[ -f "$f" ]] && kill_one "$(basename "$f" .pid)"; done; kill_one mediamtx ;;
esac
