#!/usr/bin/env bash
# Sightline CCTV simulator: MediaMTX (RTSP server) + ffmpeg publishers that look like IP-camera streams.
#   bash tools/camera_sim/start_sim.sh            # server + all camera streams (detached)
#   bash tools/camera_sim/start_sim.sh pub 101    # (re)start one publisher, e.g. after a disconnect test
#   bash tools/camera_sim/start_sim.sh server     # only the RTSP server
# Viewer URL:  rtsp://admin:Sightline2026@<host>:8554/Streams/101   (see README.md for the stream list)
# Binaries:    $CAMSIM_BIN (default /mnt/f/models/camsim/bin), downloaded on first run.
# Logs + PIDs: $CAMSIM_LOG (default /mnt/f/models/logs/camsim).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BIN="${CAMSIM_BIN:-/mnt/f/models/camsim/bin}"
LOG="${CAMSIM_LOG:-/mnt/f/models/logs/camsim}"
PORT="${CAMSIM_PORT:-8554}"
PUB="rtsp://publisher:pub-local-only@127.0.0.1:$PORT"
DEMO="${CAMSIM_DEMO:-$REPO/data/demo/sightline_demo.mp4}"
CLIP="${CAMSIM_CLIP:-$(ls "$REPO"/data/raw/voxel51_safe_unsafe/data/*.mp4 2>/dev/null | head -1 || true)}"
[[ -n "$CLIP" ]] || CLIP="$DEMO"
FONT="${CAMSIM_FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf}"
mkdir -p "$BIN" "$LOG"

fetch() {
  if [[ ! -x "$BIN/mediamtx" ]]; then
    local u; u=$(curl -s https://api.github.com/repos/bluenviron/mediamtx/releases/latest | grep -o 'https://[^"]*linux_amd64.tar.gz' | head -1)
    curl -sL "$u" | tar xz -C "$BIN" mediamtx LICENSE && mv -f "$BIN/LICENSE" "$BIN/MEDIAMTX_LICENSE"
  fi
  if [[ ! -x "$BIN/ffmpeg" ]]; then   # static build with libx264, libx265 and drawtext
    curl -sL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz \
      | tar xJ -C "$BIN" --strip-components=2 --wildcards '*/bin/ffmpeg' '*/bin/ffprobe'
  fi
}

alive() { [[ -f "$LOG/$1.pid" ]] && kill -0 "$(cat "$LOG/$1.pid")" 2>/dev/null; }

server() {
  alive mediamtx && { echo "mediamtx already running (pid $(cat "$LOG/mediamtx.pid"))"; return; }
  setsid nohup "$BIN/mediamtx" "$HERE/mediamtx.yml" >> "$LOG/mediamtx.log" 2>&1 < /dev/null &
  echo $! > "$LOG/mediamtx.pid"
  for _ in $(seq 50); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { echo "mediamtx up on :$PORT (pid $(cat "$LOG/mediamtx.pid"))"; return; }; sleep 0.2; done
  echo "mediamtx did not start; see $LOG/mediamtx.log"; exit 1
}

# camera-style OSD: name + wall clock, burned into the picture
osd() { echo "drawtext=fontfile=$FONT:text='$1 %{localtime}':x=20:y=20:fontsize=$2:fontcolor=white:box=1:boxcolor=black@0.5"; }

pub() {   # pub <id>
  local id="$1" in vf enc
  case "$id" in
    101) in="$DEMO"; vf="scale=1920:1080,fps=25,$(osd 'CAM01 MAIN' 36)"
         enc="-c:v libx264 -profile:v main -preset veryfast -tune zerolatency -b:v 4M -maxrate 4M -bufsize 2M -g 50 -bf 0" ;;
    102) in="$DEMO"; vf="scale=640:360,fps=15,$(osd 'CAM01 SUB' 16)"
         enc="-c:v libx264 -profile:v main -preset veryfast -tune zerolatency -b:v 512k -g 30 -bf 0" ;;
    201) in="$CLIP"; vf="scale=1280:720,fps=15,$(osd 'CAM02 MAIN' 24)"
         enc="-c:v libx264 -profile:v main -preset veryfast -tune zerolatency -b:v 2M -g 30 -bf 0" ;;
    301) in="$DEMO"; vf="scale=1280:720,fps=20,$(osd 'CAM03 HEVC' 24)"
         enc="-c:v libx265 -preset ultrafast -tune zerolatency -b:v 2M -x265-params keyint=40:log-level=error -tag:v hvc1" ;;
    *) echo "unknown stream $id (101 102 201 301)"; exit 2 ;;
  esac
  alive "pub_$id" && { echo "pub $id already running"; return; }
  setsid nohup "$BIN/ffmpeg" -hide_banner -loglevel warning -re -stream_loop -1 -i "$in" -an -vf "$vf" $enc \
      -pix_fmt yuv420p -f rtsp -rtsp_transport tcp "$PUB/Streams/$id" >> "$LOG/pub_$id.log" 2>&1 < /dev/null &
  echo $! > "$LOG/pub_$id.pid"
  echo "pub $id -> rtsp://admin:***@127.0.0.1:$PORT/Streams/$id (pid $!)"
}

fetch
case "${1:-all}" in
  server) server ;;
  pub) shift; server; for id in "$@"; do pub "$id"; done ;;
  all) server; for id in ${CAMSIM_STREAMS:-101 102 201 301}; do pub "$id"; done ;;
  *) echo "usage: $0 [all|server|pub <id>...]"; exit 2 ;;
esac
