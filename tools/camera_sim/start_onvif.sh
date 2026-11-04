#!/usr/bin/env bash
# Optional ONVIF layer for the simulator: daniela-hase/onvif-server (MIT, Node.js) advertises a virtual
# ONVIF Profile S camera (WS-Discovery on 239.255.255.250:3702, device + media services on :8081) whose
# main/sub profiles point at MediaMTX Streams/101 and Streams/102 (through its RTSP proxy on :8564).
#   bash tools/camera_sim/start_onvif.sh        (start_sim.sh must be running)
#   bash tools/camera_sim/stop_sim.sh           stops it too
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK="${SIGHTLINE_WORK_DIR:-$REPO/work}"
BIN="${CAMSIM_BIN:-$WORK/camsim/bin}"; ROOT="$(dirname "$BIN")"
LOG="${CAMSIM_LOG:-$WORK/logs/camsim}"; mkdir -p "$LOG"
NODE="$ROOT/node/bin"; SRV="$ROOT/onvif-server"
if [[ ! -x "$NODE/node" ]]; then
  V=$(curl -s https://nodejs.org/dist/index.json | grep -o '"version":"v22[^"]*"' | head -1 | cut -d'"' -f4)
  curl -sL "https://nodejs.org/dist/$V/node-$V-linux-x64.tar.xz" | tar xJ -C "$ROOT" && mv "$ROOT/node-$V-linux-x64" "$ROOT/node"
fi
if [[ ! -d "$SRV/node_modules" ]]; then
  git clone -q --depth 1 https://github.com/daniela-hase/onvif-server "$SRV"
  (cd "$SRV" && PATH="$NODE:$PATH" npm install --no-audit --no-fund >/dev/null)
fi
IFACE="${CAMSIM_IFACE:-eth0}"
MAC=$(cat "/sys/class/net/$IFACE/address")
cat > "$LOG/onvif.yaml" <<YAML
onvif:
  - mac: $MAC
    ports: {server: 8081, rtsp: 8564}
    name: SightlineSimCam
    uuid: 5f1c2d3e-4b5a-4c6d-8e7f-9a0b1c2d3e4f
    highQuality: {rtsp: /Streams/101, width: 1920, height: 1080, framerate: 25, bitrate: 4096, quality: 4}
    lowQuality: {rtsp: /Streams/102, width: 640, height: 360, framerate: 15, bitrate: 512, quality: 1}
    target: {hostname: 127.0.0.1, ports: {rtsp: 8554}}
YAML
if [[ -f "$LOG/onvif.pid" ]] && kill -0 "$(cat "$LOG/onvif.pid")" 2>/dev/null; then echo "onvif already running"; exit 0; fi
cd "$SRV"
setsid nohup "$NODE/node" main.js "$LOG/onvif.yaml" >> "$LOG/onvif.log" 2>&1 < /dev/null &
echo $! > "$LOG/onvif.pid"
sleep 2
IP=$(ip -4 -o addr show "$IFACE" | awk '{print $4}' | cut -d/ -f1)
echo "onvif device: http://$IP:8081/onvif/device_service (pid $(cat "$LOG/onvif.pid"))"
