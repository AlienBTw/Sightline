# Camera simulator (CCTV test rig)

A reproducible stand-in for real IP cameras / an NVR, used to show that Sightline works with real CCTV
protocols, and to test what happens when cameras misbehave.

| Piece | What | License |
|---|---|---|
| [MediaMTX](https://github.com/bluenviron/mediamtx) v1.21.3 | RTSP server acting as the camera/NVR (RTSP over TCP and UDP/RTP, Basic + Digest auth) | MIT |
| [FFmpeg](https://ffmpeg.org) static build ([BtbN](https://github.com/BtbN/FFmpeg-Builds), GPL) | the camera encoders: H.264 / H.265, OSD clock, packet loss | GPL (tool only, not shipped) |

Binaries are downloaded on first run to `$CAMSIM_BIN` (a directory outside the repo).

## Streams it publishes

Credentials and paths follow the Hikvision style (`/Streams/<channel><stream>`; 01 = main, 02 = sub):

| URL (`rtsp://admin:Sightline2026@<host>:8554/...`) | Mimics | Codec | Size / fps |
|---|---|---|---|
| `Streams/101` | channel 1 main stream | H.264 main, CBR 4 Mbit/s, GOP 2 s, no B-frames | 1920×1080 @ 25 |
| `Streams/102` | channel 1 sub-stream | H.264 main, 512 kbit/s | 640×360 @ 15 |
| `Streams/201` | channel 2 main (a plain Voxel51 clip) | H.264 main, 2 Mbit/s | 1280×720 @ 15 |
| `Streams/301` | an H.265 camera (common on modern NVRs) | HEVC (x265) | 1280×720 @ 20 |
| `Streams/901` | latency probe (barcode timestamp in every frame) | H.264 | 1280×720 @ 25 |

Every stream has a camera-style OSD (name + clock). Both transports work: the backend's
`rtsp_transport` per camera is `tcp` (default, best through Wi-Fi/NAT), `udp`, or `auto`.

## Run it

```powershell
# Windows (runs inside WSL, detached; logs and PIDs stay outside the repo)
powershell -ExecutionPolicy Bypass -File tools\camera_sim\start_sim.ps1           # server + 4 cameras
powershell -ExecutionPolicy Bypass -File tools\camera_sim\stop_sim.ps1 -Distro Ubuntu-24.04 pub 201    # unplug camera 2
powershell -ExecutionPolicy Bypass -File tools\camera_sim\start_sim.ps1 -Distro Ubuntu-24.04 pub 201   # plug it back in
powershell -ExecutionPolicy Bypass -File tools\camera_sim\stop_sim.ps1            # stop everything
```

Pass `-Distro` whenever you add stream arguments: the first positional argument binds to `-Distro`, so `start_sim.ps1 pub 101` fails with `WSL_E_DISTRO_NOT_FOUND`. A publisher started from a short-lived `wsl.exe` stops when that `wsl.exe` exits; for a long demo start it from one that stays open, for example `wsl.exe -d Ubuntu-24.04 -- bash -c "bash tools/camera_sim/start_sim.sh pub 101; exec sleep infinity"`. `CAMSIM_DEMO=<file>` swaps the clip that streams 101, 102 and 301 play.

```bash
# WSL
bash tools/camera_sim/start_sim.sh            # or: server | pub 101 301
~/vjepa-venv/bin/python tools/camera_sim/scenarios.py                 # robustness scenarios (no GPU)
~/vjepa-venv/bin/python tools/camera_sim/latency_probe.py publish --seconds 120 &
~/vjepa-venv/bin/python tools/camera_sim/latency_probe.py ingest --seconds 30
~/vjepa-venv/bin/python tools/camera_sim/latency_probe.py pipeline --seconds 60 --device cuda:1
~/vjepa-venv/bin/python -m sightline_pipeline.run --source rtsp://admin:Sightline2026@127.0.0.1:8554/Streams/101 --max-seconds 60
bash tools/camera_sim/stop_sim.sh
```

Add a simulated camera to the backend like a real one (admin account):

```bash
curl -X POST http://127.0.0.1:8000/api/cameras -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"name":"CAM01 main","uri":"rtsp://admin:Sightline2026@127.0.0.1:8554/Streams/101","rtsp_transport":"tcp"}'
```

`tools/camera_sim/backend_check.py` does this for three cameras (H.264 TCP, H.265 TCP, sub-stream UDP),
checks snapshot + MJPEG live view, status and events, then deletes them. `run_backend_test.sh` starts a
throwaway backend for it (a random admin password in a credentials file outside the repo, never `.env`).

Automated test: `pytest -m integration sightline_pipeline/tests/test_camera_sim_integration.py` starts
MediaMTX + an ffmpeg camera on port 18554 and checks TCP, UDP, wrong password and wrong path. It is
skipped when the binaries are not installed.

## Results (1 Nov 2026, RTX 3090 ×2, WSL2)

Frame rates and the 3-camera run are from a second pass with the CPU otherwise idle; the failure
scenarios were run first while the PC was also building the Android app (they passed both times).

| Scenario | Result | Notes |
|---|---|---|
| H.264 1080p25 main, TCP | pass | 25.0 of 25 fps steady, full size; first frame after ~4 s (connect + stream probe + first keyframe) |
| H.264 640×360 sub-stream | pass | 14.9 of 15 fps |
| H.264 720p15 (plain clip) | pass | 10.7-15.2 fps; the source clip is a few seconds long, so the publisher pauses at every loop restart (simulator artifact) |
| H.265 / HEVC 720p20 | pass | 20.0 of 20 fps; a few `hevc: Could not find ref` warnings when joining mid-GOP, then clean |
| UDP (RTP) transport | pass | 25.0 of 25 fps, max gap 0.08 s |
| Wrong password | pass | `authentication failed (wrong username or password)` in 1.7 s; password never logged; 1 attempt per 2 min while retrying (no account lock-out) |
| Wrong stream path | pass | `stream path not found` (RTSP 404) in 0.1 s |
| Unreachable host | pass | black-hole IP: clear timeout error in 13 s; closed port: `connection refused` in 0.2 s; bad DNS name: 0.9 s |
| Stop a camera that never connects | pass | ends in under 0.05 s (used to hang forever) |
| Camera unplugged and back (publisher killed 6-9 s) | pass | state `reconnecting` with reason; back 4.5-11.7 s after the camera returns (depends on where the backoff is), then full fps |
| NVR / RTSP server restart | pass | `connection refused` while down; back 11.5 s after restart |
| Stream stalls (TCP open, no packets) | pass | detected in 4.6 s (read timeout), recovered 3.9 s after the stream resumed |
| Resolution change (720p → 360p on reconnect) | pass | frames scaled back to the session size, so clips and the model window stay consistent |
| 3 % packet loss (dropped H.264 packets) | pass | 15.4 fps of 20, max gap 0.44 s, no reconnects |
| 3 cameras through the backend, both GPUs (H.264 1080p TCP, HEVC TCP, 360p UDP) | pass | all `running` in 21 s; real time: 28.5 / 19.5 / 15.4 fps mean into the pipeline, frame age 0.25 / 0.64 / 0.21 s mean (max 0.78 / 2.03 / 0.50 s); snapshot + MJPEG with circles; 5 events each, visible in the API |

Packet loss is simulated at the encoder (FFmpeg `noise` bitstream filter drops packets) because `tc netem`
needs root in WSL. Latency numbers: see the latency table below.

## Real cameras

**Android IP-camera app (real camera test):** install an RTSP camera app (e.g. "IP Webcam" or
"RTSP Camera Server"), start the server, and add the URL it shows, e.g. `rtsp://<phone-ip>:8080/h264_ulaw.sdp`
(IP Webcam; MJPEG also works: `http://<phone-ip>:8080/video`). Phone and PC must be on the same Wi-Fi.
Inside WSL, the phone's LAN IP is reachable directly (outbound connections from WSL are NATed).

Common URL formats (`<user>:<pass>@` may need `%40` for `@` in the password):

| Brand | Main stream | Sub-stream |
|---|---|---|
| Hikvision / HiLook | `rtsp://user:pass@IP:554/Streaming/Channels/101` | `.../Channels/102` |
| Dahua / Amcrest / Lorex | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` | `...&subtype=1` |
| Reolink | `rtsp://user:pass@IP:554/h264Preview_01_main` (H.265 models: `h265Preview_01_main`) | `.../h264Preview_01_sub` |
| Axis | `rtsp://user:pass@IP/axis-media/media.amp?videocodec=h264&resolution=1920x1080` | lower `resolution=` |
| Uniview | `rtsp://user:pass@IP:554/media/video1` | `.../video2` |
| Generic ONVIF | ask the camera: ONVIF `GetStreamUri` (Media service) returns the RTSP URL | |

Tips: prefer the sub-stream (Sightline samples at a few fps and resizes anyway), keep TCP unless the
network is clean, and give Sightline its own read-only camera user.

## Latency (single camera, 1280�720 @ 25, H.264, cuda:1)

Barcode timestamp burned into each frame at encode time, read back after decode (`latency_probe.py`):

| Path | p50 | p95 | max |
|---|---|---|---|
| capture ? frame in Sightline (RTSP/TCP) | 327 ms | 329 ms | 1489 ms (first frames) |
| capture ? frame in Sightline (RTSP/UDP) | 327 ms | 328 ms | 1412 ms (first frames) |
| capture ? model step done (alert decision) | 879 ms | 1504 ms | 1968 ms |
| model step (V-JEPA 2.1 + head + circles) | 546 ms | 559 ms | |

1,876 frames, 0 dropped. A saved event clip is written after the incident's post-roll (by design), so
"event created" is later than the alert decision.

Three cameras spread over both GPUs (the default) kept up in real time (see the table; GPU 28 % busy,
backend process ~2 CPU cores). In an earlier run with all three on ONE GPU while the PC was busy building
the Android app, the workers fell behind (~3 fps delivered, frames ~5 s old) but stayed up and created
events. So: keep the default GPU spread, and prefer sub-streams when running many cameras.

## ONVIF (optional)

Most CCTV cameras and NVRs speak ONVIF Profile S, so users can add a camera without knowing the
brand-specific RTSP path.

- **Simulator:** [daniela-hase/onvif-server](https://github.com/daniela-hase/onvif-server) (MIT, Node.js),
  started by `bash tools/camera_sim/start_onvif.sh` after `start_sim.sh`. It answers WS-Discovery on
  239.255.255.250:3702 and serves the device + media services on `http://<wsl-ip>:8081/onvif/device_service`,
  with a main profile (→ `Streams/101`, 1080p25 H.264) and a sub profile (→ `Streams/102`, 360p15) through its
  RTSP proxy on port 8564. Node 22 and the server are fetched next to the simulator binaries on first run.
- **Client:** `backend/onvif.py`, plain SOAP over httpx (no WSDL/zeep): WS-Discovery Probe (multicast,
  plus optional unicast to listed hosts for networks that drop multicast), `GetCapabilities`,
  `GetDeviceInformation`, `GetProfiles`, `GetStreamUri`, with WS-Security UsernameToken (PasswordDigest).
- **API (admin only):**
  - `POST /api/onvif/discover` `{"timeout_s": 3, "hosts": []}` → devices (name, hardware, XAddrs)
  - `POST /api/onvif/profiles` `{"xaddr", "username", "password"}` → device info + profiles with stream URIs
  - `POST /api/onvif/add` `{"xaddr", "username", "password", "profile_token"?, "name"?, "rtsp_transport"?}`
    → adds the camera like `POST /api/cameras` (credentials are inserted into the RTSP URL server-side and
    never returned).
- **Check:** `~/vjepa-venv/bin/python tools/camera_sim/onvif_check.py` (discover → profiles → read frames).

Result on 1 Nov 2026: multicast discovery found the simulated camera in 3 s; profiles returned
`MainStream` (H.264 1920×1080 @ 25, `rtsp://<ip>:8564/Streams/101`) and `SubStream` (640×360 @ 15,
`.../Streams/102`); Sightline's ingest opened both returned URIs with the credentials and decoded frames.

Limits: WS-Discovery is link-local multicast, so it only finds cameras on the backend's own LAN segment
(inside WSL2's NAT it sees WSL only; use `hosts` for unicast probes, or run the backend on the camera LAN).
PTZ, events (motion) and Profile T are not used.
