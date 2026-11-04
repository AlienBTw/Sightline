# backend: the Sightline API (FastAPI)

The product backend around the live pipeline, written during the event window. It's what the Android app (React Native + Expo) and the dev console talk to.

```
Android app (Expo) ──HTTPS / WSS── ngrok (free dev domain) ── FastAPI backend (WSL, 127.0.0.1:8000)
dev console (/console)  ─────────┘                               │
                                                                 ├─ camera manager: one SightlinePipeline worker per active camera
                                                                 │    RTSP / HTTP cameras and uploaded videos (virtual cameras)
                                                                 │    through the same VideoSource ingest path, 3 per RTX 3090
                                                                 ├─ events: the pipeline's SQLite index + event folders (clips, thumbs, JSON)
                                                                 ├─ WebSocket alerts + Expo push (exp.host) on every new event
                                                                 └─ backend SQLite: cameras, refresh tokens, push devices, uploads, event actions
```

| File | What it does |
|---|---|
| `app.py` | App factory (`create_app`), security headers, CSP, request log without query strings, per-IP API rate limit, demo-camera seeding, autostart, graceful shutdown |
| `settings.py` | All configuration from env vars (and the repo `.env` through python-dotenv) |
| `security.py` | scrypt password hashes (stdlib), JWTs (access, refresh, signed media links, WebSocket tickets), sliding-window rate limiter |
| `cameras.py` | `CameraManager`, `CameraWorker` threads, a per-GPU model pool, `TappedSource` (latest frame for the live view) |
| `routes/` | `auth`, `cameras` (registry, upload, live view), `events` (feed, detail, status, range-streamed media), `misc` (health, push devices, WebSocket) |
| `media.py` | File responses with HTTP Range (206 / 416 / HEAD / ETag) for Android video playback |
| `push.py` | Expo push sender (batches, cooldown per camera + factor, prunes `DeviceNotRegistered` tokens) |
| `hub.py` | Thread-safe fan-out from pipeline threads to WebSocket clients |
| `preview.py` | Live-view frames: latest frame, downscaled, with the current step's circles drawn |
| `static/console/` | The dev console (plain HTML + JS, strict CSP, same login) |
| `scripts/` | Start/stop (WSL + Windows), ngrok, watchdog, scheduled tasks |
| `hash_password.py` | Generates the password hash and JWT secret for `.env` |

The pipeline itself is unchanged apart from one optional hook, `process(..., on_step=None)`, which hands each model step's circles to the live view. Thresholds and the model are untouched.

## Setup (once)

Everything runs in WSL (Ubuntu-24.04) with `~/vjepa-venv`, which already has FastAPI, uvicorn, PyJWT, httpx and websockets. No new packages are needed: password hashing uses the standard library's scrypt.

1. Make the secrets and put them in `Sightline/.env` (never committed). Run these from the repo root:
   ```bash
   PY=~/vjepa-venv/bin/python
   $PY -m backend.hash_password --secret            # -> SIGHTLINE_JWT_SECRET=...
   $PY -m backend.hash_password --generate          # judge password (shown once) -> SIGHTLINE_DEMO_PASSWORD_HASH=...
   $PY -m backend.hash_password --admin             # your own password, prompted -> SIGHTLINE_ADMIN_PASSWORD_HASH=...
   ```
   Then add `SIGHTLINE_DEMO_USER=judge` (or another name) and `SIGHTLINE_ADMIN_USER=<you>`. The hash format is `scrypt:N:r:p:salt:hash`, with no `$`, so it's safe in `.env` files and shells.
2. Optional: `MODEL_API_KEY` turns on the SAM 3.1 and Muse Spark 1.3 Contributor (`muse-spark-1.3-contributor`) escalations and the F2/F5 verify gates. Without it they're recorded as `skipped_no_key`, and F2/F5 are stored but never alerted. Every call is capped: 1 per incident, 2 per minute, 3 per camera session, 30 per day (`SIGHTLINE_ESC_MAX_PER_SESSION`, `SIGHTLINE_ESC_MAX_PER_DAY`); see [`sightline_pipeline`](../sightline_pipeline/README.md).

## Run

From Windows PowerShell, in the repo root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File backend\scripts\start_backend.ps1   # detached, waits for /health
powershell -NoProfile -ExecutionPolicy Bypass -File backend\scripts\stop_backend.ps1    # graceful stop (forced after 45 s)
```

Or inside WSL: `bash backend/scripts/run_backend.sh` (foreground), `bash backend/scripts/stop_backend.sh` and `bash backend/scripts/status_backend.sh`.

- The backend listens on `127.0.0.1:8000` in WSL. WSL forwards localhost, so Windows (and ngrok) reach it at `http://127.0.0.1:8000`.
- The dev console is at `http://127.0.0.1:8000/console/`.
- Logs go to the directory in `SIGHTLINE_LOG_DIR` (Windows scripts also read `SIGHTLINE_LOG_DIR_WIN`):
  - `backend.log` is the app log, rotated at 10 MB, keeping 5 files.
  - `backend_stdout.log` holds uvicorn warnings and tracebacks, rotated at 20 MB when the backend starts.
  - The PID file is `sightline_backend.pid`.
- A camera loads its V-JEPA model on first start, which takes about 8 s. Models stay pooled on their GPU after a camera stops (about 1.3 GB of VRAM each), so restarting is instant.
- If a camera was running when the backend stopped, it starts again on the next boot (`SIGHTLINE_AUTOSTART=1`). An uploaded camera stops itself after `SIGHTLINE_UPLOAD_MAX_RUNTIME_S` (30 min by default), so a judge's upload can't fill the event list for days.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SIGHTLINE_JWT_SECRET` | required | HS256 key, at least 32 characters. `SIGHTLINE_DEV=1` allows a random per-process key for local development |
| `SIGHTLINE_DEMO_USER` / `SIGHTLINE_DEMO_PASSWORD_HASH` | none | Judge login, role `demo`: read, upload, start/stop uploaded cameras, acknowledge/dismiss events |
| `SIGHTLINE_ADMIN_USER` / `SIGHTLINE_ADMIN_PASSWORD_HASH` | none | Admin login: also RTSP cameras, edit and delete |
| `SIGHTLINE_ACCESS_TTL_S` / `SIGHTLINE_REFRESH_TTL_S` / `SIGHTLINE_MEDIA_TTL_S` | 900 / 604800 / 900 | Lifetimes of the access token, the refresh token and signed media links |
| `SIGHTLINE_GPUS` / `SIGHTLINE_CAMERAS_PER_GPU` | `cuda:0,cuda:1` / 3 | Where workers run and the cap on active cameras (6 by default) |
| `SIGHTLINE_UPLOAD_MAX_MB` / `SIGHTLINE_UPLOAD_MAX_SECONDS` | 300 / 600 | Upload size and duration limits |
| `SIGHTLINE_UPLOADS_PER_HOUR` / `SIGHTLINE_MAX_VIRTUAL_CAMERAS` | 12 / 30 | Upload rate per user, and the total number of uploaded cameras |
| `SIGHTLINE_UPLOAD_MAX_RUNTIME_S` | 1800 | An uploaded (looping) camera stops after this. 0 means no limit |
| `SIGHTLINE_UPLOAD_DIR` / `SIGHTLINE_BACKEND_DIR` | `data/uploads` / `data/backend` | Storage (gitignored). Events stay in the pipeline's `data/live` |
| `SIGHTLINE_SEED_DEMO_CAMERA` / `SIGHTLINE_DEMO_VIDEO` | 1 / `data/demo/sightline_demo.mp4` | Pre-registers "Demo workshop (sample video)", stopped until someone presses start |
| `SIGHTLINE_AUTOSTART` | 1 | Restarts cameras that were running |
| `SIGHTLINE_PUSH` / `EXPO_ACCESS_TOKEN` / `SIGHTLINE_PUSH_COOLDOWN_S` | on / none / 60 | Expo push. The token is only needed if push security is turned on for the Expo project. The cooldown applies per camera and factor |
| `SIGHTLINE_CORS_ORIGINS` | none | Extra web origins. The native app and the same-origin console need none |
| `SIGHTLINE_ALLOWED_HOSTS` | none | For example `localhost,127.0.0.1,<name>.ngrok-free.app`, which enables Host-header checks |
| `SIGHTLINE_TRUST_PROXY` | 1 | Uses `X-Forwarded-For` only when the peer is 127.0.0.1 (ngrok) |
| `SIGHTLINE_LOGIN_PER_MINUTE` / `SIGHTLINE_API_PER_MINUTE` | 5 / 600 | Rate limits per IP. Login also allows 30 per hour per IP and 10 per 15 min per username |
| `SIGHTLINE_MJPEG_MAX_SECONDS` / `SIGHTLINE_PREVIEW_QUALITY` | 300 / 60 | Length of one live-view session, and JPEG quality |
| `SIGHTLINE_API_DOCS` | 0 | `1` serves OpenAPI at `/api/docs` |
| `SIGHTLINE_LOG_DIR` | a directory outside the repo | Rotating app log. The PowerShell scripts use `SIGHTLINE_LOG_DIR_WIN` for the same folder |
| `SIGHTLINE_HOST` / `SIGHTLINE_PORT` | 127.0.0.1 / 8000 | Bind address, used by `run_backend.sh` |

The pipeline's own variables still apply: `MODEL_API_KEY`, `SAM_BACKEND`, `MUSE_BACKEND`, `MUSE_SPARK_MODEL` (Muse Spark 1.3 Contributor, `muse-spark-1.3-contributor`), `SIGHTLINE_ESCALATION`, `SIGHTLINE_VERIFY_GATES`, `SIGHTLINE_VERIFY_GATE_F2`, `SIGHTLINE_VERIFY_GATE_F5` and `SIGHTLINE_DATA_DIR`. See `.env.example`. F2 (no head protection) and F5 (no hi-vis vest) are published only after Muse Spark confirms them; the event's `verified_by` field and `escalation.verify_gates` carry the badge.

## API

Every `/api/*` route needs `Authorization: Bearer <access token>` except login, refresh and logout. Media routes also accept the signed `?st=` links that the API hands out. All paths are relative, so the app prefixes its base URL (the ngrok domain).

| Method and path | Purpose |
|---|---|
| `POST /api/auth/login` `{username, password}` | Returns `{access_token, refresh_token, expires_in, user{username, role}}`. Rate-limited |
| `POST /api/auth/refresh` `{refresh_token}` | Rotates the pair. Reusing a rotated refresh token revokes the whole session |
| `POST /api/auth/logout` `{refresh_token}` · `GET /api/auth/me` | Logout revokes the session. `me` returns the role and permissions |
| `GET /api/cameras` | Cameras with live status (state, GPU, fps, last frame age, active factors, last error), signed preview links and capacity |
| `POST /api/cameras` `{name, uri, start}` (admin) | Registers an RTSP or HTTP camera. Credentials stay on the server, and admins see a redacted URI |
| `GET` / `PATCH` / `DELETE /api/cameras/{id}` | Detail, edit (admin) and delete (admin). Deleting keeps the camera's events |
| `POST /api/cameras/{id}/start` and `/stop` | Starts or stops the pipeline worker. Returns 409 when all GPU slots are busy |
| `POST /api/cameras/upload?filename=&name=&start=` | Streamed single-request upload of the raw body, which becomes a looping virtual camera |
| `POST /api/uploads` `{filename, size}`, then `PUT /api/uploads/{id}?offset=`, `GET /api/uploads/{id}`, `POST /api/uploads/{id}/complete`, `DELETE /api/uploads/{id}` | Resumable chunked upload (8 MB chunks). A wrong offset gets 409 with an `Upload-Offset` header to resume from |
| `GET /api/cameras/{id}/live.mjpeg?fps=1&w=480` | Live view as one long MJPEG response with circles drawn (see below) |
| `GET /api/cameras/{id}/snapshot.jpg?w=480` | Latest frame with circles, for grid thumbnails (fetched once, not polled) |
| `GET /api/events?camera_id=&status=&factor=&min_risk=&since=&until=&limit=&offset=` | Alerts feed, newest first, with `total` and `next_offset`, plus a signed `thumb_url` per item |
| `GET /api/events/{id}` | Full event: segments, `overlays` (normalised circles per `t_ms` for drawing on the clip), escalation results, signed `media.{clip, clip_circles, thumb}.url` with `ready` flags, and the action history |
| `GET` / `HEAD /api/events/{id}/clip.mp4`, `clip_circles.mp4`, `thumb.jpg` | Range streaming (206, 416, ETag). Returns 503 with `Retry-After` while `clip_circles.mp4` is still rendering |
| `POST /api/events/{id}/ack`, `/dismiss`, `/status` `{status}` | Status is one of `open`, `acknowledged`, `dismissed` or `fix_pending`. Changes are broadcast over the WebSocket |
| `POST /api/devices` `{token, platform}` · `GET` / `DELETE /api/devices` · `POST /api/devices/test` | Expo push tokens per user (at most 10), plus a test push |
| `POST /api/ws/ticket`, then `WS /api/ws?ticket=` | One-time 60 s ticket, so the access token never appears in a URL |
| `GET /health` (public) · `GET /api/health` | Liveness for the watchdog (numbers only), and a detailed status |

**WebSocket messages:**
- `hello`: the cameras and their states.
- `event`: a new alert, as the same summary used in the list.
- `event_status`: an event's status changed.
- `camera`: a camera's state changed.
- `alert_state`: a camera's active factors changed (live badge).
- `camera_added` and `camera_removed`.
- `ping`, every 20 s. Clients may send `ping` and get `pong` back.

**Push payload:** a title with the factor name and a body with the risk percentage. `data` is `{event_id, factor, risk}`, with Android channel `alerts`. It carries no images and no camera names, and the app fetches details over the authed API when the push is tapped.

## Live view: why MJPEG at 1 fps

ngrok's free plan has **1 GB of data out per month and 20,000 HTTP requests per month** ([ngrok free plan limits](https://ngrok.com/docs/pricing-limits/free-plan-limits)). So the live view is built for few requests and few bytes:

- **One long `multipart/x-mixed-replace` response per viewing session**, rather than JPEG polling, which would cost 3,600 requests an hour per viewer. It runs at 1 fps and 480 px, about 20 KB per frame (roughly 70 MB per hour of viewing), and ends after 5 minutes. The app shows a "keep watching" button to reconnect.
- **Circles are drawn on the server** from the latest model step, so the app needs no overlay logic for live.
- **In Expo:** Android's `<Image>` can't decode MJPEG, so use `react-native-webview` with `<img src="{signed mjpeg_url}">`. The signed link means no headers are needed, and the WebView only needs network permission. For grid tiles, fetch `snapshot.jpg` once when the screen opens.
- **HLS was rejected** for now: it needs an ffmpeg or MediaMTX relay per camera, segment polling (many requests) and several seconds of latency, and it can't show the circles without re-encoding.
- **Flagged clips** are the main artifact. They're H.264 with faststart at 960 px, about 1 MB for a 20 s clip, and they're range-streamed so `expo-video` can seek.

## ngrok runbook

ngrok isn't installed on the build PC yet (checked Oct 31).

1. Install it with `winget install ngrok.ngrok`, then sign up at ngrok.com.
2. Store the authtoken in ngrok's own config, outside the repo: `ngrok config add-authtoken <token>`, which writes `%LOCALAPPDATA%\ngrok\ngrok.yml`. Alternatively, set it as a user env var, `NGROK_AUTHTOKEN`.
3. Start the tunnel with `backend\scripts\start_ngrok.ps1`, which runs `ngrok http 8000` on your account's free static dev domain (`https://<name>.ngrok-free.app`). Set `$env:SIGHTLINE_NGROK_URL` to pin it. The log is `ngrok.log` in the log directory.
4. Put that HTTPS URL in the app build as the API base URL. The app must use HTTPS/WSS only.
5. Browsers get ngrok's interstitial page once (it's then remembered for 7 days). Apps and API clients aren't affected, and the header `ngrok-skip-browser-warning: 1` skips it as well.
6. Watch your usage on the ngrok dashboard. If judges use it heavily, upgrade the plan for Nov 4-8 rather than shrinking the demo.

## Judging uptime (Nov 4-8)

- **Scheduled tasks:** `backend\scripts\register_tasks.ps1` registers three tasks for the current user:
  - "Sightline Backend": starts the backend at logon.
  - "Sightline ngrok": starts ngrok at logon.
  - "Sightline Watchdog": runs every 5 minutes. After 2 failed `/health` checks it restarts the backend, and it restarts ngrok if ngrok isn't running. It logs to `watchdog.log` in the log directory, rotated at 5 MB.
  - To remove them: `Unregister-ScheduledTask -TaskName "Sightline*" -Confirm:$false`.
- **Power:** set the Windows power plan to never sleep or hibernate, pause Windows Update for Nov 4-8, and use mains power and a wired network if possible.
- **External check:** point a free uptime monitor at `https://<name>.ngrok-free.app/health`. The endpoint is public and returns numbers only.
- **Pre-seeded data:** the events under `data/live` (from the demo runs) appear in the feed, and the "Demo workshop (sample video)" camera is registered and ready to start.
- **Submission notes:** include the demo username and password, the APK link, and a note that live access depends on the PC being up.

## Security notes

- **Logins:** passwords are stored only as scrypt hashes in env vars, and checks are constant-time, with a dummy hash for unknown users. Login is rate-limited per IP and per username.
- **Tokens:**
  - Access tokens last 15 min. Refresh tokens last 7 days, rotate on every use and are stored server-side, so reuse revokes the session.
  - JWTs carry `iss`, `aud`, `typ` and `jti`, and a token of one type is rejected where another type is expected.
  - Removing a user from the env invalidates their tokens.
- **Roles:** the demo role can read, upload, start and stop uploaded cameras, and acknowledge or dismiss events. It can't add RTSP URLs (so the backend can't be pointed at internal hosts), delete or edit cameras.
- **Media:** signed links are scoped to one event or camera path and last 15 min. Event IDs and names are matched against strict patterns, so there's no path traversal. Local file paths and stream URLs are removed from API output.
- **Uploads:**
  - Every upload passes an extension allowlist, a content-type check, a magic-byte check and an FFmpeg decode probe, plus duration, resolution and size limits.
  - Uploads are streamed to disk with a running size check and are never served raw.
  - Uploads are also rate-limited per user, and the server checks free disk space.
- **Headers:** responses get `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer` and CORP. The console gets a strict CSP (no inline scripts), and HSTS is sent behind HTTPS.
- **CORS** is off unless `SIGHTLINE_CORS_ORIGINS` is set.
- **Logging:** request logs never include query strings (signed links and tickets), and uvicorn runs at `--log-level warning` for the same reason. No secret is ever logged.
- **Not done yet:** per-workshop multi-tenancy (there's one workshop for the demo), per-camera ACLs, and footage retention or expiry.

## Tests

`python -m pytest backend/tests -q` runs 28 tests with a fake pipeline (no GPU):
- auth: hashing, login, rate limit, every `/api` route requiring a token, expired or foreign tokens, refresh rotation and reuse, logout, roles
- uploads: type, size, magic bytes, the decode probe and duration, the resumable chunked protocol
- GPU capacity cap, and RTSP CRUD with credential redaction
- live snapshot, MJPEG and signed-link scoping
- events: filters, pagination, status changes, Range/HEAD/416/ETag streaming
- the WebSocket ticket, event delivery, and the Expo push payload and token pruning

The whole suite (`tests`, `sightline_pipeline/tests`, `training/tests` and `backend/tests`) runs with plain `python -m pytest`.

## Smoke test with the real pipeline (Oct 31, 23:15-23:18 Tunis)

The backend was started detached in WSL with a throwaway login (process env only), and `data/demo/sightline_demo.mp4` was uploaded through `POST /api/cameras/upload`:
- **Upload:** 11.8 MB in 0.38 s, which created virtual camera `cam_69a308` (59.4 s, 1280x720 at 25 fps, looping).
- **Startup:** it was running on `cuda:0` 8.2 s later, with the V-JEPA 2.1 model load included.
- **Live view:** the snapshot was 19.8 KB, and MJPEG delivered 5 frames (104 KB) in 3.5 s at `fps=1`.
- **Events:** the first event arrived about 35 s after the upload, and three were delivered over the WebSocket (F7+F9, F9+F8+F7, F9). Escalations were recorded as `skipped_no_key` (F7 high severity, F9 unsure at 1.33x threshold).
- **Clips:** `clip.mp4` (973 KB) and `clip_circles.mp4` (950 KB) both served 200, `bytes=0-1023` returned 206, the 2048-byte suffix returned 206 with the correct bytes, HEAD returned the length, and an out-of-range request returned 416. The `moov` atom comes before `mdat` (faststart). An unsigned request returned 401.
- **Status and shutdown:** acknowledge worked, and the clean stop freed the GPU.
- **Networking:** Windows reached the WSL backend at `127.0.0.1:8000`.
