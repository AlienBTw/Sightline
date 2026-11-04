# Sightline

**A live workshop hazard watcher that circles the danger in the frame.**

Sightline is being built for the **Meta Global AI Developer Hackathon** (Indie track, Oct 26 – Nov 4 2026). Meta's V-JEPA 2.1 encoder runs locally on the camera stream. A trained concept-bottleneck head predicts 11 named hazard factors, then a risk score, and the per-patch activations are drawn as a circle around the hazard area. SAM 3.1 and Muse Spark 1.3 Contributor (`muse-spark-1.3-contributor`, Meta Model API) are called only when something is flagged.

> **Disclaimer:** Sightline is a decision-support demo. It is **not** a certified safety system and must not replace formal EHS controls, inspections, or regulatory compliance.

> **Status (Nov 4 2026):** the live pipeline, the backend, the Android app and a CCTV camera simulator were built during the event and run end to end: a simulated RTSP camera feeds the backend, the pipeline circles the hazard, and the app gets the alert with the circled clip. The legacy MVP sections further down (written Oct 13-14, before the event) are kept for reference only.

## What alerts today

| Factor | When it alerts | Circle |
|---|---|---|
| **F1** hand in a machine's danger zone | V-JEPA 2.1 features + the concept-bottleneck head (`h3_wide`), smoothed with hysteresis | Head patch map, trained on Fraunhofer hand boxes; SAM 3.1 snaps it to the hand on escalation |
| **F2** no head protection, **F5** no hi-vis vest | The head score is only a trigger (F2 0.5, F5 0.15). One **Muse Spark 1.3 Contributor** request with up to 3 frames must find a person without the protection on at least 2 of 3 frames. A rejection, timeout, bad answer, missing key or budget cap fails closed: the clip is stored, no alert is sent. | **SAM 3.1** `head` box (F2) and `person` torso (F5), drawn within +-1.5 s of the frame SAM saw; the event thumbnail is that frame. No SAM box: the head-map circle, marked `head_map_fallback` |
| F6-F10 | Scored and stored, muted (they raised false positives on live video) | - |
| F3, F4, F11 | Hidden (no usable labels, or not a head output) | - |

## Meta Model API budget

Every escalation (SAM 3.1 + Muse Spark, counted as one slot) is capped: **1 per incident, 2 per minute per camera, 3 per camera session** (camera start to stop; a looping video stays in one session) and **30 per local day** across cameras and restarts. `SIGHTLINE_ESC_MAX_PER_SESSION` and `SIGHTLINE_ESC_MAX_PER_DAY` override the last two (`0` means no calls). Only submitted calls count. A refused check is still saved, as `skipped_budget` with the most permanent reason first (`daily_cap`, `session_cap`, `incident_cap`, `minute_cap`); with no API key it is `skipped_no_key`. The daily counter lives in `data/live/escalation_budget.json` (gitignored).

## Meta tooling

| Model | Where it runs | What it does |
|---|---|---|
| **V-JEPA 2.1** ViT-L/16 384 (open weights, MIT) | Locally on the GPU, always on | Video features for the concept-bottleneck head (16-frame windows) |
| **SAM 3.1** (`sam-3.1`) | Meta Model API on escalation; local weights for offline boxes | Snaps live circles to the hand (F1), head (F2) and torso (F5). Stage 2 labelling: boxes on the PPE video clips |
| **Muse Spark 1.3 Contributor** (`muse-spark-1.3-contributor`) | Meta Model API, on escalation and for offline PPE labels | Verdict for flagged incidents, the F2/F5 2-of-3 verify gate, and the stage 2 relabel of the PPE video clips (helmet/vest), together with the SAM 3.1 boxes ([`training/08-video-ppe`](training/08-video-ppe/README.md)) |
| **Muse Glimmer 30B** 4-bit, `RedHatAI/Muse-Glimmer-30B-INT4` (Apache 2.0) | Locally, offline, on vLLM (tensor parallel), served as `muse-glimmer-30b-int4-vllm` | Stage 1: weak F1–F10 labels for the initial head ([`training/02-labeling`](training/02-labeling/README.md)) |

Labelling was two stages. **Stage 1:** local Muse Glimmer 30B 4-bit (`RedHatAI/Muse-Glimmer-30B-INT4`), served with vLLM split across both GPUs (tensor parallel) as `muse-glimmer-30b-int4-vllm`, gave weak F1–F10 labels for the initial head. **Stage 2:** SAM 3.1 boxes plus Muse Spark 1.3 Contributor (Meta Model API) relabelled the PPE video clips (helmet and vest) to fix the source shortcut. Ollama (`muse-glimmer:30b`) and SGLang were earlier attempts and did not produce the full label set.

## How it was built

The event code was written with Cursor, mostly through the headless Cursor CLI (`cursor-agent -p`) driven from prompt files, with every change checked by the test suites: `pytest` in WSL (pipeline, backend, training) and `tsc` + Jest for the app.

> **Product direction:** after the models are trained, a mobile app lets workshop supervisors watch their existing cameras live, get hazard push notifications and review flagged clips with the circle overlay. The Streamlit UI in this repo becomes the internal/dev console.

## Live pipeline (built during the event)

[`sightline_pipeline/`](sightline_pipeline/README.md) is the camera-to-alert path:
- RTSP, video file or uploaded video, through one ingest path
- V-JEPA 2.1 + concept-bottleneck head on sliding 16-frame windows (263 ms per step on an RTX 3090)
- smoothed per-factor incidents, with hazard circles
- SAM 3.1 and Muse Spark 1.3 Contributor (`muse-spark-1.3-contributor`) escalation through the Meta Model API
- H.264 clips (raw and with circles), event JSON and a SQLite index

```bash
~/vjepa-venv/bin/python -m sightline_pipeline.run --source data/demo/sightline_demo.mp4 --camera-id cam_demo
```

## Backend API (built during the event)

[`backend/`](backend/README.md) is the FastAPI service that the Android app and the dev console use:
- JWT login for judges (demo role) and an admin
- RTSP cameras and uploaded videos as looping virtual cameras, with one pipeline worker per camera (3 per GPU)
- streamed and resumable uploads
- an events feed with range-streamed clips (raw and with circles)
- a low-bandwidth MJPEG live view with circles
- WebSocket alerts and Expo push notifications
- a web dev console at `/console/`
- ngrok, watchdog and scheduled-task scripts for judging uptime

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File backend\scripts\start_backend.ps1   # then open http://127.0.0.1:8000/console/
```

## Android app (built during the event)

[`mobile/`](mobile/README.md) is the supervisor app (Expo SDK 58, React Native, TypeScript):
- sign in to your own backend URL; tokens kept in the Android Keystore, auto-refresh, HTTPS only
- a camera grid with live hazard chips, video upload, and RTSP cameras for admins
- a live view with the server-drawn circles
- a real-time alerts feed over WebSocket, with notifications that open the event
- clip review with the circles in sync with playback, the plain-language factors, the SAM 3.1 / Muse Spark 1.3 Contributor verdict, and acknowledge / dismiss

```powershell
cd mobile; npm install; npx expo run:android                      # development build
powershell -ExecutionPolicy Bypass -File scripts\build-apk.ps1    # release APK
```

## Target stack (Meta build)

| Stage | Tech |
|---|---|
| Always-on encoder | Meta **V-JEPA 2.1** (MIT), ViT-B 80M or ViT-L 300M, 384 px, 64-frame clips, dense per-patch features |
| Hazard head | Concept-bottleneck: 11 hazard factors (F1–F11) → risk; per-patch activations → circle/ellipse overlay |
| Anomaly (F11) | V-JEPA prediction error |
| Escalation only | **SAM 3.1** (mask) + **Muse Spark 1.3 Contributor** (`muse-spark-1.3-contributor`, reasoning) via the Meta Model API |
| Offline labeling (stage 1) | **Muse Glimmer 30B** 4-bit (`RedHatAI/Muse-Glimmer-30B-INT4`, Apache 2.0) served with **vLLM**, tensor parallel, name `muse-glimmer-30b-int4-vllm`. Weak F1–F10 labels for the initial head |
| Offline PPE video labels (stage 2) | **SAM 3.1** boxes + **Muse Spark 1.3 Contributor** (`muse-spark-1.3-contributor`, Meta Model API) relabel helmet/vest clips to fix the source shortcut ([`training/08-video-ppe`](training/08-video-ppe/README.md)) |
| UI / API | Streamlit + FastAPI + SQLite |

## Legacy MVP stack (pre-event reference)

| Stage | Tech |
|---|---|
| Detect | LLMDet (Hugging Face) — config-swappable backends |
| Measure | Homography + zones (metres) |
| Evidence gate | Any OpenAI-compatible VLM (sparse; attributes only, not final verdict) — local Muse Glimmer via Ollama by default |
| Decide | HistGradientBoosting + isotonic calibration |
| Report | One configured backend: `ollama` (local Muse Glimmer) **or** `meta_api` (Meta Model API) |
| UI | Streamlit (no model weights on the UI host) |
| API | FastAPI + SQLite |

MIT-compliant code. **No** Ultralytics / YOLO-World / YOLOv8 (AGPL/GPL).

## Legacy MVP quick start (pre-event reference)

```bash
cp .env.example .env
# edit .env — set SIGHTLINE_MODE, DETECTOR_BACKEND, REPORT_BACKEND, secrets

python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -e ".[dev]"

# Local Muse Glimmer (report writer / optional VLM gate)
ollama pull muse-glimmer:30b

# API
uvicorn sightline.api.main:app --reload --port 8000

# UI (separate terminal)
streamlit run app/streamlit_app.py
```

Or with Docker (local):

```bash
docker compose up --build
# Optional containerised Ollama:
# docker compose --profile ollama up --build
```

### Offline replay demo (no GPU)

```bash
# .env: SIGHTLINE_MODE=replay, DETECTOR_BACKEND=replay, TORCH_DEVICE=cpu
python scripts/demo_e2e_replay.py
uvicorn sightline.api.main:app --port 8000
streamlit run app/streamlit_app.py
```

## Configuration

Copy `.env.example` → `.env`. Required:

- `SIGHTLINE_MODE` — `live` or `replay` (explicit; no auto-switch)
- `DETECTOR_BACKEND` — exactly one of `llmdet`, `owlv2`, `rtdetr`, `dfine`, `replay`
- `REPORT_BACKEND` — exactly one of `ollama`, `meta_api`
- `API_URL` — where Streamlit calls the API (point at a remote host later without changing UI code)

Optional:

- `OLLAMA_BASE_URL` / `OLLAMA_MODEL` — default `http://localhost:11434/v1` / `muse-glimmer:30b`
- `META_API_BASE_URL` / `META_API_KEY` / `META_API_MODEL` — Meta Model API (Muse Spark)
- `VLM_BASE_URL` / `VLM_MODEL` / `VLM_API_KEY` — sparse evidence gate; empty URL = gate off

**Fail-loud:** misconfiguration or dependency failure returns structured errors. There are no silent fallbacks.

## Architecture (hosting-ready)

```
Streamlit (UI) ──HTTPS──► FastAPI (API)
                              ├── detector / geometry / rules / HistGB
                              ├── VLM_BASE_URL (optional, OpenAI-compatible: Ollama Muse Glimmer)
                              ├── Ollama Muse Glimmer (report, default)
                              └── Meta Model API (report, optional)
```

The UI never loads detector or VLM weights. `GET /system` reports torch/device info and stage latencies.

## Limitations and future work

- **F2/F5 need Muse Spark.** Retraining the head on Muse-labelled video learned the video source, not the helmet: on held-out Pexels clips F2 AUROC stayed at 0.19-0.58 while the head separated Voxel51 from Pexels clips at domain AUROC 0.54-1.00 (most runs 0.97-1.00). Stage 2 relabelled those PPE clips with SAM 3.1 boxes and Muse Spark 1.3 Contributor. Details: [`training/04-head/results/PPE_SHORTCUT.md`](training/04-head/results/PPE_SHORTCUT.md) and [`training/08-video-ppe`](training/08-video-ppe/README.md). `h3_wide` stays live and the verify gates decide F2/F5.
- **Data we still need:** same-camera, paired helmet / no-helmet and vest / no-vest footage, so the source of a clip cannot predict its label, plus a small hand-labelled out-of-distribution test set from real workshop cameras.
- **Model:** the V-JEPA 2.1 trunk is frozen. A partial fine-tune of its last blocks is planned once paired data exists.
- **Only F1 is validated from the head alone.** F6-F10 are muted. The demo footage comes from the held-out test splits of the training datasets plus Pexels clips; real-workshop accuracy is untested.
- **Verify gates cost API calls and seconds.** One check per incident, capped per session and day. A slow reply (30 s timeout) fails closed, so an incident can go unreported. A gate check and an F1 escalation share the one slot per incident.
- **SAM circles are a snapshot.** They are drawn only within +-1.5 s of the frame SAM saw, so the rest of a long clip shows the label without a circle.

## Licence + model weights

Code: MIT (see [LICENSE](LICENSE)). **Verify weight licences before going public.**

| Component | Package / weights | Notes |
|---|---|---|
| V-JEPA 2.1 | `facebookresearch/vjepa2` | MIT |
| Muse Glimmer 30B (stage 1 labels) | `RedHatAI/Muse-Glimmer-30B-INT4`, served as `muse-glimmer-30b-int4-vllm` | Apache 2.0. Ollama `muse-glimmer:30b` was an earlier attempt. The legacy report writer and VLM gate can still use that Ollama tag |
| SAM 3.1 / Muse Spark 1.3 Contributor | Meta Model API | `muse-spark-1.3-contributor`. Follow the API terms |
| LLMDet | Hugging Face `iSEE-Laboratory/llmdet_*` | Check model card licence |
| OWLv2 | Hugging Face OWL-ViT / OWLv2 | Apache-2.0 typical — confirm card |
| RT-DETR / D-FINE | HF checkpoints | Confirm card; avoid AGPL YOLO |
| HistGB | scikit-learn (BSD) | Local `.joblib` artifact |
| **Banned** | Ultralytics / YOLO-World / YOLOv8 | AGPL/GPL — not used |

Datasets: only permissive licences (CC BY / CC0 / MIT) because the hackathon rules grant the organisers a licence to submissions.

**Configuration.** The detector backend is chosen with the `DETECTOR_BACKEND` environment variable.

## Calibration & clips

- Floor-plan metres: `config/sites/demo_workshop/cameras.json` + `zones.json`
- Validate: `python scripts/validate_calibration.py`
- Capture checklist: [`docs/capture_checklist.md`](docs/capture_checklist.md)
- Clips go under `data/clips/` (gitignored). Only film people who agreed to be filmed.

## Security

Secrets (API keys, camera passwords) go in a local `.env` file. Copy `.env.example` to start.
