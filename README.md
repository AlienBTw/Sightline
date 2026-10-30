# Sightline

**A live workshop hazard watcher that circles the danger in the frame.**

Sightline is being built for the **Meta Global AI Developer Hackathon** (Indie track, Oct 26 – Nov 4 2026). Meta's V-JEPA 2.1 encoder runs locally on the camera stream. A trained concept-bottleneck head predicts 11 named hazard factors, then a risk score, and the per-patch activations are drawn as a circle around the hazard area. SAM 3.1 and Muse Spark (Meta Model API) are called only when something is flagged.

> **Disclaimer:** Sightline is a decision-support demo. It is **not** a certified safety system and must not replace formal EHS controls, inspections, or regulatory compliance.

> **Status:** the code in this folder is the **pre-event MVP** (written Oct 13–14, before the Meta event started on Oct 26). It is kept as **reference only**. The Meta build plan in [`../PLAN.md`](../PLAN.md) rebuilds the core pipeline during the event window (the participant guide reportedly requires new work).

## Target stack (Meta build)

| Stage | Tech |
|---|---|
| Always-on encoder | Meta **V-JEPA 2.1** (MIT), ViT-B 80M or ViT-L 300M, 384 px, 64-frame clips, dense per-patch features |
| Hazard head | Concept-bottleneck: 11 hazard factors (F1–F11) → risk; per-patch activations → circle/ellipse overlay |
| Anomaly (F11) | V-JEPA prediction error |
| Escalation only | **SAM 3.1** (mask) + **Muse Spark** (reasoning) via the Meta Model API |
| Offline labeling | **Muse Glimmer 30B** (Apache 2.0, Q4) through **Ollama**, OpenAI-compatible at `http://localhost:11434/v1` |
| UI / API | Streamlit + FastAPI + SQLite |

## Current MVP stack (reference)

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

## Quick start

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

## Licence + model weights

Code: MIT (see [LICENSE](LICENSE)). **Verify weight licences before going public.**

| Component | Package / weights | Notes |
|---|---|---|
| V-JEPA 2.1 | `facebookresearch/vjepa2` | MIT |
| Muse Glimmer 30B | Ollama `muse-glimmer:30b` | Apache 2.0 |
| SAM 3.1 / Muse Spark | Meta Model API | Follow the API terms |
| LLMDet | Hugging Face `iSEE-Laboratory/llmdet_*` | Check model card licence |
| OWLv2 | Hugging Face OWL-ViT / OWLv2 | Apache-2.0 typical — confirm card |
| RT-DETR / D-FINE | HF checkpoints | Confirm card; avoid AGPL YOLO |
| HistGB | scikit-learn (BSD) | Local `.joblib` artifact |
| **Banned** | Ultralytics / YOLO-World / YOLOv8 | AGPL/GPL — not used |

Datasets: only permissive licences (CC BY / CC0 / MIT) because the hackathon rules grant the organisers an any-purpose licence to submissions. See [`../17-meta-hackathon-rules.md`](../17-meta-hackathon-rules.md).

Detector backends are selected only via `DETECTOR_BACKEND` — never auto-cascaded.

## Calibration & clips

- Floor-plan metres: `config/sites/demo_workshop/cameras.json` + `zones.json`
- Validate: `python scripts/validate_calibration.py`
- Capture checklist: [`docs/capture_checklist.md`](docs/capture_checklist.md)
- Clips go under `data/clips/` (gitignored). Only film people who agreed to be filmed.

## Development

- All product code lives in this repository.
- Local `progress/` trackers are gitignored (team status only).
- Never commit `.env`, API keys, RTSP passwords, or raw site footage.
