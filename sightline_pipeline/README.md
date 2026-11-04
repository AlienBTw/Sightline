# sightline_pipeline: live inference with hazard circles

The live part of Sightline, written during the event window. It takes any camera source, runs Meta **V-JEPA 2.1** plus the trained concept-bottleneck head on sliding windows, raises hazard incidents, circles the flagged area, escalates selected incidents to **SAM 3.1** and **Muse Spark 1.3 Contributor** (`muse-spark-1.3-contributor`), and writes clips, event JSON and a SQLite index for the backend and the mobile app.

The code lives in a new top-level package, `sightline_pipeline/`. The older `sightline/` package (pre-event MVP, Oct 13-14) stays reference-only. The package reuses the training code directly: `training/03-features/extract_features.py` for the encoder and preprocessing, and `training/04-head/train_head.py` for the head. Live inference therefore uses the same resize, normalisation, token pooling and head as training.

```
camera (RTSP / HTTP) ─┐
video file / upload ──┴─> ingest.VideoSource ──> ring buffer (clips) + WindowSampler (16 frames, every 0.16 s)
                                                     │  about once per second (stride_s)
                                                     v
                         model.HazardModel: V-JEPA 2.1 ViT-L/16 384 (fp16) -> 12x21 tokens -> head (MIL over last 2 windows)
                                                     │  per-factor probability + patch maps
                                                     v
                         events.IncidentTracker: EMA, per-factor thresholds + hysteresis, incident merge
                                                     │
                 localize: map -> circle         escalation (policy): SAM 3.1 refine + Muse Spark verdict (async); F2/F5 verify gates
                                                     v
                         storage: clip.mp4, clip_circles.mp4, thumb.jpg, event.json, sightline_events.db
                                                     v
                         on_event(record) callback -> backend (push notification, alerts feed)
```

## Modules

| File | What it does |
|---|---|
| `ingest.py` | `VideoSource`: one OpenCV/FFmpeg ingest path for `rtsp://`, `http(s)://` and files (an uploaded video is a file). Files are paced to real time (`Pacer`), so an upload behaves like a live camera. `--loop` turns a file into a never-ending virtual camera. Live streams reconnect with exponential backoff (RTSP over TCP). A reader thread drops the oldest frames when the consumer falls behind, like a real camera, and drops nothing in offline mode. Credentials in RTSP URLs are redacted in logs and records. |
| `model.py` | `HazardModel`: loads the V-JEPA 2.1 encoder (fp32 weights, fp16 autocast) and the `h3_wide` head checkpoint, and runs one step per window. Its outputs are factor probabilities, per-factor evidence and attention maps on the 12 x 21 grid, and latency. F1 adds a single-frame image pass (see Localization). |
| `events.py` | `IncidentTracker`: EMA smoothing (alpha 0.6), a factor turns on after 2 steps at or above its threshold and off after 2 steps below 0.7 x threshold. An incident is one alert and one clip. It closes 4 s after the last active factor and is split at 20 s so clips stay short. |
| `localize.py` | Turns a map into a circle: upsample, keep the connected region of at least 0.7 x peak around the peak, take the mass-weighted centre, set the radius from the region's area, then clamp it. Also holds the circle/box geometry used for evaluation. |
| `escalation.py` | `EscalationPolicy`, `SamRefiner` (SAM 3.1), `MuseVerifier` (Muse Spark) and `Escalator` (thread pool, never blocks the live loop). |
| `verify_gate.py` | F2 and F5 Muse Spark verify gates: one batched request from the last few above-trigger seconds, 2-of-3 vote, fail closed, SAM 3.1 circles for the head and the torso. |
| `verify_gate_eval.py` | CLI that scores that same prompt on a folder of still frames (`--factor F2` or `F5`). |
| `storage.py` | Ring buffer, H.264 clip writer (ffmpeg pipe, OpenCV fallback), `EventStore` (folders and SQLite). |
| `service.py` | `SightlinePipeline.process(source, camera_id, on_event, stop)`: the reusable service the backend calls, one instance per camera. |
| `run.py` | The CLI. |
| `make_demo.py` | Builds the demo video from held-out CC BY 4.0 test footage, with attribution. |
| `eval_circles.py` | Measures circle quality on the Zenodo test hand boxes. |
| `config.yaml` | All defaults (thresholds, smoothing, clip length, localization, escalation policy). |

## How to run

Run these in WSL (Ubuntu-24.04) with `~/vjepa-venv`, from the repo root. The venv needs torch, `~/vjepa2` on disk, `opencv`, `av`, `imageio-ffmpeg` and `python-dotenv`.

```bash
cd "/mnt/c/projects/Meta Global AI Developer Hackathon/Sightline"
PY=~/vjepa-venv/bin/python
$PY -m sightline_pipeline.make_demo                                   # -> data/demo/sightline_demo.mp4 + ATTRIBUTION.md
$PY -m sightline_pipeline.run --source data/demo/sightline_demo.mp4 --camera-id cam_demo \
    --session-video data/live/session_demo.mp4                        # paced like a live camera
$PY -m sightline_pipeline.run --source data/demo/sightline_demo.mp4 --fast   # as fast as possible
$PY -m sightline_pipeline.run --source data/demo/sightline_demo.mp4 --loop   # virtual camera, runs until stopped
$PY -m sightline_pipeline.run --source rtsp://<phone-ip>:8554/<path> --camera-id phone_cam   # Android IP-camera app
$PY -m sightline_pipeline.eval_circles                                # circle metrics -> sightline_pipeline/results/
$PY -m sightline_pipeline.verify_gate_eval --frames <dir> --labels labels.csv --factor F5 --dry-run
# labels.csv is file,vest for F5 (1 = wearing hi-vis) or file,head for F2 (1 = wearing head protection).
# Omit --dry-run to call Muse Spark 1.3 Contributor (muse-spark-1.3-contributor).
# --model overrides that id. --batch-size 1 or 3. --show-raw prints the model answer, not the key.
```

Options: `--device cuda:1`, `--no-escalation`, `--max-seconds N`, `--config my.yaml` (overrides `config.yaml`), `--data-dir`. Detached launcher with logs in `F:\models\logs`: `F:\models\wsl\run_pipeline_jobs.sh demo|eval|run ...`.

Using it from the backend:

```python
from sightline_pipeline.config import load_config
from sightline_pipeline.model import HazardModel
from sightline_pipeline.service import SightlinePipeline
from sightline_pipeline.ingest import VideoSource
cfg = load_config(); pipe = SightlinePipeline(cfg, HazardModel(cfg))
pipe.process(VideoSource("rtsp://..."), camera_id="cam_02", on_event=send_push, stop=stop_event)
```

`process()` also takes an optional `on_step(info)` callback. The backend uses it for the live view: after each model step it gets `{camera_id, t, wall, probs, ema, active, incident}`, where `active` holds the circles of the factors that are currently on. An error inside the hook is logged and never stops the loop. The product backend that runs one pipeline per camera is in [`backend/`](../backend/README.md).

## Output (under `data/live/`, gitignored)

```
events/<event_id>/clip.mp4           raw clip, 4 s before the incident to 4 s after (H.264, 960 px wide, faststart)
events/<event_id>/clip_circles.mp4   the same clip with circles burned in
events/<event_id>/thumb.jpg          peak frame with circles
events/<event_id>/event.json         event record
sightline_events.db                  SQLite index: event_id, camera_id, times, risk_max, top_factor, factors, status, escalated
runs/<stamp>_<camera>_summary.json   latency / throughput summary; runs/..._steps.jsonl per-step scores
```

`event.json` follows the shape in the mobile plan: `event_id`, `camera_id`, `started_at`/`ended_at` (ISO with offset), `risk_max`, `top_factor`, `factors`, `segments` (per factor: start, end, peak, threshold), `status`, `clip`/`clip_circles`/`thumbnail`, `escalation` (policy, backends, calls, skipped), and `overlays`. `overlays` is a list of `{t_ms, factor, score, raw, risk, circle{cx, cy, r, w, h, norm{cx, cy, rx, ry, angle_deg}}}`, so the app can draw circles client-side over the clean clip.

**Risk:** each active factor's probability is mapped to a confidence where its own threshold equals 0.5, and the confidences are combined with noisy-OR. Thresholds vary a lot between factors, so raw sigmoids can't be compared.

## Escalation: SAM 3.1 and Muse Spark

The policy is explicit, set in `config.yaml` under `escalation`. A factor is escalated when it turns on inside an incident and either:

- it's **high severity** (F1 hand in danger zone, F6 fall, F7 unstable load, F10 fire), or
- the head is **unsure**, meaning its smoothed score is between 1.0 and 1.6 times its threshold.

There is at most 1 escalation per incident and 2 per minute per camera. Everything else is decided by V-JEPA and the head alone. A hard Meta API budget also caps submitted escalations at 3 per camera session and 30 per local day (`max_per_session` / `max_per_day`, env `SIGHTLINE_ESC_MAX_PER_SESSION` / `SIGHTLINE_ESC_MAX_PER_DAY`); over the cap the event is still saved and the check is recorded as `skipped_budget` with no API call.

1. **SAM 3.1 refinement.** It segments the factor's concept (`hand`, `fire`, `forklift`, `open electrical panel`, `person`) in the peak frame. The head circle is then snapped to the enclosing circle of the SAM box whose centre lies within 1.5 radii of it (`source: "sam3.1"`). Backends:
   - `SAM_BACKEND=api` (default): Meta Model API, `POST /v1/responses` with model `sam-3.1`. The streamed box tokens are parsed (`<|box;x1=..;y1=..|>`) and masks are not decoded.
   - `SAM_BACKEND=local`: open weights from `facebook/sam3.1` (gated, SAM License) through the `sam3` package, with checkpoint `SAM3_CHECKPOINT`.
   - `SAM_BACKEND=off`.
2. **Muse Spark verdict.** The peak frame, with the circle drawn, goes to Muse Spark 1.3 Contributor (`muse-spark-1.3-contributor`) with the factor definition (`/v1/chat/completions`, image input). It answers with JSON: `{confirmed, severity 1-5, reason, action}`. `MUSE_BACKEND=api|off`. The model id is `MUSE_SPARK_MODEL` if that is set, otherwise `escalation.muse_model`, otherwise `muse-spark-1.3-contributor`.
3. **F2 and F5 verify gates.** `escalation.verify_gates` (env `SIGHTLINE_VERIFY_GATES=on|off`, and `SIGHTLINE_VERIFY_GATE_F2` / `SIGHTLINE_VERIFY_GATE_F5` for one factor). F2 and F5 are not in `muted_factors`, but the tracker still treats them as gated: they are scored and stored, and they can open or join an incident only through the gate. They are published only after Muse Spark confirms them. F1 is unchanged.

   When the smoothed score crosses the factor's trigger (F5 default 0.15, F2 default 0.5) the pipeline keeps that frame, with the stream time and the smoothed score. F2's threshold is 0.5 because h3_wide F2 scores sit around 0.72-0.76 on the Pexels bare-head clip and 0.89-1.0 on the demo; Glimmer labelled F2 present on about 80% of clips, so the head score is only a person-present trigger and Muse decides. At incident close, factors that triggered and have not been checked yet go out in **one** Muse Spark 1.3 Contributor (`muse-spark-1.3-contributor`) chat completion. The frames are not spread across the whole incident or the pre-roll. Up to `max_frames` (default 3) JPEGs, long side 768, come only from where every requested factor's smoothed score was still at or above its trigger, spread across the last `recent_s` seconds (default 3) of that stretch: the frames just before the gate fires. If none qualify, the factor fails closed with `no_frames`. The prompt asks only the questions for those factors. Per frame the JSON is `{person_visible, wearing_hi_vis_vest}` for F5 and/or `{person_visible, wearing_head_protection}` for F2. Head protection means a hard hat, safety helmet or bump cap. A normal cap, hood, hair net or bare head is not head protection. If no person or head is visible, or the model is unsure, it answers true (protected) so the vote fails closed.

   A factor alerts only when at least `min_votes` (default 2) frames have `person_visible: true` and the protective item false. Frames with no visible person do not count. A missing key, HTTP error, timeout, empty reply, bad JSON or the wrong frame count fails closed and does not publish that factor. The reply is not retried. On a failure the gate record and the Muse call store a diagnostic: HTTP status, `finish_reason`, usage token counts when the response included them, and the first 1500 characters of the assistant content and of `reasoning` or `reasoning_content`. Headers and the API key are not stored. `max_tokens` defaults to 4000 because reasoning tokens count against that budget. The verdict is cached per incident and per factor, so the same factor is not asked twice. The request takes one slot through `EscalationPolicy.reserve` and `EscalationBudget` (`max_per_incident` 1, `max_per_minute` 2, `max_per_session` 3, `max_per_day` 30). A refusal is stored as `skipped_budget` with reason `incident_cap`, `minute_cap`, `session_cap` or `daily_cap`, and the factor is not published. No API key records `skipped_no_key` and spends nothing.

   After Muse confirms a factor, SAM 3.1 runs in that same budget slot, on one frame (a frame Muse marked hazardous, otherwise the newest sample). Prompts are short nouns: the live demo got no box for `bare head` and `person head without helmet`, while `head` and `person` return boxes. F2 tries `head`, then `person`. A `head` box is the F2 circle. A `person` box becomes an F2 circle on the top 20% of the box, centred horizontally, with the radius taken from the box width. F5 tries `person` and circles the torso: the box's centre x, 40% of the way down the box, radius from the box width, clamped to the localization min and max radius. Boxes are cached per frame and prompt inside that evaluation, so `person` is sent once even when F2 and F5 both need it. The highest-confidence box is used, and the largest box breaks a tie. `escalation.verify_gates.api_calls.sam` counts real SAM requests, not cache hits. The frame SAM ran on is stored as `sam_t_stream`. Its circle replaces the head-map circle only on overlays within `sam_hold_s` (default 1.5 s) of that frame, which is what `event.json` and `clip_circles.mp4` both draw. Overlays farther away keep the factor and drop the circle, so a person who has moved is not circled on empty floor. If SAM returns nothing or errors, the head-map circle is kept on every overlay with source `head_map_fallback`. Each factor records `circle_source` (`sam3.1` or `head_map_fallback`), `sam_t_stream`, `sam_status`, `sam_prompt`, and how many boxes each prompt returned. When every published factor is gated and SAM drew a circle, `thumb.jpg` is that SAM frame with the SAM circles, and `peak_t_stream` is `sam_t_stream`. An F1 incident still uses its risk-peak frame.

   The record is `escalation.verify_gates`: `{factors: {F2|F5: {status, votes, n_frames, reason, verified_by, circle_source, sam_t_stream, sam_status, sam_prompt, sam_boxes}}, api_calls: {muse, sam}, suppressed: [...]}`. A failure also stores `diagnostic` on that object and on the Muse call. `status` is `confirmed`, `rejected`, `failed`, `skipped_budget` or `skipped_no_key`. A confirmed factor sets top-level `verified_by` to `Muse Spark`. If every factor in the incident was gated and none was confirmed, the clip is still written with `status: suppressed` and `on_event` is not called. The gate does nothing when escalation is disabled (`verify_gates_ignored`).

Auth for both is `MODEL_API_KEY` (`META_API_KEY` is accepted as an alias). The base URL comes from `META_API_BASE_URL` and defaults to `https://api.meta.ai/v1`. Without a key, every escalation is still recorded with the reason it fired and `status: "skipped_no_key"`. Network errors are stored as `status: "error"`. Escalation never stops the live loop. With the verify gates on and no key, F2 and F5 are recorded and not published.

## Results (2026-10-31, RTX 3090)

**Demo run, paced in real time** (`data/demo/sightline_demo.mp4`, 59.4 s, 1280x720 at 25 fps, GPU 0):

| Metric | Value |
|---|---|
| Stream vs wall time | 59.4 s of video processed in 62.0 s, including writing the final clip; 0 frames dropped; ingest 24 fps |
| Model step (16-frame window, ViT-L 384x672, fp16) | encoder p50 **220 ms**; head 1.2 ms; F1 image pass 42 ms; **total p50 263 ms / p95 283 ms** |
| Capacity | 3.6 model steps/s per 3090. With stride 1 s, one GPU uses about 26% for one camera, so about 3 cameras per GPU at 1 step/s each |
| Offline (`--fast`) | the same video in 28.6 s (about 2x real time, decode-bound) |
| Events | 3 incidents (one per 20 s window of hazards), each with a raw clip, a circles clip, a thumbnail and JSON |

On the second GPU the same step measured 546 ms p50, which is still under the 1 s stride. That's probably a slower PCIe slot or clocks; it hasn't been investigated.

**Circle quality, F1 hand in danger zone** (`eval_circles.py`): 3,590 Zenodo **test** images that contain hands (never trained on), 1920x1080, the F1 head map on the saved V-JEPA 2.1 features (T=1 image pass, the same as live). Results are in `results/circles_zenodo_test.json`.

| Map, region threshold | Centre inside a hand box | Hand-box centres inside circle | IoU circle vs box | IoU vs box's enclosing circle | Radius (/ diagonal) |
|---|---|---|---|---|---|
| **evidence, 0.7 (live default)** | **0.937** | **0.838** | **0.466** | **0.669** | 0.103 |
| evidence, 0.55 | 0.895 | 0.871 | 0.364 | 0.584 | 0.119 |
| evidence, 0.4 | 0.863 | 0.909 | 0.291 | 0.482 | 0.136 |
| attention, 0.55 | 0.994 | 0.526 | 0.359 | 0.281 | 0.042 |
| baseline: same radius at the mean training hand position | 0.140 | 0.260 | 0.109 | - | - |

Attention points at the hand almost every time, but its circles are too small. The evidence map at 0.7 is the best trade-off. In the live demo the F1 map is computed from a still-image pass of the newest frame, because F1's patch loss was trained on single images. Without that pass, the 16-frame video window drew the F1 circle on the forearm instead of the hand.

**Visual check of the demo:** PNG frames were pulled from `clip_circles.mp4`.
- **F1:** the circle lands on the hand at the saw.
- **F8:** the circle sits on the walkway.
- **F7 is off.** It circles a bright floor reflection instead of the forklift load.
- **F9 is inconsistent:** sometimes it's on the machine, sometimes on a person.

**Escalation in the demo:** 6 escalations fired across 3 incidents:
- **high severity:** F7 twice, F1 once
- **unsure score:** F9 three times, at score/threshold 1.33-1.45

The policy skipped the rest: F9 at 5.1x and F8 at 1.66x its threshold were confident, and F6 was rate-limited by the 6-per-minute cap. All are recorded with `skipped_no_key`, because no Meta Model API key is configured. That run had no API key. SAM 3.1 and Muse Spark have run live since Nov 4 (next section).

## Live verify-gate results (2026-11-04, simulated RTSP camera into the backend)

| Clip | Gate result | Circles |
|---|---|---|
| Factory scene of `data/demo/sightline_demo.mp4` (workers without helmets or vests) | F2 and F5 confirmed, 3 of 3 frames | SAM 3.1 `head` circle on the worker's bare head |
| Pexels 8987012, angle grinder (`data/demo_extra/`) | F2 and F5 confirmed, 3 of 3 frames; F1 from the head | SAM head circle, SAM torso circle, F1 circle on the hand |
| Table-saw close-up of the demo video (only hands visible) | Rejected, 0 of 3 frames | none (correct fail-closed) |

- **Budget:** with the session cap of 3, exactly 3 Meta API slots were used per camera session; every later incident was stored as `skipped_budget` (`session_cap`) while F1 kept alerting.
- **Bugs found live and fixed:** frames sampled from the pre-roll (empty frames led to rejections), long SAM prompts returning no box, one SAM circle pasted over a whole clip with moving people, and a misleading `minute_cap` reason after the session cap.
- **Timeout:** one Muse request hit the 30 s timeout and failed closed (`failed`, not published).

## Configuration choices for live use (and why)

- **Thresholds** are the validation-chosen ones saved in the head checkpoint (`h3_wide`), with a floor of 0.05. **F6 and F7 are raised to 0.5**: their validation thresholds (0.026 and 0.007) were picked on whole clips with up to 8 windows pooled, from very few validation positives, while live runs pool 2 windows.
- **F2 (no head protection) and F5 (no hi-vis) are verify-gated.** They're scored and stored, and they can open an incident only through `escalation.verify_gates`. About 80% of their training labels are "present", so the head score is only a person-present trigger. Muse Spark decides, and a factor is published only when Muse confirms it. F6-F10 stay in `muted_factors`.
- **F3 and F4 are hidden** because they're untrained. F11 isn't a head output.

## Known limits

- **Only F1 (hand in danger zone) alerts from the head score alone;** F2 and F5 need a Muse Spark confirmation. In our Oct 31 demo run, the other factors (F6-F9) raised confident (0.98-0.99) false positives. Their offline AUROC was measured on same-source test clips with weak labels and did not carry over. Scores are not calibrated. Treat non-F1 alerts as experimental.
- **Circles are validated only for F1** (table-saw hands). The F7/F8/F9/F6/F10 circles come from the head's MIL attention, which was trained with clip-level labels only. Looking at the demo, F7 and F9 circles can land on the wrong object, such as a bright floor reflection instead of the forklift load. SAM 3.1 now runs live with a key and gives good F1 hand, F2 head and F5 torso circles; the F6-F10 circles are still unvalidated. Each event records which factors have validated localization (`localization`).
- **False positives on the demo:** F9 (open panel) fires at the start of a safe-walkway clip and on the table saw, and F6 (fall) fires on the table-saw frames.
- The demo footage comes from the same two datasets the head was trained on, from their held-out test splits. Out-of-distribution performance on real workshop cameras is untested.

## Future improvements

- F2/F5 from the head alone: retrains on Muse-labelled video learned the video source instead of the helmet (see [`../training/08-video-ppe`](../training/08-video-ppe/README.md) and [`PPE_SHORTCUT.md`](../training/04-head/results/PPE_SHORTCUT.md)). Next: same-camera, paired helmet / no-helmet and vest / no-vest footage, then a partial V-JEPA 2.1 fine-tune.
- Fire/smoke data with a permissive licence for F10.
- A small hand-labeled, out-of-distribution test set from real workshop cameras, giving honest per-factor and circle numbers.
- Train F3 (eye protection) and F4 (gloves).
- Patch-level supervision (boxes or SAM 3.1 pseudo-masks) for F6-F10, so their circles can be validated the way F1's are.
- F11 abnormal machine behaviour from the V-JEPA 2.1 predictor (surprise score).
- A smoother live overlay. The backend live view draws the circles server-side at 1 fps over MJPEG today; WebRTC would allow a higher frame rate.

## Tests

`python -m pytest sightline_pipeline/tests -q` runs the offline tests with no GPU: ingest pacing, loop and reconnect, events (hysteresis, merging, max length, muted/hidden factors), circle geometry, storage and SQLite, escalation policy and parsing, the no-key fallback, the F2/F5 verify gates with mocked HTTP (2-of-3 vote, frames without a person, fail closed on errors and timeouts, one combined request, per-incident cache, budget caps and skip reasons, SAM head/torso circles and the head-map fallback, the +-1.5 s circle window, the SAM-frame thumbnail, and the Muse model id), an end-to-end service run with a fake model that writes real clips and events, and the `on_step` live-view hook.

## Data and licences

The demo video uses held-out test clips from Voxel51 Safe_and_Unsafe_Behaviours (Önal & Dandıl 2024, CC BY 4.0) and test frames from the Fraunhofer IPA sliding-table-saw hand dataset (Steckhan et al. 2026, Zenodo, CC BY 4.0). Per-segment attribution is written next to the video in `ATTRIBUTION.md`. SAM 3.1 is used under the SAM License, through the Meta Model API or the gated open weights.
