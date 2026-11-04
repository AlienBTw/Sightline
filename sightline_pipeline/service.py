"""SightlinePipeline: the reusable service the backend calls (one instance per camera).

    pipe = SightlinePipeline(cfg, model=HazardModel(cfg))
    summary = pipe.process(VideoSource(uri), camera_id="cam_01", on_event=callback, stop=threading.Event())

Per frame: ring buffer + 16-frame sampler. About every `stride_s` of stream time: one model
step -> IncidentTracker -> circles for active factors -> escalation on factor-on (policy) ->
clip writing. When an incident closes (+ post roll) the event folder and SQLite row are written
and `on_event(record)` is called (the backend turns that into a push notification).
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import localize, render
from .config import FACTOR_LABELS, FACTORS_YAML, effective_thresholds, resolve_path
from .escalation import EscalationPolicy, Escalator, MuseVerifier, SamRefiner
from .events import IncidentTracker
from .ingest import VideoSource, redact
from .risk import confidence, noisy_or
from .storage import ClipWriter, EventStore, RingBuffer, resize_w


class WindowSampler:
    """Keeps the last `n` frames sampled every `interval` seconds of stream time (training used
    frame step 4 at 25 fps = 0.16 s), already resized for the encoder."""

    def __init__(self, n: int, interval: float, prep):
        self.n, self.interval, self.prep = n, interval, prep
        self.frames: collections.deque = collections.deque(maxlen=n)
        self.next_t = None

    def push(self, t: float, img) -> bool:
        if self.next_t is None or t >= self.next_t:
            self.frames.append(self.prep(img))
            self.next_t = t + self.interval if self.next_t is None or t - self.next_t > self.interval else self.next_t + self.interval
            return True
        return False

    def ready(self) -> bool:
        return len(self.frames) == self.n

    def window(self) -> np.ndarray:
        return np.stack(self.frames)


@dataclass
class OpenIncident:
    inc: object
    event_id: str
    clip_t0: float
    raw: ClipWriter
    clip_size: tuple
    overlays: list = field(default_factory=list)
    escalations: list = field(default_factory=list)   # (factor, t, reason, future)
    skipped: list = field(default_factory=list)
    peak: tuple = (0.0, None, None, [])               # (risk, t, full-res frame, overlays at full res)
    risk_max: float = 0.0
    wall0: str = ""
    frames: int = 0


def _factor_defs() -> dict:
    try:
        fs = yaml.safe_load(open(FACTORS_YAML, encoding="utf-8"))["factors"]
        return {f["id"]: (f["name"].replace("_", " "), f["description"]) for f in fs}
    except OSError:
        return {}


class SightlinePipeline:
    def __init__(self, cfg: dict, model, store: EventStore | None = None, escalator: Escalator | None = None):
        self.cfg, self.model = cfg, model
        self.store = store or EventStore(resolve_path(cfg["storage"]["data_dir"]))
        ev = cfg["events"]
        self.hidden = set(ev.get("hidden_factors", []))
        self.thr = effective_thresholds({f: t for f, t in model.ckpt_thresholds.items()}, cfg)
        self.alert_factors = [f for f in model.factors if f not in self.hidden]
        es = cfg["escalation"]
        self.escalator = escalator or Escalator(
            EscalationPolicy(tuple(es["high_severity"]), tuple(es["uncertain_band"]), es["max_per_incident"],
                             es["max_per_minute"]),
            SamRefiner(), MuseVerifier(_factor_defs()), enabled=es["enabled"], timeout_s=es["timeout_s"])
        self._finalizers: list[threading.Thread] = []

    # ------------------------------------------------------------------ helpers
    def _map_mode(self, f: str) -> str:
        mm = self.cfg["localization"].get("map_mode", {})
        return mm.get(f, mm.get("default", "attention_x_evidence"))

    def circles_for(self, out: dict, f: str, w: int, h: int) -> list:
        lc = self.cfg["localization"]
        m = self.model.factor_map(out, f, self._map_mode(f))
        return localize.circles_from_map(m, w, h, lc["rel_threshold"], lc["min_radius_frac"], lc["max_radius_frac"],
                                         lc["radius_scale"], lc["max_circles"])

    def _draw_one(self, frame, f, circle, score):
        ov = [{"factor": f, "score": score, "circle": circle.to_dict()}]
        return render.draw_overlays(frame, ov)

    # ------------------------------------------------------------------ main loop
    def process(self, source: VideoSource, camera_id: str = "cam_01", on_event=None, stop: threading.Event | None = None,
                max_seconds: float | None = None, session_video: str | None = None, log=print, on_step=None) -> dict:
        """on_step(info) is optional (live preview): called after every model step with
        {"camera_id", "t", "wall", "probs", "ema", "active": [overlay dicts, normalised circles], "incident"}."""
        s, c = self.cfg["sampling"], self.cfg["clips"]
        sampler = WindowSampler(self.model.n_frames, float(s["sample_interval_s"]), self.model.preprocess)
        ev = self.cfg["events"]
        tracker = IncidentTracker(self.thr, ev["ema_alpha"], ev["min_on_steps"], ev["min_off_steps"], ev["off_ratio"],
                                  ev["merge_gap_s"], ev["max_incident_s"], muted=tuple(ev.get("muted_factors", [])),
                                  hidden=tuple(self.hidden))
        ring = RingBuffer(float(c["pre_s"]) + 1.0)
        stride = float(s["stride_s"])
        next_step, last_out, last_step_t = None, None, None
        open_inc: OpenIncident | None = None
        closing: list[OpenIncident] = []        # closed incidents still recording post-roll
        lat, steps, events = [], [], []
        session_w, active_ov = None, []
        wall_start, t_first, t_last, n_frames = time.time(), None, 0.0, 0
        clip_w = int(c["width"])
        log(json.dumps({"msg": "start", "camera": camera_id, "source": redact(source.uri), "kind": source.kind,
                        "realtime": source.realtime, "thresholds": {f: round(v, 4) for f, v in self.thr.items()},
                        "escalation": self.escalator.status()}))
        try:
            for fr in source:
                if stop is not None and stop.is_set():
                    break
                t = fr.t
                if t_first is None:
                    t_first = t
                t_last = t
                n_frames += 1
                if max_seconds and t - t_first >= max_seconds:
                    break
                small = resize_w(fr.image, clip_w)
                ring.push(t, small)
                sampler.push(t, fr.image)
                fps = source.fps or 25.0
                # ---- model step
                if sampler.ready() and (next_step is None or t >= next_step):
                    next_step = t + stride if next_step is None or t - next_step > stride else next_step + stride
                    out = self.model.step(sampler.window())
                    total_ms = sum(out["latency_ms"].values())
                    lat.append(out["latency_ms"] | {"total": total_ms})
                    probs = {f: p for f, p in out["probs"].items() if f not in self.hidden}
                    trans = tracker.update(t, probs)
                    sm = tracker.smoothed()
                    steps.append({"t": round(t, 2), "p": {f: round(v, 4) for f, v in probs.items()},
                                  "ema": {f: round(sm[f], 4) for f in probs}})
                    last_out, last_step_t = out, t
                    H, W = fr.image.shape[:2]
                    for tr in trans:
                        if tr[0] == "open":
                            inc = tr[1]
                            stamp = dt.datetime.fromtimestamp(fr.wall).strftime("%Y%m%dT%H%M%S")
                            eid = self.store.next_id(camera_id, stamp)
                            pre = ring.since(t - float(c["pre_s"]))
                            sz = (small.shape[1], small.shape[0])
                            w = ClipWriter(self.store.event_dir(eid) / "clip.mp4", min(fps, float(c["fps_cap"])), sz)
                            for _, im in pre:
                                w.write(im)
                            open_inc = OpenIncident(inc, eid, pre[0][0] if pre else t, w, sz,
                                                    wall0=dt.datetime.fromtimestamp(fr.wall - (t - (pre[0][0] if pre else t))).astimezone().isoformat(timespec="seconds"),
                                                    frames=len(pre))
                            log(json.dumps({"msg": "incident_open", "event_id": eid, "t": round(t, 2)}))
                        elif tr[0] == "factor_on" and open_inc is not None:
                            seg = tr[2]
                            f = seg.factor
                            circ = self.circles_for(out, f, W, H)
                            n_esc = len(open_inc.escalations)
                            if circ and self.escalator.enabled:
                                go, why = self.escalator.policy.decide(f, sm[f], self.thr[f], n_esc, time.time())
                                if go:
                                    fut = self.escalator.submit(fr.image.copy(), f, circ[0], sm[f], self.thr[f], self._draw_one)
                                    open_inc.escalations.append((f, t, why, fut))
                                else:
                                    open_inc.skipped.append({"factor": f, "t": round(t, 2), "reason": why})
                            log(json.dumps({"msg": "factor_on", "event_id": open_inc.event_id, "factor": f,
                                            "score": round(sm[f], 4), "thr": round(self.thr[f], 4), "t": round(t, 2)}))
                        elif tr[0] == "close" and open_inc is not None:
                            closing.append(open_inc)
                            log(json.dumps({"msg": "incident_close", "event_id": open_inc.event_id, "t": round(t, 2)}))
                            open_inc = None
                    # circles for factors that are on (current incident)
                    active_ov = []
                    on = [f for f, st in tracker.state.items() if st.on]
                    for f in on:
                        for ci in self.circles_for(out, f, small.shape[1], small.shape[0]):
                            active_ov.append({"factor": f, "score": round(sm[f], 4), "raw": round(probs[f], 4),
                                              "circle": ci.to_dict()})
                    if on_step is not None:
                        try:
                            on_step({"camera_id": camera_id, "t": round(t, 2), "wall": fr.wall,
                                     "probs": steps[-1]["p"], "ema": steps[-1]["ema"], "active": list(active_ov),
                                     "incident": open_inc.event_id if open_inc is not None else None})
                        except Exception as e:   # a preview hook must never stop the live loop
                            log(json.dumps({"msg": "on_step_error", "error": repr(e)[:200]}))
                    if open_inc is not None and on:
                        risk = noisy_or(confidence(sm[f], self.thr[f]) for f in on)
                        for o in active_ov:
                            open_inc.overlays.append({"t_ms": int(round((t - open_inc.clip_t0) * 1000)),
                                                      "t_stream": round(t, 2), **o, "risk": round(risk, 4)})
                        if risk > open_inc.risk_max:
                            full = []
                            for f in on:
                                for ci in self.circles_for(out, f, W, H):
                                    full.append({"factor": f, "score": round(sm[f], 4), "circle": ci.to_dict()})
                            open_inc.risk_max = risk
                            open_inc.peak = (risk, t, fr.image.copy(), full)
                # ---- clip recording
                for oi in ([open_inc] if open_inc else []) + closing:
                    oi.raw.write(small)
                    oi.frames += 1
                for oi in list(closing):
                    if t >= oi.inc.end + float(c["post_s"]):
                        closing.remove(oi)
                        self._finalize(oi, camera_id, source, fps, on_event, events, log)
                # ---- optional whole-session annotated video (live overlay preview)
                if session_video:
                    if session_w is None:
                        session_w = ClipWriter(Path(session_video), min(fps, float(c["fps_cap"])),
                                               (small.shape[1], small.shape[0]))
                    hold = last_step_t is not None and t - last_step_t <= stride * 1.5
                    hdr = f"{camera_id}  t={t:6.1f}s  " + ("ALERT " + ",".join(sorted({o['factor'] for o in active_ov})) if active_ov else "ok")
                    session_w.write(render.draw_overlays(small, active_ov if hold else [], header=hdr))
        finally:
            for tr in tracker.flush(t_last):
                if tr[0] == "close" and open_inc is not None:
                    closing.append(open_inc)
                    open_inc = None
            for oi in closing:
                self._finalize(oi, camera_id, source, source.fps or 25.0, on_event, events, log)
            if session_w:
                session_w.close()
            source.stop()
            for th in self._finalizers:
                th.join(timeout=300)
        wall = time.time() - wall_start
        tot = [l["total"] for l in lat]
        enc = [l["encoder"] for l in lat]
        summary = {
            "camera_id": camera_id, "source": redact(source.uri), "kind": source.kind, "realtime": source.realtime,
            "frames": n_frames, "stream_seconds": round(t_last - (t_first or 0.0), 2), "wall_seconds": round(wall, 2),
            "ingest_fps": round(n_frames / wall, 2) if wall else 0, "source_fps": source.fps,
            "dropped_frames": source.stats["dropped"], "reconnects": source.stats["reconnects"],
            "model_steps": len(lat),
            "latency_ms": {"total_p50": _pct(tot, 50), "total_p95": _pct(tot, 95), "encoder_p50": _pct(enc, 50),
                           "head_p50": _pct([l["head"] for l in lat], 50),
                           "image_pass_p50": _pct([l["image_pass"] for l in lat if "image_pass" in l], 50)},
            "model_steps_per_s_capacity": round(1000 / statistics.mean(tot), 2) if tot else 0,
            "events": [e["event_id"] for e in events], "escalation": self.escalator.status(),
            "thresholds": self.thr, "error": source.error,
        }
        return {"summary": summary, "steps": steps, "events": events}

    # ------------------------------------------------------------------ finalize one incident
    def _finalize(self, oi: OpenIncident, camera_id, source, fps, on_event, events, log):
        codec = oi.raw.close()
        inc = oi.inc
        d = self.store.event_dir(oi.event_id)
        esc_calls = []
        for f, t, why, fut in oi.escalations:
            try:
                r = fut.result(timeout=self.escalator.timeout_s)
            except Exception as e:
                r = {"factor": f, "status": "error", "error": repr(e)[:200]}
            r.update({"t_stream": round(t, 2), "reason": why})
            esc_calls.append(r)
        top_f = max(inc.segments, key=lambda sg: confidence(sg.peak, sg.threshold)).factor if inc.segments else None
        segs = [{"factor": s.factor, "label": FACTOR_LABELS.get(s.factor), "start_s": round(s.start, 2),
                 "end_s": round(s.end if s.end is not None else inc.end, 2), "peak_score": round(s.peak, 4),
                 "peak_t": round(s.peak_t, 2), "threshold": round(s.threshold, 4)} for s in inc.segments]
        start_dt = dt.datetime.fromisoformat(oi.wall0)
        rec = {
            "event_id": oi.event_id, "camera_id": camera_id, "source": redact(source.uri), "source_kind": source.kind,
            "started_at": oi.wall0,
            "ended_at": (start_dt + dt.timedelta(seconds=oi.frames / max(fps, 1e-6))).isoformat(timespec="seconds"),
            "stream_start_s": round(inc.start, 2), "stream_end_s": round(inc.end, 2), "clip_t0_s": round(oi.clip_t0, 2),
            "risk_max": round(oi.risk_max, 4), "top_factor": top_f, "factors": inc.factors, "segments": segs,
            "status": "open", "clip": "clip.mp4", "clip_circles": "clip_circles.mp4", "thumbnail": "thumb.jpg",
            "clip_codec": codec, "clip_size": list(oi.clip_size), "clip_fps": round(min(fps, self.cfg["clips"]["fps_cap"]), 3),
            "escalation": {"policy": {"high_severity": list(self.escalator.policy.high_severity),
                                      "uncertain_band": list(self.escalator.policy.uncertain_band)},
                           "backends": self.escalator.status(), "calls": esc_calls, "skipped": oi.skipped},
            "overlays": oi.overlays,
            "localization": {f: ("validated: F1 patch evidence, Zenodo test hand boxes" if f == "F1" else
                                 "weak: MIL attention map, not validated (refined by SAM 3.1 on escalation)")
                             for f in inc.factors},
            "model": {"encoder": "V-JEPA 2.1 ViT-L/16 384", "head": Path(str(self.cfg["model"]["head_checkpoint"])).stem,
                      "mil_windows": self.cfg["sampling"]["mil_windows"], "stride_s": self.cfg["sampling"]["stride_s"]},
        }
        # thumbnail: peak frame with circles (SAM-refined circle where escalation returned one)
        risk, pt, frame, full = oi.peak
        if frame is not None:
            refined = {r["factor"]: r["final_circle"] for r in esc_calls if r.get("final_circle")
                       and r.get("sam3.1", {}).get("circle")}
            ovs = [dict(o, circle=refined.get(o["factor"], o["circle"])) for o in full]
            import cv2
            cv2.imwrite(str(d / "thumb.jpg"), resize_w(render.draw_overlays(frame, ovs), 960),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            rec["peak_t_stream"] = round(pt, 2)
        self.store.save(rec)
        th = threading.Thread(target=render_circles_clip, args=(d, rec), daemon=True)
        th.start()
        self._finalizers.append(th)
        events.append(rec)
        log(json.dumps({"msg": "event_saved", "event_id": oi.event_id, "factors": inc.factors,
                        "risk_max": rec["risk_max"], "escalations": len(esc_calls), "dir": str(d)}))
        if on_event:
            try:
                on_event(rec)
            except Exception as e:
                log(json.dumps({"msg": "on_event_error", "error": repr(e)}))


def overlays_at(overlays: list, t_ms: int, hold_ms: int = 1500) -> list:
    """Overlays to draw at clip time t_ms: the most recent step at or before t_ms (within hold)."""
    best_t = None
    for o in overlays:
        if o["t_ms"] <= t_ms and t_ms - o["t_ms"] <= hold_ms and (best_t is None or o["t_ms"] > best_t):
            best_t = o["t_ms"]
    return [o for o in overlays if o["t_ms"] == best_t] if best_t is not None else []


def render_circles_clip(d: Path, rec: dict):
    """Burn circles into a copy of the raw clip (the app draws them client-side from `overlays`)."""
    import cv2
    cap = cv2.VideoCapture(str(d / "clip.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or rec.get("clip_fps", 25)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = ClipWriter(d / "clip_circles.mp4", fps, (w, h))
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        t_ms = int(round(i * 1000 / fps))
        hdr = f"{rec['camera_id']}  {rec['event_id']}  +{t_ms/1000:5.1f}s"
        out.write(render.draw_overlays(im, overlays_at(rec["overlays"], t_ms), header=hdr))
        i += 1
    cap.release()
    out.close()


def _pct(v, q):
    if not v:
        return None
    return round(float(np.percentile(v, q)), 1)
