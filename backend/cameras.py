"""Camera manager: one SightlinePipeline worker thread per active camera, spread over the GPUs.

Each worker owns one HazardModel (it keeps the MIL window history per camera). Models are pooled
per GPU so stopping and restarting a camera does not reload V-JEPA. GPU work releases the GIL,
so threads are enough; the cap (cameras_per_gpu x GPUs, about 3 per RTX 3090 at 1 step/s) keeps
every camera at real time. RTSP/HTTP cameras and uploaded videos (virtual cameras, looped and
paced to real time) go through the same `VideoSource` ingest path.
"""
from __future__ import annotations

import json
import logging
import threading
import time

from sightline_pipeline.ingest import VideoSource, redact

log = logging.getLogger("sightline.cameras")


class CapacityError(RuntimeError):
    pass


class TappedSource(VideoSource):
    """VideoSource that hands every delivered frame to `tap` (latest frame for the live preview)."""

    def __init__(self, *a, tap=None, **kw):
        super().__init__(*a, **kw)
        self.tap = tap

    def __iter__(self):
        for fr in super().__iter__():
            if self.tap is not None:
                self.tap(fr)
            yield fr


class RealPipelineFactory:
    """Loads V-JEPA 2.1 + the head (WSL, ~/vjepa-venv) and builds one SightlinePipeline per worker."""

    def __init__(self, config_path: str | None = None, store=None):
        from sightline_pipeline.config import load_config, resolve_path
        from sightline_pipeline.storage import EventStore
        self.cfg = load_config(config_path)
        self.store = store or EventStore(resolve_path(self.cfg["storage"]["data_dir"]))
        self.stride_s = float(self.cfg["sampling"]["stride_s"])

    def load_model(self, device: str):
        import copy
        from sightline_pipeline.model import HazardModel
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["device"] = device
        return HazardModel(cfg)

    def make_pipeline(self, model):
        from sightline_pipeline.service import SightlinePipeline
        return SightlinePipeline(self.cfg, model, store=self.store)


class ModelPool:
    def __init__(self, factory):
        self.factory = factory
        self._free: dict[str, list] = {}
        self._lock = threading.Lock()
        self._load_locks: dict[str, threading.Lock] = {}
        self.loaded: dict[str, int] = {}

    def acquire(self, device: str):
        with self._lock:
            free = self._free.setdefault(device, [])
            if free:
                m = free.pop()
                if hasattr(m, "history"):
                    m.history.clear()            # fresh MIL history for the new camera
                return m
            ll = self._load_locks.setdefault(device, threading.Lock())
        with ll:                                 # one model load at a time per GPU
            m = self.factory.load_model(device)
        with self._lock:
            self.loaded[device] = self.loaded.get(device, 0) + 1
        return m

    def release(self, device: str, model):
        if model is None:
            return
        with self._lock:
            self._free.setdefault(device, []).append(model)


class CameraWorker(threading.Thread):
    def __init__(self, cam: dict, device: str, pool: ModelPool, factory, on_event, on_status, on_alert_state=None,
                 stride_s: float = 1.0):
        super().__init__(daemon=True, name=f"cam-{cam['id']}")
        self.cam, self.device, self.pool, self.factory = cam, device, pool, factory
        self.on_event_cb, self.on_status_cb, self.on_alert_state = on_event, on_status, on_alert_state
        self.stride_s = stride_s
        self.stop_ev = threading.Event()
        self.state = "starting"
        self.started_at = time.time()
        self.last_error: str | None = None
        self.restarts = 0
        self.events = 0
        self.frames = 0
        self.latest = None                       # (frame index, wall, image)
        self.last_step = None                    # on_step info + wall
        self.active_factors: tuple = ()
        self.last_summary: dict | None = None
        self.finished_reason: str | None = None
        self.src = None                          # current VideoSource (live connection state)
        self._fps_t, self._fps_n, self.fps = time.time(), 0, 0.0

    # ---- callbacks from the pipeline thread
    def _tap(self, fr):
        self.latest = (fr.index, fr.wall, fr.image)
        self.frames += 1
        self._fps_n += 1
        now = time.time()
        if now - self._fps_t >= 2.0:
            self.fps = round(self._fps_n / (now - self._fps_t), 1)
            self._fps_t, self._fps_n = now, 0

    def _on_step(self, info):
        info["wall_rx"] = time.time()
        self.last_step = info
        af = tuple(sorted({o["factor"] for o in info.get("active", [])}))
        if af != self.active_factors:
            self.active_factors = af
            if self.on_alert_state:
                self.on_alert_state(self.cam["id"], list(af), info.get("incident"))

    def _on_event(self, rec):
        self.events += 1
        self.on_event_cb(rec)

    def _log(self, m):
        log.info("%s %s", self.cam["id"], m if isinstance(m, str) else json.dumps(m))

    def _set(self, state):
        self.state = state
        try:
            self.on_status_cb(self.cam["id"], self.status())
        except Exception:
            log.exception("status callback failed")

    def run(self):
        model = None
        try:
            self._set("loading_model")
            model = self.pool.acquire(self.device)
            pipe = self.factory.make_pipeline(model)
            backoff = 2.0
            while not self.stop_ev.is_set():
                kind = self.cam["kind"]
                meta = self.cam.get("meta") or {}
                src = TappedSource(self.cam["uri"], realtime=True if kind == "virtual" else None,
                                   loop=bool(self.cam.get("loop")) and kind == "virtual", tap=self._tap,
                                   rtsp_transport=meta.get("rtsp_transport", "tcp"), cancel=self.stop_ev)
                self.src = src
                self._set("running")
                res = pipe.process(src, camera_id=self.cam["id"], on_event=self._on_event, stop=self.stop_ev,
                                   max_seconds=self.cam.get("max_runtime_s") or None, log=self._log,
                                   on_step=self._on_step)
                s = res.get("summary", {})
                self.last_summary = {k: s.get(k) for k in ("frames", "stream_seconds", "wall_seconds", "model_steps",
                                                           "dropped_frames", "reconnects", "latency_ms", "error")}
                if self.stop_ev.is_set():
                    break
                maxrt = self.cam.get("max_runtime_s")
                if maxrt and (s.get("stream_seconds") or 0) >= float(maxrt) - 1.0:
                    self.finished_reason = "max_runtime"
                    break
                if kind == "virtual" and not self.cam.get("loop") and not s.get("error"):
                    self.finished_reason = "end_of_file"
                    break
                if s.get("error") and kind == "virtual":
                    self.last_error = str(s.get("error"))[:300]
                    self._set("error")
                    return
                # live stream gave up (VideoSource already retried): back off and reopen
                self.last_error = str(s.get("error") or "stream ended")[:300]
                self.restarts += 1
                self._set("reconnecting")
                if self.stop_ev.wait(backoff):
                    break
                backoff = min(60.0, backoff * 2)
            self._set("finished" if self.finished_reason else "stopped")
        except Exception as e:
            self.last_error = redact(repr(e))[:300]
            log.exception("camera %s worker crashed", self.cam["id"])
            self._set("error")
        finally:
            esc = getattr(locals().get("pipe"), "escalator", None)
            if esc is not None and hasattr(esc, "shutdown"):
                try:
                    esc.shutdown()
                except Exception:
                    pass
            self.pool.release(self.device, model)

    def stop(self):
        self.stop_ev.set()

    def status(self) -> dict:
        lat = self.latest
        return {"state": self.state, "device": self.device, "started_at": self.started_at,
                "uptime_s": round(time.time() - self.started_at, 1), "fps": self.fps, "frames": self.frames,
                "last_frame_age_s": round(time.time() - lat[1], 2) if lat else None, "events": self.events,
                "restarts": self.restarts, "last_error": self.last_error, "active_factors": list(self.active_factors),
                "finished_reason": self.finished_reason, "last_run": self.last_summary, "source": self._source_status()}

    def _source_status(self) -> dict | None:
        src = self.src
        if src is None:
            return None
        return {"state": src.state, "last_error": src.last_error, "reconnects": src.stats.get("reconnects", 0),
                "resolution": list(src.size), "fps": round(src.fps, 2),
                "resolution_changes": src.stats.get("resolution_changes", 0)}


class CameraManager:
    def __init__(self, db, factory, gpus: list, per_gpu: int, on_event=None, on_status=None, on_alert_state=None):
        self.db, self.factory = db, factory
        self.gpus, self.per_gpu = list(gpus) or ["cuda:0"], int(per_gpu)
        self.pool = ModelPool(factory)
        self.workers: dict[str, CameraWorker] = {}
        self._lock = threading.Lock()
        self.on_event, self.on_status, self.on_alert_state = on_event, on_status, on_alert_state

    @property
    def capacity(self) -> int:
        return len(self.gpus) * self.per_gpu

    def _alive(self) -> dict:
        return {k: w for k, w in self.workers.items() if w.is_alive()}

    def active_count(self) -> int:
        with self._lock:
            return len(self._alive())

    def _pick_device(self) -> str:
        load = {g: 0 for g in self.gpus}
        for w in self._alive().values():
            load[w.device] = load.get(w.device, 0) + 1
        g = min(self.gpus, key=lambda d: (load[d], self.gpus.index(d)))
        if load[g] >= self.per_gpu:
            raise CapacityError(f"all {self.capacity} camera slots are busy ({self.per_gpu} per GPU)")
        return g

    def start(self, cid: str) -> dict:
        cam = self.db.camera(cid)
        if cam is None:
            raise KeyError(cid)
        with self._lock:
            w = self.workers.get(cid)
            if w is not None and w.is_alive() and not w.stop_ev.is_set():
                return w.status()
            dev = self._pick_device()
            w = CameraWorker(cam, dev, self.pool, self.factory, self._event, self._status, self._alert_state,
                             stride_s=getattr(self.factory, "stride_s", 1.0))
            self.workers[cid] = w
            w.start()
        self.db.update_camera(cid, enabled=1)
        return w.status()

    def stop(self, cid: str, wait_s: float = 0.0, disable: bool = True) -> dict | None:
        with self._lock:
            w = self.workers.get(cid)
        if disable:
            self.db.update_camera(cid, enabled=0)
        if w is None:
            return None
        w.stop()
        if wait_s:
            w.join(timeout=wait_s)
        return w.status()

    def status(self, cid: str) -> dict:
        w = self.workers.get(cid)
        if w is None:
            return {"state": "stopped", "device": None, "active_factors": []}
        st = w.status()
        if not w.is_alive() and st["state"] in ("running", "loading_model", "starting", "reconnecting"):
            st["state"] = "stopped"
        return st

    def worker(self, cid: str) -> CameraWorker | None:
        w = self.workers.get(cid)
        return w if w is not None and w.is_alive() else None

    def autostart(self) -> list:
        started = []
        for cam in self.db.cameras():
            if cam["enabled"]:
                try:
                    self.start(cam["id"])
                    started.append(cam["id"])
                except CapacityError:
                    log.warning("autostart: no capacity for %s", cam["id"])
                    self.db.update_camera(cam["id"], enabled=0)
        return started

    def shutdown(self, wait_s: float = 30.0):
        with self._lock:
            ws = list(self.workers.values())
        for w in ws:
            w.stop()
        t0 = time.time()
        for w in ws:
            w.join(timeout=max(0.1, wait_s - (time.time() - t0)))

    # ---- fan-out
    def _event(self, rec):
        if self.on_event:
            self.on_event(rec)

    def _status(self, cid, st):
        if st.get("state") in ("finished",) or (st.get("state") == "error" and cid in self.workers
                                                and self.workers[cid].cam["kind"] == "virtual"):
            self.db.update_camera(cid, enabled=0)      # do not auto-restart it on the next boot
        if self.on_status:
            self.on_status(cid, st)

    def _alert_state(self, cid, factors, incident):
        if self.on_alert_state:
            self.on_alert_state(cid, factors, incident)
