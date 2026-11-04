"""Escalation to Meta models, only when something is flagged.

Policy (EscalationPolicy, pure): when a factor turns on inside an incident, escalate it if
  * the factor is high severity (default F1 hand in danger zone, F6 fall, F7 unstable load,
    F10 fire), or
  * the head is unsure: peak smoothed score / threshold falls inside `uncertain_band`
    (default 1.0-1.6x the threshold),
capped per incident and per minute per camera. Everything else is decided by V-JEPA + head alone.

Escalation steps (each behind a small interface, configured by env vars, never fatal):
  1. SAM 3.1 refine: segment the factor's concept ("hand", "fire", ...) in the peak frame and
     snap the head circle to the SAM box that overlaps it (tighter circle, source "sam3.1").
       SAM_BACKEND=api   (default) Meta Model API, model SAM_MODEL (default sam-3.1)
       SAM_BACKEND=local open weights (facebook/sam3.1, gated on Hugging Face) via the `sam3`
                         package, checkpoint SAM3_CHECKPOINT
       SAM_BACKEND=off
  2. Muse Spark verdict: the peak frame with the circle drawn + the factor definition, asking
     for a JSON verdict {confirmed, severity 1-5, reason, action}.
       MUSE_BACKEND=api  (default) Meta Model API, model MUSE_SPARK_MODEL (default muse-spark-1.3)
       MUSE_BACKEND=off
  Auth for both API backends: MODEL_API_KEY (alias META_API_KEY); base URL META_API_BASE_URL
  (default https://api.meta.ai/v1). With no key the step records status "skipped_no_key".
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import json
import os
import re
import time
from dataclasses import dataclass, field

from .localize import Circle, box_to_circle

DEFAULT_BASE = "https://api.meta.ai/v1"

SAM_PROMPTS = {
    "F1": "hand", "F2": "person", "F3": "face", "F4": "hand", "F5": "person", "F6": "person",
    "F7": "forklift", "F8": "person", "F9": "open electrical panel", "F10": "fire",
}


# ------------------------------------------------------------------ policy (pure)
@dataclass
class EscalationPolicy:
    high_severity: tuple = ("F1", "F6", "F7", "F10")
    uncertain_band: tuple = (1.0, 1.6)
    max_per_incident: int = 3
    max_per_minute: int = 6
    _recent: list = field(default_factory=list)

    def decide(self, factor: str, score: float, threshold: float, n_in_incident: int, now: float) -> tuple[bool, str]:
        if n_in_incident >= self.max_per_incident:
            return False, "incident_cap"
        self._recent = [t for t in self._recent if now - t < 60.0]
        if len(self._recent) >= self.max_per_minute:
            return False, "rate_limited"
        ratio = score / max(threshold, 1e-6)
        if factor in self.high_severity:
            reason = "high_severity"
        elif self.uncertain_band[0] <= ratio <= self.uncertain_band[1]:
            reason = f"uncertain(score/threshold={ratio:.2f})"
        else:
            return False, f"confident(score/threshold={ratio:.2f})"
        self._recent.append(now)
        return True, reason


# ------------------------------------------------------------------ helpers
def api_key() -> str | None:
    return os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY") or None


def api_base() -> str:
    return (os.environ.get("META_API_BASE_URL") or DEFAULT_BASE).rstrip("/")


def jpeg_data_url(img, max_w: int = 1280, quality: int = 88) -> str:
    import cv2
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


BOX_RE = re.compile(r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([0-9.]+))?\|>")


def parse_sam_boxes(text: str) -> list[dict]:
    """SAM 3.1 Model API special-token output -> [{id, x1, y1, x2, y2, w, h, c}] (masks ignored)."""
    out = []
    for m in BOX_RE.finditer(text or ""):
        oid, x1, y1, x2, y2, w, h, c = m.groups()
        out.append({"id": int(oid), "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "w": int(w), "h": int(h), "c": float(c) if c else None})
    return out


def pick_box(boxes: list[dict], circle: Circle) -> dict | None:
    """The SAM box that best matches the head circle: centre inside the circle (closest first),
    else the nearest box centre within 1.5 radii."""
    best, best_d = None, None
    for b in boxes:
        sx, sy = circle.w / b["w"], circle.h / b["h"]
        bx, by = (b["x1"] + b["x2"]) / 2 * sx, (b["y1"] + b["y2"]) / 2 * sy
        d = ((bx - circle.cx) ** 2 + (by - circle.cy) ** 2) ** 0.5
        if d <= 1.5 * circle.r and (best_d is None or d < best_d):
            best, best_d = b, d
    return best


def refine_circle(circle: Circle, box: dict) -> Circle:
    sx, sy = circle.w / box["w"], circle.h / box["h"]
    c = box_to_circle(box["x1"] * sx, box["y1"] * sy, box["x2"] * sx, box["y2"] * sy, circle.w, circle.h,
                      source="sam3.1")
    c.score = circle.score
    return c


def parse_verdict(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {"confirmed": None, "raw": (text or "")[:300]}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"confirmed": None, "raw": (text or "")[:300]}
    return {"confirmed": d.get("confirmed"), "severity": d.get("severity"), "reason": d.get("reason"),
            "action": d.get("action")}


# ------------------------------------------------------------------ SAM 3.1
class SamRefiner:
    def __init__(self):
        self.backend = (os.environ.get("SAM_BACKEND") or "api").lower()
        self.model = os.environ.get("SAM_MODEL") or "sam-3.1"
        self._local = None

    def status(self) -> str:
        if self.backend == "off":
            return "off"
        if self.backend == "api":
            return "ready" if api_key() else "skipped_no_key"
        return "local:" + (os.environ.get("SAM3_CHECKPOINT") or "no SAM3_CHECKPOINT")

    def _api(self, img, prompt: str) -> list[dict]:
        import requests
        body = {"model": self.model, "stream": True, "metadata": {"include_confidence": "true"},
                "input": [{"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": jpeg_data_url(img, max_w=1920)}]}]}
        r = requests.post(api_base() + "/responses", json=body, stream=True, timeout=60,
                          headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        text = ""
        ctype = r.headers.get("content-type", "")
        if "event-stream" in ctype:
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "response.output_text.delta":
                    text += ev.get("delta", "")
                elif ev.get("type") == "response.completed" and not text:
                    text = json.dumps(ev.get("response", {}))
        else:
            text = r.text
        return parse_sam_boxes(text)

    def _local_boxes(self, img, prompt: str) -> list[dict]:
        # Open-weight path (untested here: needs gated facebook/sam3.1 access + `pip install sam3`).
        import numpy as np
        from PIL import Image
        if self._local is None:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
            ck = os.environ.get("SAM3_CHECKPOINT")
            model = build_sam3_image_model(checkpoint_path=ck) if ck else build_sam3_image_model()
            self._local = Sam3Processor(model)
        h, w = img.shape[:2]
        st = self._local.set_image(Image.fromarray(img[:, :, ::-1]))
        out = self._local.set_text_prompt(state=st, prompt=prompt)
        boxes = []
        for i, (b, s) in enumerate(zip(np.asarray(out["boxes"].cpu()), np.asarray(out["scores"].cpu()))):
            boxes.append({"id": i, "x1": int(b[0]), "y1": int(b[1]), "x2": int(b[2]), "y2": int(b[3]),
                          "w": w, "h": h, "c": float(s)})
        return boxes

    def refine(self, img, factor: str, circle: Circle) -> dict:
        st = self.status()
        if st in ("off", "skipped_no_key"):
            return {"status": st}
        prompt = SAM_PROMPTS.get(factor, "object")
        t0 = time.time()
        try:
            boxes = self._api(img, prompt) if self.backend == "api" else self._local_boxes(img, prompt)
        except Exception as e:
            return {"status": "error", "error": repr(e)[:300], "backend": self.backend}
        res = {"status": "ok", "backend": self.backend, "model": self.model, "prompt": prompt,
               "n_matches": len(boxes), "latency_s": round(time.time() - t0, 2)}
        b = pick_box(boxes, circle)
        if b is None:
            res["status"] = "no_match"
            return res
        res["box"] = b
        res["circle"] = refine_circle(circle, b).to_dict()
        return res


# ------------------------------------------------------------------ Muse Spark
VERDICT_PROMPT = """You are a workshop safety reviewer. An automatic hazard watcher flagged this camera frame.
Flagged factor: {fid} - {fname}: {fdesc}
Model score: {score:.2f} (alert threshold {thr:.2f}). The flagged area is circled in the image.
Look at the circled area and the whole frame. Answer with JSON only:
{{"confirmed": true|false, "severity": 1-5, "reason": "<one short sentence>", "action": "<one short corrective action>"}}"""


class MuseVerifier:
    def __init__(self, factor_defs: dict | None = None):
        self.backend = (os.environ.get("MUSE_BACKEND") or "api").lower()
        self.model = os.environ.get("MUSE_SPARK_MODEL") or "muse-spark-1.3"
        self.defs = factor_defs or {}

    def status(self) -> str:
        if self.backend == "off":
            return "off"
        return "ready" if api_key() else "skipped_no_key"

    def verify(self, img_with_circle, factor: str, score: float, thr: float) -> dict:
        st = self.status()
        if st != "ready":
            return {"status": st}
        import requests
        name, desc = self.defs.get(factor, (factor, ""))
        prompt = VERDICT_PROMPT.format(fid=factor, fname=name, fdesc=desc, score=score, thr=thr)
        body = {"model": self.model, "max_tokens": 400, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": jpeg_data_url(img_with_circle)}}]}]}
        t0 = time.time()
        try:
            r = requests.post(api_base() + "/chat/completions", json=body, timeout=60,
                              headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
            if r.status_code >= 400:
                return {"status": "error", "error": f"HTTP {r.status_code}: {r.text[:200]}"}
            text = r.json()["choices"][0]["message"].get("content") or ""
        except Exception as e:
            return {"status": "error", "error": repr(e)[:300]}
        out = {"status": "ok", "backend": "meta_api", "model": self.model, "latency_s": round(time.time() - t0, 2)}
        out.update(parse_verdict(text))
        return out


# ------------------------------------------------------------------ async runner
class Escalator:
    """Runs escalations in a small thread pool so the live loop never waits on the network."""

    def __init__(self, policy: EscalationPolicy, sam: SamRefiner, muse: MuseVerifier, enabled: bool = True,
                 timeout_s: float = 30.0):
        self.policy, self.sam, self.muse = policy, sam, muse
        self.enabled, self.timeout_s = enabled, timeout_s
        self.pool = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="sightline-escalate")

    def status(self) -> dict:
        return {"enabled": self.enabled, "sam3.1": self.sam.status(), "muse_spark": self.muse.status()}

    def submit(self, frame, factor: str, circle: Circle, score: float, thr: float, draw_fn):
        def job():
            out = {"factor": factor, "score": round(score, 4), "threshold": round(thr, 4)}
            ref = self.sam.refine(frame, factor, circle)
            out["sam3.1"] = ref
            final = circle
            if ref.get("circle"):
                c = ref["circle"]
                final = Circle(c["cx"], c["cy"], c["r"], c["w"], c["h"], c.get("score", 0.0), "sam3.1")
            out["muse_spark"] = self.muse.verify(draw_fn(frame, factor, final, score), factor, score, thr)
            out["final_circle"] = final.to_dict()
            return out
        return self.pool.submit(job)

    def shutdown(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
