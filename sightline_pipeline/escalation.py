"""Escalation to Meta models, only when something is flagged.

Policy (EscalationPolicy, pure): when a factor turns on inside an incident, escalate it if
  * the factor is high severity (default F1 hand in danger zone, F6 fall, F7 unstable load,
    F10 fire), or
  * the head is unsure: peak smoothed score / threshold falls inside `uncertain_band`
    (default 1.0-1.6x the threshold),
capped per incident and per minute per camera. Everything else is decided by V-JEPA + head alone.

A hard Meta API budget (EscalationBudget) caps submissions per camera session (default 3) and per
local day across cameras and restarts (default 30). The session counter starts at camera start and
is not reset when a looped video replays. Caps are checked most-permanent first: daily, then
session, then per incident, then per minute. Only a call that is actually submitted to the API
takes a per-minute slot or a session/day slot. A refusal takes none of them. The caller counts an
incident slot only when ``allow`` or ``begin_verify`` returns true. No MODEL_API_KEY still records
skipped_no_key and does not spend budget or a per-minute slot. Over-cap attempts are not submitted;
they are stored as skipped_budget. F1 skips and verify-gate skips use the same reason names
(``minute_cap``, not the old ``rate_limited``).

F2 (no head protection) and F5 (no hi-vis vest) are not escalated here. `verify_gate.py` keeps
them muted until Muse Spark confirms them. That check reserves one slot with `reserve` plus
`EscalationBudget`, and the F2 SAM 3.1 circle uses the same slot.

Escalation steps (each behind a small interface, configured by env vars, never fatal):
  1. SAM 3.1 refine: segment the factor's concept ("hand", "fire", ...) in the peak frame and
     snap the head circle to the SAM box that overlaps it (tighter circle, source "sam3.1").
       SAM_BACKEND=api   (default) Meta Model API, model SAM_MODEL (default sam-3.1)
       SAM_BACKEND=local open weights (facebook/sam3.1, gated on Hugging Face) via the `sam3`
                         package, checkpoint SAM3_CHECKPOINT
       SAM_BACKEND=off
  2. Muse Spark verdict: the peak frame with the circle drawn + the factor definition, asking
     for a JSON verdict {confirmed, severity 1-5, reason, action}.
       MUSE_BACKEND=api  (default) Meta Model API, model MUSE_SPARK_MODEL if set,
                         else escalation.muse_model, else muse-spark-1.3-contributor
                         (Muse Spark 1.3 Contributor)
       MUSE_BACKEND=off
  Auth for both API backends: MODEL_API_KEY (alias META_API_KEY); base URL META_API_BASE_URL
  (default https://api.meta.ai/v1). With no key the step records status "skipped_no_key".
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .localize import Circle, box_to_circle

DEFAULT_BASE = "https://api.meta.ai/v1"
# Meta Model API id for Muse Spark 1.3 Contributor (GET /v1/models).
DEFAULT_MUSE_MODEL = "muse-spark-1.3-contributor"

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

    def decide(self, factor: str, score: float, threshold: float, n_in_incident: int, now: float,
               *, commit: bool = True) -> tuple[bool, str]:
        """Whether this factor would escalate. A per-minute refusal is ``minute_cap``.

        ``commit`` records the timestamp. ``Escalator.allow`` passes ``commit=False`` and calls
        ``note_submission`` only after the Meta API budget accepts the call, so a later refusal
        does not occupy the per-minute window. Direct callers keep the old "True means counted"
        behaviour.
        """
        blocked = self._slot_block(n_in_incident, now)
        if blocked:
            return False, blocked
        ratio = score / max(threshold, 1e-6)
        if factor in self.high_severity:
            reason = "high_severity"
        elif self.uncertain_band[0] <= ratio <= self.uncertain_band[1]:
            reason = f"uncertain(score/threshold={ratio:.2f})"
        else:
            return False, f"confident(score/threshold={ratio:.2f})"
        if commit:
            self.note_submission(now)
        return True, reason

    def reserve(self, n_in_incident: int, now: float, *, commit: bool = True) -> tuple[bool, str]:
        """Take one per-incident and per-minute slot, with no severity check.

        The verify gate uses this so its one Muse request (and the F2 SAM call that follows)
        shares the same caps as other escalations. A refusal does not record a timestamp.
        The per-minute refusal is ``minute_cap`` (older events may still say ``rate_limited``).
        ``commit=False`` leaves the window unchanged until ``note_submission``.
        """
        blocked = self._slot_block(n_in_incident, now)
        if blocked:
            return False, blocked
        if commit:
            self.note_submission(now)
        return True, "verify_gate"

    def note_submission(self, now: float) -> None:
        """Record one API submission in the per-minute window. Refusals must not call this."""
        self._recent = [t for t in self._recent if now - t < 60.0]
        self._recent.append(now)

    def _slot_block(self, n_in_incident: int, now: float) -> str | None:
        """Incident cap, then minute cap. Does not record a timestamp. Drops expired ones."""
        if n_in_incident >= self.max_per_incident:
            return "incident_cap"
        self._recent = [t for t in self._recent if now - t < 60.0]
        if len(self._recent) >= self.max_per_minute:
            return "minute_cap"
        return None


# ------------------------------------------------------------------ helpers
def api_key() -> str | None:
    return os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY") or None


def api_base() -> str:
    return (os.environ.get("META_API_BASE_URL") or DEFAULT_BASE).rstrip("/")


def jpeg_data_url(img, max_w: int = 1280, quality: int = 88, long_side: int | None = None) -> str:
    import cv2
    h, w = img.shape[:2]
    if long_side is not None:
        longest = max(h, w)
        if longest > long_side:
            scale = long_side / float(longest)
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    elif w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
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

    def segment(self, img, prompt: str) -> dict:
        """One SAM request for `prompt`. ``sent`` is false when no HTTP call was made.

        Boxes are in the image's coordinate frame. The verify gate picks the box; this method
        does not snap a circle.
        """
        st = self.status()
        if st in ("off", "skipped_no_key"):
            return {"status": st, "boxes": [], "prompt": prompt, "sent": False}
        try:
            boxes = self._api(img, prompt) if self.backend == "api" else self._local_boxes(img, prompt)
        except Exception as e:
            return {"status": "error", "error": repr(e)[:300], "boxes": [], "prompt": prompt,
                    "sent": True, "backend": self.backend}
        return {"status": "ok" if boxes else "no_match", "boxes": boxes or [], "prompt": prompt,
                "sent": True, "backend": self.backend, "model": self.model}

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


def resolve_muse_model(configured: str | None = None, override: str | None = None) -> str:
    """Model id for a Muse Spark request.

    An explicit ``override`` (the eval CLI ``--model``) wins. Otherwise
    ``MUSE_SPARK_MODEL`` if it is set, then ``configured`` (``escalation.muse_model``),
    then Muse Spark 1.3 Contributor.
    """
    if override is not None and str(override).strip():
        return str(override).strip()
    env = os.environ.get("MUSE_SPARK_MODEL")
    if env is not None and str(env).strip():
        return str(env).strip()
    if configured is not None and str(configured).strip():
        return str(configured).strip()
    return DEFAULT_MUSE_MODEL


# ------------------------------------------------------------------ Muse Spark
VERDICT_PROMPT = """You are a workshop safety reviewer. An automatic hazard watcher flagged this camera frame.
Flagged factor: {fid} - {fname}: {fdesc}
Model score: {score:.2f} (alert threshold {thr:.2f}). The flagged area is circled in the image.
Look at the circled area and the whole frame. Answer with JSON only:
{{"confirmed": true|false, "severity": 1-5, "reason": "<one short sentence>", "action": "<one short corrective action>"}}"""


class MuseVerifier:
    def __init__(self, factor_defs: dict | None = None, model: str | None = None):
        self.backend = (os.environ.get("MUSE_BACKEND") or "api").lower()
        # ``model`` is escalation.muse_model. MUSE_SPARK_MODEL still wins.
        self.model = resolve_muse_model(model)
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
                return {"status": "error", "error": f"HTTP {r.status_code}: {r.text[:200]}",
                        "backend": "meta_api", "model": self.model}
            text = r.json()["choices"][0]["message"].get("content") or ""
        except Exception as e:
            return {"status": "error", "error": repr(e)[:300], "backend": "meta_api", "model": self.model}
        out = {"status": "ok", "backend": "meta_api", "model": self.model,
               "latency_s": round(time.time() - t0, 2)}
        out.update(parse_verdict(text))
        return out


# ------------------------------------------------------------------ Meta API budget
_BUDGET_LOCKS: dict[str, threading.Lock] = {}
_BUDGET_LOCKS_GUARD = threading.Lock()
# Most permanent first. When several caps refuse, the earliest one is the stored reason.
CAP_PRIORITY = ("daily_cap", "session_cap", "incident_cap", "minute_cap")
_CAP_RANK = {reason: i for i, reason in enumerate(CAP_PRIORITY)}
BUDGET_REASONS = CAP_PRIORITY


def canonical_skip_reason(reason: str | None) -> str | None:
    """Map the old per-minute name onto ``minute_cap``. Other reasons pass through."""
    if reason == "rate_limited":
        return "minute_cap"
    return reason


def strictest_cap(*reasons: str | None) -> str | None:
    """The most permanent cap among ``reasons``, or None when none of them is a cap."""
    found = [r for r in reasons if r in _CAP_RANK]
    if not found:
        return None
    return min(found, key=_CAP_RANK.__getitem__)


def _budget_file_lock(path) -> threading.Lock:
    """One lock per budget file so two cameras in this process cannot lose an increment."""
    key = os.path.normcase(os.path.abspath(path))
    with _BUDGET_LOCKS_GUARD:
        lock = _BUDGET_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _BUDGET_LOCKS[key] = lock
        return lock


def skip_entry(factor: str, t, reason: str) -> dict:
    """Skipped-escalation record. Cap refusals carry status skipped_budget.

    ``rate_limited`` is stored as ``minute_cap``. Old event files may still contain
    ``rate_limited``; readers accept both.
    """
    reason = canonical_skip_reason(reason) or reason
    entry = {"factor": factor, "t": t, "reason": reason}
    if reason in BUDGET_REASONS:
        entry["status"] = "skipped_budget"
    return entry


class EscalationBudget:
    """Session counter plus a shared daily file counter for Meta API escalations.

    A session is one camera start until stop (one pipeline instance). Call ``reset_session``
    at that start; do not reset it when a looped file replays. The daily file lives at
    ``<data_dir>/escalation_budget.json`` and rolls over at local midnight. ``try_consume``
    re-reads the file under a lock and writes it with a temp file plus ``os.replace``.
    A missing file counts as 0. A corrupt file counts as 0 and is overwritten.
    """

    def __init__(self, data_dir, max_per_session: int = 3, max_per_day: int = 30, today=None):
        self.path = Path(data_dir) / "escalation_budget.json"
        self.max_per_session = int(max_per_session)
        self.max_per_day = int(max_per_day)
        self._today = today or (lambda: dt.date.today().isoformat())
        self._lock = threading.Lock()
        self._session_used = 0

    def reset_session(self) -> None:
        with self._lock:
            self._session_used = 0

    def snapshot(self) -> dict:
        with self._lock:
            with _budget_file_lock(self.path):
                today, count, repair = self._load()
                if repair:
                    self._write(today, 0)
                    count = 0
            return {"session_used": self._session_used, "session_max": self.max_per_session,
                    "day_used": count, "day_max": self.max_per_day}

    def blocking_reason(self) -> str | None:
        """The most permanent cap that would refuse a new submission, without taking a slot.

        ``daily_cap`` wins over ``session_cap``. A clear budget returns None. Nothing is written.
        """
        with self._lock:
            session_blocked = self.max_per_session <= 0 or self._session_used >= self.max_per_session
            with _budget_file_lock(self.path):
                _today, count, _repair = self._load()
            if self.max_per_day <= 0 or count >= self.max_per_day:
                return "daily_cap"
            if session_blocked:
                return "session_cap"
            return None

    def try_consume(self) -> tuple[bool, str | None]:
        """Take one slot. ``(True, None)`` on success, else ``(False, 'daily_cap'|'session_cap')``.

        A max of 0 refuses every call. The file is re-read inside the lock immediately before
        the increment. Daily is checked before session, so both being exhausted reports
        ``daily_cap``. Nothing is written when a cap refuses, except a corrupt file is repaired
        when the daily cap is the reason.
        """
        with self._lock:
            session_blocked = self.max_per_session <= 0 or self._session_used >= self.max_per_session
            with _budget_file_lock(self.path):
                today, count, repair = self._load()
                if self.max_per_day <= 0 or count >= self.max_per_day:
                    if repair:
                        self._write(today, 0)
                    return False, "daily_cap"
                if session_blocked:
                    return False, "session_cap"
                self._write(today, count + 1)
            self._session_used += 1
            return True, None

    def _load(self) -> tuple[str, int, bool]:
        """``(today, count, repair)``. Caller holds the file lock. Stale dates count as 0."""
        today = self._today()
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return today, 0, False
        except OSError:
            return today, 0, True
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return today, 0, True
        if not isinstance(data, dict):
            return today, 0, True
        date, count = data.get("date"), data.get("count")
        count_ok = isinstance(count, int) and not isinstance(count, bool) and count >= 0
        date_ok = isinstance(date, str) and len(date) == 10
        if not count_ok or not date_ok:
            return today, 0, True
        if date != today:
            return today, 0, False
        return today, count, False

    def _write(self, today: str, count: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmpname = tempfile.mkstemp(prefix=".escalation_budget.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"date": today, "count": int(count)}, fh)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmpname, self.path)
        except Exception:
            try:
                os.unlink(tmpname)
            except OSError:
                pass
            raise


# ------------------------------------------------------------------ async runner
class Escalator:
    """Runs escalations in a small thread pool so the live loop never waits on the network."""

    def __init__(self, policy: EscalationPolicy, sam: SamRefiner, muse: MuseVerifier, enabled: bool = True,
                 timeout_s: float = 30.0, budget: EscalationBudget | None = None):
        self.policy, self.sam, self.muse = policy, sam, muse
        self.enabled, self.timeout_s = enabled, timeout_s
        self.budget = budget
        self.pool = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="sightline-escalate")

    def status(self) -> dict:
        out = {"enabled": self.enabled, "sam3.1": self.sam.status(), "muse_spark": self.muse.status()}
        if self.budget is not None:
            out["budget"] = self.budget.snapshot()
        return out

    def reset_session(self) -> None:
        if self.budget is not None:
            self.budget.reset_session()

    def spends_meta_api(self) -> bool:
        """True when submit would call the Meta Model API. No key, or both backends off, does not."""
        if not api_key():
            return False
        return self.sam.status() == "ready" or self.muse.status() == "ready"

    def _budget_block(self, go: bool, why: str) -> str | None:
        """Daily or session refusal, and only when this attempt would otherwise be sent."""
        if self.budget is None or not self.spends_meta_api():
            return None
        if not go and why not in ("incident_cap", "minute_cap"):
            return None
        return self.budget.blocking_reason()

    def allow(self, factor: str, score: float, threshold: float, n_in_incident: int, now: float) -> tuple[bool, str]:
        """Policy, then the budget. ``(True, reason)`` means the caller should ``submit``.

        Caps are reported most-permanent first (daily, session, incident, minute). The per-minute
        window and the session/day counters move only when the Meta API will actually be called.
        A refusal is ``(False, reason)`` and leaves those counters unchanged.
        """
        go, why = self.policy.decide(factor, score, threshold, n_in_incident, now, commit=False)
        why = canonical_skip_reason(why) or why
        cap = strictest_cap(self._budget_block(go, why), None if go else why)
        if cap:
            return False, cap
        if not go:
            return False, why
        if not self.spends_meta_api():
            return True, why
        if self.budget is not None:
            ok, reason = self.budget.try_consume()
            if not ok:
                return False, canonical_skip_reason(reason) or reason or "session_cap"
        self.policy.note_submission(now)
        return True, why

    def begin_verify(self, n_in_incident: int, now: float) -> tuple[bool, str]:
        """Reserve one shared cap slot and one Meta API budget unit for a verify-gate request.

        A missing key or a Muse backend that is off fails closed and spends nothing. Caps and
        budget are taken only when the HTTP call is about to happen. ``(True, 'verify_gate')``
        means the caller should send exactly one Muse request. The F2 SAM call is part of this
        same slot and must not reserve again. Refusals use the same names as F1 skips, most
        permanent first, and do not count toward the per-minute or per-incident caps.
        """
        if not self.enabled:
            return False, "escalation_disabled"
        if not api_key():
            return False, "skipped_no_key"
        if self.muse.status() != "ready":
            return False, "muse_off"
        go, why = self.policy.reserve(n_in_incident, now, commit=False)
        why = canonical_skip_reason(why) or why
        block = self.budget.blocking_reason() if self.budget is not None and (go or why in ("incident_cap", "minute_cap")) else None
        cap = strictest_cap(block, None if go else why)
        if cap:
            return False, cap
        if not go:
            return False, why
        if self.budget is not None:
            ok, reason = self.budget.try_consume()
            if not ok:
                return False, canonical_skip_reason(reason) or reason or "session_cap"
        self.policy.note_submission(now)
        return True, why

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


