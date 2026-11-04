"""Per-factor Muse Spark verify gates (F2 no head protection, F5 no hi-vis vest).

A gated factor stays muted in the incident tracker. It can open or join an incident only
through this gate, and it is published only after Muse Spark confirms it. One incident gets
at most one Muse Spark request: up to three JPEG frames, long side 768, covering every gated
factor that triggered and has not been checked yet. Those frames are the last few seconds
where each factor's smoothed score stayed at or above its trigger, never the pre-roll from
before the trigger. Each factor alerts only when at least ``min_votes`` frames show a person
who is missing that protection. Anything else fails closed, and a failed reply is not retried.

Once Muse confirms a factor, the same escalation slot runs SAM 3.1 with short noun concepts.
F2 tries ``head``, then ``person``. A head box is the F2 circle; a person box becomes a circle
on the top of that box. F5 tries ``person`` and circles the torso. Boxes are cached per frame
and prompt for that one evaluation. No box, or a SAM error, keeps the head-map circle and
marks it ``head_map_fallback``. A SAM circle replaces that factor's circle only on overlays
within ``sam_hold_s`` of the frame SAM ran on. Farther overlays keep the factor and drop
the circle, so a person who has moved is not circled on empty floor.

``python -m sightline_pipeline.verify_gate_eval`` asks the same prompt on a folder of stills.
"""
from __future__ import annotations

import copy
import logging
import math
import re
import threading

from .escalation import api_base, api_key, canonical_skip_reason, jpeg_data_url, resolve_muse_model
from .localize import Circle, box_to_circle
from .risk import confidence, noisy_or

log = logging.getLogger("sightline.escalation")

VERIFIED_BY = "Muse Spark"
FACTOR_FIELD = {"F5": "wearing_hi_vis_vest", "F2": "wearing_head_protection"}
BUDGET_REASONS = ("incident_cap", "minute_cap", "session_cap", "daily_cap")
RECENT_WINDOW_S = 3.0
DIAG_LIMIT = 1500
# Top band of a person box treated as the head, and how far down the torso circle sits.
HEAD_BAND = 0.20
TORSO_Y = 0.40
# How long a SAM circle stays on the clip, either side of the frame SAM was run on.
DEFAULT_SAM_HOLD_S = 1.5

# F5, unchanged from the vest gate. `{n}` is the number of images in that request.
_VEST_PROMPT = """You check workshop camera frames for high-visibility vests and nothing else.
A high-visibility vest is fluorescent yellow, orange, or lime clothing on the torso, with or without reflective strips.
There are {n} images, in the order they are attached. For each image decide:
- person_visible: true only if a person is clearly visible, otherwise false
- wearing_hi_vis_vest: true only if that person is wearing a high-visibility vest. If no person is visible, or you cannot tell, use true
Reply with JSON only, no markdown, exactly this shape and exactly {n} objects in frames, in the same order as the images:
{{"frames":[{example}]}}"""

_HEAD_PROMPT = """You check workshop camera frames for head protection and nothing else.
Head protection means a hard hat, a safety helmet, or a bump cap.
A normal cap, a hood, a hair net, or a bare head is not head protection.
There are {n} images, in the order they are attached. For each image decide:
- person_visible: true only if a person is clearly visible, otherwise false
- wearing_head_protection: true only if that person is wearing head protection. If no person is visible, the head is not visible, or you cannot tell, use true
Reply with JSON only, no markdown, exactly this shape and exactly {n} objects in frames, in the same order as the images:
{{"frames":[{example}]}}"""

_BOTH_PROMPT = """You check workshop camera frames for protective clothing and nothing else.
A high-visibility vest is fluorescent yellow, orange, or lime clothing on the torso, with or without reflective strips.
Head protection means a hard hat, a safety helmet, or a bump cap. A normal cap, a hood, a hair net, or a bare head is not head protection.
There are {n} images, in the order they are attached. For each image decide:
- person_visible: true only if a person is clearly visible, otherwise false
- wearing_hi_vis_vest: true only if that person is wearing a high-visibility vest. If no person is visible, or you cannot tell, use true
- wearing_head_protection: true only if that person is wearing head protection (hard hat, safety helmet, or bump cap). If no person is visible, the head is not visible, or you cannot tell, use true
Reply with JSON only, no markdown, exactly this shape and exactly {n} objects in frames, in the same order as the images:
{{"frames":[{example}]}}"""

DEFAULT_FACTORS = {
    "F5": {
        "enabled": True,
        "trigger_threshold": 0.15,
        "sam_circle": True,
        "sam_prompts": ["person"],
    },
    "F2": {
        "enabled": True,
        "trigger_threshold": 0.5,
        "sam_circle": True,
        "sam_prompts": ["head", "person"],
    },
}


def fill_verify_defaults(gate: dict) -> dict:
    """Fill missing verify-gate keys. Values already set are left alone."""
    gate.setdefault("enabled", True)
    gate.setdefault("max_frames", 3)
    gate.setdefault("long_side", 768)
    gate.setdefault("min_votes", 2)
    gate.setdefault("timeout_s", 30)
    gate.setdefault("max_tokens", 4000)
    gate.setdefault("recent_s", RECENT_WINDOW_S)
    gate.setdefault("sam_hold_s", DEFAULT_SAM_HOLD_S)
    factors = gate.get("factors")
    if not isinstance(factors, dict):
        factors = {}
        gate["factors"] = factors
    for fid, defaults in DEFAULT_FACTORS.items():
        spec = factors.get(fid)
        if not isinstance(spec, dict):
            spec = {}
            factors[fid] = spec
        for key, value in defaults.items():
            if key not in spec:
                spec[key] = copy.deepcopy(value)
        spec.setdefault("enabled", True)
    return gate


def _example(fields: list[str]) -> str:
    body = {"person_visible": False}
    for field in fields:
        body[field] = True
    import json
    return json.dumps(body, separators=(",", ":"))


def vest_prompt(n: int) -> str:
    example = ",".join([_example(["wearing_hi_vis_vest"])] * max(n, 1))
    return _VEST_PROMPT.format(n=n, example=example)


def head_prompt(n: int) -> str:
    example = ",".join([_example(["wearing_head_protection"])] * max(n, 1))
    return _HEAD_PROMPT.format(n=n, example=example)


def both_prompt(n: int) -> str:
    example = ",".join([_example(["wearing_hi_vis_vest", "wearing_head_protection"])] * max(n, 1))
    return _BOTH_PROMPT.format(n=n, example=example)


def fields_for(factors) -> list[str]:
    ordered = [f for f in ("F5", "F2") if f in set(factors or [])]
    return [FACTOR_FIELD[f] for f in ordered]


def prompt_for(factors, n: int) -> str:
    wanted = [f for f in ("F5", "F2") if f in set(factors or [])]
    if wanted == ["F2"]:
        return head_prompt(n)
    if wanted == ["F5"] or not wanted:
        return vest_prompt(n)
    return both_prompt(n)


def sample_evenly(items, k: int) -> list:
    """Up to `k` items spread across the sequence, first and last included when k >= 2."""
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        return list(items)
    if k == 1:
        return [items[n // 2]]
    idxs, seen, out = [], set(), []
    for i in range(k):
        idxs.append((i * (n - 1)) // (k - 1))
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out


def _time_of(sample):
    if isinstance(sample, dict):
        t = sample.get("t")
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            return float(t)
    return None


def _score_at(sample, factor: str):
    if not isinstance(sample, dict):
        return None
    scores = sample.get("scores")
    if not isinstance(scores, dict) or factor not in scores:
        return None
    try:
        return float(scores.get(factor, 0.0))
    except (TypeError, ValueError):
        return 0.0


def select_gate_frames(samples, factors, thresholds, max_frames: int, recent_s: float = RECENT_WINDOW_S) -> list:
    """Frames for one verify request, from the end of the above-trigger stretch.

    A plain image is already chosen and is kept. A dict sample is kept only when it is not
    pre-roll and every requested factor is at or above its trigger. When the kept frames
    have timestamps, only the last ``recent_s`` seconds (the frames just before the gate
    fires) are used. Up to ``max_frames`` are then spread across that window, including the
    most recent frame. Frames from before the trigger never get here: pre-roll is dropped,
    and a score under the trigger is dropped.
    """
    wanted = [f for f in (factors or []) if f in FACTOR_FIELD]
    eligible = []
    for sample in samples or []:
        if isinstance(sample, dict):
            if sample.get("pre_roll"):
                continue
            if wanted and isinstance(sample.get("scores"), dict):
                below = False
                for fid in wanted:
                    score = _score_at(sample, fid)
                    if score is None:
                        score = 0.0
                    thr = thresholds.get(fid, 0.5) if isinstance(thresholds, dict) else 0.5
                    if score < float(thr):
                        below = True
                        break
                if below:
                    continue
        eligible.append(sample)
    timed = [(sample, _time_of(sample)) for sample in eligible]
    if timed and any(t is not None for _, t in timed):
        tmax = max(t for _, t in timed if t is not None)
        cutoff = tmax - float(recent_s)
        eligible = [sample for sample, t in timed if t is not None and t >= cutoff]
    return sample_evenly(eligible, max_frames)


def downscale_bgr(img, long_side: int):
    """BGR image whose longer side is at most `long_side`. The input is returned when it already fits."""
    import cv2
    h, w = img.shape[:2]
    longest = max(h, w)
    if long_side <= 0 or longest <= long_side:
        return img
    scale = long_side / float(longest)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


_TRUE_WORDS = {"yes", "true"}
_FALSE_WORDS = {"no", "false"}
_IMAGE_KEY = re.compile(
    r"^(?:image|img|frame|picture|photo|item)?[_\s-]*(\d+)(?:\.[A-Za-z0-9]+)?$",
    re.IGNORECASE,
)


def _as_bool(value):
    """JSON bool, or yes/no/true/false. Anything else cannot be voted on."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_WORDS:
            return True
        if token in _FALSE_WORDS:
            return False
    return None


def _strip_fences(raw: str) -> str:
    """Drop markdown code fences so the JSON inside can be parsed."""
    return re.sub(r"```[A-Za-z0-9_+-]*", "", raw).strip()


def _frame_dict(item, fields: list[str]) -> dict | None:
    if not isinstance(item, dict):
        return None
    person = _as_bool(item.get("person_visible"))
    if person is None:
        return None
    out = {"person_visible": person}
    for field in fields:
        value = _as_bool(item.get(field))
        if value is None:
            return None
        out[field] = value
    return out


def _frame_sort_key(index: int, key) -> tuple:
    m = re.search(r"(\d+)", str(key))
    if m:
        return (0, int(m.group(1)), index)
    return (1, index, 0)


def _frames_from_map(data: dict, fields: list[str]) -> list | None:
    """Dict keyed by image index or name, e.g. ``{"image_1": {...}}``. Values stay in index order."""
    items = [(i, k, v) for i, (k, v) in enumerate(data.items()) if isinstance(v, dict)]
    if not items:
        return None
    indexed = [it for it in items if _IMAGE_KEY.match(str(it[1]).strip())]
    chosen = indexed if indexed else items
    if not indexed and any(_frame_dict(v, fields) is None for _, _, v in chosen):
        return None
    chosen = sorted(chosen, key=lambda it: _frame_sort_key(it[0], it[1]))
    return [v for _, _, v in chosen]


def parse_answers(text: str, n: int | None = None, fields: list[str] | None = None) -> list[dict] | None:
    """Parse the model answer into one dict per image. None means it cannot be voted on.

    Accepts a JSON list, ``{"frames": [...]}``, one object, or a dict keyed by image index
    or name (``{"image_1": {...}}``). Markdown fences are stripped. Bool fields may be JSON
    booleans or the strings yes/no/true/false. A missing required key, or the wrong number
    of frames, fails closed.
    """
    import json
    wanted = list(fields or ["wearing_hi_vis_vest"])
    raw = _strip_fences((text or "").strip())
    if not raw:
        return None
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", raw, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    frames = None
    if isinstance(data, list):
        frames = data
    elif isinstance(data, dict):
        for key in ("frames", "images", "results", "answers"):
            if key not in data:
                continue
            val = data[key]
            if isinstance(val, list):
                frames = val
                break
            if isinstance(val, dict):
                mapped = _frames_from_map(val, wanted)
                frames = mapped if mapped is not None else []
                break
        if frames is None and "person_visible" in data and all(f in data for f in wanted):
            frames = [data]
        if frames is None:
            frames = _frames_from_map(data, wanted)
    if not isinstance(frames, list) or not frames:
        return None
    out = []
    for item in frames:
        parsed = _frame_dict(item, wanted)
        if parsed is None:
            return None
        out.append(parsed)
    if n is not None and len(out) != n:
        return None
    return out


def hazard_votes(answers: list[dict] | None, field: str) -> int:
    """Frames where a person is visible and the protective item is absent.

    A frame with ``person_visible`` false does not count, even when the item is false.
    """
    n = 0
    for answer in answers or []:
        if answer.get("person_visible") is True and answer.get(field) is False:
            n += 1
    return n


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("content") or ""))
        return "".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content", "reasoning", "reasoning_content"):
            val = content.get(key)
            if isinstance(val, str):
                return val
        return ""
    return "" if content is None else str(content)


def _assistant_text(message) -> str:
    """Message content, or reasoning / reasoning_content when the content is empty."""
    if not isinstance(message, dict):
        return _message_text(message).strip()
    content = _message_text(message.get("content")).strip()
    if content:
        return content
    for key in ("reasoning", "reasoning_content"):
        alt = _message_text(message.get(key)).strip()
        if alt:
            return alt
    return ""


def _is_timeout(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    return isinstance(exc, TimeoutError) or "timeout" in name


def _snip(value, limit: int = DIAG_LIMIT) -> str:
    text = value if isinstance(value, str) else _message_text(value)
    return (text or "")[:limit]


def _usage_counts(usage) -> dict | None:
    """Token counts only. Strings, headers, and anything else on the payload are dropped."""
    if not isinstance(usage, dict):
        return None
    out = {}
    for key, val in usage.items():
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            out[key] = val
        elif isinstance(val, dict):
            nested = {nk: nv for nk, nv in val.items()
                      if isinstance(nv, (int, float)) and not isinstance(nv, bool)}
            if nested:
                out[key] = nested
    return out or None


def _reasoning_text(message, choice) -> str:
    parts = []
    sources = []
    if isinstance(message, dict):
        sources.append(message)
    if isinstance(choice, dict):
        sources.append(choice)
    for source in sources:
        for key in ("reasoning", "reasoning_content"):
            text = _message_text(source.get(key)).strip()
            if text and text not in parts:
                parts.append(text)
    return "\n".join(parts)


def muse_diagnostic(status_code, payload, raw_body: str = "") -> dict:
    """What a failed Muse reply contained. Never headers, and never the API key.

    ``content`` is the assistant message (empty when reasoning used up the reply).
    ``reasoning`` is ``reasoning`` or ``reasoning_content``, whichever the payload carried.
    Both are capped at 1500 characters. ``usage`` is present only when token counts were.
    """
    finish = None
    message = None
    choice = None
    usage = None
    if isinstance(payload, dict):
        usage = _usage_counts(payload.get("usage"))
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            finish = choice.get("finish_reason")
            if isinstance(choice.get("message"), dict):
                message = choice["message"]
    if message is not None:
        content = _snip(_message_text(message.get("content")))
        reasoning = _snip(_reasoning_text(message, choice))
    else:
        content = _snip(raw_body)
        reasoning = ""
    diag = {"http_status": status_code, "finish_reason": finish, "content": content, "reasoning": reasoning}
    if usage:
        diag["usage"] = usage
    return diag


def _image_of(sample):
    if isinstance(sample, dict):
        return sample.get("image")
    return sample


def _score_of(sample, factor: str) -> float:
    if isinstance(sample, dict):
        try:
            return float((sample.get("scores") or {}).get(factor, 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


class VerifyClient:
    """One OpenAI-compatible chat completion. `post` is injectable so tests never touch the network."""

    def __init__(self, post=None, timeout_s: float = 30.0, model: str | None = None, long_side: int = 768,
                 max_tokens: int = 4000, model_override: str | None = None):
        self._post = post
        self.timeout_s = float(timeout_s)
        # ``model`` is escalation.muse_model. ``model_override`` is the eval CLI --model.
        self.model = resolve_muse_model(model, override=model_override)
        self.long_side = int(long_side)
        self.max_tokens = int(max_tokens)

    def _send(self, url: str, body: dict, headers: dict):
        if self._post is not None:
            return self._post(url, json=body, timeout=self.timeout_s, headers=headers)
        import requests
        return requests.post(url, json=body, timeout=self.timeout_s, headers=headers)

    def ask(self, images: list, factors=("F5",)) -> dict:
        """Send every image in one request. Questions are only for `factors`.

        Status is `ok`, `skipped_no_key`, `timeout`, `parse_error`, or `error`. No network
        when the key is missing. A parsed answer with the wrong frame count is `parse_error`.
        A failure is returned once, with a diagnostic, and is not retried.
        """
        model = self.model
        factors = [f for f in ("F5", "F2") if f in set(factors or [])] or ["F5"]
        fields = fields_for(factors)
        if not images:
            return {"status": "error", "error": "no images", "raw": None, "answers": None, "model": model,
                    "factors": factors}
        key = api_key()
        if not key:
            return {"status": "skipped_no_key", "raw": None, "answers": None, "model": model, "factors": factors}
        content = [{"type": "text", "text": prompt_for(factors, len(images))}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": jpeg_data_url(img, long_side=self.long_side)}})
        body = {"model": model, "max_tokens": self.max_tokens, "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": content}]}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            response = self._send(api_base() + "/chat/completions", body, headers)
        except Exception as e:
            # One attempt. A timeout or a transport error is not sent again.
            diag = {"http_status": None, "finish_reason": None, "content": "", "reasoning": ""}
            if _is_timeout(e):
                return {"status": "timeout", "error": "timeout", "raw": None, "answers": None, "model": model,
                        "factors": factors, "diagnostic": diag}
            return {"status": "error", "error": repr(e)[:300], "raw": None, "answers": None, "model": model,
                    "factors": factors, "diagnostic": diag}
        raw_body = getattr(response, "text", "") or ""
        status_code = getattr(response, "status_code", 200)
        try:
            payload = response.json()
        except Exception:
            payload = None
        diagnostic = muse_diagnostic(status_code, payload, raw_body)
        if status_code >= 400:
            return {"status": "error", "error": f"HTTP {status_code}: {raw_body[:200]}",
                    "raw": raw_body[:4000], "answers": None, "model": model, "factors": factors,
                    "diagnostic": diagnostic}
        try:
            message = payload["choices"][0]["message"]
            text = _assistant_text(message if isinstance(message, dict) else {"content": message})
        except Exception as e:
            return {"status": "parse_error", "error": repr(e)[:300], "raw": raw_body[:4000],
                    "answers": None, "model": model, "factors": factors, "diagnostic": diagnostic}
        answers = parse_answers(text, n=len(images), fields=fields)
        if answers is None:
            loose = parse_answers(text, n=None, fields=fields)
            reason = "wrong frame count" if loose is not None else "answer was not the expected JSON"
            return {"status": "parse_error", "error": reason, "raw": (text or "")[:4000],
                    "answers": None, "model": model, "factors": factors, "diagnostic": diagnostic}
        return {"status": "ok", "raw": text[:4000], "answers": answers, "model": model, "factors": factors}


class FactorTrigger:
    """Hysteresis for one gated factor. Same shape as the incident tracker, separate threshold."""

    def __init__(self, threshold: float, min_on: int, min_off: int, off_ratio: float):
        self.threshold = float(threshold)
        self.min_on, self.min_off = int(min_on), int(min_off)
        self.off_ratio = float(off_ratio)
        self.above = 0
        self.below = 0
        self.on = False

    def update(self, score: float) -> str | None:
        """Return ``'on'`` or ``'off'`` on an edge, else None. ``on`` stays true between them."""
        if score >= self.threshold:
            self.above += 1
            self.below = 0
        elif score < self.threshold * self.off_ratio:
            self.below += 1
            self.above = 0
        else:
            self.above = 0
            self.below = 0
        if not self.on and self.above >= self.min_on:
            self.on = True
            return "on"
        if self.on and self.below >= self.min_off:
            self.on = False
            return "off"
        return None


def _blank(factor: str, status: str, reason: str, n_frames: int = 0, votes: int = 0, **extra) -> dict:
    out = {"status": status, "votes": int(votes), "n_frames": int(n_frames), "reason": reason,
           "verified_by": VERIFIED_BY if status == "confirmed" else None}
    out.update(extra)
    return out


def _reason_text(why: str) -> tuple[str, str]:
    """``(status, reason)`` for a begin() refusal or a transport failure.

    ``rate_limited`` is the old per-minute name. New skips use ``minute_cap``, the same
    name F1 escalation skips store.
    """
    why = canonical_skip_reason(why) or why
    if why in BUDGET_REASONS:
        return "skipped_budget", why
    if why == "skipped_no_key":
        return "skipped_no_key", "MODEL_API_KEY is not set"
    return "failed", _FAIL_REASON.get(why, why or "not sent")


_FAIL_REASON = {
    "incident_cap": "per-incident escalation cap",
    "minute_cap": "per-minute escalation cap",
    "rate_limited": "per-minute escalation cap",
    "session_cap": "Meta API session budget exhausted",
    "daily_cap": "Meta API daily budget exhausted",
    "skipped_no_key": "MODEL_API_KEY is not set",
    "muse_off": "Muse Spark backend is off",
    "escalation_disabled": "escalation is disabled",
    "timeout": "Muse Spark timed out",
    "parse_error": "Muse Spark answer was not the expected JSON",
    "error": "Muse Spark request failed",
    "no_frames": "no frames sampled while the factor was above the trigger",
    "too_few_frames": "not enough frames in the incident window to vote",
}


def gate_skip_entry(factor: str, t, reason: str) -> dict:
    """Skipped-escalation record. Budget refusals carry status skipped_budget, like skip_entry.

    ``rate_limited`` is stored as ``minute_cap`` so gate skips match F1 skips. Readers still
    accept ``rate_limited`` on older event files.
    """
    reason = canonical_skip_reason(reason) or reason
    entry = {"factor": factor, "t": t, "reason": reason}
    if reason in BUDGET_REASONS:
        entry["status"] = "skipped_budget"
    return entry


def _confidence(box: dict) -> float:
    conf = box.get("c")
    if isinstance(conf, (int, float)) and not isinstance(conf, bool):
        return float(conf)
    return 0.0


def _box_area(box: dict) -> float:
    try:
        return max(0.0, float(box["x2"]) - float(box["x1"])) * max(0.0, float(box["y2"]) - float(box["y1"]))
    except (KeyError, TypeError, ValueError):
        return 0.0


def best_sam_box(boxes: list[dict]) -> dict | None:
    """Highest-confidence box. The same confidence keeps the larger box."""
    if not boxes:
        return None
    return max(boxes, key=lambda box: (_confidence(box), _box_area(box)))


def circle_from_box(box: dict) -> dict:
    w, h = int(box["w"]), int(box["h"])
    circle = box_to_circle(box["x1"], box["y1"], box["x2"], box["y2"], w, h, source="sam3.1")
    circle.score = _confidence(box)
    return circle.to_dict()


def _clamp_radius(radius: float, w: int, h: int, min_frac: float, max_frac: float) -> float:
    diag = math.hypot(max(int(w), 1), max(int(h), 1))
    lo = float(min_frac) * diag
    hi = max(float(max_frac) * diag, lo)
    return min(max(float(radius), lo), hi)


def _region_circle(cx: float, cy: float, radius: float, w: int, h: int, score: float, clamp: bool,
                   min_frac: float, max_frac: float) -> dict:
    if clamp:
        radius = _clamp_radius(radius, w, h, min_frac, max_frac)
    return Circle(float(cx), float(cy), float(radius), int(w), int(h), score=float(score), source="sam3.1").to_dict()


def circle_for_factor(factor: str, prompt: str, box: dict, min_frac: float, max_frac: float) -> dict:
    """Turn one SAM box into the factor's circle.

    ``head`` is the F2 circle as the box stands. ``person`` on F2 is the top 20% of the box
    (the head), centred horizontally, with the radius taken from the box width. ``person``
    on F5 is the torso: box centre x, 40% of the way down the box, radius from the box
    width, clamped to the localization min and max radius.
    """
    w, h = int(box["w"]), int(box["h"])
    x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    score = _confidence(box)
    if factor == "F2" and prompt == "person":
        return _region_circle((x1 + x2) / 2.0, y1 + (HEAD_BAND / 2.0) * bh, 0.5 * bw, w, h, score,
                              False, min_frac, max_frac)
    if factor == "F5" and prompt == "person":
        return _region_circle((x1 + x2) / 2.0, y1 + TORSO_Y * bh, 0.5 * bw, w, h, score,
                              True, min_frac, max_frac)
    return circle_from_box(box)


def _hazard_frame(answer: dict, factor: str) -> bool:
    if not isinstance(answer, dict):
        return False
    return answer.get("person_visible") is True and answer.get(FACTOR_FIELD[factor]) is False


def pick_sam_frame(picked: list, answers: list[dict] | None, factors: list[str]):
    """One frame for every SAM prompt in this evaluation, so a shared prompt is sent once.

    Prefer a frame Muse marked hazardous for every factor that needs a circle, then any
    hazardous frame, then the newest frame in the sample.
    """
    paired = list(zip(picked, answers or []))
    if not paired:
        return None
    wanted = [fid for fid in factors if fid in FACTOR_FIELD]

    def pool(pred):
        return [sample for sample, answer in paired if pred(sample, answer)]

    both = pool(lambda _s, answer: all(_hazard_frame(answer, fid) for fid in wanted)) if wanted else []
    any_hit = pool(lambda _s, answer: any(_hazard_frame(answer, fid) for fid in wanted)) if wanted else []
    chosen = both or any_hit or [sample for sample, _a in paired]

    def key(sample):
        t = _time_of(sample)
        return (t if t is not None else -1.0, _score_of(sample, "F2") + _score_of(sample, "F5"))

    return max(chosen, key=key)


class _SamCache:
    """Boxes for one gate evaluation, keyed by frame and prompt. A repeated prompt is not sent again."""

    def __init__(self, sam, image):
        self.sam = sam
        self.image = image
        self._hit: dict[tuple, dict] = {}
        self.requests = 0

    def fetch(self, prompt: str) -> tuple[dict, int]:
        key = (id(self.image), prompt)
        if key in self._hit:
            return self._hit[key], 0
        sent = 0
        try:
            res = self.sam.segment(self.image, prompt)
        except Exception as e:
            sent = 1
            self.requests += 1
            log.warning("verify_gate SAM error on %r: %s", prompt, e)
            stored = {"status": "error", "boxes": [], "sent": True, "n": 0}
            self._hit[key] = stored
            return stored, sent
        if not isinstance(res, dict):
            sent = 1
            self.requests += 1
            stored = {"status": "error", "boxes": [], "sent": True, "n": 0}
            self._hit[key] = stored
            return stored, sent
        if res.get("sent"):
            sent = 1
            self.requests += 1
        boxes = list(res.get("boxes") or [])
        stored = {"status": res.get("status") or "error", "boxes": boxes, "sent": bool(res.get("sent")),
                  "n": len(boxes)}
        self._hit[key] = stored
        return stored, sent


def resolve_sam_circles(sam, image, jobs: list[tuple[str, list]], min_frac: float, max_frac: float) -> tuple[dict, int]:
    """SAM circles for every confirmed factor, sharing one frame's prompt cache.

    ``jobs`` is ``[(factor, prompts)]``. The first prompt that returns a box wins for that
    factor. A missing key or a backend that is off does not send and does not try again.
    The returned request count is how many ``segment`` calls actually went out.
    """
    cache = _SamCache(sam, image)
    out = {}
    for fid, prompts in jobs:
        info = {"circle": None, "sam_status": "no_match", "sam_prompt": None, "sam_boxes": {}, "sam_requests": 0}
        for prompt in prompts or []:
            res, sent = cache.fetch(prompt)
            info["sam_requests"] += sent
            info["sam_boxes"][prompt] = int(res["n"])
            status = res["status"]
            if status in ("off", "skipped_no_key") and not res["sent"]:
                info["sam_status"] = status
                break
            if status == "error" or not res["boxes"]:
                info["sam_status"] = "error" if status == "error" else "no_match"
                continue
            box = best_sam_box(res["boxes"])
            if box is None:
                info["sam_status"] = "no_match"
                continue
            info["circle"] = circle_for_factor(fid, prompt, box, min_frac, max_frac)
            info["sam_status"] = "ok"
            info["sam_prompt"] = prompt
            break
        out[fid] = info
    return out, cache.requests


def _as_float(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def overlay_stream_time(ov: dict, record: dict | None = None):
    """Stream time of one overlay. ``t_stream`` wins; otherwise ``clip_t0_s + t_ms``."""
    if not isinstance(ov, dict):
        return None
    t = _as_float(ov.get("t_stream"))
    if t is not None:
        return t
    t_ms = _as_float(ov.get("t_ms"))
    if t_ms is None:
        return None
    t0 = 0.0
    if isinstance(record, dict):
        base = _as_float(record.get("clip_t0_s"))
        if base is not None:
            t0 = base
    return t0 + t_ms / 1000.0


def within_sam_hold(ov, sam_t, hold_s, record: dict | None = None) -> bool:
    """True when the overlay is within ``hold_s`` seconds of the frame SAM ran on."""
    sam = _as_float(sam_t)
    hold = _as_float(hold_s)
    if sam is None or hold is None or hold < 0:
        return False
    t = overlay_stream_time(ov, record)
    if t is None:
        return False
    return abs(t - sam) <= hold + 1e-6


def _circle_on_frame(circle: dict, dest: dict | None) -> dict:
    """SAM circle in the overlay's pixel frame. Normalised centre and radius stay the same."""
    if not isinstance(dest, dict):
        return copy.deepcopy(circle)
    try:
        dw, dh = int(dest["w"]), int(dest["h"])
        sw, sh = int(circle["w"]), int(circle["h"])
    except (KeyError, TypeError, ValueError):
        return copy.deepcopy(circle)
    if dw == sw and dh == sh:
        return copy.deepcopy(circle)
    fitted = Circle(float(circle["cx"]), float(circle["cy"]), float(circle["r"]), sw, sh,
                    score=float(circle.get("score") or 0.0), source=str(circle.get("source") or "sam3.1")).scaled(dw, dh)
    return fitted.to_dict()


def _paint_factor(overlays, factor: str, info: dict, record: dict | None = None,
                  hold_s: float = DEFAULT_SAM_HOLD_S) -> None:
    """Write one factor's circle onto overlays of that factor near the SAM frame.

    A SAM circle replaces the head-map circle only when the overlay's stream time is within
    ``hold_s`` of ``sam_t_stream``. Farther overlays keep the factor and drop the circle
    (the label and badge stay; the head-map circle is not drawn in its place). A fallback
    only marks the source and leaves the geometry on every overlay.
    """
    if overlays is None:
        return
    source = info.get("circle_source")
    circle = info.get("circle")
    sam_t = info.get("sam_t_stream")
    found = False
    for ov in overlays:
        if ov.get("factor") != factor:
            continue
        found = True
        if source == "sam3.1" and isinstance(circle, dict):
            if within_sam_hold(ov, sam_t, hold_s, record):
                prev = ov.get("circle") if isinstance(ov.get("circle"), dict) else None
                ov["circle"] = _circle_on_frame(circle, prev)
            else:
                ov["circle"] = None
        elif source == "head_map_fallback" and isinstance(ov.get("circle"), dict):
            ov["circle"]["source"] = "head_map_fallback"
    if source == "sam3.1" and isinstance(circle, dict) and not found:
        sam = _as_float(sam_t)
        if sam is None:
            return
        t0 = 0.0
        score = 0.0
        if isinstance(record, dict):
            base = _as_float(record.get("clip_t0_s"))
            if base is not None:
                t0 = base
            for seg in record.get("segments") or []:
                if seg.get("factor") == factor:
                    score = float(seg.get("peak_score") or 0.0)
                    break
        t_ms = int(round((sam - t0) * 1000))
        overlays.append({"t_ms": max(0, t_ms), "t_stream": sam, "factor": factor,
                         "score": round(score, 4), "circle": copy.deepcopy(circle)})


class VerifyGates:
    """One cached verdict per incident per factor. `begin` reserves caps and budget before the request."""

    def __init__(self, enabled: bool = True, max_frames: int = 3, long_side: int = 768, min_votes: int = 2,
                 timeout_s: float = 30.0, max_tokens: int = 4000, factors: dict | None = None,
                 client: VerifyClient | None = None, model: str | None = None, recent_s: float = RECENT_WINDOW_S,
                 min_radius_frac: float = 0.04, max_radius_frac: float = 0.35,
                 sam_hold_s: float = DEFAULT_SAM_HOLD_S):
        self.enabled = bool(enabled)
        self.max_frames = int(max_frames)
        self.long_side = int(long_side)
        self.min_votes = int(min_votes)
        self.max_tokens = int(max_tokens)
        self.recent_s = float(recent_s)
        self.sam_hold_s = float(sam_hold_s)
        self.min_radius_frac = float(min_radius_frac)
        self.max_radius_frac = float(max_radius_frac)
        self.specs = {}
        raw = factors if isinstance(factors, dict) else {}
        for fid, defaults in DEFAULT_FACTORS.items():
            spec = copy.deepcopy(defaults)
            if isinstance(raw.get(fid), dict):
                spec.update(raw[fid])
            if "sam_prompts" in spec and not isinstance(spec["sam_prompts"], (list, tuple)):
                spec["sam_prompts"] = list(defaults.get("sam_prompts") or [])
            self.specs[fid] = spec
        for fid, spec in raw.items():
            if fid not in self.specs and isinstance(spec, dict):
                self.specs[fid] = dict(spec)
        self.configured_ids = tuple(fid for fid in self.specs if fid in FACTOR_FIELD)
        self.active_ids = tuple(fid for fid in self.configured_ids
                                if self.enabled and bool(self.specs[fid].get("enabled", True)))
        self.client = client or VerifyClient(timeout_s=timeout_s, long_side=long_side, max_tokens=self.max_tokens,
                                             model=model)
        self._cache: dict[tuple, dict] = {}
        self._api: dict[str, dict] = {}
        self._raw: dict[str, str | None] = {}
        self._answers: dict[str, list | None] = {}
        self._diag: dict[str, dict | None] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_cfg(cls, cfg: dict | None, enabled: bool | None = None, localization: dict | None = None,
                 model: str | None = None) -> "VerifyGates":
        g = fill_verify_defaults(dict(cfg or {}))
        gate_on = bool(g.get("enabled", True))
        if enabled is not None:
            gate_on = gate_on and bool(enabled)
        loc = localization if isinstance(localization, dict) else {}
        return cls(enabled=gate_on, max_frames=int(g.get("max_frames", 3)), long_side=int(g.get("long_side", 768)),
                   min_votes=int(g.get("min_votes", 2)), timeout_s=float(g.get("timeout_s", 30)),
                   max_tokens=int(g.get("max_tokens", 4000)), factors=g.get("factors"),
                   recent_s=float(g.get("recent_s", RECENT_WINDOW_S)),
                   sam_hold_s=float(g.get("sam_hold_s", DEFAULT_SAM_HOLD_S)),
                   min_radius_frac=float(loc.get("min_radius_frac", 0.04)),
                   max_radius_frac=float(loc.get("max_radius_frac", 0.35)),
                   model=model)

    def threshold(self, factor: str) -> float | None:
        spec = self.specs.get(factor)
        if not spec:
            return None
        return float(spec.get("trigger_threshold", 0.5))

    def reset(self) -> None:
        self._cache.clear()
        self._api.clear()
        self._raw.clear()
        self._answers.clear()
        self._diag.clear()

    def evaluate(self, incident_id: str, factors, samples, begin, t_stream=None, sam=None) -> dict:
        """Ask once per factor. `begin()` is ``(ok, reason)`` and is not called on a cache hit.

        `begin` must already have decided the request is allowed and, when it returns True,
        consumed one shared cap slot and one Meta API budget unit. That slot also covers the
        SAM circles. A second call for a factor already checked in this incident does not
        send another request.
        """
        if isinstance(factors, str):
            factors = [factors]
        wanted = [f for f in ("F5", "F2") if f in set(factors or []) and f in self.active_ids]
        with self._lock:
            pending = [f for f in wanted if (incident_id, f) not in self._cache]
            if wanted and not pending:
                return self._cached_bundle(incident_id, wanted)
            result = self._once(incident_id, pending, list(samples or []), begin, t_stream, sam)
            for fid, info in result["factors"].items():
                self._cache[(incident_id, fid)] = copy.deepcopy(info)
            return result

    def _cached_bundle(self, incident_id: str, factors: list[str]) -> dict:
        return {"cached": True,
                "factors": {fid: copy.deepcopy(self._cache[(incident_id, fid)]) for fid in factors},
                "api_calls": copy.deepcopy(self._api.get(incident_id, {"muse": 0, "sam": 0})),
                "raw": self._raw.get(incident_id),
                "answers": copy.deepcopy(self._answers.get(incident_id)),
                "diagnostic": copy.deepcopy(self._diag.get(incident_id)),
                "sam_hold_s": self.sam_hold_s,
                "skipped": [], "calls": []}

    def _once(self, incident_id: str, factors: list[str], samples: list, begin, t_stream, sam) -> dict:
        if not factors:
            return {"cached": False, "factors": {}, "api_calls": {"muse": 0, "sam": 0},
                    "raw": None, "answers": None, "diagnostic": None, "skipped": [], "calls": []}
        thresholds = {fid: self.threshold(fid) for fid in factors}
        picked = select_gate_frames(samples, factors, thresholds, self.max_frames, self.recent_s)
        images = [im for im in (_image_of(s) for s in picked) if im is not None]
        n_frames = len(images)
        if n_frames == 0:
            return self._finish(incident_id, factors, self._refused(factors, "no_frames", n_frames), None, None, 0, 0, t_stream, False)
        if n_frames < self.min_votes:
            return self._finish(incident_id, factors, self._refused(factors, "too_few_frames", n_frames), None, None, 0, 0, t_stream, False)
        try:
            ok, why = begin()
        except Exception as e:
            log.warning("verify_gate begin failed: %s", e)
            ok, why = False, "error"
        if not ok:
            refused = self._refused(factors, why or "error", n_frames)
            return self._finish(incident_id, factors, refused, None, None, 0, 0, t_stream, True)
        asked = self.client.ask(images, factors)
        raw, answers = asked.get("raw"), asked.get("answers")
        diagnostic = asked.get("diagnostic")
        status = asked.get("status")
        muse_calls = 0 if status == "skipped_no_key" else 1
        if status != "ok" or not answers:
            # Fail closed on this single reply. Credits are limited, so it is not retried.
            why = "skipped_no_key" if status == "skipped_no_key" else (status or "error")
            if status == "parse_error" and asked.get("error") == "wrong frame count":
                why = "wrong frame count"
            refused = self._refused(factors, why, n_frames, detail=asked.get("error"))
            return self._finish(incident_id, factors, refused, raw, answers, muse_calls, 0, t_stream, True,
                                diagnostic=diagnostic, model_used=asked.get("model"))
        decided = {}
        for fid in factors:
            votes = hazard_votes(answers, FACTOR_FIELD[fid])
            noun = "hi-vis vest" if fid == "F5" else "head protection"
            reason = f"{votes} of {len(answers)} frames show a person with no {noun}"
            if votes >= self.min_votes:
                log.info("verify_gate %s confirmed by %s (%s of %s frames)", fid, VERIFIED_BY, votes, len(answers))
                decided[fid] = _blank(fid, "confirmed", reason, n_frames=len(answers), votes=votes)
            else:
                log.warning("verify_gate %s rejected: %s", fid, reason)
                decided[fid] = _blank(fid, "rejected", reason, n_frames=len(answers), votes=votes)
        jobs = []
        for fid in factors:
            if decided[fid]["status"] != "confirmed":
                continue
            spec = self.specs.get(fid) or {}
            if not spec.get("sam_circle"):
                continue
            prompts = [p for p in (spec.get("sam_prompts") or []) if isinstance(p, str) and p.strip()]
            if prompts:
                jobs.append((fid, prompts))
        sam_requests = 0
        sam_image = None
        sam_t_stream = None
        if jobs:
            usable = [sample for sample in picked if _image_of(sample) is not None]
            frame = pick_sam_frame(usable, answers, [fid for fid, _p in jobs])
            image = _image_of(frame) if frame is not None else (images[0] if images else None)
            if sam is not None and image is not None:
                sam_image = image
                sam_t_stream = _time_of(frame) if frame is not None else None
                located, sam_requests = resolve_sam_circles(
                    sam, image, jobs, self.min_radius_frac, self.max_radius_frac)
            else:
                located = {fid: {"circle": None, "sam_status": "no_sam", "sam_prompt": None,
                                 "sam_boxes": {}, "sam_requests": 0} for fid, _p in jobs}
            for fid, info in located.items():
                decided[fid]["sam_status"] = info.get("sam_status")
                if info.get("sam_boxes"):
                    decided[fid]["sam_boxes"] = dict(info["sam_boxes"])
                decided[fid]["sam_requests"] = int(info.get("sam_requests") or 0)
                if sam_t_stream is not None:
                    decided[fid]["sam_t_stream"] = float(sam_t_stream)
                if info.get("circle"):
                    decided[fid]["circle_source"] = "sam3.1"
                    decided[fid]["circle"] = info["circle"]
                    decided[fid]["sam_prompt"] = info.get("sam_prompt")
                else:
                    decided[fid]["circle_source"] = "head_map_fallback"
        return self._finish(incident_id, factors, decided, raw, answers, muse_calls, sam_requests, t_stream, True,
                            model_used=asked.get("model"), sam_image=sam_image, sam_t_stream=sam_t_stream)

    def _refused(self, factors: list[str], why: str, n_frames: int, detail: str | None = None) -> dict:
        if why == "wrong frame count":
            status, reason = "failed", "Muse Spark returned the wrong number of frames"
        elif why in ("timeout", "parse_error", "error", "no_frames", "too_few_frames", "muse_off", "escalation_disabled"):
            status, reason = _reason_text(why)
            if why == "too_few_frames":
                reason = f"{n_frames} frame(s) in the incident window, need {self.min_votes} to vote"
            elif why == "parse_error":
                reason = detail or reason
            elif why == "error":
                reason = detail or reason
        else:
            status, reason = _reason_text(why)
        if status == "failed":
            log.warning("verify_gate fail closed: %s (%s)", why, reason)
        out = {}
        for fid in factors:
            out[fid] = _blank(fid, status, reason, n_frames=n_frames, votes=0)
        return out

    def _finish(self, incident_id, factors, decided, raw, answers, muse_calls, sam_calls, t_stream, attempted,
                diagnostic=None, model_used=None, sam_image=None, sam_t_stream=None) -> dict:
        prev = self._api.get(incident_id, {"muse": 0, "sam": 0})
        api_calls = {"muse": int(prev.get("muse", 0)) + int(muse_calls), "sam": int(prev.get("sam", 0)) + int(sam_calls)}
        self._api[incident_id] = dict(api_calls)
        self._raw[incident_id] = raw
        self._answers[incident_id] = answers
        self._diag[incident_id] = copy.deepcopy(diagnostic) if isinstance(diagnostic, dict) else None
        skipped = []
        if attempted:
            for fid, info in decided.items():
                if info["status"] in ("skipped_budget", "skipped_no_key"):
                    skipped.append(gate_skip_entry(fid, t_stream, info["reason"] if info["status"] == "skipped_budget" else "skipped_no_key"))
        calls = []
        if muse_calls:
            confirmed = [fid for fid, info in decided.items() if info["status"] == "confirmed"]
            call = {
                "factor": "+".join(factors), "reason": "verify_gate", "t_stream": t_stream,
                "n_frames": decided[factors[0]]["n_frames"] if factors else 0,
                "verified_by": VERIFIED_BY if confirmed else None,
                "muse_spark": {"status": "ok" if any(info["status"] in ("confirmed", "rejected") for info in decided.values()) else "error",
                               "raw": raw, "answers": answers,
                               "model": model_used or getattr(self.client, "model", None),
                               "prompt": "verify_gate",
                               "reason": "verified by Muse Spark" if confirmed else decided[factors[0]]["reason"]},
            }
            if isinstance(diagnostic, dict):
                call["diagnostic"] = copy.deepcopy(diagnostic)
                call["muse_spark"]["diagnostic"] = copy.deepcopy(diagnostic)
            calls.append(call)
        if int(sam_calls):
            for fid, info in decided.items():
                if not info.get("sam_status"):
                    continue
                sam_call = {"factor": fid, "reason": "verify_gate", "t_stream": t_stream,
                            "sam3.1": {"status": info.get("sam_status"),
                                       "n_requests": int(info.get("sam_requests") or 0),
                                       "prompt": info.get("sam_prompt"),
                                       "boxes": dict(info.get("sam_boxes") or {}),
                                       "backend": "api"}}
                if info.get("circle"):
                    sam_call["sam3.1"]["circle"] = info["circle"]
                    sam_call["final_circle"] = info["circle"]
                calls.append(sam_call)
        out = {"cached": False, "factors": decided, "api_calls": api_calls, "raw": raw, "answers": answers,
               "diagnostic": copy.deepcopy(diagnostic) if isinstance(diagnostic, dict) else None,
               "sam_hold_s": self.sam_hold_s, "skipped": skipped, "calls": calls}
        if sam_image is not None:
            out["sam_image"] = sam_image
        if sam_t_stream is not None:
            out["sam_t_stream"] = float(sam_t_stream)
        return out


def _stored_factor(info: dict) -> dict:
    out = {
        "status": info.get("status"),
        "votes": int(info.get("votes") or 0),
        "n_frames": int(info.get("n_frames") or 0),
        "reason": info.get("reason"),
        "verified_by": info.get("verified_by") if info.get("status") == "confirmed" else None,
    }
    for key in ("circle_source", "sam_status", "sam_prompt", "sam_boxes"):
        if info.get(key) is not None:
            out[key] = copy.deepcopy(info[key]) if key == "sam_boxes" else info[key]
    sam_t = _as_float(info.get("sam_t_stream"))
    if sam_t is not None:
        out["sam_t_stream"] = sam_t
    return out


def apply_verify_result(record: dict, result: dict | None, extra_overlays=None) -> bool:
    """Keep a gated factor only when Muse confirmed it. Returns whether `on_event` should run.

    A missing result leaves the record alone. Unconfirmed gated factors are removed. An incident
    that then has no factors is saved as ``suppressed`` and is not published. Confirmed factors
    set ``verified_by`` to Muse Spark. A confirmed SAM circle is written onto overlays of that
    factor whose stream time is within ``sam_hold_s`` of ``sam_t_stream``. ``clip_circles.mp4``
    and ``event.json`` both draw that list. Overlays outside the window keep the factor and
    lose the circle.
    """
    if not result or not result.get("factors"):
        return True
    hold = _as_float(result.get("sam_hold_s"))
    if hold is None:
        hold = DEFAULT_SAM_HOLD_S
    present = set(record.get("factors") or [])
    relevant = {fid: info for fid, info in result["factors"].items() if fid in present}
    if not relevant and not present:
        relevant = dict(result["factors"])
    for fid, info in result["factors"].items():
        if info.get("status") != "confirmed" or not info.get("circle_source"):
            continue
        if fid not in present and fid not in relevant:
            continue
        _paint_factor(record.get("overlays"), fid, info, record, hold_s=hold)
        if extra_overlays is not None:
            _paint_factor(extra_overlays, fid, info, record, hold_s=hold)
    esc = record.setdefault("escalation", {})
    suppressed = [fid for fid, info in relevant.items() if info.get("status") != "confirmed"]
    stored_factors = {fid: _stored_factor(info) for fid, info in result["factors"].items() if fid in present or fid in relevant}
    if not stored_factors:
        stored_factors = {fid: _stored_factor(info) for fid, info in result["factors"].items()}
    esc["verify_gates"] = {
        "factors": stored_factors,
        "api_calls": dict(result.get("api_calls") or {"muse": 0, "sam": 0}),
        "suppressed": list(suppressed),
    }
    if result.get("raw"):
        esc["verify_gates"]["raw"] = result.get("raw")
    if result.get("answers"):
        esc["verify_gates"]["answers"] = result.get("answers")
    if isinstance(result.get("diagnostic"), dict):
        esc["verify_gates"]["diagnostic"] = copy.deepcopy(result["diagnostic"])
    confirmed = [fid for fid, info in relevant.items() if info.get("status") == "confirmed"]
    for fid in confirmed:
        loc = record.get("localization")
        if isinstance(loc, dict):
            loc[fid] = "verified by Muse Spark"
    if confirmed:
        record["verified_by"] = VERIFIED_BY
    else:
        record.pop("verified_by", None)
    if not suppressed:
        return True
    segments = [s for s in (record.get("segments") or []) if s.get("factor") not in suppressed]
    record["segments"] = segments
    record["overlays"] = [o for o in (record.get("overlays") or []) if o.get("factor") not in suppressed]
    loc = record.get("localization")
    if isinstance(loc, dict):
        for fid in suppressed:
            loc.pop(fid, None)
    if segments:
        record["factors"] = list(dict.fromkeys(s["factor"] for s in segments))
        record["top_factor"] = max(segments, key=lambda s: confidence(float(s["peak_score"]), float(s["threshold"])))["factor"]
        record["risk_max"] = round(noisy_or(confidence(float(s["peak_score"]), float(s["threshold"])) for s in segments), 4)
        if confirmed:
            record["verified_by"] = VERIFIED_BY
        return True
    record["factors"] = []
    record["top_factor"] = None
    record["risk_max"] = 0.0
    record["status"] = "suppressed"
    record.pop("verified_by", None)
    return False
