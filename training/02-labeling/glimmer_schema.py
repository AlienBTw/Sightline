"""Parsing, validation and clip aggregation for Glimmer weak labels.

Pure Python (no Ollama, no OpenCV) so it can be unit-tested anywhere.
"""
from __future__ import annotations

import json
import re
from typing import Iterable, Mapping

GLIMMER_FACTORS: tuple[str, ...] = tuple(f"F{i}" for i in range(1, 11))  # F11 is anomaly-scored, never labeled here
STATES = ("present", "absent", "uncertain")
EVIDENCE_MAX = 200

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


class LabelValidationError(ValueError):
    """The model answer is not valid JSON or does not match the schema."""


def extract_json_object(text: str) -> dict:
    """Pull the first JSON object out of a model reply.

    Tolerates code fences, <think> blocks and chatter before/after the object.
    Raises LabelValidationError if no object can be decoded.
    """
    if not isinstance(text, str) or not text.strip():
        raise LabelValidationError("empty reply")
    cleaned = _THINK.sub("", text)
    cleaned = _FENCE.sub("", cleaned.strip())
    start = cleaned.find("{")
    if start < 0:
        raise LabelValidationError("no JSON object in reply")
    decoder = json.JSONDecoder()
    while start >= 0:
        try:
            obj, _ = decoder.raw_decode(cleaned, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = cleaned.find("{", start + 1)
    raise LabelValidationError("reply contains '{' but no decodable JSON object")


def _norm_state(v: object, fid: str) -> str:
    if not isinstance(v, str):
        raise LabelValidationError(f"{fid}.state must be a string, got {type(v).__name__}")
    s = v.strip().lower()
    if s not in STATES:
        raise LabelValidationError(f"{fid}.state '{v}' not in {list(STATES)}")
    return s


def _norm_conf(v: object, fid: str) -> float:
    if isinstance(v, bool):
        raise LabelValidationError(f"{fid}.confidence must be a number")
    if isinstance(v, str):
        try:
            v = float(v.strip())
        except ValueError:
            raise LabelValidationError(f"{fid}.confidence '{v}' is not a number") from None
    if not isinstance(v, (int, float)):
        raise LabelValidationError(f"{fid}.confidence must be a number, got {type(v).__name__}")
    c = float(v)
    if c != c or not 0.0 <= c <= 1.0:  # NaN or out of range (e.g. 85 meaning 85%)
        raise LabelValidationError(f"{fid}.confidence {v} outside [0, 1]")
    return c


def validate_frame_labels(obj: object, factor_ids: Iterable[str] = GLIMMER_FACTORS) -> dict:
    """Validate one frame answer and return a normalized copy:
    {"factors": {"F1": {"state": ..., "confidence": float, "evidence": str}, ...}}
    """
    ids = tuple(factor_ids)
    if not isinstance(obj, Mapping):
        raise LabelValidationError("top level must be a JSON object")
    factors = obj.get("factors")
    if factors is None and all(k in obj for k in ids):
        factors = obj  # model dropped the wrapper; accept
    if not isinstance(factors, Mapping):
        raise LabelValidationError('missing "factors" object')
    keys = {str(k).strip().upper(): v for k, v in factors.items()}
    missing = [f for f in ids if f not in keys]
    extra = sorted(k for k in keys if k not in ids)
    if missing:
        raise LabelValidationError(f"missing factors: {missing}")
    if extra:
        raise LabelValidationError(f"unexpected factors: {extra} (only {list(ids)} are allowed)")
    out: dict[str, dict] = {}
    for fid in ids:
        entry = keys[fid]
        if not isinstance(entry, Mapping):
            raise LabelValidationError(f"{fid} must be an object with state/confidence")
        if "state" not in entry or "confidence" not in entry:
            raise LabelValidationError(f"{fid} needs both 'state' and 'confidence'")
        ev = entry.get("evidence", "")
        if ev is None:
            ev = ""
        if not isinstance(ev, str):
            ev = str(ev)
        out[fid] = {
            "state": _norm_state(entry["state"], fid),
            "confidence": _norm_conf(entry["confidence"], fid),
            "evidence": ev.strip()[:EVIDENCE_MAX],
        }
    return {"factors": out}


def parse_and_validate(text: str, factor_ids: Iterable[str] = GLIMMER_FACTORS) -> dict:
    return validate_frame_labels(extract_json_object(text), factor_ids)


def aggregate_clip(frames: list[dict], factor_ids: Iterable[str] = GLIMMER_FACTORS,
                   present_threshold: float = 0.5) -> dict:
    """Clip-level weak label from per-frame answers (multiple-instance view).

    A clip is a bag of frames:
      present   - at least one frame says present with confidence >= threshold
                  (confidence = max over those frames)
      absent    - every frame says absent (confidence = min over frames, conservative)
      uncertain - anything else (confidence = mean over frames)
    `frames` is a list of validated answers (output of validate_frame_labels), each optionally
    carrying "frame_index".
    """
    ids = tuple(factor_ids)
    if not frames:
        raise ValueError("aggregate_clip needs at least one frame")
    result: dict[str, dict] = {}
    for fid in ids:
        states = [f["factors"][fid]["state"] for f in frames]
        confs = [f["factors"][fid]["confidence"] for f in frames]
        idx = [f.get("frame_index", i) for i, f in enumerate(frames)]
        strong = [(c, i) for s, c, i in zip(states, confs, idx) if s == "present" and c >= present_threshold]
        if strong:
            label, conf = "present", max(c for c, _ in strong)
        elif all(s == "absent" for s in states):
            label, conf = "absent", min(confs)
        else:
            label, conf = "uncertain", sum(confs) / len(confs)
        result[fid] = {
            "label": label,
            "confidence": round(conf, 4),
            "votes": {s: states.count(s) for s in STATES},
            "frames_present": [i for _, i in strong],
        }
    return result


def hint_agreement(clip_factors: Mapping[str, dict], hints: Mapping[str, str]) -> dict:
    """Compare Glimmer clip labels with dataset class hints, e.g. {"F8": "present"}.
    Returns {fid: {"hint": ..., "glimmer": ..., "agree": bool|None}} (None = Glimmer uncertain)."""
    out = {}
    for fid, hint in (hints or {}).items():
        got = (clip_factors.get(fid) or {}).get("label")
        out[fid] = {"hint": hint, "glimmer": got, "agree": None if got in (None, "uncertain") else got == hint}
    return out
