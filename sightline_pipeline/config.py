"""Pipeline configuration: config.yaml defaults + optional user YAML + SIGHTLINE_* env vars."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
FACTORS_YAML = REPO / "training" / "configs" / "factors.yaml"

FACTOR_LABELS = {
    "F1": "Hand in danger zone", "F2": "No head protection", "F3": "No eye/face protection",
    "F4": "Wrong glove state", "F5": "No hi-vis vest", "F6": "Fall or slip",
    "F7": "Unstable load", "F8": "Person in vehicle path", "F9": "Open guard/panel",
    "F10": "Fire/smoke/sparks", "F11": "Abnormal machine behaviour",
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve_path(p: str | os.PathLike) -> Path:
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else REPO / p


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(open(PKG / "config.yaml", encoding="utf-8"))
    if path:
        cfg = _merge(cfg, yaml.safe_load(open(path, encoding="utf-8")) or {})
    env = {
        "SIGHTLINE_HEAD_CHECKPOINT": ("model", "head_checkpoint"),
        "SIGHTLINE_ENCODER_WEIGHTS": ("model", "encoder_weights"),
        "SIGHTLINE_VJEPA_REPO": ("model", "vjepa_repo"),
        "SIGHTLINE_DEVICE": ("model", "device"),
        "SIGHTLINE_DATA_DIR": ("storage", "data_dir"),
    }
    for var, (sec, key) in env.items():
        if os.environ.get(var):
            cfg[sec][key] = os.environ[var]
    if os.environ.get("SIGHTLINE_ESCALATION", "").lower() in ("0", "off", "false", "no"):
        cfg["escalation"]["enabled"] = False
    return _merge(cfg, overrides or {})


def factor_ids() -> list[str]:
    """Head output order (F1..F10) from training/configs/factors.yaml (F11 is not a head output)."""
    try:
        fs = yaml.safe_load(open(FACTORS_YAML, encoding="utf-8"))["factors"]
        return [f["id"] for f in fs if f.get("supervision") != "anomaly"]
    except OSError:
        return [f"F{i}" for i in range(1, 11)]


def effective_thresholds(ckpt_thresholds: dict, cfg: dict) -> dict:
    ev = cfg["events"]
    out = {}
    for f, t in ckpt_thresholds.items():
        t = float(ev.get("threshold_overrides", {}).get(f, t))
        out[f] = max(t, float(ev.get("min_threshold", 0.0)))
    return out
