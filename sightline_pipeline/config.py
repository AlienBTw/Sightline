"""Pipeline configuration: config.yaml defaults + optional user YAML + SIGHTLINE_* env vars."""
from __future__ import annotations

import copy
import os
import warnings
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
    _apply_verify_gates(cfg)
    _apply_escalation_budget(cfg)
    cfg = _merge(cfg, overrides or {})
    esc = cfg.get("escalation")
    if isinstance(esc, dict):
        esc.pop("f5_muse_gate", None)
        esc.pop("f5_vlm_gate", None)
        if isinstance(esc.get("verify_gates"), dict):
            from .verify_gate import fill_verify_defaults
            fill_verify_defaults(esc["verify_gates"])
    _apply_muse_model(cfg)
    return cfg


def _parse_budget_env(name: str) -> int | None:
    """Integer env override. Unset stays None. 0 is valid (no escalations). Invalid values warn and are ignored."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip()
    try:
        val = int(text)
    except ValueError:
        warnings.warn(f"ignoring invalid {name}={raw!r} (expected an integer)", stacklevel=2)
        return None
    if val < 0:
        warnings.warn(f"ignoring invalid {name}={raw!r} (expected an integer >= 0)", stacklevel=2)
        return None
    return val


def _on_off(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    token = str(raw).strip().lower()
    if token in ("1", "on", "true", "yes"):
        return True
    if token in ("0", "off", "false", "no"):
        return False
    warnings.warn(f"ignoring invalid {name}={raw!r} (expected on or off)", stacklevel=2)
    return None


def _apply_verify_gates(cfg: dict) -> None:
    """Defaults for escalation.verify_gates, then the SIGHTLINE_VERIFY_GATE* switches.

    SIGHTLINE_VERIFY_GATES=on|off is the global switch. SIGHTLINE_VERIFY_GATE_F2 and
    SIGHTLINE_VERIFY_GATE_F5 turn one factor on or off. A missing key is filled from the
    defaults (max_tokens 4000, recent_s 3, sam_hold_s 1.5, F2 prompts head/person, F5 prompt person); a
    value already in the file is kept. The old f5_muse_gate / f5_vlm_gate keys are dropped.
    Gated factors stay scored, and they are published only after Muse confirms.
    """
    from .verify_gate import fill_verify_defaults
    esc = cfg.setdefault("escalation", {})
    esc.pop("f5_muse_gate", None)
    esc.pop("f5_vlm_gate", None)
    gate = esc.get("verify_gates")
    if not isinstance(gate, dict):
        gate = {}
    esc["verify_gates"] = fill_verify_defaults(gate)
    global_switch = _on_off("SIGHTLINE_VERIFY_GATES")
    if global_switch is not None:
        gate["enabled"] = global_switch
    for fid, var in (("F2", "SIGHTLINE_VERIFY_GATE_F2"), ("F5", "SIGHTLINE_VERIFY_GATE_F5")):
        choice = _on_off(var)
        if choice is None or not isinstance(gate.get("factors"), dict):
            continue
        spec = gate["factors"].get(fid)
        if isinstance(spec, dict):
            spec["enabled"] = choice


def _apply_muse_model(cfg: dict) -> None:
    """Default escalation.muse_model to Muse Spark 1.3 Contributor.

    A blank value is filled. ``MUSE_SPARK_MODEL`` is not copied in here; each Muse
    request applies that override itself.
    """
    from .escalation import DEFAULT_MUSE_MODEL
    esc = cfg.setdefault("escalation", {})
    if not isinstance(esc, dict):
        return
    raw = esc.get("muse_model")
    esc["muse_model"] = raw.strip() if isinstance(raw, str) and raw.strip() else DEFAULT_MUSE_MODEL


def _apply_escalation_budget(cfg: dict) -> None:
    """Defaults for older configs, then SIGHTLINE_ESC_MAX_PER_SESSION / SIGHTLINE_ESC_MAX_PER_DAY."""
    esc = cfg.setdefault("escalation", {})
    esc.setdefault("max_per_session", 3)
    esc.setdefault("max_per_day", 30)
    session = _parse_budget_env("SIGHTLINE_ESC_MAX_PER_SESSION")
    if session is not None:
        esc["max_per_session"] = session
    day = _parse_budget_env("SIGHTLINE_ESC_MAX_PER_DAY")
    if day is not None:
        esc["max_per_day"] = day


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
