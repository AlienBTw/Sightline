import pytest

from sightline.config import Settings
from sightline.errors import SightlineError
from sightline.vision.vlm_gate import VLMGate


def _settings(**kwargs) -> Settings:
    base = dict(
        sightline_mode="live",
        detector_backend="replay",
        report_backend="ollama",
        vlm_base_url="",
        vlm_model="",
        vlm_api_key="",
        ollama_base_url="http://localhost:11434/v1",
        ollama_model="muse-glimmer:30b",
        meta_api_base_url="",
        meta_api_key="",
        meta_api_model="",
        torch_device="cpu",
        sightline_db="./data/sightline.db",
        sample_fps=2.0,
        api_url="http://127.0.0.1:8000",
    )
    base.update(kwargs)
    return Settings.model_construct(**base)  # type: ignore[arg-type]


def test_should_gate_requires_url():
    gate = VLMGate(_settings(vlm_base_url=""))
    assert gate.should_gate("fire_heat", 0.99) is False


def test_should_gate_always_families():
    gate = VLMGate(_settings(vlm_base_url="http://127.0.0.1:8001", vlm_model="muse-glimmer:30b"))
    # with empty always_gate in thresholds, mid-band still applies
    assert gate.should_gate("slip_trip", 0.6) is True
    assert gate.should_gate("slip_trip", 0.9) is False


def test_analyze_fails_without_url():
    gate = VLMGate(_settings(vlm_base_url=""))
    with pytest.raises(SightlineError) as ei:
        gate.analyze_crop("abc", family="fire_heat")
    assert ei.value.error_code == "vlm_not_configured"


def test_parse_json_strict():
    gate = VLMGate(_settings(vlm_base_url="http://x", vlm_model="muse-glimmer:30b"))
    ok = gate._parse_json(
        '{"source_active": true, "source_type": "heater", "material": "cardboard",'
        ' "material_combustible": true, "liquid_present": false,'
        ' "uncertainty": "low", "evidence": "box near heater"}'
    )
    assert ok["source_active"] is True
    with pytest.raises(SightlineError) as ei:
        gate._parse_json('{"source_active": true}')
    assert ei.value.error_code == "vlm_schema_incomplete"
