"""Offline tests for Glimmer answer parsing/validation/aggregation (no Ollama needed)."""
import json

import pytest

from glimmer_schema import (
    GLIMMER_FACTORS, LabelValidationError, aggregate_clip, extract_json_object,
    hint_agreement, parse_and_validate, validate_frame_labels,
)


def answer(**over):
    f = {fid: {"state": "absent", "confidence": 0.9, "evidence": ""} for fid in GLIMMER_FACTORS}
    for k, v in over.items():
        f[k] = v
    return {"factors": f}


def test_f11_not_in_glimmer_factors():
    assert GLIMMER_FACTORS == tuple(f"F{i}" for i in range(1, 11))


def test_valid_plain_json():
    out = parse_and_validate(json.dumps(answer(F9={"state": "present", "confidence": 0.8, "evidence": "panel open"})))
    assert out["factors"]["F9"] == {"state": "present", "confidence": 0.8, "evidence": "panel open"}


@pytest.mark.parametrize("wrap", [
    "```json\n{}\n```",
    "Sure! Here is the JSON:\n{}\nHope this helps.",
    "<think>let me look at the frame {{maybe}}</think>{}",
])
def test_tolerates_fences_chatter_and_think(wrap):
    text = wrap.replace("{}", json.dumps(answer()), 1) if "{}" in wrap else wrap
    assert parse_and_validate(text)["factors"]["F1"]["state"] == "absent"


def test_normalizes_case_and_string_confidence():
    a = answer(F2={"state": " Present ", "confidence": "0.7"})
    a["factors"]["f3"] = a["factors"].pop("F3")  # lower-case key
    out = validate_frame_labels(a)
    assert out["factors"]["F2"] == {"state": "present", "confidence": 0.7, "evidence": ""}
    assert "F3" in out["factors"]


def test_accepts_missing_wrapper():
    assert validate_frame_labels(answer()["factors"])["factors"]["F10"]["state"] == "absent"


@pytest.mark.parametrize("bad, msg", [
    ("", "empty"),
    ("no json here", "no JSON"),
    ("{not json", "no decodable"),
    ('["a list"]', "no JSON object"),
])
def test_rejects_non_json(bad, msg):
    with pytest.raises(LabelValidationError, match=msg):
        parse_and_validate(bad)


def test_rejects_missing_factor():
    a = answer()
    del a["factors"]["F4"]
    with pytest.raises(LabelValidationError, match="missing factors"):
        validate_frame_labels(a)


def test_rejects_f11_and_unknown_factors():
    a = answer()
    a["factors"]["F11"] = {"state": "present", "confidence": 1}
    with pytest.raises(LabelValidationError, match="unexpected factors"):
        validate_frame_labels(a)


@pytest.mark.parametrize("entry, msg", [
    ({"state": "maybe", "confidence": 0.5}, "not in"),
    ({"state": "present", "confidence": 85}, "outside"),
    ({"state": "present", "confidence": -0.1}, "outside"),
    ({"state": "present", "confidence": True}, "number"),
    ({"state": "present", "confidence": "high"}, "not a number"),
    ({"state": "present"}, "needs both"),
    ("present", "must be an object"),
])
def test_rejects_bad_entries(entry, msg):
    with pytest.raises(LabelValidationError, match=msg):
        validate_frame_labels(answer(F1=entry))


def test_evidence_truncated_not_rejected():
    out = validate_frame_labels(answer(F1={"state": "absent", "confidence": 0.5, "evidence": "x" * 500}))
    assert len(out["factors"]["F1"]["evidence"]) == 200


def test_aggregate_mil_rules():
    frames = [
        dict(validate_frame_labels(answer(F8={"state": "present", "confidence": 0.8})), frame_index=10),
        dict(validate_frame_labels(answer(F8={"state": "absent", "confidence": 0.6},
                                          F6={"state": "uncertain", "confidence": 0.4})), frame_index=20),
        dict(validate_frame_labels(answer(F9={"state": "present", "confidence": 0.3})), frame_index=30),
    ]
    agg = aggregate_clip(frames)
    assert agg["F8"]["label"] == "present" and agg["F8"]["confidence"] == 0.8
    assert agg["F8"]["frames_present"] == [10]
    assert agg["F1"]["label"] == "absent" and agg["F1"]["confidence"] == 0.9
    assert agg["F6"]["label"] == "uncertain"
    assert agg["F9"]["label"] == "uncertain"  # present but below threshold
    assert agg["F8"]["votes"] == {"present": 1, "absent": 2, "uncertain": 0}


def test_hint_agreement():
    agg = {"F8": {"label": "present"}, "F9": {"label": "uncertain"}, "F7": {"label": "absent"}}
    out = hint_agreement(agg, {"F8": "present", "F9": "present", "F7": "present"})
    assert out["F8"]["agree"] is True and out["F9"]["agree"] is None and out["F7"]["agree"] is False


def test_trailing_special_token_ignored():
    # seen from muse-glimmer:30b via Ollama on 2026-10-30 with reasoning_effort=none
    assert parse_and_validate(json.dumps(answer()) + "<|eot|>")["factors"]["F2"]["state"] == "absent"
