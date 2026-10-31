"""Offline tests for the weak-label run summary."""
import json

import summarize_labels as sl
from glimmer_schema import GLIMMER_FACTORS


def row(cid, labels=None, hints=None, votes=None, nk=4):
    labels = labels or {}
    facs = {}
    for f in GLIMMER_FACTORS:
        lab = labels.get(f, "absent")
        v = (votes or {}).get(f) or {"present": nk if lab == "present" else 0,
                                     "absent": nk if lab == "absent" else 0,
                                     "uncertain": nk if lab == "uncertain" else 0}
        facs[f] = {"label": lab, "confidence": 0.8, "votes": v, "frames_present": []}
    return {"clip_id": cid, "source": "voxel51_safe_unsafe", "split": "train", "source_label": "X",
            "factor_hints": hints or {}, "keyframes": list(range(nk)), "factors": facs, "duplicate_of": []}


def test_counts_clip_and_frame_level():
    s = sl.summarize([row("a", {"F2": "present"}), row("b", {"F2": "uncertain"}), row("c")])
    assert s["valid_rows"] == 3 and s["frames"] == 12 and not s["malformed"]
    assert s["clip_level"]["F2"] == {"present": 1, "absent": 1, "uncertain": 1}
    assert s["frame_level"]["F2"] == {"present": 4, "absent": 4, "uncertain": 4}


def test_agreement_confusion():
    rows = [row("p1", {"F8": "present"}, {"F8": "present"}),
            row("p2", {"F8": "uncertain"}, {"F8": "present"}),
            row("n1", {"F8": "present"}, {"F8": "absent"}),
            row("n2", {}, {"F8": "absent"})]
    a = sl.summarize(rows)["agreement"]["F8"]
    assert (a["tp"], a["fp"], a["fn"], a["tn"], a["uncertain"]) == (1, 1, 1, 1, 1)
    assert a["precision"] == 0.5 and a["recall"] == 0.5 and a["agree_when_decided"] == 0.667


def test_malformed_rows_and_bad_json(tmp_path):
    bad_votes = row("v", votes={"F1": {"present": 1, "absent": 1, "uncertain": 0}})
    no_f10 = row("m")
    del no_f10["factors"]["F10"]
    bad_label = row("l")
    bad_label["factors"]["F3"]["label"] = "maybe"
    p = tmp_path / "w.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [row("ok"), bad_votes, no_f10, bad_label]) + "\n{oops\n",
                 encoding="utf-8")
    rows, unparseable = sl.read_rows(p)
    s = sl.summarize(rows, unparseable)
    assert s["valid_rows"] == 1 and s["rows"] == 5
    probs = {m["clip_id"]: " ".join(m["problems"]) for m in s["malformed"]}
    assert "expected 4 keyframes" in probs["v"]
    assert "F10" in probs["m"] and "maybe" in probs["l"] and "invalid JSON" in probs[None]
    assert "Malformed rows" in sl.to_markdown(s, "t")