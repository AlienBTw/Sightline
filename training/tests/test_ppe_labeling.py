"""Split rule, F2/F5 geometry, merge, Muse parsing, and the call budget."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAB = str(ROOT / "tools" / "labeling")
if LAB not in sys.path:
    sys.path.insert(0, LAB)

import extract_features as ef
import muse_verify
import ppe_rules
import ppe_video as pv


def test_group_is_id_bucket_and_held_out_ids_share_it():
    assert pv.pexels_group(8987012) == 898
    assert pv.pexels_group(8987022) == pv.pexels_group(8987023) == 898
    assert pv.pexels_group(10817415) == 1081
    assert pv.pexels_group(5100044) == pv.pexels_group(5100048) == 510
    assert pv.held_out_groups() == {898, 429, 510, 745, 1081}
    assert pv.page_url(8987012) == "https://www.pexels.com/video/8987012/"


def test_held_out_groups_never_enter_train_and_groups_do_not_split():
    sizes = {898: 5, 429: 3, 510: 1, 12: 4, 12: 4, 80: 2, 81: 6}
    sizes[12] = 4
    out = pv.assign_group_splits(sizes, seed=0)
    assert out == pv.assign_group_splits(sizes, seed=0)
    for group in pv.held_out_groups():
        if group in out:
            assert out[group] == "demo_eval"
    assert out[898] == "demo_eval" and out[12] != "demo_eval"
    rows = [{"id": f"pexels/{g}0000", "group": g, "split": split} for g, split in out.items()]
    # A second clip in the same group cannot be given another split.
    rows.append({"id": "pexels/8987999", "group": 898, "split": "demo_eval"})
    assert pv.leakage_errors(rows) == []
    leaked = list(rows)
    leaked.append({"id": "pexels/120001", "group": 12, "split": "test" if out[12] != "test" else "train"})
    assert pv.leakage_errors(leaked)
    held_as_train = [{"id": "pexels/8987012", "group": 898, "split": "train"}]
    assert pv.leakage_errors(held_as_train)


def test_hash_split_is_about_70_15_15_and_test_has_a_floor():
    assert pv.split_from_unit(0.0) == "train"
    assert pv.split_from_unit(0.699999) == "train"
    assert pv.split_from_unit(0.70) == "val"
    assert pv.split_from_unit(0.849999) == "val"
    assert pv.split_from_unit(0.85) == "test"
    held = pv.held_out_groups()
    sizes = {g: 1 for g in range(1, 401) if g not in held}
    out = pv.assign_group_splits(sizes, seed=0)
    n = len(sizes)
    counts = {split: sum(1 for s in out.values() if s == split) for split in ("train", "val", "test", "demo_eval")}
    assert counts["demo_eval"] == 0
    assert counts["test"] >= pv.MIN_TEST_CLIPS
    assert 0.60 * n < counts["train"] < 0.80 * n
    assert 0.08 * n < counts["val"] < 0.25 * n
    assert 0.08 * n < counts["test"] < 0.25 * n
    assert counts["train"] + counts["val"] + counts["test"] == n


def test_test_floor_moves_whole_groups_and_stays_put_when_hash_already_has_test():
    sizes = {i: 1 for i in range(20)}
    out = pv.assign_group_splits(sizes, unit_fn=lambda g: 0.1, min_test=3)
    assert sum(split == "test" for split in out.values()) == 3
    assert out[19] == "test" and out[18] == "test" and out[17] == "test"
    assert out[0] == "train" and all(out[i] != "demo_eval" for i in sizes)
    # Highest hash moves first, even if it is currently val.
    mixed = pv.assign_group_splits(
        {1: 2, 2: 2, 3: 5},
        unit_fn=lambda g: {1: 0.1, 2: 0.75, 3: 0.2}[g],
        min_test=3,
    )
    assert mixed[2] == "test"  # closest to the test bucket
    assert mixed[3] == "train" or mixed[1] == "train"
    assert sum(size for g, size in {1: 2, 2: 2, 3: 5}.items() if mixed[g] == "test") >= 3


def test_voxel_dedup_matches_the_first_sha_and_the_weak_label_set():
    rows = [
        {"id": "voxel51/a", "source": "voxel51_safe_unsafe", "sha256": "h", "split": "train"},
        {"id": "voxel51/b", "source": "voxel51_safe_unsafe", "sha256": "h", "split": "test"},
        {"id": "other/c", "source": "zenodo_tablesaw", "sha256": "h"},
        {"id": "voxel51/c", "source": "voxel51_safe_unsafe", "sha256": "i", "split": "test"},
    ]
    assert [r["id"] for r in pv.dedup_voxel51(rows)] == ["voxel51/a", "voxel51/c"]
    weak = [{"clip_id": "voxel51/b"}, {"clip_id": "voxel51/c"}, {"clip_id": "voxel51/missing"}]
    picked = pv.voxel_rows_for_labeling(rows, weak)
    assert [r["id"] for r in picked] == ["voxel51/b", "voxel51/c"]
    assert picked[0]["split"] == "test"


def test_ppe_source_is_opt_in_and_does_not_change_voxel_items(tmp_path):
    assert ef.build_arg_parser().parse_args([]).sources == "voxel51,zenodo"
    (tmp_path / "data").mkdir()
    (tmp_path / "02-labeling" / "outputs").mkdir(parents=True)
    (tmp_path / "data" / "manifest.jsonl").write_text("")
    (tmp_path / "02-labeling" / "outputs" / "weak_labels_v2_vllm.jsonl").write_text("")
    assert ef.build_items(tmp_path, ["voxel51", "zenodo"]) == []
    row = {
        "id": "pexels/42", "source": "ppe_video", "kind": "video", "path": "/mnt/f/clips/pexels_42.mp4",
        "split": "train", "frames": 100,
    }
    man = tmp_path / "ppe.jsonl"
    man.write_text(json.dumps(row) + "\n")
    items = ef.build_items(tmp_path, ["ppe_video"], ppe_manifest=man)
    assert len(items) == 1 and items[0]["source"] == "ppe_video" and items[0]["kind"] == "video"
    assert items[0]["frames"] == 100 and items[0]["key"].startswith("pexels_") and "/" not in items[0]["key"]
    with pytest.raises(SystemExit):
        ef.build_items(tmp_path, ["ppe_video"], ppe_manifest=tmp_path / "missing.jsonl")
    bad = dict(row, source="shwd")
    man.write_text(json.dumps(bad) + "\n")
    with pytest.raises(SystemExit):
        ef.build_items(tmp_path, ["ppe_video"], ppe_manifest=man)


def test_keyframes_are_first_middle_last_of_the_feature_window():
    n = 400
    for start in ef.window_starts(n):
        indices = ef.window_indices(start, n)
        assert len(indices) == ef.FRAMES_PER_WINDOW
        assert ppe_rules.keyframe_indices(indices) == [indices[0], indices[len(indices) // 2], indices[-1]]


def test_head_and_torso_geometry_and_small_people():
    person = (100.0, 50.0, 220.0, 650.0)
    assert ppe_rules.head_region(person) == pytest.approx((124.0, 50.0, 196.0, 182.0))
    assert ppe_rules.torso_region(person) == pytest.approx((100.0, 170.0, 220.0, 410.0))
    assert ppe_rules.person_is_confident(person, 800)
    assert not ppe_rules.person_is_confident((0, 0, 30, 30), 800)
    assert not ppe_rules.person_is_confident((0, 0, 50, 50), 1000)  # 50 px but 5% of the frame
    assert ppe_rules.person_is_confident((0, 0, 80, 80), 1000)
    head = ppe_rules.head_region(person)
    inside = (130.0, 60.0, 190.0, 170.0)
    assert ppe_rules.ioa(inside, head) == pytest.approx(1.0)
    assert ppe_rules.ioa((0.0, 0.0, 10.0, 10.0), head) == 0.0


def test_head_box_is_bare_unless_a_hat_covers_it():
    person = {"x1": 10, "y1": 10, "x2": 110, "y2": 410, "c": 0.9}
    vest = {"x1": 10, "y1": 100, "x2": 110, "y2": 250, "c": 0.8}
    head = {"x1": 40, "y1": 20, "x2": 80, "y2": 70, "c": 0.88}
    bare = ppe_rules.label_frame([person], [], [vest], 200, 500, 0.5, heads=[head])
    assert bare["F2"] == "missing" and bare["F5"] == "ok"
    assert bare["boxes"]["F2_pos"] == [[0.2, 0.04, 0.4, 0.14]]
    assert bare["boxes"]["F2_neg"] == []
    hat = {"x1": 40, "y1": 20, "x2": 80, "y2": 55, "c": 0.77}
    covered = ppe_rules.label_frame([person], [hat], [vest], 200, 500, 0.5, heads=[head])
    assert covered["F2"] == "ok" and covered["boxes"]["F2_pos"] == []
    assert len(covered["boxes"]["F2_neg"]) == 2
    skim = {"x1": 40, "y1": 20, "x2": 80, "y2": 30, "c": 0.7}
    still_bare = ppe_rules.label_frame([person], [skim], [vest], 200, 500, 0.5, heads=[head])
    assert still_bare["F2"] == "missing"
    assert still_bare["boxes"]["F2_pos"] == [[0.2, 0.04, 0.4, 0.14]]
    assert len(still_bare["boxes"]["F2_neg"]) == 1
    # No head box: the top-of-person region is the bare head, same as heads omitted.
    geometric = ppe_rules.label_frame([person], [], [vest], 200, 500, 0.5, heads=[])
    omitted = ppe_rules.label_frame([person], [], [vest], 200, 500, 0.5)
    assert geometric["F2"] == omitted["F2"] == "missing"
    assert geometric["boxes"]["F2_pos"] == omitted["boxes"]["F2_pos"]
    assert geometric["boxes"]["F2_pos"] != bare["boxes"]["F2_pos"]


def test_prompt_set_and_normalised_scores_and_pexels_order():
    assert ppe_rules.SAM_PROMPTS == (
        "head", "hard hat", "helmet", "safety vest", "high-visibility vest", "person",
    )
    norm = ppe_rules.normalize_box_records(
        [{"x1": 10, "y1": 20, "x2": 30, "y2": 40, "c": 0.81234}], 100, 200,
    )
    assert norm == [{"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.2, "c": 0.8123}]
    back = ppe_rules.scale_box_records(norm, 100, 200)
    assert back[0]["x1"] == pytest.approx(10) and back[0]["c"] == pytest.approx(0.8123)
    persons, heads, hats, vests = ppe_rules.collect_detections({
        "person": [1], "head": [2], "hard hat": [3], "helmet": [4],
        "safety vest": [5], "high-visibility vest": [6],
    })
    assert persons == [1] and heads == [2] and hats == [3, 4] and vests == [5, 6]
    import sam_label
    assert sam_label.resolve_local_gpu(1, None) == (0, "1")
    assert sam_label.resolve_local_gpu(1, "1") == (0, None)
    assert sam_label.resolve_local_gpu(0, "1") == (0, None)
    assert sam_label.RULE_VERSION == 2
    rows = [
        {"id": "voxel51/a", "source": "voxel51_safe_unsafe", "split": "test", "window": 0, "key": "k", "start": 0},
        {"id": "pexels/9", "source": "ppe_video", "split": "train", "window": 0, "key": "p", "start": 0},
    ]
    assert [clip["id"] for clip in sam_label.jobs_from_index(rows, None, None)] == ["pexels/9", "voxel51/a"]


def test_frame_and_window_rules():
    # 200x500 frame. Person is tall enough. No hat -> bare head. Vest covers the torso.
    person = {"x1": 10, "y1": 10, "x2": 110, "y2": 410, "c": 0.9}
    vest = {"x1": 10, "y1": 100, "x2": 110, "y2": 250, "c": 0.8}
    labeled = ppe_rules.label_frame([person], [], [vest], 200, 500, score_threshold=0.5)
    assert labeled["F2"] == "missing" and labeled["F5"] == "ok"
    assert len(labeled["boxes"]["F2_pos"]) == 1 and labeled["boxes"]["F2_neg"] == []
    assert labeled["boxes"]["F5_pos"] == [] and len(labeled["boxes"]["F5_neg"]) == 1
    # Low score is ignored, so this person does not count.
    faint = dict(person, c=0.1)
    assert ppe_rules.label_frame([faint], [], [], 200, 500, 0.5)["F2"] == "none"
    tiny = {"x1": 0, "y1": 0, "x2": 20, "y2": 30, "c": 0.99}
    assert ppe_rules.label_frame([tiny], [], [], 200, 500, 0.5)["F2"] == "none"
    hat = {"x1": 30, "y1": 12, "x2": 90, "y2": 90, "c": 0.7}
    protected = ppe_rules.label_frame([person], [hat], [vest], 200, 500, 0.5)
    assert protected["F2"] == "ok" and protected["boxes"]["F2_pos"] == []
    assert len(protected["boxes"]["F2_neg"]) == 1
    assert ppe_rules.window_vote(["missing", "missing", "ok"]) == 1
    assert ppe_rules.window_vote(["ok", "ok", "ok"]) == 0
    assert ppe_rules.window_vote(["ok", "ok", "missing"]) is None
    assert ppe_rules.window_vote(["none", "none", "none"]) is None
    assert ppe_rules.window_vote(["ok", "ok", "none"]) == 0
    assert ppe_rules.window_vote(["missing", "none", "none"]) is None
    assert ppe_rules.window_vote(["missing", "missing", "none"]) == 1
    summary = ppe_rules.aggregate_window([labeled, labeled, protected])
    assert summary["F2"] == 1 and summary["F5"] == 0
    assert len(summary["boxes"]["F2_pos"]) == 2


def _win(w, f2, f5):
    return {"w": w, "start": w * 64, "F2": f2, "F5": f5, "boxes": ppe_rules.empty_boxes()}


def test_merge_rules():
    meta = {"id": "pexels/1", "source": "ppe_video", "split": "test", "path": "a.mp4",
            "license": "Pexels License", "attribution": "https://www.pexels.com/video/1/", "url": "u"}
    sam = {"windows": [_win(0, 1, 0), _win(1, 0, 1), _win(2, None, None)]}
    present = ppe_rules.merge_record(meta, sam, {"F2": "present", "F5": "absent", "reason": "bare head"})
    assert present["label_source"] == "muse"
    assert present["clip"] == {"F2": 1, "F5": 0}
    assert present["muse"]["F2"] == 1 and present["muse"]["F5"] == 0
    assert [w["F2"] for w in present["windows"]] == [1, None, None]
    assert [w["F5"] for w in present["windows"]] == [0, 0, 0]
    unsure = ppe_rules.merge_record(meta, sam, {"F2": "uncertain", "F5": "uncertain", "reason": ""})
    assert unsure["clip"] == {"F2": None, "F5": None}
    assert all(w["F2"] is None and w["F5"] is None for w in unsure["windows"])
    sam_only = ppe_rules.merge_record(meta, sam, None)
    assert sam_only["label_source"] == "sam" and sam_only["muse"] is None
    assert sam_only["clip"] == {"F2": 1, "F5": 1}
    all_clear = ppe_rules.merge_record(meta, {"windows": [_win(0, 0, 0), _win(1, 0, 0)]}, None)
    assert all_clear["clip"] == {"F2": 0, "F5": 0}
    mixed = ppe_rules.merge_record(meta, {"windows": [_win(0, 0, None)]}, None)
    assert mixed["clip"] == {"F2": 0, "F5": None}
    empty = ppe_rules.merge_record(meta, None, None)
    assert empty["label_source"] == "none" and empty["windows"] == [] and empty["clip"] == {"F2": None, "F5": None}
    specs = [{"w": 0, "start": 4}, {"w": 1, "start": 80}]
    muse_only = ppe_rules.merge_record(meta, None, {"F2": "present", "F5": "absent", "reason": "clip"}, specs)
    assert muse_only["label_source"] == "muse"
    assert [(w["w"], w["start"], w["F2"], w["F5"], w["boxes"]) for w in muse_only["windows"]] == [
        (0, 4, 1, 0, ppe_rules.empty_boxes()),
        (1, 80, 1, 0, ppe_rules.empty_boxes()),
    ]


def test_budget_refusal():
    with pytest.raises(ppe_rules.BudgetRefusal) as exc:
        ppe_rules.apply_claim(400, 400, 0, 2)
    assert exc.value.reason == "budget_total" and exc.value.total == 400
    assert ppe_rules.apply_claim(399, 400, 0, 2) == 400
    with pytest.raises(ppe_rules.BudgetRefusal) as exc:
        ppe_rules.apply_claim(10, 400, 2, 2)
    assert exc.value.reason == "max_calls"
    with pytest.raises(ppe_rules.BudgetRefusal) as exc:
        ppe_rules.apply_claim(0, 400, 0, 0)
    assert exc.value.reason == "max_calls"
    assert ppe_rules.api_calls_allowed(0, 0) is False
    assert ppe_rules.api_calls_allowed(3, 4) is True
    assert ppe_rules.api_calls_allowed(4, 4) is False


def test_muse_json_parsing():
    ok = ppe_rules.parse_muse_json('{"F2":"present","F5":"absent","reason":"one bare head"}')
    assert ok["parsed"] is True and ok["F2"] == "present" and ok["F5"] == "absent"
    fenced = ppe_rules.parse_muse_json('```json\n{"F2": "uncertain", "F5": "present", "reason": "vest"}\n```')
    assert fenced["parsed"] is True and fenced["F2"] == "uncertain" and fenced["F5"] == "present"
    for text in ("not json", '{"F2":"yes","F5":"absent"}', "", '{"F2":"present"}'):
        bad = ppe_rules.parse_muse_json(text)
        assert bad["parsed"] is False
        assert bad["F2"] == "uncertain" and bad["F5"] == "uncertain"
    assert ppe_rules.verdict_bit("present") == 1
    assert ppe_rules.verdict_bit("absent") == 0
    assert ppe_rules.verdict_bit("uncertain") is None


def test_muse_frame_choice_prefers_a_positive_window():
    windows = [
        {"w": 0, "paths": ["a0", "a1", "a2"]},
        {"w": 1, "paths": ["b0", "b1", "b2"]},
        {"w": 2, "paths": ["c0", "c1", "c2"]},
    ]
    sam = {0: {"F2": 0, "F5": 0}, 2: {"F2": 1, "F5": None}}
    assert ppe_rules.choose_muse_paths(windows, sam) == ["c0", "c1", "c2"]
    assert ppe_rules.choose_muse_paths(windows, {}) == ["a0", "b1", "c2"]
    index_style = [{"window": row["w"], "paths": row["paths"]} for row in windows]
    assert ppe_rules.choose_muse_paths(index_style, {}) == ["a0", "b1", "c2"]
    assert ppe_rules.priority_key({"split": "test", "id": "b"}) < ppe_rules.priority_key({"split": "train", "id": "a"})


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _label_tree(tmp_path: Path, clips: list[dict]) -> dict:
    paths = {
        "manifest": tmp_path / "ppe.jsonl",
        "voxel": tmp_path / "voxel.jsonl",
        "weak": tmp_path / "weak.jsonl",
        "index": tmp_path / "index.jsonl",
        "sam": tmp_path / "sam.jsonl",
        "out": tmp_path / "out.jsonl",
        "budget": tmp_path / "budget.json",
        "log": tmp_path / "calls.jsonl",
        "ids": tmp_path / "ids.txt",
    }
    _write_jsonl(paths["manifest"], [row for row in clips if row.get("source") == pv.SOURCE])
    _write_jsonl(paths["voxel"], [row for row in clips if row.get("source") == pv.VOXEL_SOURCE])
    _write_jsonl(paths["index"], [
        {"id": row["id"], "window": 0, "paths": [f"{row['id']}-0", f"{row['id']}-1", f"{row['id']}-2"]}
        for row in clips
    ])
    return paths


def _argv(paths: dict, extra: list[str]) -> list[str]:
    return [
        "--manifest", str(paths["manifest"]),
        "--voxel-manifest", str(paths["voxel"]),
        "--weak-labels", str(paths["weak"]),
        "--index", str(paths["index"]),
        "--sam", str(paths["sam"]),
        "--out", str(paths["out"]),
        "--budget", str(paths["budget"]),
        "--call-log", str(paths["log"]),
        "--source", "all",
        *extra,
    ]


def test_muse_model_precedence_is_cli_then_env_then_contributor(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SIGHTLINE_NO_DOTENV", "1")
    monkeypatch.delenv("MUSE_SPARK_MODEL", raising=False)
    args = muse_verify.build_arg_parser().parse_args([])
    assert args.max_tokens == 4000 and args.order == "priority" and args.redo_parse_errors is False
    assert muse_verify.DEFAULT_MUSE_MODEL == "muse-spark-1.3-contributor"
    assert muse_verify.resolve_model(None) == "muse-spark-1.3-contributor"
    assert muse_verify.resolve_model("  ") == "muse-spark-1.3-contributor"
    paths = _label_tree(tmp_path, [])
    assert muse_verify.main(_argv(paths, ["--dry-run", "--max-calls", "1"])) == 0
    assert "muse model=muse-spark-1.3-contributor" in capsys.readouterr().out
    monkeypatch.setenv("MUSE_SPARK_MODEL", "   ")
    assert muse_verify.resolve_model(None) == "muse-spark-1.3-contributor"
    monkeypatch.setenv("MUSE_SPARK_MODEL", "from-env")
    assert muse_verify.resolve_model(None) == "from-env"
    assert muse_verify.resolve_model("from-cli") == "from-cli"
    assert muse_verify.main(_argv(paths, ["--dry-run", "--max-calls", "1", "--model", "from-cli"])) == 0
    assert "muse model=from-cli" in capsys.readouterr().out
    assert muse_verify.main(_argv(paths, ["--dry-run", "--max-calls", "1"])) == 0
    assert "muse model=from-env" in capsys.readouterr().out
    assert not paths["budget"].exists()
    assert not paths["log"].exists()


def test_answer_text_falls_back_to_reasoning_when_content_has_no_json(monkeypatch):
    verdict = '{"F2":"absent","F5":"present","reason":"vest"}'
    assert verdict in muse_verify.answer_text({"content": "", "reasoning": "notes " + verdict})
    assert muse_verify.answer_text({"content": "no braces"}, {"reasoning_content": verdict}) == verdict
    listed = muse_verify.answer_text({
        "content": [{"type": "text", "text": '{"F2":"present","F5":"absent","reason":"content"}'}],
        "reasoning": verdict,
    })
    assert '"F2":"present"' in listed
    kept = muse_verify.answer_text({
        "content": '{"F2":"nope","F5":"absent","reason":"bad"}',
        "reasoning": verdict,
    })
    assert "nope" in kept
    message_text, assistant_text, reasoning_text = muse_verify._text_helpers()
    from sightline_pipeline.verify_gate import _assistant_text, _message_text, _reasoning_text
    assert (message_text, assistant_text, reasoning_text) == (_message_text, _assistant_text, _reasoning_text)
    monkeypatch.setattr(muse_verify, "_text_helpers", lambda: (
        muse_verify._message_text_local,
        muse_verify._assistant_text_local,
        muse_verify._reasoning_text_local,
    ))
    replica = muse_verify.answer_text(
        {"content": None, "reasoning_content": '{"F2":"present","F5":"absent","reason":"x"}'},
    )
    assert '"F2":"present"' in replica


def test_ask_muse_reads_reasoning_and_matches_verify_client_limits(monkeypatch):
    monkeypatch.setenv("SIGHTLINE_NO_DOTENV", "1")
    secret = "unit-test-key-not-for-logs"
    monkeypatch.setenv("MODEL_API_KEY", secret)
    monkeypatch.delenv("META_API_KEY", raising=False)
    import requests
    import sightline_pipeline.escalation as escalation
    import types
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.imread = lambda path, flag=None: object()
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(escalation, "jpeg_data_url", lambda image, long_side=768: "data:image/jpeg;base64,AA")
    answer = '{"F2":"present","F5":"absent","reason":"bare head"}'
    box = [{
        "choices": [{
            "finish_reason": "length",
            "message": {"content": "", "reasoning_content": secret + (" thinking" * 40) + answer},
        }],
        "usage": {"completion_tokens": 400, "completion_tokens_details": {"reasoning_tokens": 398}},
    }]
    bodies = []

    class _Resp:
        status_code = 200

        @property
        def text(self):
            return json.dumps(box[0])

        def json(self):
            return box[0]

    def fake_post(url, json=None, timeout=None, headers=None):
        bodies.append(json)
        assert headers["Authorization"] == f"Bearer {secret}"
        assert url.endswith("/chat/completions")
        return _Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    result = muse_verify.ask_muse(["a.jpg", "b.jpg", "c.jpg"], "muse-spark-1.3-contributor", 12.0)
    assert bodies[0]["max_tokens"] == 4000 and bodies[0]["temperature"] == 0
    assert bodies[0]["model"] == "muse-spark-1.3-contributor"
    assert result["verdict"]["parsed"] is True and result["verdict"]["F2"] == "present"
    assert result["finish_reason"] == "length"
    assert len(result["raw_snippet"]) == 300
    assert secret not in result["raw_snippet"] and "[redacted]" in result["raw_snippet"]
    assert secret not in json.dumps(result)
    box[0] = {"choices": [{"finish_reason": "length", "message": {
        "content": "no object in the content",
        "reasoning_content": '{"F2":"uncertain","F5":"present","reason":"vest"}',
    }}]}
    result = muse_verify.ask_muse(["a.jpg", "b.jpg", "c.jpg"], "muse-spark-1.3-contributor", 12.0, 2222)
    assert bodies[-1]["max_tokens"] == 2222
    assert result["verdict"]["F2"] == "uncertain" and result["verdict"]["F5"] == "present"
    assert result["raw_snippet"].startswith("no object") is False
    assert "vest" in result["raw_snippet"]
    box[0] = {"choices": [{"finish_reason": "length", "message": {"content": "", "reasoning": "still thinking"}}]}
    result = muse_verify.ask_muse(["a.jpg", "b.jpg", "c.jpg"], "muse-spark-1.3-contributor", 12.0)
    assert result["verdict"]["parsed"] is False and result["verdict"]["reason"] == "parse_error"
    assert result["finish_reason"] == "length" and result["raw_snippet"] == "still thinking"


def test_limit_source_split_order_and_ids(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SIGHTLINE_NO_DOTENV", "1")
    monkeypatch.delenv("MUSE_SPARK_MODEL", raising=False)
    clips = [
        {"id": "a-train", "source": pv.SOURCE, "split": "train"},
        {"id": "b-test", "source": pv.SOURCE, "split": "test"},
        {"id": "v-test", "source": pv.VOXEL_SOURCE, "split": "test", "sha256": "vtest"},
        {"id": "v-train", "source": pv.VOXEL_SOURCE, "split": "train", "sha256": "vtrain"},
    ]
    paths = _label_tree(tmp_path, clips)
    with pytest.raises(SystemExit):
        muse_verify.parse_source_splits("voxel51_safe_unsafe")
    pairs = muse_verify.parse_source_splits("voxel51_safe_unsafe:test, ppe_video:train")
    picked = muse_verify.select_clips(
        paths["manifest"], "all", None, paths["voxel"], paths["weak"], pairs, "priority",
    )
    assert [row["id"] for row in picked] == ["v-test", "a-train"]
    assert [row["id"] for row in muse_verify.select_clips(
        paths["manifest"], pv.SOURCE, None, paths["voxel"], paths["weak"], None, "priority",
    )] == ["b-test", "a-train"]
    assert [row["id"] for row in muse_verify.select_clips(
        paths["manifest"], pv.SOURCE, None, paths["voxel"], paths["weak"], None, "id",
    )] == ["a-train", "b-test"]
    paths["ids"].write_text("v-train\nv-test\n# comment\n", encoding="utf-8")
    only = muse_verify.parse_source_splits("voxel51_safe_unsafe:test")
    assert [row["id"] for row in muse_verify.select_clips(
        paths["manifest"], "all", muse_verify.read_ids(paths["ids"]), paths["voxel"], paths["weak"], only, "id",
    )] == ["v-test"]
    assert muse_verify.main(_argv(paths, [
        "--dry-run", "--max-calls", "1", "--limit-source-split", "voxel51_safe_unsafe:test", "--ids", str(paths["ids"]),
    ])) == 0
    out = capsys.readouterr().out
    assert "muse model=muse-spark-1.3-contributor" in out and "max_tokens=4000" in out
    assert "id=v-test" in out and "id=v-train" not in out and "id=b-test" not in out
    assert muse_verify.main(_argv(paths, ["--dry-run", "--max-calls", "2", "--source", "ppe_video", "--order", "id"])) == 0
    out = capsys.readouterr().out
    assert out.index("id=a-train") < out.index("id=b-test")
    assert muse_verify.main(_argv(paths, [
        "--dry-run", "--max-calls", "2", "--source", "ppe_video", "--order", "priority",
    ])) == 0
    out = capsys.readouterr().out
    assert out.index("id=b-test") < out.index("id=a-train")
    assert not paths["budget"].exists() and not paths["log"].exists()


def test_redo_parse_errors_spend_budget_and_default_skips_them(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGHTLINE_NO_DOTENV", "1")
    monkeypatch.delenv("MUSE_SPARK_MODEL", raising=False)
    secret = "unit-test-key-not-for-logs"
    monkeypatch.setenv("MODEL_API_KEY", secret)
    clips = [
        {"id": "fresh", "source": pv.SOURCE, "split": "train"},
        {"id": "retry", "source": pv.SOURCE, "split": "test"},
        {"id": "keep", "source": pv.SOURCE, "split": "test"},
        {"id": "transport", "source": pv.SOURCE, "split": "val"},
    ]
    paths = _label_tree(tmp_path, clips)
    verdicts = [
        {"id": "retry", "source": pv.SOURCE, "split": "test", "F2": "uncertain", "F5": "uncertain",
         "reason": "parse_error", "parsed": False, "model": "old"},
        {"id": "keep", "source": pv.SOURCE, "split": "test", "F2": "absent", "F5": "absent",
         "reason": "hats and vests", "parsed": True, "model": "old"},
        {"id": "transport", "source": pv.SOURCE, "split": "val", "F2": "uncertain", "F5": "uncertain",
         "reason": "request_error", "parsed": False, "model": "old"},
    ]
    seen = []

    def fake_ask(frame_paths, model, timeout, max_tokens=4000):
        seen.append((frame_paths[0].split("-")[0], model, max_tokens))
        return {
            "http_status": 200, "latency_s": 0.01, "usage": {"completion_tokens": 20},
            "verdict": {"F2": "present", "F5": "absent", "reason": "bare head", "parsed": True},
            "error": None, "finish_reason": "stop", "raw_snippet": '{"F2":"present"',
        }

    monkeypatch.setattr(muse_verify, "ask_muse", fake_ask)

    def reset():
        seen.clear()
        _write_jsonl(paths["out"], verdicts)
        paths["budget"].write_text(json.dumps({"total": 7}) + "\n", encoding="utf-8")
        if paths["log"].exists():
            paths["log"].unlink()

    reset()
    assert muse_verify.main(_argv(paths, ["--max-calls", "5", "--source", pv.SOURCE, "--budget-total", "400"])) == 0
    assert seen == [("fresh", "muse-spark-1.3-contributor", 4000)]
    assert json.loads(paths["budget"].read_text())["total"] == 8
    reset()
    assert muse_verify.main(_argv(paths, [
        "--max-calls", "5", "--source", pv.SOURCE, "--budget-total", "400", "--redo-parse-errors",
    ])) == 0
    assert seen == [
        ("retry", "muse-spark-1.3-contributor", 4000),
        ("fresh", "muse-spark-1.3-contributor", 4000),
    ]
    assert json.loads(paths["budget"].read_text())["total"] == 9
    logged = [json.loads(line) for line in paths["log"].read_text().splitlines()]
    assert [row["id"] for row in logged] == ["retry", "fresh"]
    assert all(row["finish_reason"] == "stop" and row["raw_snippet"] == '{"F2":"present"' for row in logged)
    assert all(secret not in json.dumps(row) for row in logged)
    stored = {row["id"]: row for row in pv.read_jsonl(paths["out"])}
    assert stored["retry"]["reason"] == "bare head" and stored["retry"]["parsed"] is True
    assert stored["keep"]["reason"] == "hats and vests"
    assert stored["transport"]["reason"] == "request_error"
    assert stored["fresh"]["F2"] == "present"


def test_redo_parse_error_still_stops_at_the_budget_cap(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SIGHTLINE_NO_DOTENV", "1")
    monkeypatch.delenv("MUSE_SPARK_MODEL", raising=False)
    clips = [{"id": "retry", "source": pv.SOURCE, "split": "test"}]
    paths = _label_tree(tmp_path, clips)
    _write_jsonl(paths["out"], [{
        "id": "retry", "source": pv.SOURCE, "split": "test", "F2": "uncertain", "F5": "uncertain",
        "reason": "parse_error", "parsed": False, "model": "old",
    }])
    paths["budget"].write_text(json.dumps({"total": 400}) + "\n", encoding="utf-8")
    called = []
    monkeypatch.setattr(muse_verify, "ask_muse", lambda *args, **kwargs: called.append(args))
    assert muse_verify.main(_argv(paths, [
        "--max-calls", "2", "--source", pv.SOURCE, "--budget-total", "400", "--redo-parse-errors",
    ])) == 0
    assert called == []
    assert json.loads(paths["budget"].read_text())["total"] == 400
    assert pv.read_jsonl(paths["out"])[0]["reason"] == "parse_error"
    before = paths["budget"].read_text()
    assert muse_verify.main(_argv(paths, [
        "--dry-run", "--max-calls", "1", "--source", pv.SOURCE, "--redo-parse-errors",
    ])) == 0
    assert paths["budget"].read_text() == before
    assert "id=retry" in capsys.readouterr().out
    assert not paths["log"].exists()
