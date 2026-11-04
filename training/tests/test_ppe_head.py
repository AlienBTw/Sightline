"""PPE real-video labels, patch targets, freeze-except fine-tune, and eval."""
import importlib.util
import json
from pathlib import Path

import pytest

import train_head as th

REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _weak(clip_id, split, f1="absent"):
    factors = {f: {"label": "absent"} for f in th.GLIMMER_FACTORS}
    factors["F1"]["label"] = f1
    return {"clip_id": clip_id, "path": "x.mp4", "split": split, "source_label": "",
            "factors": factors, "factor_hints": {"F7": "absent", "F8": "absent", "F9": "absent"}}


def _voxel(clip_id, f2=1.0, f5=1.0):
    y, m = [0.0] * 10, [0.0] * 10
    y[0], m[0] = 1.0, 1.0
    y[1], m[1] = f2, 1.0
    y[4], m[4] = f5, 1.0
    return {"id": clip_id, "source": "voxel51", "split": "train", "set": "test", "n": 1, "y": y, "m": m}


def test_ppe_labels_mask_other_factors_and_hold_out_demo_eval():
    row = {"id": "pexels/1", "source": "ppe_video", "split": "train", "n": 1, "starts": [0],
           "y": [1.0] * 10, "m": [1.0] * 10}
    label = {
        "id": "pexels/1", "source": "pexels", "split": "train", "width": 672, "height": 384,
        "clip": {"F2": 1, "F5": None},
        "windows": [{"w": 0, "start": 0, "F2": 1, "F5": None,
                     "boxes": {"F2_pos": [[0.0, 0.0, 0.5, 1.0]], "F2_neg": [[0.0, 0.0, 0.25, 1.0]],
                               "F5_pos": [[0.0, 0.0, 1.0, 1.0]], "F5_neg": []}}],
    }
    th.apply_pexels_row(row, label, gh=2, gw=4, min_frac=0.30, neg_weight=2.0)
    assert row["ppe_item"] is True and row["ppe_source"] == "pexels"
    assert row["m"][0] == 0.0 and row["m"][5] == 0.0          # F1 and F6 stay masked
    assert row["y"][1] == 1.0 and row["m"][1] == 1.0          # clip F2
    assert row["m"][4] == 0.0                                 # clip F5 null
    spec = row["ppe_windows"][0]
    assert spec["F2"] == 1.0 and spec["F5"] is None
    assert "F5" not in spec["patch"] and "F2" in spec["patch"]
    assert spec["patch"]["F2"]["y"][0] == 0.0 and spec["patch"]["F2"]["w"][0] == 2.0
    assert spec["patch"]["F2"]["y"][1] == 1.0 and spec["patch"]["F2"]["w"][1] == 1.0
    assert th.assign_split(row, 0.9, 0) == "train"
    assert th.assign_split(row, 0.9, 0, cv=5, fold=0) == "train"
    label["split"] = "demo_eval"
    th.apply_pexels_row(row, label, 2, 4, 0.30, 2.0)
    assert row["split"] == "demo_eval"
    assert th.assign_split(row, 0.15, 0) == "skip"
    assert th.assign_split(row, 0.15, 0, cv=5, fold=4) == "skip"
    splits = [th.assign_split({"id": "pexels/z", "source": "ppe_video", "split": s}, 0.5, 0)
              for s in ("train", "val", "validation", "test", "demo_eval")]
    assert splits == ["train", "val", "val", "test", "skip"]


def test_voxel_override_and_glimmer_mask():
    rows = [_voxel("voxel51/a"), _voxel("voxel51/b", f2=1, f5=0)]
    labels = {"voxel51/a": {"id": "voxel51/a", "source": "voxel51", "split": "test",
                            "clip": {"F2": 0, "F5": 1}, "width": 672, "height": 384}}
    stats = th.apply_voxel_ppe_policy(rows, labels, True, True, 2, 2, 0.30, 2.0)
    assert stats == {"overridden": 1, "cleared": 1}
    assert rows[0]["y"][1] == 0.0 and rows[0]["m"][1] == 1.0
    assert rows[0]["y"][4] == 1.0 and rows[0]["m"][4] == 1.0
    assert rows[0]["y"][0] == 1.0 and rows[0]["m"][0] == 1.0   # F1 untouched
    assert rows[0]["ppe_source"] == "voxel51" and rows[0].get("ppe_item") is not True
    assert rows[1]["m"][1] == 0.0 and rows[1]["m"][4] == 0.0
    assert rows[1]["m"][0] == 1.0
    kept = [_voxel("voxel51/a")]
    th.apply_voxel_ppe_policy(kept, labels, override=False, mask=True, gh=1, gw=1, min_frac=0.3, neg_weight=2)
    assert kept[0]["m"][1] == 1.0 and kept[0]["y"][1] == 1.0   # a PPE row is not masked without override
    with pytest.raises(SystemExit):
        th.ppe_binary("maybe")


def test_ppe_patch_targets_follow_the_letterbox():
    target, weight = th.ppe_patch_from_boxes(
        [[0.0, 0.0, 0.5, 1.0]], [[0.0, 0.0, 0.25, 1.0]],
        672, 384, gh=2, gw=4, min_frac=0.30, neg_weight=2.0)
    assert target[0] == 0.0 and weight[0] == 2.0
    assert target[1] == 1.0 and weight[1] == 1.0
    assert weight[2] == 0.0 and weight[3] == 0.0
    assert target[4] == 0.0 and weight[4] == 2.0 and target[5] == 1.0
    assert th.ppe_patch_from_boxes([], [], 672, 384, 2, 4) is None
    # Square frame: side padding is ignored, the centre cell is the positive head.
    full, fw = th.ppe_patch_from_boxes([[0, 0, 1, 1]], [], 672, 672, gh=1, gw=21, min_frac=0.30, neg_weight=2.0)
    assert full[0] == 0.0 and fw[0] == 0.0 and full[10] == 1.0 and fw[10] == 1.0
    mapped = th.feature_mod().map_xyxy_letterbox([[0, 0, 1, 1]], 672, 672)
    mask = th.box_patch_mask(th.xyxy_to_cxcywh(mapped), 1, 21, 0.30)
    assert full[10] == mask[0][10]


def test_ppe_bg_neg_marks_only_patches_beyond_the_margin():
    # 4x4 grid, cell edges on multiples of 0.25. One positive cell at (1, 1).
    pos = [[0.25, 0.25, 0.5, 0.5]]
    target, weight = th.ppe_patch_from_boxes(pos, [], 672, 384, gh=4, gw=4, min_frac=0.30, neg_weight=2.0,
                                             bg_neg=0.5, bg_margin=1)
    assert target[5] == 1.0 and weight[5] == 1.0          # the box
    assert weight[6] == 0.0 and weight[1] == 0.0          # one patch away, still unlabeled
    assert target[7] == 0.0 and weight[7] == 0.5          # two patches away
    assert weight[15] == 0.5 and target[15] == 0.0
    # An explicit negative keeps its own weight, even when that cell is far from the positive.
    both_y, both_w = th.ppe_patch_from_boxes(
        pos, [[0.0, 0.0, 0.25, 0.25]], 672, 384, 4, 4, 0.30, 2.0, bg_neg=0.5, bg_margin=1)
    assert both_y[0] == 0.0 and both_w[0] == 2.0
    assert both_w[3] == 0.5
    # margin 0: the adjacent cell is a background negative too
    _y0, w0 = th.ppe_patch_from_boxes(pos, [], 672, 384, 4, 4, 0.30, 2.0, bg_neg=0.5, bg_margin=0)
    assert w0[6] == 0.5 and w0[5] == 1.0
    # default weight leaves the old ignore-outside-boxes targets
    old_y, old_w = th.ppe_patch_from_boxes(pos, [], 672, 384, 4, 4, min_frac=0.30, neg_weight=2.0)
    assert old_w[7] == 0.0 and old_y[5] == 1.0
    assert th.ppe_patch_from_boxes([], [], 672, 384, 4, 4, bg_neg=0.5) is None
    with pytest.raises(SystemExit):
        th.ppe_patch_from_boxes(pos, [], 672, 384, 4, 4, bg_neg=-0.1)


def _ppe_label(clip_id, source, f2_boxes, f5_boxes, f2=1, f5=1):
    return {
        "id": clip_id, "source": source, "split": "train", "width": 672, "height": 384,
        "clip": {"F2": f2, "F5": f5},
        "windows": [{"w": 0, "start": 0, "F2": f2, "F5": f5, "boxes": {
            "F2_pos": f2_boxes[0], "F2_neg": f2_boxes[1],
            "F5_pos": f5_boxes[0], "F5_neg": f5_boxes[1]}}],
    }


def test_bg_neg_applies_to_pexels_and_voxel_windows_that_have_a_box():
    pos = [[0.25, 0.25, 0.5, 0.5]]
    row = {"id": "pexels/1", "source": "ppe_video", "split": "train", "n": 1, "starts": [0],
           "y": [0.0] * 10, "m": [0.0] * 10}
    th.apply_pexels_row(row, _ppe_label("pexels/1", "pexels", (pos, []), ([], [])),
                        4, 4, 0.30, 2.0, bg_neg=0.5, bg_margin=1)
    spec = row["ppe_windows"][0]
    assert spec["patch"]["F2"]["w"][7] == 0.5 and spec["patch"]["F2"]["y"][7] == 0.0
    assert "F5" not in spec["patch"]
    empty = {"id": "pexels/2", "source": "ppe_video", "split": "train", "n": 1, "starts": [0],
             "y": [0.0] * 10, "m": [0.0] * 10}
    th.apply_pexels_row(empty, _ppe_label("pexels/2", "pexels", ([], []), ([], [])),
                        4, 4, 0.30, 2.0, bg_neg=0.5, bg_margin=1)
    assert empty["ppe_windows"][0]["patch"] == {}
    # A null window label stays masked even when SAM filled a box.
    unlabeled = _ppe_label("pexels/3", "pexels", (pos, []), ([], []))
    unlabeled["windows"][0]["F2"] = None
    row_u = {"id": "pexels/3", "source": "ppe_video", "split": "train", "n": 1, "starts": [0],
             "y": [0.0] * 10, "m": [0.0] * 10}
    th.apply_pexels_row(row_u, unlabeled, 4, 4, 0.30, 2.0, bg_neg=0.5, bg_margin=1)
    assert "F2" not in row_u["ppe_windows"][0]["patch"]
    voxel = _voxel("voxel51/a")
    voxel["n"] = 1
    voxel["starts"] = [0]
    lab = _ppe_label("voxel51/a", "voxel51", (pos, []), ([], [[0.0, 0.0, 0.25, 0.25]]))
    th.apply_voxel_ppe_policy([voxel], {"voxel51/a": lab}, True, False, 4, 4, 0.30, 2.0,
                              bg_neg=0.5, bg_margin=1)
    assert voxel["ppe_source"] == "voxel51"
    assert voxel["ppe_windows"][0]["patch"]["F2"]["w"][15] == 0.5
    f5 = voxel["ppe_windows"][0]["patch"]["F5"]
    assert f5["y"][0] == 0.0 and f5["w"][0] == 2.0 and f5["w"][3] == 0.5
    # Without the override, a Voxel51 row is not given PPE background negatives.
    plain = _voxel("voxel51/a")
    plain["n"] = 1
    th.apply_voxel_ppe_policy([plain], {"voxel51/a": lab}, False, False, 4, 4, 0.30, 2.0,
                              bg_neg=0.5, bg_margin=1)
    assert plain.get("ppe_windows") is None


def test_window_max_bce_uses_non_null_window_labels():
    torch = pytest.importorskip("torch")
    p_count = 4
    row = {"id": "pexels/t", "n": 2, "n_patches": p_count, "ppe_windows": [
        {"F2": 1.0, "F5": None, "patch": {}},
        {"F2": 0.0, "F5": 1.0, "patch": {}},
    ]}
    s = torch.full((1, 2 * p_count, 10), -8.0)
    s[0, 1, 1] = 8.0                                  # window 0, F2 max matches label 1
    s[0, p_count:, 4] = 8.0                           # window 1, F5 max matches label 1
    valid = torch.ones(1, 2 * p_count, dtype=torch.bool)
    matched = th.ppe_window_bce(s, valid, [row], [[0, 1]], s.device)
    assert float(matched) < 0.01
    # A present clip's empty frame (window label 0) is a negative on the window max.
    missed = s.detach().clone()
    missed[0, p_count:, 1] = 8.0
    penalised = th.ppe_window_bce(missed, valid, [row], [[0, 1]], s.device)
    assert float(penalised) > 1.0
    # Invalid patches do not set the max. The only valid F2 logit is -8, label 1.
    masked = torch.zeros(1, p_count, 10)
    masked[0, 0, 1] = 20.0
    masked[0, 1:, 1] = -8.0
    vmask = torch.tensor([[False, True, True, True]])
    one = {"n": 1, "n_patches": p_count, "ppe_windows": [{"F2": 1.0, "F5": None, "patch": {}}]}
    hidden = th.ppe_window_bce(masked, vmask, [one], [[0]], masked.device)
    assert float(hidden) > 1.0
    # A logit of 0 against a positive label is plain BCE, not a class-weighted one.
    zeros = torch.zeros(1, p_count, 10)
    exact = th.ppe_window_bce(zeros, torch.ones(1, p_count, dtype=torch.bool), [one], [[0]], zeros.device)
    assert abs(float(exact) - torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(0.0), torch.tensor(1.0))) < 1e-5
    # Dropped windows and null labels are not targets.
    assert th.ppe_window_bce(s, valid, [row], [[1]], s.device) is not None
    only_second = th.ppe_window_bce(s[:, p_count:], valid[:, p_count:], [row], [[1]], s.device)
    assert float(only_second) < 0.01
    blank = {"n": 1, "n_patches": p_count, "ppe_windows": [{"F2": None, "F5": None, "patch": {}}]}
    assert th.ppe_window_bce(zeros, torch.ones(1, p_count, dtype=torch.bool), [blank], [[0]], zeros.device) is None


def test_shwd_rows_stay_beside_ppe_rows_and_keep_cache_boxes():
    sh = {"id": "shwd/a", "source": "shwd", "split": "train", "y": [0.0] * 10, "m": [0.0] * 10,
          "bare_boxes": [[0.0, 0.0, 1.0, 1.0]], "hat_boxes": [], "width": 672, "height": 384, "patch": None}
    ppe = {"id": "pexels/1", "source": "ppe_video", "y": [0.0] * 10, "m": [0.0] * 10}
    out = th.apply_train_sources([sh, ppe], f2_box_source="shwd", cache_label="shwd-cache")
    assert [r["source"] for r in out] == ["shwd", "ppe_video"]
    assert out[0]["patch_factor"] == 1 and out[0]["y"][1] == 1.0
    # A manifest row without geometry must not hide the boxes already stored on the cache row.
    thin = {"id": "shwd/a", "source": "shwd"}
    kept = th.apply_train_sources([dict(sh)], f2_box_source="shwd", shwd_rows=[thin])
    assert kept[0]["patch_factor"] == 1
    with pytest.raises(SystemExit, match="no shwd rows"):
        th.apply_train_sources([ppe], f2_box_source="shwd", cache_label="/tmp/default-cache")


def test_ppe_sampling_share_does_not_change_the_zenodo_weight():
    base_rows = [{"source": "voxel51"}] * 70 + [{"source": "zenodo"}] * 30
    base = th.source_weights(base_rows, 0.3)
    mixed = base_rows + [{"source": "ppe_video", "ppe_item": True}] * 10
    full = th.source_weights(mixed, 0.3, ppe_share=0.25)
    assert abs(full["zenodo"] - base["zenodo"]) < 1e-12
    assert full["voxel51"] == 1.0
    mass = 70 * 1.0 + 30 * full["zenodo"]
    share = 10 * full["ppe_video"] / (mass + 10 * full["ppe_video"])
    assert abs(share - 0.25) < 1e-9
    only = [{"source": "voxel51"}] * 75 + [{"source": "ppe_video", "ppe_item": True}] * 25
    w = th.source_weights(only, 0.3, ppe_share=0.25)
    assert abs(w["ppe_video"] - 1.0) < 1e-9


def test_cache_source_filter_leaves_the_root_index_alone(tmp_path):
    (tmp_path / "index_r0.jsonl").write_text("{}\n")
    (tmp_path / "ppe_video").mkdir()
    (tmp_path / "ppe_video" / "index_r0.jsonl").write_text("{}\n")
    root_only = th.index_files_for(tmp_path, [])
    assert root_only == [tmp_path / "index_r0.jsonl"]
    both = th.index_files_for(tmp_path, ["ppe_video"])
    assert len(both) == 2
    cache = th.build_arg_parser().parse_args(["cache"])
    assert cache.sources == "" and cache.out == ""
    assert cache.cache.endswith("vjepa2_1_vitl_384_p2")
    train = th.build_arg_parser().parse_args(["train"])
    assert train.ppe_labels == "" and train.ppe_cache == "" and train.ppe_weight == 0.25
    assert train.ppe_override_voxel is False and train.mask_glimmer_f2f5 is False
    assert train.ppe_patch_loss == 0.0 and train.init_checkpoint == "" and train.freeze_except == ""
    assert train.ppe_bg_neg == 0.0 and train.ppe_bg_margin == 1.0
    assert train.ppe_window_labels is False and train.ppe_window_weight == 0.5
    assert train.f2_box_source == "" and train.hidden == 256 and train.ignore == "F3,F4"


def test_freeze_except_changes_only_f2_and_f5_rows():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = th.build_model(dim=32, hidden=8, drop=0.0)
    blob = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    model.load_state_dict(blob)
    keep = th.configure_freeze(model, ["F2", "F5"])
    assert keep == [1, 4]
    snap = th.snapshot_factor_rows(model)
    proj = model.proj[0].weight.detach().clone()
    norm = model.norm.weight.detach().clone()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-1, weight_decay=0.1)
    x = torch.randn(4, 6, 32)
    valid = torch.ones(4, 6, dtype=torch.bool)
    z, s, _w = model(x, valid)
    loss = z.sum() + s.sum()
    opt.zero_grad()
    loss.backward()
    th.mask_factor_grads(model, keep)
    opt.step()
    th.restore_factor_rows(model, snap, keep)
    assert torch.equal(model.proj[0].weight, proj)
    assert torch.equal(model.norm.weight, norm)
    assert model.proj[0].weight.requires_grad is False
    for layer, src in ((model.att.weight, snap["att_w"]), (model.att.bias, snap["att_b"]),
                       (model.val.weight, snap["val_w"]), (model.val.bias, snap["val_b"])):
        for i in range(layer.shape[0]):
            if i in (1, 4):
                assert not torch.allclose(layer[i], src[i])
            else:
                assert torch.equal(layer[i], src[i])


def test_eval_subcommand_writes_metrics_without_a_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    np.save(cache / "tokens.npy", rng.normal(size=(4, 4, 1024)).astype(np.float16))
    meta = []
    for i, split in enumerate(("train", "train", "test", "test")):
        meta.append({"id": f"voxel51/c{i}", "source": "voxel51", "split": split, "off": i, "n": 1,
                     "y": [0.0] * 10, "m": [0.0] * 10, "patch": None})
    (cache / "meta.jsonl").write_text("".join(json.dumps(r) + "\n" for r in meta), encoding="utf-8")
    labels = [_weak("voxel51/c0", "train", "present"), _weak("voxel51/c1", "train", "absent"),
              _weak("voxel51/c2", "test", "present"), _weak("voxel51/c3", "test", "absent")]
    lab = tmp_path / "labels.jsonl"
    lab.write_text("".join(json.dumps(r) + "\n" for r in labels), encoding="utf-8")
    model = th.build_model(hidden=8, drop=0.0)
    ckpt = tmp_path / "tiny.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": {"hidden": 8, "dropout": 0.0},
                "factors": th.FACTORS,
                "thresholds": {f: 0.5 for f in th.FACTORS}}, ckpt)
    results = tmp_path / "results"
    th.main(["eval", "--checkpoint", str(ckpt), "--name", "tiny_eval_case",
             "--cache", str(cache), "--labels", str(lab), "--results-dir", str(results), "--gpu", "0"])
    out = json.loads((results / "tiny_eval_case_eval.json").read_text(encoding="utf-8"))
    assert out["eval_only"] is True
    assert out["test_by_source"]["voxel51"]["F1"]["n"] == 2
    assert out["test_by_source"]["voxel51"]["F1"]["pos"] == 1
    assert "auroc" in out["test_by_source"]["voxel51"]["F1"]
    assert "F1" in out["test"] and "thresholds" in out
    assert not (results / "tiny_eval_case.pt").exists()
    assert th.guarded_output(results / "tiny_eval_case_eval.json")
    with pytest.raises(SystemExit):
        th.guarded_output(results / "h3_wide.json")
    with pytest.raises(SystemExit):
        th.guarded_output(tmp_path / "h3_wide.pt")


def test_replay_scores_incidents_and_circle_distance(tmp_path):
    replay = _load("replay_eval", REPO / "tools" / "labeling" / "replay_eval.py")
    steps = [{"t": 0, "p": {"F1": 0.1, "F2": 0.9, "F5": 0.2}},
             {"t": 1, "p": {"F1": 0.4, "F2": 0.8, "F5": 0.7}}]
    scores = replay.summarize_steps(steps, {"F1": 0.3, "F2": 0.5, "F5": 0.6})
    assert scores["F2"]["max"] == 0.9 and scores["F2"]["frac_above"] == 1.0
    assert scores["F1"]["frac_above"] == 0.5 and scores["F5"]["frac_above"] == 0.5
    assert replay.point_box_distance(50, 50, [0.0, 0.0, 1.0, 1.0], 100, 100) == 0.0
    assert abs(replay.point_box_distance(0, 50, [0.25, 0.0, 0.75, 1.0], 100, 100) - 25) < 1e-6
    mean = replay.mean_circle_distance(
        [{"cx": 50, "cy": 50, "w": 100, "h": 100}, {"cx": 0, "cy": 50, "w": 100, "h": 100}],
        [[0.25, 0.0, 0.75, 1.0]])
    assert abs(mean - 12.5) < 1e-6
    event = tmp_path / "events" / "evt_demo" 
    event.mkdir(parents=True)
    (event / "event.json").write_text(json.dumps({
        "factors": ["F1", "F2"],
        "segments": [{"factor": "F1"}, {"factor": "F5"}],
        "overlays": [
            {"factor": "F2", "t_ms": 0, "circle": {"cx": 10, "cy": 10, "w": 100, "h": 80}},
            {"factor": "F2", "t_ms": 0, "circle": {"cx": 10, "cy": 10, "w": 100, "h": 80}},
            {"factor": "F5", "t_ms": 100, "circle": {"cx": 80, "cy": 40, "w": 100, "h": 80}},
        ],
    }), encoding="utf-8")
    counts = replay.count_incidents(tmp_path)
    assert counts == {"F1": 1, "F2": 1, "F5": 1}
    circles = replay.circles_by_factor(tmp_path)
    assert len(circles["F2"]) == 1 and len(circles["F5"]) == 1
    ref = {"split": "demo_eval", "id": "pexels/1", "path": "data/demo_extra/clip.mp4",
           "windows": [{"boxes": {"F2_pos": [[0.0, 0.0, 0.2, 0.2]], "F5_pos": [[0.5, 0.0, 1.0, 1.0]]}}]}
    body = replay.summarize_clip(steps, {"F1": 0.3, "F2": 0.5, "F5": 0.6}, tmp_path, ref)
    assert body["circle_distance_px"]["F2"] == 0.0
    assert body["incidents"]["F1"] == 1
    yaml = replay.override_yaml("training/04-head/checkpoints/ppe_ft.pt")
    assert "muted_factors: [F6, F7, F8, F9, F10]" in yaml and "enabled: false" in yaml
    assert "head_checkpoint: training/04-head/checkpoints/ppe_ft.pt" in yaml
    clips = replay.load_demo_clips(tmp_path / "missing.jsonl")
    paths = [c["path"] for c in clips]
    assert "data/demo/sightline_demo.mp4" in paths
    assert "data/demo_extra/pexels_8987012_bare_head_grinder.mp4" in paths
    with pytest.raises(SystemExit):
        replay.assert_not_live(REPO / "data" / "live" / "events")


def test_compare_pass_rule(tmp_path):
    compare = _load("compare_heads", REPO / "tools" / "labeling" / "compare_heads.py")
    base = {"f1_auroc": 0.947, "zenodo_auroc": 0.992, "ppe_clip_all": {"F2": None, "F5": None}}
    good = {"f1_auroc": 0.940, "zenodo_auroc": 0.985, "ppe_clip_all": {"F2": 0.80, "F5": 0.75}}
    assert compare.passes(good, base)
    assert not compare.passes({"f1_auroc": 0.93, "zenodo_auroc": 0.992, "ppe_clip_all": {"F2": 0.9, "F5": 0.9}}, base)
    assert not compare.passes({"f1_auroc": 0.947, "zenodo_auroc": 0.992, "ppe_clip_all": {"F2": 0.74, "F5": 0.9}}, base)
    results = tmp_path / "results"
    results.mkdir()
    doc = {"test_by_source": {"voxel51": {"F1": {"auroc": 0.947, "ap": 0.14, "pos": 1, "n": 115}},
                              "zenodo": {"F1_patch_vs_boxes": {"auroc": 0.992, "ap": 0.94}}},
           "ppe": {"clip": {"all": {"F2": {"auroc": 0.8, "ap": 0.7, "pos": 4, "n": 8},
                                   "F5": {"auroc": 0.76, "ap": 0.6, "pos": 3, "n": 8}},
                            "pexels": {"F2": {"n": 0}, "F5": {"n": 0}},
                            "voxel51": {"F2": {"n": 0}, "F5": {"n": 0}}},
                   "window": {"all": {"F2": {"n": 0}, "F5": {"n": 0}},
                              "pexels": {"F2": {"n": 0}, "F5": {"n": 0}},
                              "voxel51": {"F2": {"n": 0}, "F5": {"n": 0}}}}}
    (results / "h3_wide.json").write_text(json.dumps({
        "test_by_source": doc["test_by_source"]}), encoding="utf-8")
    (results / "ppe_ft.json").write_text(json.dumps(doc), encoding="utf-8")
    compare.main(["--results", str(results), "--replay-root", str(tmp_path / "noreplay"),
                  "--out", str(results / "PPE_COMPARISON.md"), "--baseline", "h3_wide"])
    text = (results / "PPE_COMPARISON.md").read_text(encoding="utf-8")
    assert "| ppe_ft | PASS |" in text and "| h3_wide | baseline |" in text
