"""SHWD manifest, F2 box targets, and the opt-in defaults around them."""
import json
from copy import deepcopy

import pytest

import build_shwd_manifest as bsm
import extract_features as ef
import train_head as th


def _xml(name, w, h, objects):
    body = [
        f"<annotation><filename>{name}</filename>"
        f"<size><width>{w}</width><height>{h}</height><depth>3</depth></size>"
    ]
    for cls, (x0, y0, x1, y1) in objects:
        body.append(
            "<object><name>%s</name><difficult>0</difficult><bndbox>"
            "<xmin>%s</xmin><ymin>%s</ymin><xmax>%s</xmax><ymax>%s</ymax>"
            "</bndbox></object>" % (cls, x0, y0, x1, y1)
        )
    body.append("</annotation>")
    return "".join(body)


def _voc(root, trainval, test, bare):
    voc = root / "VOC2028"
    ann, jpeg, sets = voc / "Annotations", voc / "JPEGImages", voc / "ImageSets" / "Main"
    for p in (ann, jpeg, sets):
        p.mkdir(parents=True)
    for stem in list(trainval) + list(test):
        (jpeg / f"{stem}.jpg").write_bytes(b"\xff")
        objs = [("person", (10, 10, 40, 40))] if stem in bare else [("hat", (12, 12, 30, 36))]
        (ann / f"{stem}.xml").write_text(_xml(f"{stem}.jpg", 100, 80, objs))
    (sets / "trainval.txt").write_text("\n".join(trainval) + "\n")
    (sets / "test.txt").write_text("\n".join(test) + "\n")


def test_parse_voc_person_and_hat(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text(_xml("a.jpg", 200, 100, [("person", (10, 20, 110, 70)), ("hat", (120, 10, 180, 60))]))
    d = bsm.parse_voc_xml(p)
    assert d["width"] == 200 and d["height"] == 100
    assert d["bare_boxes"] == [[0.05, 0.2, 0.55, 0.7]]
    assert d["hat_boxes"] == [[0.6, 0.1, 0.9, 0.6]]
    assert d["unknown"] == {}
    p.write_text(_xml("a.jpg", 10, 10, [("helmet", (1, 1, 4, 4)), ("person", (1, 1, 5, 5))]))
    d = bsm.parse_voc_xml(p)
    assert d["unknown"] == {"helmet": 1}
    assert len(d["bare_boxes"]) == 1 and d["hat_boxes"] == []


def test_trainval_hash_is_stable_and_about_15_percent():
    stems = [f"{i:04d}" for i in range(400)]
    splits = [bsm.trainval_split(s) for s in stems]
    assert splits == [bsm.trainval_split(s, 0.15, 0) for s in stems]
    n_val = splits.count("validation")
    assert 40 < n_val < 80
    assert bsm.trainval_split("same", 0.0, 0) == "train"


def test_stratified_subsample_is_half_bare_and_stable():
    rows = [{"id": f"shwd/{i:03d}", "n_bare": int(i < 30)} for i in range(60)]
    kept = bsm.subsample_stratified(rows, 10, 0)
    assert len(kept) == 10
    assert sum(r["n_bare"] >= 1 for r in kept) == 5
    assert bsm.subsample_stratified(rows, 10, 0) == kept
    short = [{"id": f"p{i}", "n_bare": 1} for i in range(3)] + [{"id": f"n{i}", "n_bare": 0} for i in range(20)]
    filled = bsm.subsample_stratified(short, 10, 0)
    assert len(filled) == 10 and sum(r["n_bare"] >= 1 for r in filled) == 3
    assert bsm.subsample_stratified(rows, 0, 0) == rows


def test_manifest_holds_out_test_and_stratifies_train(tmp_path):
    trainval = [f"{i:04d}" for i in range(40)]
    test = [f"t{i:02d}" for i in range(8)]
    bare = set(trainval[:20])
    root = tmp_path / "shwd"
    _voc(root, trainval, test, bare)
    rows = bsm.build_manifest(root, max_train=10, max_eval=4, val_frac=0.0, seed=0)
    assert rows == bsm.build_manifest(root, max_train=10, max_eval=4, val_frac=0.0, seed=0)
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)
        assert r["source"] == "shwd" and r["kind"] == "image" and r["id"] == "shwd/" + r["id"].split("/", 1)[1]
        assert set(r) == {
            "id", "source", "kind", "path", "width", "height", "split",
            "bare_boxes", "hat_boxes", "n_bare", "n_hat",
        }
    train = by_split["train"]
    assert len(train) == 10
    assert sum(r["n_bare"] >= 1 for r in train) == 5
    assert len(by_split["test"]) == 4
    assert by_split.get("validation", []) == []
    test_ids = {f"shwd/{s}" for s in test}
    assert {r["id"] for r in by_split["test"]} <= test_ids
    assert not any(r["id"] in test_ids for r in train)
    # A second seed-stable build with the default val fraction still never trains on test.txt.
    mixed = bsm.build_manifest(root, max_train=3000, max_eval=1000, val_frac=0.15, seed=0)
    assert all(r["split"] == "test" for r in mixed if r["id"] in test_ids)
    assert any(r["split"] == "validation" for r in mixed)


def test_manifest_cli_defaults_and_print(tmp_path, capsys):
    args = bsm.build_arg_parser().parse_args([])
    assert args.max_train == 3000 and args.max_eval == 1000 and args.val_frac == 0.15
    root = tmp_path / "shwd"
    _voc(root, ["0001", "0002"], ["0009"], {"0001"})
    out = tmp_path / "shwd_manifest.jsonl"
    bsm.main(["--root", str(root), "--out", str(out), "--max-train", "10", "--max-eval", "10"])
    text = capsys.readouterr().out
    assert "train" in text and "validation" in text and "test" in text
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["split"] for r in rows if r["id"] == "shwd/0009"} == {"test"}
    assert rows[0]["path"].endswith("0001.jpg") or rows[-1]["path"].endswith(".jpg")


def test_letterbox_and_f2_patch_mask():
    assert ef.letterbox_content(672, 384) == (0, 0, 672, 384)
    assert ef.letterbox_content(672, 672) == (144, 0, 384, 384)
    assert ef.letterbox_content(200, 100) == (0, 24, 672, 336)
    box = [(0.1, 0.2, 0.4, 0.5)]
    mapped = ef.map_xyxy_letterbox(box, 672, 384)
    assert len(mapped) == 1
    assert all(abs(a - b) < 1e-9 for a, b in zip(mapped[0], box[0]))
    assert th.feature_mod().map_xyxy_letterbox(box, 800, 600) == ef.map_xyxy_letterbox(box, 800, 600)
    # default min_frac stays 0.25; F2 asks for 0.30
    assert th.box_patch_mask([(0.5, 0.5, 0.5, 0.5)], 1, 1) == [[1.0]]
    assert th.box_patch_mask([(0.5, 0.5, 0.5, 0.5)], 1, 1, min_frac=0.30) == [[0.0]]
    target, weight, clip = th.shwd_patch_targets(
        [(0.0, 0.0, 0.5, 1.0)], [(0.0, 0.0, 0.25, 1.0)], 672, 384, gh=2, gw=4, min_frac=0.30, hat_weight=2.0)
    assert clip == 1.0
    assert target == [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    assert weight[0] == [2.0, 1.0, 1.0, 1.0]
    wiped, w2, clip0 = th.shwd_patch_targets(
        [(0.0, 0.0, 1.0, 1.0)], [(0.0, 0.0, 1.0, 1.0)], 672, 384, 2, 4, 0.30, 2.0)
    assert clip0 == 0.0
    assert all(v == 0.0 for row in wiped for v in row)
    assert all(v == 2.0 for row in w2 for v in row)
    # square frame: side padding is a plain negative, the centre is a bare head
    padded, pw, pclip = th.shwd_patch_targets([(0, 0, 1, 1)], [], 672, 672, gh=1, gw=21, min_frac=0.30)
    assert pclip == 1.0 and padded[0][0] == 0.0 and padded[0][10] == 1.0 and pw[0][0] == 1.0


def test_read_still_window_letterboxes_and_repeats(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :] = (0, 0, 255)
    path = tmp_path / "a.png"
    assert cv2.imwrite(str(path), img)
    arr = ef.read_still_window(str(path))
    assert arr.shape == (ef.FRAMES_PER_WINDOW, ef.IN_H, ef.IN_W, 3)
    assert np.array_equal(arr[0], arr[-1])
    pad_x, pad_y, _, _ = ef.letterbox_content(200, 100)
    assert arr[0, 0, 0].tolist() == [0, 0, 0]
    assert arr[0, pad_y, pad_x].tolist() == [255, 0, 0]


def _row(source, clip_id, y1=1.0, m1=1.0, split_set="train"):
    y, m = [0.0] * 10, [0.0] * 10
    y[1], m[1] = y1, m1
    m[0] = 1.0
    return {"id": clip_id, "source": source, "split": "train", "set": split_set, "y": y, "m": m, "patch": [[1, 0]]}


def test_f2_defaults_leave_voxel_and_zenodo_unchanged():
    voxel = _row("voxel51", "v")
    zen = _row("zenodo", "z")
    zen["patch"] = [[1, 0], [0, 0]]
    sh = _row("shwd", "shwd/a")
    sh.update(bare_boxes=[[0, 0, 1, 1]], hat_boxes=[], width=672, height=384, patch=None)
    snap_v, snap_z = deepcopy(voxel), deepcopy(zen)
    out = th.apply_train_sources([voxel, zen, sh])
    assert [r["source"] for r in out] == ["voxel51", "zenodo"]
    assert out[0]["y"] == snap_v["y"] and out[0]["m"] == snap_v["m"]
    assert out[1]["patch"] == snap_z["patch"] and out[1]["y"] == snap_z["y"]
    bare = th.apply_train_sources([deepcopy(sh)], f2_box_source="shwd")
    assert bare[0]["m"] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert bare[0]["y"][1] == 1.0 and bare[0]["patch_factor"] == 1
    empty = deepcopy(sh)
    empty["bare_boxes"] = []
    neg = th.apply_train_sources([empty], f2_box_source="shwd")
    assert neg[0]["y"][1] == 0.0 and neg[0]["m"][1] == 1.0 and sum(neg[0]["m"]) == 1
    with pytest.raises(SystemExit):
        th.apply_train_sources([_row("voxel51", "v")], f2_box_source="shwd")
    dropped = th.apply_train_sources([_row("voxel51", "v"), _row("zenodo", "z"), _row("shwd", "shwd/a")], use_zenodo=0)
    assert [r["source"] for r in dropped] == ["voxel51"]


def test_drop_glimmer_f2_clears_train_only():
    rows = []
    for split_set in ("train", "val", "test"):
        rows.append(_row("voxel51", f"v-{split_set}", split_set=split_set))
    rows.append(_row("zenodo", "z"))
    assert th.mask_glimmer_f2(rows) == 1
    assert rows[0]["m"][1] == 0.0 and rows[0]["y"][1] == 0.0
    assert rows[0]["m"][0] == 1.0
    assert rows[1]["m"][1] == 1.0 and rows[2]["m"][1] == 1.0 and rows[3]["m"][1] == 1.0


def test_source_weights_keep_the_zenodo_formula():
    rows = [{"source": "voxel51"}] * 70 + [{"source": "zenodo"}] * 30
    w = th.source_weights(rows, 0.3)
    assert abs(w["zenodo"] - 1.0) < 1e-9 and "shwd" not in w
    w2 = th.source_weights(rows + [{"source": "shwd"}], 0.3, 0.5)
    assert abs(w2["zenodo"] - w["zenodo"]) < 1e-12 and w2["shwd"] == 0.5


def test_shwd_splits_follow_the_manifest_and_stay_out_of_cv():
    assert th.assign_split({"id": "shwd/a", "source": "shwd", "split": "train"}, 0.15, 0) == "train"
    assert th.assign_split({"id": "shwd/a", "source": "shwd", "split": "validation"}, 0.15, 0) == "val"
    assert th.assign_split({"id": "shwd/a", "source": "shwd", "split": "test"}, 0.15, 0, cv=5, fold=0) == "test"
    assert th.assign_split({"id": "z", "source": "zenodo", "split": "validation"}, 0.2, 0) == "val"


def test_cli_defaults_and_shwd_items_are_opt_in(tmp_path):
    train = th.build_arg_parser().parse_args(["train"])
    assert train.f2_box_source == "" and train.drop_glimmer_f2 is False
    assert train.shwd_weight == 1.0 and train.f2_min_frac == 0.30 and train.f2_hatneg_weight == 2.0
    assert train.use_zenodo == 1 and train.zenodo_share == 0.3 and train.zenodo_f9_frac == 1.0
    assert train.hidden == 256 and train.patch_loss == 0.5 and train.ignore == "F3,F4"
    ext = ef.build_arg_parser().parse_args([])
    assert ext.sources == "voxel51,zenodo"
    (tmp_path / "data").mkdir()
    (tmp_path / "02-labeling/outputs").mkdir(parents=True)
    (tmp_path / "data/manifest.jsonl").write_text("")
    (tmp_path / "02-labeling/outputs/weak_labels_v2_vllm.jsonl").write_text("")
    row = {"id": "shwd/0001", "source": "shwd", "kind": "image", "path": "/mnt/f/x.jpg",
           "split": "train", "width": 8, "height": 8, "bare_boxes": [], "hat_boxes": [],
           "n_bare": 0, "n_hat": 0}
    man = tmp_path / "data/shwd_manifest.jsonl"
    man.write_text(json.dumps(row) + "\n")
    assert ef.build_items(tmp_path, ["voxel51", "zenodo"]) == []
    items = ef.build_items(tmp_path, ["shwd"], shwd_manifest=man)
    assert len(items) == 1 and items[0]["key"].startswith("shwd_") and "/" not in items[0]["key"]
    assert ef.item_key("shwd/0001") != ef.item_key("zenodo_tablesaw/0001")
    with pytest.raises(SystemExit):
        ef.build_items(tmp_path, ["shwd"], shwd_manifest=tmp_path / "missing.jsonl")


def test_summarize_shwd_table_is_opt_in(tmp_path):
    import summarize_head as sh
    base = {"test_by_source": {"voxel51": {}, "zenodo": {}}, "test": {}}
    (tmp_path / "h3_wide.json").write_text(json.dumps(base))
    sh.main(tmp_path)
    assert "SHWD:" not in (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    run = {**base, "shwd": {
        "validation": {"F2_patch_vs_boxes": {"auroc": 0.81, "ap": 0.70}},
        "test": {"F2_patch_vs_boxes": {"auroc": 0.77, "ap": 0.60}},
    }}
    (tmp_path / "h3_wide_f2.json").write_text(json.dumps(run))
    sh.main(tmp_path)
    md = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert "F2 patch map vs bare-head boxes" in md
    assert "0.81" in md and "0.77" in md and "0.70" in md and "0.60" in md
