"""Offline tests for phase 03 (feature extraction planning) and 04 (head labels/metrics helpers)."""
import json

import extract_features as ef
import train_head as th


def test_item_key_stable_and_safe():
    k = ef.item_key("zenodo_tablesaw/project kreissäge 1.1/obj_Train_data/frame_000001")
    assert k == ef.item_key("zenodo_tablesaw/project kreissäge 1.1/obj_Train_data/frame_000001")
    assert k.startswith("zenodo_tablesaw_") and k.isascii() and "/" not in k and " " not in k
    assert ef.item_key("voxel51/a") != ef.item_key("voxel51/b")


def test_window_starts_cover_clip():
    span = ef.FRAMES_PER_WINDOW * ef.FRAME_STEP
    assert ef.window_starts(10) == [0]
    assert ef.window_starts(span) == [0]
    s = ef.window_starts(200)
    assert len(s) == 3 and s[0] == 0 and s[-1] == 200 - span
    assert len(ef.window_starts(100000)) == ef.MAX_WINDOWS
    assert all(0 <= x <= 520 - span for x in ef.window_starts(520))


def test_window_indices_short_and_long():
    w = ef.window_indices(0, 1000)
    assert len(w) == ef.FRAMES_PER_WINDOW and w[1] - w[0] == ef.FRAME_STEP
    short = ef.window_indices(0, 35)
    assert len(short) == ef.FRAMES_PER_WINDOW and max(short) <= 34 and short == sorted(short)
    tiny = ef.window_indices(0, 5)
    assert max(tiny) == 4 and len(tiny) == ef.FRAMES_PER_WINDOW


def test_shard_partitions():
    items = list(range(11))
    parts = [ef.shard(items, r, 2) for r in range(2)]
    assert sorted(parts[0] + parts[1]) == items and not set(parts[0]) & set(parts[1])


def test_build_items_dedup_and_splits(tmp_path):
    (tmp_path / "data").mkdir(); (tmp_path / "02-labeling/outputs").mkdir(parents=True)
    man = [
        {"id": "voxel51/a", "source": "voxel51_safe_unsafe", "kind": "video", "path": "a.mp4", "split": "train", "frames": 100},
        {"id": "voxel51/b", "source": "voxel51_safe_unsafe", "kind": "video", "path": "b.mp4", "split": "test", "frames": 50},
        {"id": "zenodo_tablesaw/x/frame_000001", "source": "zenodo_tablesaw", "kind": "image", "path": "x.PNG", "split": "train", "n_hand_boxes": 1},
        {"id": "zenodo_tablesaw/x/frame_000002", "source": "zenodo_tablesaw", "kind": "image", "path": "y.PNG", "split": "augmentation"},
        {"id": "zenodo_tablesaw/x/frame_000003", "source": "zenodo_tablesaw", "kind": "image", "path": "z.PNG", "split": "inpaint"},
    ]
    (tmp_path / "data/manifest.jsonl").write_text("\n".join(json.dumps(m) for m in man))
    # weak labels hold one row per unique clip (duplicate b-copy not listed)
    (tmp_path / "02-labeling/outputs/weak_labels_v2_vllm.jsonl").write_text(
        json.dumps({"clip_id": "voxel51/a", "path": "a.mp4", "split": "train"}))
    items = ef.build_items(tmp_path, ["voxel51", "zenodo"])
    ids = [i["id"] for i in items]
    assert ids == ["voxel51/a", "zenodo_tablesaw/x/frame_000001"]
    assert items[0]["frames"] == 100 and items[0]["kind"] == "video" and all("key" in i for i in items)


def _row(labels=None, hints=None):
    labels = labels or {}
    return {"factors": {f: {"label": labels.get(f, "absent")} for f in th.GLIMMER_FACTORS},
            "factor_hints": hints or {}}


def test_voxel_targets_sources_and_masks():
    y, m = th.voxel_targets(_row({"F2": "present", "F6": "uncertain"}, {"F8": "present", "F9": "absent"}))
    i = {f: k for k, f in enumerate(th.FACTORS)}
    assert y[i["F2"]] == 1 and m[i["F2"]] == 1
    assert m[i["F6"]] == 0                       # uncertain -> ignored
    assert m[i["F3"]] == 0 and m[i["F4"]] == 0   # ignored by default
    assert y[i["F8"]] == 1 and m[i["F8"]] == 1   # from dataset hint
    assert y[i["F9"]] == 0 and m[i["F9"]] == 1
    assert y[i["F7"]] == 0 and m[i["F7"]] == 1   # no hint -> explicit negative
    y2, m2 = th.voxel_targets(_row({"F3": "present"}), ignore=[])
    assert y2[i["F3"]] == 1 and m2[i["F3"]] == 1


def test_zenodo_targets():
    y, m = th.zenodo_targets(2)
    assert y[0] == 1 and m[0] == 1
    for f in th.ZENODO_NEGATIVE:
        i = th.FACTORS.index(f)
        assert y[i] == 0 and m[i] == 1
    assert m[1] == 0 and m[2] == 0  # F2/F3 stay unlabeled on a saw frame
    y0, m0 = th.zenodo_targets(0)
    assert y0[0] == 0 and m0[0] == 1 and sum(m0) == 1 + len(th.ZENODO_NEGATIVE)


def _clip(fid_y, fid_m, source="voxel51", label="", clip_id="c"):
    y, m = [0.0] * 10, [0.0] * 10
    i = th.FACTORS.index("F9")
    y[i], m[i] = fid_y, fid_m
    return {"id": clip_id, "source": source, "y": y, "m": m, "source_label": label, "set": "train"}


def test_sampling_weights_boost_hard_negatives_and_closed_panels():
    src = {"voxel51": 1.0, "zenodo": 0.2}
    neg = _clip(0, 1, clip_id="neg")
    easy = _clip(0, 1, clip_id="easy")
    closed = _clip(0, 1, label="Closed Panel Cover", clip_id="closed")
    opened = _clip(1, 1, label="Opened Panel Cover", clip_id="opened")
    saw = _clip(0, 1, source="zenodo", clip_id="saw")
    rows = [neg, easy, closed, opened, saw]
    scores = {"neg": 0.91, "easy": 0.2, "closed": 0.1, "opened": 0.99, "saw": 0.99}
    w = th.sampling_weights(rows, src, scores, "F9", 0.8)
    assert w == [5.0, 1.0, 5.0, 1.0, 0.2]
    assert th.sampling_weights(rows, src, None) == [1.0, 1.0, 1.0, 1.0, 0.2]


def test_zenodo_f9_frac_clears_train_only():
    def row(i, split_set):
        y, m = th.zenodo_targets(1)
        return {"id": f"z{i}", "source": "zenodo", "y": y, "m": m, "set": split_set}
    rows = [row(0, "train"), row(1, "train"), row(2, "val"), row(3, "test")]
    assert th.mask_zenodo_f9(rows, 0.0) == 2
    i = th.FACTORS.index("F9")
    assert rows[0]["m"][i] == 0 and rows[1]["m"][i] == 0
    assert rows[2]["m"][i] == 1 and rows[3]["m"][i] == 1
    kept = [row(k, "train") for k in range(4)]
    assert th.mask_zenodo_f9(kept, 1.0) == 0
    assert all(r["m"][i] == 1 for r in kept)
    assert th.zenodo_f9_keep("same", 0.5, 0) == th.zenodo_f9_keep("same", 0.5, 0)


def test_precision_threshold_prefers_the_precise_cut():
    y = [1, 1, 0, 0]
    p = [0.9, 0.4, 0.7, 0.1]
    assert th.precision_threshold(y, p, 0.8) == 0.9
    assert th.best_f1_threshold(y, p) == 0.4


def test_box_patch_mask_and_yolo(tmp_path):
    p = tmp_path / "f.txt"; p.write_text("0 0.25 0.5 0.5 1.0\n")
    boxes = th.read_yolo(str(p))
    assert boxes == [(0.25, 0.5, 0.5, 1.0)]
    m = th.box_patch_mask(boxes, 2, 4)
    assert m == [[1, 1, 0, 0], [1, 1, 0, 0]]
    assert th.box_patch_mask([], 3, 3) == [[0] * 3] * 3
    assert th.read_yolo(None) == []


def test_assign_split_keeps_test_and_is_stable():
    assert th.assign_split({"id": "v/a", "source": "voxel51", "split": "test"}, 0.5, 0) == "test"
    a = [th.assign_split({"id": f"v/{k}", "source": "voxel51", "split": "train"}, 0.2, 0) for k in range(500)]
    assert set(a) == {"train", "val"} and 50 < a.count("val") < 150
    assert a == [th.assign_split({"id": f"v/{k}", "source": "voxel51", "split": "train"}, 0.2, 0) for k in range(500)]
    assert th.assign_split({"id": "z", "source": "zenodo", "split": "validation"}, 0.2, 0) == "val"
    assert th.assign_split({"id": "z", "source": "zenodo", "split": "augmentation"}, 0.2, 0) == "skip"


def test_pos_weights_and_threshold():
    ys = [[1] + [0] * 9, [0] * 10, [0] * 10, [0] * 10]
    ms = [[1] * 10] * 4
    pw = th.pos_weights(ys, ms, cap=20)
    assert pw[0] == 3.0 and pw[1] == 1.0
    assert th.pos_weights([[1] * 10] + [[0] * 10] * 100, [[1] * 10] * 101, cap=20)[0] == 20.0
    t = th.best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9])
    assert 0.2 < t <= 0.7


def test_cv_folds_partition_voxel_only():
    ids = [f"voxel51/c{k}" for k in range(300)]
    tests = [sum(th.assign_split({"id": i, "source": "voxel51", "split": "train"}, 0.15, 0, cv=5, fold=f) == "test"
                 for i in ids) for f in range(5)]
    assert sum(tests) == 300 and min(tests) > 30
    # cv ignores the dataset split for Voxel51, never for Zenodo
    assert th.assign_split({"id": ids[0], "source": "voxel51", "split": "test"}, 0.15, 0, cv=5,
                           fold=(th.cv_fold(ids[0], 5) + 1) % 5) in ("train", "val")
    assert th.assign_split({"id": "z", "source": "zenodo", "split": "test"}, 0.15, 0, cv=5, fold=0) == "test"


def test_summarize_head_tables(tmp_path):
    import summarize_head as sh
    res = {"F2": {"n": 10, "pos": 10}, "F8": {"n": 10, "pos": 4, "auroc": 0.9, "ap": 0.8}}
    run = {"test_by_source": {"voxel51": res, "zenodo": {"F1_patch_vs_boxes": {"auroc": 0.99, "ap": 0.9}}},
           "test": {"F1": {"f1": 0.99}}}
    (tmp_path / "h1.json").write_text(json.dumps(run))
    preds = [{"id": str(k), "p": [0.1 * (k % 10)] * 10, "y": [float(k % 2)] * 10, "m": [1.0] * 10} for k in range(20)]
    for f in range(2):
        (tmp_path / f"h1_cv2_f{f}.json").write_text(json.dumps({**run, "test_predictions_voxel51": preds[f::2]}))
    sh.main(tmp_path)
    md = (tmp_path / "SUMMARY.md").read_text()
    assert "0.90 / 0.80 (4/10)" in md and "one class (10/10)" in md and "h1_cv2" in md and "not trained" in md
