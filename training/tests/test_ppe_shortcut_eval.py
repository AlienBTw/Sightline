"""Shortcut metrics: one-class n/a, held-out bootstrap, and the Voxel51-vs-Pexels gap."""
import pytest

import ppe_shortcut_eval as pse
import train_head as th


def _rec(split, y, p):
    return {"split": split, "y": y, "p": p}


def _separated():
    """Held-out Pexels is perfectly ranked. A train leak would spoil that."""
    pexels = [
        _rec("train", 0, 0.99),
        _rec("val", 1, 0.6),
        _rec("val", 0, 0.4),
        _rec("test", 1, 0.9),
        _rec("test", 1, 0.8),
        _rec("test", 0, 0.2),
        _rec("demo_eval", 1, 0.7),
        _rec("demo_eval", 0, 0.1),
    ]
    voxel = [_rec("val", 1, 0.95), _rec("test", 1, 0.99), _rec("test", 0, 0.05)]
    return pexels, voxel


def test_one_class_is_na_and_held_out_excludes_train_and_val():
    pexels, voxel = _separated()
    report = pse.factor_report(pexels, voxel, [1.0, 0.0], [0.9, 0.1])
    held = report["splits"]["held_out"]
    assert held["pos"] == 3 and held["neg"] == 2
    assert held["auroc"] == 1.0 and held["ap"] == 1.0
    assert held["auroc_ci95"] == [1.0, 1.0]
    assert report["splits"]["test"]["pos"] == 2 and report["splits"]["test"]["neg"] == 1
    assert report["splits"]["train"]["n"] == 1
    assert report["distributions"]["pexels_pos"]["n"] == 3
    assert report["distributions"]["pexels_pos"]["mean"] == pytest.approx(0.8)
    assert report["distributions"]["pexels_neg"]["median"] == pytest.approx(0.15)
    assert report["distributions"]["voxel51_pos"]["n"] == 2
    assert report["shortcut"]["domain_auroc"] == 1.0
    assert report["shortcut"]["domain_n_voxel51"] == 2
    assert report["shortcut"]["domain_n_pexels"] == 3
    assert report["shortcut"]["val_threshold_fit"] is True
    assert report["shortcut"]["heldout_pos_above"] == 1.0
    assert report["shortcut"]["heldout_neg_above"] == 0.0
    assert report["window_held_out"]["auroc"] == 1.0
    only_pos = pse.ranking_metrics([1, 1, 1], [0.2, 0.9, 0.4])
    assert only_pos["pos"] == 3 and only_pos["neg"] == 0
    assert only_pos["auroc"] == "n/a" and only_pos["ap"] == "n/a"
    empty = pse.factor_report([], [], [], [])
    assert empty["splits"]["demo_eval"]["auroc"] == "n/a"
    assert empty["splits"]["held_out"]["auroc_ci95"] == "n/a"
    assert empty["distributions"]["voxel51_pos"]["mean"] == "n/a"
    assert empty["shortcut"]["domain_auroc"] == "n/a"
    assert empty["shortcut"]["heldout_pos_above"] == "n/a"


def test_bootstrap_interval_is_deterministic():
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    p = [0.1, 0.4, 0.2, 0.05, 0.55, 0.7, 0.95, 0.8]
    first = pse.bootstrap_auroc(y, p)
    second = pse.bootstrap_auroc(y, p)
    assert first == second
    lo, hi = first["auroc_ci95"]
    point = pse.ranking_metrics(y, p)["auroc"]
    assert first["auroc_boot_used"] > 900
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= point <= hi


def test_domain_auroc_falls_when_voxel_scores_lower():
    high = pse.domain_auroc([0.9, 0.8], [0.2, 0.1])
    low = pse.domain_auroc([0.1, 0.2], [0.8, 0.9])
    assert high["domain_auroc"] == 1.0
    assert low["domain_auroc"] == 0.0
    assert pse.domain_auroc([0.5], [])["domain_auroc"] == "n/a"


def test_markdown_table_shows_na_and_the_checkpoint():
    pexels, voxel = _separated()
    f5 = pse.factor_report(
        [_rec("test", 1, 0.4), _rec("test", 1, 0.6), _rec("val", 1, 0.5)],
        [_rec("test", 1, 0.7)],
        [1.0, 1.0],
        [0.4, 0.6],
    )
    report = {
        "clip_score": pse.CLIP_SCORE,
        "window_score": pse.WINDOW_SCORE,
        "data": {
            "pexels_clips": {"test": 3, "demo_eval": 2, "val": 2, "train": 1},
            "voxel51_clips": {"val": 1, "test": 2},
            "voxel_ppe": {"overridden": 4, "cleared": 1, "val_test_labeled": 3},
        },
        "checkpoints": [{
            "name": "toy",
            "path": "toy.pt",
            "hidden": 8,
            "dropout": 0.0,
            "factors": {"F2": pse.factor_report(pexels, voxel, [1.0, 0.0], [0.9, 0.1]), "F5": f5},
        }],
    }
    text = pse.render_markdown(report)
    assert "| toy | F2 | 3/2 | 1.000 | 1.000 | [1.000, 1.000] |" in text
    assert "| toy | F5 | 2/0 | n/a | n/a | n/a |" in text
    assert "| F5 | test | 2 | 0 | n/a | n/a |  |" in text
    assert "Pexels present (test+demo_eval)" in text
    assert "Voxel51 present (val+test)" in text


def test_write_outputs_round_trip(tmp_path):
    report = {
        "clip_score": pse.CLIP_SCORE,
        "window_score": pse.WINDOW_SCORE,
        "data": {"pexels_clips": {}, "voxel51_clips": {}, "voxel_ppe": {}},
        "checkpoints": [{
            "name": "toy",
            "path": "toy.pt",
            "hidden": 8,
            "dropout": 0.0,
            "factors": {
                "F2": pse.factor_report([], [], [], []),
                "F5": pse.factor_report([], [], [], []),
            },
        }],
    }
    json_path = tmp_path / "shortcut_eval.json"
    md_path = tmp_path / "PPE_SHORTCUT.md"
    pse.write_outputs(report, json_path, md_path)
    text = md_path.read_text(encoding="utf-8")
    assert "| toy | F2 | 0/0 | n/a | n/a | n/a |" in text
    stored = json_path.read_text(encoding="utf-8")
    assert '"auroc": "n/a"' in stored


def test_discover_checkpoints_keeps_h3_wide_and_ppe_only(tmp_path):
    (tmp_path / "h3_wide.pt").write_bytes(b"x")
    (tmp_path / "h1_base_s0.pt").write_bytes(b"x")
    (tmp_path / "ppe_b.pt").write_bytes(b"x")
    (tmp_path / "ppe_a.pt").write_bytes(b"x")
    assert [p.name for p in pse.discover_checkpoints(tmp_path)] == ["h3_wide.pt", "ppe_a.pt", "ppe_b.pt"]
    assert [p.name for p in pse.checkpoints_from_args("", tmp_path)] == ["h3_wide.pt", "ppe_a.pt", "ppe_b.pt"]
    with pytest.raises(SystemExit):
        pse.checkpoints_from_args("", tmp_path / "empty")
    lone = tmp_path / "custom.pt"
    lone.write_bytes(b"x")
    assert pse.checkpoints_from_args(str(lone), tmp_path) == [lone]


def test_window_scores_match_train_head_and_keep_alignment():
    torch = pytest.importorskip("torch")
    import numpy as np
    rows = []
    for i, (split, y) in enumerate((("train", 1.0), ("test", 0.0))):
        rows.append({
            "id": f"pexels/{i}", "source": "ppe_video", "ppe_source": "pexels", "split": split,
            "n": 1, "n_patches": 2,
            "y": [0.0, y, 0.0, 0.0, y] + [0.0] * 5,
            "m": [0.0, 1.0, 0.0, 0.0, 1.0] + [0.0] * 5,
            "ppe_windows": [{"F2": y, "F5": y}],
        })
    evid = {
        1: [torch.tensor([[0.2, 0.3], [0.4, 0.8]])],
        4: [torch.tensor([[0.1, 0.2], [0.0, 0.5]])],
    }
    probs = np.zeros((2, 10))
    _clip, window, _patch = th.collect_ppe_scores(rows, probs, evid)
    y, p = pse.labeled_windows(rows, evid, "F2")
    assert y == window["pexels"]["F2"]["y"]
    assert p == window["pexels"]["F2"]["p"]
    held_y, held_p = pse.labeled_windows(rows, evid, "F2", {"test", "demo_eval"})
    assert held_y == [0.0]
    assert held_p == pytest.approx([0.8])
    short = [{"id": "pexels/z", "split": "test", "n": 1, "n_patches": 1, "ppe_windows": [{"F2": 1.0}]}] * 33
    with pytest.raises(SystemExit):
        pse.labeled_windows(short, {1: [torch.zeros(32, 1)]}, "F2")
