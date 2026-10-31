"""Offline tests for the Glimmer eval helpers."""
import pytest

pytest.importorskip("yaml")
import eval_labels as ev  # noqa: E402


def rows():
    out = []
    for fid, state, n in [("F8", "present", 8), ("F8", "absent", 6), ("F9", "present", 6),
                          ("F9", "absent", 5), ("F7", "present", 6), ("F7", "absent", 5)]:
        for i in range(n):
            out.append({"id": f"voxel51/{fid}{state}{i}", "source": "voxel51_safe_unsafe",
                        "factor_hints": {fid: state}, "duplicate_of": [], "split": "train"})
    out.append({"id": "voxel51/dup", "source": "voxel51_safe_unsafe", "factor_hints": {"F8": "present"},
                "duplicate_of": ["voxel51/x"]})
    out.append({"id": "voxel51/two", "source": "voxel51_safe_unsafe", "factor_hints": {"F8": "absent", "F9": "absent"},
                "duplicate_of": []})
    return out


def test_select_is_deterministic_unique_and_sized():
    a, b = ev.select_eval_set(rows()), ev.select_eval_set(rows())
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert len(a) == sum(ev.DEFAULT_BUCKETS.values()) == len({x["id"] for x in a})
    assert not {"voxel51/dup", "voxel51/two"} & {x["id"] for x in a}


def test_select_fails_loudly_when_short():
    with pytest.raises(SystemExit):
        ev.select_eval_set(rows()[:3])


def rec(cid, **labels):
    f = {k: {"label": "absent"} for k in ev.GLIMMER_FACTORS}
    for k, v in labels.items():
        f[k] = {"label": v}
    return {"clip_id": cid, "factors": f}


def test_metrics():
    es = [{"id": "a", "factor_hints": {"F8": "present"}}, {"id": "b", "factor_hints": {"F8": "present"}},
          {"id": "c", "factor_hints": {"F8": "absent"}}, {"id": "d", "factor_hints": {"F9": "present"}}]
    recs = [rec("a", F8="present"), rec("b", F8="uncertain"), rec("c", F8="present"), rec("d", F9="present")]
    m = ev.compute_metrics(recs, es, [10.0, 12.0, 14.0])
    f8 = m["per_factor"]["F8"]
    assert (f8["tp"], f8["fp"], f8["fn"], f8["tn"]) == (1, 1, 1, 0)
    assert f8["precision"] == 0.5 and f8["recall"] == 0.5 and f8["uncertain_rate"] == round(1 / 3, 3)
    assert m["per_factor"]["F9"]["recall"] == 1.0 and m["per_factor"]["F7"]["n"] == 0
    assert m["sec_per_call_mean"] == 12.0 and m["clips"] == 4 and not m["missing"]
    assert "F8 P / R / unc" in ev.to_markdown({"configs": {"v1:none": m}})
