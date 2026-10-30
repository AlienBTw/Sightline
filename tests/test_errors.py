from sightline.errors import SightlineError


def test_sightline_error_shape():
    e = SightlineError(
        error_code="unit_test",
        component="tests",
        message="boom",
        hint="fix it",
        details={"a": 1},
    )
    d = e.to_dict()
    assert d["error_code"] == "unit_test"
    assert d["component"] == "tests"
    assert d["message"] == "boom"
    assert d["hint"] == "fix it"
    assert d["details"]["a"] == 1
