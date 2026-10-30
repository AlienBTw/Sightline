import pytest

from sightline.errors import SightlineError
from sightline.hazards.state_machine import transition


def test_happy_path():
    s = transition("candidate", "confirmed")
    s = transition(s, "open")
    s = transition(s, "fix_pending")
    s = transition(s, "verified")
    assert s == "verified"


def test_illegal():
    with pytest.raises(SightlineError) as ei:
        transition("candidate", "verified")
    assert ei.value.error_code == "hazard_transition_illegal"
