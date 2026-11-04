from sightline_pipeline.events import IncidentTracker
from sightline_pipeline.risk import confidence, noisy_or


def run(tr, seq, t0=0.0, dt=1.0):
    out = []
    for i, p in enumerate(seq):
        out += [(x[0], i) for x in tr.update(t0 + i * dt, p)]
    return out


def test_needs_consecutive_steps_and_hysteresis():
    tr = IncidentTracker({"F1": 0.5}, ema_alpha=1.0, min_on_steps=2, min_off_steps=2, off_ratio=0.7, merge_gap_s=2.0)
    seq = [{"F1": v} for v in [0.6, 0.2, 0.6, 0.6, 0.45, 0.45, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1]]
    ev = run(tr, seq)
    assert ("open", 3) in ev and ("factor_on", 3) in ev      # single spike at step 0 ignored
    assert ("factor_off", 7) in ev                            # 0.45 is in the hysteresis band, 0.3 < 0.35
    assert ("close", 8) in ev                                 # merge gap 2 s after the last step still on (6)


def test_flicker_merges_into_one_incident():
    tr = IncidentTracker({"F9": 0.5}, ema_alpha=1.0, min_on_steps=1, min_off_steps=1, merge_gap_s=3.0)
    seq = [{"F9": v} for v in [0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]]
    ev = run(tr, seq)
    assert [e for e in ev if e[0] == "open"] == [("open", 0)]
    assert sum(1 for e in ev if e[0] == "factor_on") == 3
    assert [e for e in ev if e[0] == "close"] == [("close", 8)]


def test_multi_factor_segments_and_peak():
    tr = IncidentTracker({"F1": 0.5, "F7": 0.1}, ema_alpha=1.0, min_on_steps=1, min_off_steps=1, merge_gap_s=1.0)
    incs = []
    for i, p in enumerate([{"F1": 0.7, "F7": 0.0}, {"F1": 0.9, "F7": 0.3}, {"F1": 0.1, "F7": 0.0},
                           {"F1": 0.1, "F7": 0.0}, {"F1": 0.1, "F7": 0.0}]):
        for x in tr.update(float(i), p):
            if x[0] == "close":
                incs.append(x[1])
    assert len(incs) == 1
    inc = incs[0]
    assert inc.factors == ["F1", "F7"] and inc.start == 0.0 and inc.end == 2.0
    assert inc.peak() == ("F1", 0.9)


def test_muted_and_hidden_never_open_and_flush():
    tr = IncidentTracker({"F2": 0.5, "F3": 0.5, "F8": 0.5}, ema_alpha=1.0, min_on_steps=1, muted=("F2",), hidden=("F3",))
    assert tr.update(0.0, {"F2": 0.99, "F3": 0.99, "F8": 0.0}) == []
    assert tr.update(1.0, {"F2": 0.99, "F3": 0.99, "F8": 0.9})[0][0] == "open"
    closed = tr.flush(5.0)
    assert closed[0][0] == "close" and closed[0][1].end == 5.0


def test_ema_smoothing():
    tr = IncidentTracker({"F1": 0.5}, ema_alpha=0.5, min_on_steps=1)
    tr.update(0, {"F1": 1.0}); tr.update(1, {"F1": 0.0})
    assert abs(tr.smoothed()["F1"] - 0.5) < 1e-9


def test_max_incident_length_splits():
    tr = IncidentTracker({"F1": 0.5}, ema_alpha=1.0, min_on_steps=1, max_incident_s=5.0)
    ev = run(tr, [{"F1": 0.9}] * 12)
    assert sum(1 for e in ev if e[0] == "open") == 2


def test_confidence_and_noisy_or():
    assert confidence(0.1, 0.1) == 0.5 and confidence(1.0, 0.1) == 1.0 and confidence(0.0, 0.3) == 0.0
    assert abs(noisy_or([0.5, 0.5]) - 0.75) < 1e-9
