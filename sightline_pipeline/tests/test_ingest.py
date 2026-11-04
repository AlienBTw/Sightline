import numpy as np

from sightline_pipeline.ingest import Backoff, Pacer, VideoSource, redact, source_kind


class FakeClock:
    def __init__(self):
        self.t = 100.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.slept.append(d)
        self.t += d


def test_source_kind_and_redact():
    assert source_kind("rtsp://u:p@10.0.0.5:8554/live") == "rtsp"
    assert source_kind("http://10.0.0.5:8080/video") == "stream"
    assert source_kind("C:/clips/upload.mp4") == "file"
    assert redact("rtsp://user:secret@cam:554/s") == "rtsp://***@cam:554/s"
    assert redact("demo.mp4") == "demo.mp4"


def test_pacer_releases_frames_at_media_time():
    c = FakeClock()
    p = Pacer(clock=c, sleep=c.sleep)
    for i in range(26):
        p.wait(i / 25)
    assert abs(c.t - 101.0) < 1e-6          # 25 frame intervals at 25 fps = 1 s of wall time
    assert all(abs(d - 0.04) < 1e-9 for d in c.slept)


def test_pacer_speed_and_late_consumer():
    c = FakeClock()
    p = Pacer(speed=2.0, clock=c, sleep=c.sleep, max_lag_s=1.0)
    p.wait(0.0)
    p.wait(1.0)
    assert abs(c.t - 100.5) < 1e-9          # 2x speed
    c.t += 5.0                              # consumer stalled 5 s
    assert p.wait(1.04) < 0                 # late -> no sleep, re-anchored
    before = c.t
    p.wait(1.08)
    assert abs((c.t - before) - 0.02) < 1e-9  # paced again from the new anchor


def test_backoff_caps():
    b = Backoff(0.5, 4.0)
    assert [b.next() for _ in range(5)] == [0.5, 1.0, 2.0, 4.0, 4.0]
    b.reset()
    assert b.next() == 0.5


class FakeCap:
    def __init__(self, n, fps=25.0, fail_open=False):
        self.n, self.i, self.fps, self.ok = n, 0, fps, not fail_open

    def isOpened(self):
        return self.ok

    def get(self, prop):
        return {5: self.fps, 3: 64, 4: 48}.get(prop, 0)

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, np.full((48, 64, 3), self.i % 255, np.uint8)

    def release(self):
        pass


def test_file_source_fast_and_loop():
    src = VideoSource("clip.mp4", realtime=False, open_fn=lambda u: FakeCap(10))
    fr = list(src)
    assert len(fr) == 10 and fr[-1].t == 9 / 25 and src.fps == 25.0
    opens = []

    def op(u):
        opens.append(1)
        return FakeCap(5)
    src = VideoSource("clip.mp4", realtime=False, loop=True, open_fn=op)
    out = []
    for f in src:
        out.append(f)
        if len(out) == 12:
            src.stop()
            break
    ts = [f.t for f in out]
    assert len(opens) >= 3 and all(b > a for a, b in zip(ts, ts[1:]))   # time keeps increasing across loops


def test_live_source_reconnects(monkeypatch):
    import sightline_pipeline.ingest as ing
    monkeypatch.setattr(ing.time, "sleep", lambda d: None)
    caps = [FakeCap(0, fail_open=True), FakeCap(3), FakeCap(2)]
    src = VideoSource("rtsp://cam/live", open_fn=lambda u: caps.pop(0) if caps else FakeCap(0, fail_open=True),
                      max_reconnects=3)
    fr = list(src)
    assert len(fr) == 5
    assert src.stats["reconnects"] >= 2
    assert [f.index for f in fr] == list(range(5))
