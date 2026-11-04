"""Temporal smoothing, per-factor hysteresis and incident merging (pure, no torch).

Each model step gives per-factor probabilities. Per factor: EMA smoothing, switch on after
`min_on_steps` smoothed scores >= threshold, off after `min_off_steps` scores below
threshold * off_ratio. An *incident* (one alert, one clip) opens when any alert factor turns
on and closes once no factor has been on for `merge_gap_s`, so a hazard that flickers is one
incident, not many. Each incident keeps per-factor segments with start/end times and peaks.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FactorState:
    ema: float | None = None
    on: bool = False
    above: int = 0
    below: int = 0


@dataclass
class Segment:
    factor: str
    start: float
    end: float | None = None
    peak: float = 0.0
    peak_t: float = 0.0
    threshold: float = 0.5


@dataclass
class Incident:
    incident_id: int
    start: float
    end: float | None = None
    last_on: float = 0.0
    segments: list[Segment] = field(default_factory=list)
    closed: bool = False

    @property
    def factors(self) -> list[str]:
        seen = []
        for s in self.segments:
            if s.factor not in seen:
                seen.append(s.factor)
        return seen

    def peak(self) -> tuple[str | None, float]:
        best = (None, 0.0)
        for s in self.segments:
            if s.peak > best[1]:
                best = (s.factor, s.peak)
        return best


class IncidentTracker:
    def __init__(self, thresholds: dict[str, float], ema_alpha: float = 0.6, min_on_steps: int = 2,
                 min_off_steps: int = 2, off_ratio: float = 0.7, merge_gap_s: float = 4.0,
                 max_incident_s: float = 60.0, muted: tuple = (), hidden: tuple = ()):
        self.thr = dict(thresholds)
        self.alpha, self.min_on, self.min_off = ema_alpha, min_on_steps, min_off_steps
        self.off_ratio, self.gap, self.max_len = off_ratio, merge_gap_s, max_incident_s
        self.ignored = set(muted) | set(hidden)
        self.state = {f: FactorState() for f in self.thr}
        self.current: Incident | None = None
        self._open_seg: dict[str, Segment] = {}
        self._next_id = 1

    def smoothed(self) -> dict[str, float]:
        return {f: (s.ema or 0.0) for f, s in self.state.items()}

    def update(self, t: float, probs: dict[str, float]) -> list[tuple]:
        """Feed one step. Returns transitions: ("open", inc), ("factor_on", inc, seg),
        ("factor_off", inc, seg), ("close", inc)."""
        out = []
        for f, p in probs.items():
            if f not in self.state:
                continue
            st = self.state[f]
            st.ema = p if st.ema is None else self.alpha * p + (1 - self.alpha) * st.ema
            thr = self.thr[f]
            if st.ema >= thr:
                st.above += 1
                st.below = 0
            elif st.ema < thr * self.off_ratio:
                st.below += 1
                st.above = 0
            else:
                st.above = 0
                st.below = 0
            if f in self.ignored:
                continue
            if not st.on and st.above >= self.min_on:
                st.on = True
                if self.current is None:
                    self.current = Incident(self._next_id, start=t, last_on=t)
                    self._next_id += 1
                    out.append(("open", self.current))
                seg = Segment(f, start=t, peak=st.ema, peak_t=t, threshold=thr)
                self.current.segments.append(seg)
                self._open_seg[f] = seg
                out.append(("factor_on", self.current, seg))
            elif st.on and st.below >= self.min_off:
                st.on = False
                seg = self._open_seg.pop(f, None)
                if seg:
                    seg.end = t
                    out.append(("factor_off", self.current, seg))
            if st.on and f in self._open_seg:
                seg = self._open_seg[f]
                if st.ema > seg.peak:
                    seg.peak, seg.peak_t = st.ema, t
        inc = self.current
        if inc is not None:
            if self._open_seg:
                inc.last_on = t
            too_long = t - inc.start >= self.max_len
            if (not self._open_seg and t - inc.last_on >= self.gap) or too_long:
                out.extend(self._close(t))
        return out

    def _close(self, t: float) -> list[tuple]:
        inc = self.current
        for f, seg in list(self._open_seg.items()):
            seg.end = t
        if self._open_seg:            # forced close (max length): factors stay on for a new incident
            self._open_seg = {}
            for f, st in self.state.items():
                if st.on:
                    st.on, st.above = False, self.min_on - 1
        inc.end = max(inc.last_on, max((s.end or t) for s in inc.segments))
        inc.closed = True
        self.current = None
        return [("close", inc)]

    def flush(self, t: float) -> list[tuple]:
        """End of stream: close any open incident."""
        if self.current is None:
            return []
        if self._open_seg:
            self.current.last_on = t
        return self._close(t)
