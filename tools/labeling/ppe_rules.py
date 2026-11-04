"""Pure F2/F5 geometry, window votes, Muse parsing, merge, and budget checks.

No video decoding and no network. 1 means the hazard is present (no helmet / no
hi-vis vest). 0 means it is absent. None is uncertain and becomes JSON null.
"""
from __future__ import annotations

FACTORS = ("F2", "F5")
BOX_KEYS = ("F2_pos", "F2_neg", "F5_pos", "F5_neg")

PERSON_PROMPT = "person"
HEAD_PROMPT = "head"
HAT_PROMPTS = ("hard hat", "helmet")
VEST_PROMPTS = ("safety vest", "high-visibility vest")
VEST_PROMPT = VEST_PROMPTS[0]
SAM_PROMPTS = (HEAD_PROMPT, *HAT_PROMPTS, *VEST_PROMPTS, PERSON_PROMPT)

MIN_PERSON_PX = 40
MIN_PERSON_FRAC = 0.06
HEAD_HEIGHT_FRAC = 0.22
HEAD_WIDTH_FRAC = 0.60
TORSO_Y0 = 0.20
TORSO_Y1 = 0.60
HAT_IOA = 0.30
VEST_COVER = 0.25
WINDOW_HITS = 2

MUSE_STATES = ("present", "absent", "uncertain")
SPLIT_RANK = {"test": 0, "demo_eval": 1, "val": 2, "validation": 2, "train": 3}


class BudgetRefusal(Exception):
    def __init__(self, reason: str, total: int):
        super().__init__(reason)
        self.reason = reason
        self.total = int(total)


def empty_boxes() -> dict:
    return {key: [] for key in BOX_KEYS}


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def area(box) -> float:
    width = max(0.0, float(box[2]) - float(box[0]))
    height = max(0.0, float(box[3]) - float(box[1]))
    return width * height


def intersection(a, b) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def iou(a, b) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def ioa(box, region) -> float:
    """Intersection over the area of ``box`` (the candidate hat or vest)."""
    denom = area(box)
    if denom <= 0.0:
        return 0.0
    return intersection(box, region) / denom


def head_region(person) -> tuple[float, float, float, float]:
    """Top 22% of the person box, width shrunk to the central 60%."""
    x0, y0, x1, y1 = (float(v) for v in person)
    width = x1 - x0
    height = y1 - y0
    cx = (x0 + x1) / 2.0
    half = (HEAD_WIDTH_FRAC * width) / 2.0
    return (cx - half, y0, cx + half, y0 + HEAD_HEIGHT_FRAC * height)


def torso_region(person) -> tuple[float, float, float, float]:
    """20% to 60% of the person height, full person width."""
    x0, y0, x1, y1 = (float(v) for v in person)
    height = y1 - y0
    return (x0, y0 + TORSO_Y0 * height, x1, y0 + TORSO_Y1 * height)


def person_is_confident(person, frame_h: float, min_px: float = MIN_PERSON_PX,
                        min_frac: float = MIN_PERSON_FRAC) -> bool:
    """False when the box is under 40 px or under 6% of the frame height."""
    height = float(person[3]) - float(person[1])
    if height < min_px or height < min_frac * float(frame_h):
        return False
    return True


def _as_xyxy(box) -> tuple[float, float, float, float] | None:
    if box is None:
        return None
    if isinstance(box, dict):
        try:
            coords = (box["x1"], box["y1"], box["x2"], box["y2"])
        except KeyError:
            return None
    else:
        if len(box) < 4:
            return None
        coords = box[:4]
    x0, y0, x1, y1 = (float(v) for v in coords)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def box_score(box) -> float | None:
    if isinstance(box, dict) and box.get("c") is not None:
        return float(box["c"])
    return None


def passes_score(box, threshold: float) -> bool:
    score = box_score(box)
    if score is None:
        return True
    return score >= float(threshold)


def head_is_helmeted(head, hats, threshold: float = HAT_IOA) -> bool:
    """True when a hat covers at least ``threshold`` of the head box (IoA of the head)."""
    return overlaps_any(head, hats, threshold, "cover")


def assign_heads(people, head_boxes, min_frac: float = HAT_IOA) -> list[list]:
    """Give each head to the person that contains the largest share of that head."""
    assigned = [[] for _ in people]
    for head in head_boxes:
        best_i = None
        best = 0.0
        for index, person in enumerate(people):
            frac = ioa(head, person)
            if frac > best:
                best = frac
                best_i = index
        if best_i is not None and best >= min_frac:
            assigned[best_i].append(head)
    return assigned


def normalize_box_records(boxes, width: float, height: float) -> list[dict]:
    """Pixel xyxy records -> normalised [0, 1] records, keeping ``c`` when present."""
    if width <= 0 or height <= 0:
        raise ValueError(f"bad frame size {width}x{height}")
    out = []
    for box in boxes or []:
        xy = _as_xyxy(box)
        if xy is None:
            continue
        rec = {
            "x1": round(xy[0] / float(width), 6),
            "y1": round(xy[1] / float(height), 6),
            "x2": round(xy[2] / float(width), 6),
            "y2": round(xy[3] / float(height), 6),
        }
        score = box_score(box)
        if score is not None:
            rec["c"] = round(float(score), 4)
        out.append(rec)
    return out


def scale_box_records(boxes, width: float, height: float) -> list[dict]:
    """Normalised xyxy records -> pixels. ``c`` is copied through."""
    out = []
    for box in boxes or []:
        xy = _as_xyxy(box)
        if xy is None:
            continue
        rec = {
            "x1": xy[0] * float(width),
            "y1": xy[1] * float(height),
            "x2": xy[2] * float(width),
            "y2": xy[3] * float(height),
        }
        score = box_score(box)
        if score is not None:
            rec["c"] = float(score)
        out.append(rec)
    return out


def collect_detections(prompt_boxes: dict) -> tuple[list, list, list, list]:
    """Person, head, hat (hard hat ∪ helmet), vest (safety vest ∪ high-visibility vest)."""
    persons = list((prompt_boxes or {}).get(PERSON_PROMPT) or [])
    heads = list((prompt_boxes or {}).get(HEAD_PROMPT) or [])
    hats = []
    for name in HAT_PROMPTS:
        hats.extend((prompt_boxes or {}).get(name) or [])
    vests = []
    for name in VEST_PROMPTS:
        vests.extend((prompt_boxes or {}).get(name) or [])
    vests.extend((prompt_boxes or {}).get("high-visibility safety vest") or [])
    return persons, heads, hats, vests


def dedupe_boxes(boxes: list, iou_thr: float = 0.5) -> list:
    kept = []
    for box in boxes:
        if all(iou(box, prev) < iou_thr for prev in kept):
            kept.append(box)
    return kept


def norm_xyxy(box, width: float, height: float) -> list[float] | None:
    if width <= 0 or height <= 0:
        raise ValueError(f"bad frame size {width}x{height}")
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((_clip01(x0 / width), _clip01(x1 / width)))
    y0, y1 = sorted((_clip01(y0 / height), _clip01(y1 / height)))
    if x1 <= x0 or y1 <= y0:
        return None
    return [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


def _norm_list(boxes, width, height) -> list:
    out = []
    for box in boxes:
        norm = norm_xyxy(box, width, height)
        if norm is not None:
            out.append(norm)
    return out


def overlaps_any(region, boxes, threshold: float, mode: str) -> bool:
    for box in boxes:
        if mode == "ioa":
            score = ioa(box, region)
        else:
            denom = area(region)
            score = 0.0 if denom <= 0.0 else intersection(box, region) / denom
        if score >= threshold:
            return True
    return False


def _kept_xyxy(raws, score_threshold: float) -> list:
    return dedupe_boxes([
        box for raw in raws
        if passes_score(raw, score_threshold) and (box := _as_xyxy(raw)) is not None
    ])


def label_frame(persons, hats, vests, frame_w: float, frame_h: float,
                score_threshold: float = 0.5, heads=None) -> dict:
    """One keyframe. States are ``missing`` (hazard), ``ok``, or ``none``.

    Persons shorter than 40 px or 6% of the frame are ignored. A frame whose
    only people are that small is ``none`` (uncertain), same as no person.

    When a person has a ``head`` box, that box is the head. It is helmeted when
    a hard-hat or helmet box covers at least 0.3 of the head (IoA of the head).
    A person with no head box uses the top-of-person region: a hat counts when
    its own area is at least 0.3 inside that region. Vest / no-vest stays the
    torso-cover rule.
    """
    people = []
    ignored = 0
    for raw in persons:
        if not passes_score(raw, score_threshold):
            continue
        box = _as_xyxy(raw)
        if box is None:
            continue
        if not person_is_confident(box, frame_h):
            ignored += 1
            continue
        people.append(box)
    head_boxes = _kept_xyxy(heads or [], score_threshold)
    hat_boxes = _kept_xyxy(hats, score_threshold)
    vest_boxes = _kept_xyxy(vests, score_threshold)
    f2_pos, f5_pos, helmeted_heads = [], [], []
    if not people:
        f2_state, f5_state = "none", "none"
    else:
        bare = False
        vest_missing = False
        for person, person_heads in zip(people, assign_heads(people, head_boxes)):
            torso = torso_region(person)
            has_vest = overlaps_any(torso, vest_boxes, VEST_COVER, "cover")
            if not has_vest:
                vest_missing = True
                f5_pos.append(torso)
            if person_heads:
                person_bare = False
                for head in person_heads:
                    if head_is_helmeted(head, hat_boxes):
                        helmeted_heads.append(head)
                    else:
                        person_bare = True
                        f2_pos.append(head)
                if person_bare:
                    bare = True
            else:
                region = head_region(person)
                has_hat = overlaps_any(region, hat_boxes, HAT_IOA, "ioa")
                if not has_hat:
                    bare = True
                    f2_pos.append(region)
        f2_state = "missing" if bare else "ok"
        f5_state = "missing" if vest_missing else "ok"
    boxes = {
        "F2_pos": _norm_list(f2_pos, frame_w, frame_h),
        "F2_neg": _norm_list(list(hat_boxes) + helmeted_heads, frame_w, frame_h),
        "F5_pos": _norm_list(f5_pos, frame_w, frame_h),
        "F5_neg": _norm_list(vest_boxes, frame_w, frame_h),
    }
    return {
        "F2": f2_state,
        "F5": f5_state,
        "boxes": boxes,
        "n_persons": len(people),
        "n_ignored_small": ignored,
    }


def window_vote(states: list[str], hits: int = WINDOW_HITS) -> int | None:
    """1 if a hazard shows in >= 2 keyframes, 0 if protection shows in >= 2 and never fails.

    ``states`` are per-keyframe ``missing`` / ``ok`` / ``none``. No confident
    person in the window is null. One failure and two protected frames is null:
    the hazard was not repeated, and protection was not consistent.
    """
    missing = sum(state == "missing" for state in states)
    ok = sum(state == "ok" for state in states)
    if missing + ok == 0:
        return None
    if missing >= hits:
        return 1
    if ok >= hits and missing == 0:
        return 0
    return None


def aggregate_window(frame_labels: list[dict]) -> dict:
    """Combine the three keyframe dicts from ``label_frame`` into one window."""
    boxes = empty_boxes()
    for frame in frame_labels:
        for key in BOX_KEYS:
            boxes[key].extend(frame["boxes"].get(key) or [])
    return {
        "F2": window_vote([frame["F2"] for frame in frame_labels]),
        "F5": window_vote([frame["F5"] for frame in frame_labels]),
        "boxes": boxes,
    }


def keyframe_indices(window_indices: list[int]) -> list[int]:
    """First, middle, and last frame of one feature window."""
    if not window_indices:
        return []
    mid = window_indices[len(window_indices) // 2]
    return [window_indices[0], mid, window_indices[-1]]


def verdict_bit(value) -> int | None:
    if value in (0, 1):
        return int(value)
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "present":
        return 1
    if text == "absent":
        return 0
    return None


def parse_muse_json(text: str) -> dict:
    """Strict F2/F5 JSON. Anything else is an uncertain parse failure.

    ``parsed`` is False when the text was not usable. Callers still count the
    HTTP call and store uncertain.
    """
    import json
    import re
    fail = {"F2": "uncertain", "F5": "uncertain", "reason": "parse_error", "parsed": False}
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return dict(fail)
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return dict(fail)
    if not isinstance(payload, dict):
        return dict(fail)
    f2 = str(payload.get("F2", "")).strip().lower()
    f5 = str(payload.get("F5", "")).strip().lower()
    if f2 not in MUSE_STATES or f5 not in MUSE_STATES:
        return dict(fail)
    reason = str(payload.get("reason") or "").strip()
    return {"F2": f2, "F5": f5, "reason": reason[:500], "parsed": True}


def claim_call(total: int, budget_total: int, used_this_run: int, max_calls: int) -> tuple[bool, str]:
    """Whether one more Muse call may be reserved. Does not mutate anything."""
    if max_calls <= 0 or used_this_run >= max_calls:
        return False, "max_calls"
    if int(total) >= int(budget_total):
        return False, "budget_total"
    return True, "ok"


def apply_claim(total: int, budget_total: int, used_this_run: int, max_calls: int) -> int:
    ok, reason = claim_call(total, budget_total, used_this_run, max_calls)
    if not ok:
        raise BudgetRefusal(reason, total)
    return int(total) + 1


def api_calls_allowed(sent: int, max_api_calls: int) -> bool:
    return max_api_calls > 0 and sent < max_api_calls


def _agree(sam_bit, muse_bit) -> int | None:
    """Window label once the clip verdict is known.

    Absent forces every window to 0. Present keeps a SAM positive and clears
    SAM negatives and unknowns (multiple-instance learning: the clip carries it).
    An uncertain verdict clears the window.
    """
    if muse_bit is None:
        return None
    if muse_bit == 0:
        return 0
    if sam_bit == 1:
        return 1
    return None


def clip_from_windows(windows: list[dict], factor: str) -> int | None:
    """1 if any window is 1, 0 if every window is 0, else null."""
    values = [verdict_bit(w.get(factor)) for w in windows]
    if any(value == 1 for value in values):
        return 1
    if values and all(value == 0 for value in values):
        return 0
    return None


def _window_out(window: dict, f2, f5) -> dict:
    boxes = window.get("boxes") or empty_boxes()
    clean = empty_boxes()
    for key in BOX_KEYS:
        clean[key] = list(boxes.get(key) or [])
    return {
        "w": int(window.get("w", 0)),
        "start": int(window.get("start", 0)),
        "F2": f2,
        "F5": f5,
        "boxes": clean,
    }


def muse_only_windows(specs: list[dict] | None, f2, f5) -> list[dict]:
    """No SAM boxes. Each known window inherits the clip label."""
    if not specs:
        return [_window_out({"w": 0, "start": 0, "boxes": empty_boxes()}, f2, f5)]
    return [
        _window_out({"w": spec.get("w", i), "start": spec.get("start", 0), "boxes": empty_boxes()}, f2, f5)
        for i, spec in enumerate(specs)
    ]


def merge_record(meta: dict, sam: dict | None, muse: dict | None,
                 window_specs: list[dict] | None = None) -> dict:
    """One output row. ``sam`` is ``{"windows": [...]}`` or None. ``muse`` is a verdict or None."""
    muse_out = None
    if muse is not None:
        f2 = verdict_bit(muse.get("F2"))
        f5 = verdict_bit(muse.get("F5"))
        muse_out = {"F2": f2, "F5": f5, "reason": str(muse.get("reason") or "")}
        if sam is None:
            windows = muse_only_windows(window_specs, f2, f5)
        else:
            windows = []
            for window in sam.get("windows") or []:
                windows.append(_window_out(
                    window,
                    _agree(verdict_bit(window.get("F2")), f2),
                    _agree(verdict_bit(window.get("F5")), f5),
                ))
            if not windows:
                windows = muse_only_windows(window_specs, f2, f5)
        clip = {"F2": f2, "F5": f5}
        label_source = "muse"
    elif sam is not None:
        windows = [
            _window_out(window, verdict_bit(window.get("F2")), verdict_bit(window.get("F5")))
            for window in (sam.get("windows") or [])
        ]
        clip = {"F2": clip_from_windows(windows, "F2"), "F5": clip_from_windows(windows, "F5")}
        label_source = "sam"
    else:
        windows = []
        clip = {"F2": None, "F5": None}
        label_source = "none"
    return {
        "id": meta["id"],
        "source": meta.get("source"),
        "split": meta.get("split"),
        "path": meta.get("path"),
        "license": meta.get("license"),
        "attribution": meta.get("attribution"),
        "url": meta.get("url"),
        "clip": clip,
        "muse": muse_out,
        "label_source": label_source,
        "windows": windows,
    }


def window_number(window: dict) -> int:
    """Index rows say ``window``; SAM rows say ``w``."""
    if "w" in window:
        return int(window["w"])
    return int(window["window"])


def pick_muse_window(windows: list[dict], sam_by_w: dict) -> dict | None:
    """Keyframes of the window with the strongest SAM signal, if any signal exists."""
    best = None
    best_key = None
    for window in windows:
        number = window_number(window)
        sam = sam_by_w.get(number) or {}
        pos = int(verdict_bit(sam.get("F2")) == 1) + int(verdict_bit(sam.get("F5")) == 1)
        known = int(verdict_bit(sam.get("F2")) is not None) + int(verdict_bit(sam.get("F5")) is not None)
        if pos == 0 and known == 0:
            continue
        key = (pos, known, -number)
        if best_key is None or key > best_key:
            best, best_key = window, key
    return best


def spread_frame_paths(windows: list[dict]) -> list[str]:
    """Three frames spread over the clip: start, middle, end."""
    if not windows:
        return []
    if len(windows) == 1:
        return [str(p) for p in (windows[0].get("paths") or [])[:3]]
    mid = windows[len(windows) // 2]
    paths = [
        (windows[0].get("paths") or [None])[0],
        (mid.get("paths") or [None, None])[1] if len(mid.get("paths") or []) > 1 else (mid.get("paths") or [None])[0],
        (windows[-1].get("paths") or [None])[-1],
    ]
    return [str(p) for p in paths if p]


def choose_muse_paths(windows: list[dict], sam_by_w: dict) -> list[str]:
    chosen = pick_muse_window(windows, sam_by_w)
    if chosen is not None:
        return [str(p) for p in (chosen.get("paths") or [])[:3]]
    return spread_frame_paths(windows)


def priority_key(row: dict):
    return (SPLIT_RANK.get(row.get("split"), 9), str(row.get("id")))


def summarize_labels(rows: list[dict]) -> list[dict]:
    """Counts per source and split: clips, windows, and 1/0/null for each factor."""
    buckets: dict[tuple, dict] = {}
    for row in rows:
        key = (row.get("source"), row.get("split"))
        bucket = buckets.setdefault(key, {
            "source": key[0], "split": key[1], "clips": 0, "windows": 0,
            "clip_F2": {1: 0, 0: 0, None: 0}, "clip_F5": {1: 0, 0: 0, None: 0},
            "win_F2": {1: 0, 0: 0, None: 0}, "win_F5": {1: 0, 0: 0, None: 0},
        })
        bucket["clips"] += 1
        clip = row.get("clip") or {}
        for factor, slot in (("F2", "clip_F2"), ("F5", "clip_F5")):
            bucket[slot][verdict_bit(clip.get(factor))] += 1
        for window in row.get("windows") or []:
            bucket["windows"] += 1
            for factor, slot in (("F2", "win_F2"), ("F5", "win_F5")):
                bucket[slot][verdict_bit(window.get(factor))] += 1
    return [buckets[key] for key in sorted(buckets, key=lambda k: (str(k[0]), SPLIT_RANK.get(k[1], 9), str(k[1])))]


def format_summary(rows: list[dict]) -> str:
    counts = summarize_labels(rows)
    header = (
        f"{'source':<24} {'split':<12} {'clips':>6} {'windows':>8} "
        f"{'cF2+':>5} {'cF2-':>5} {'cF2?':>5} {'cF5+':>5} {'cF5-':>5} {'cF5?':>5} "
        f"{'wF2+':>5} {'wF2-':>5} {'wF2?':>5} {'wF5+':>5} {'wF5-':>5} {'wF5?':>5}"
    )
    lines = [header]
    for row in counts:
        c2, c5 = row["clip_F2"], row["clip_F5"]
        w2, w5 = row["win_F2"], row["win_F5"]
        lines.append(
            f"{str(row['source']):<24} {str(row['split']):<12} {row['clips']:6d} {row['windows']:8d} "
            f"{c2[1]:5d} {c2[0]:5d} {c2[None]:5d} {c5[1]:5d} {c5[0]:5d} {c5[None]:5d} "
            f"{w2[1]:5d} {w2[0]:5d} {w2[None]:5d} {w5[1]:5d} {w5[0]:5d} {w5[None]:5d}"
        )
    if not counts:
        lines.append("(no rows)")
    return "\n".join(lines)
