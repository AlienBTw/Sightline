"""Stitch the demo video (virtual camera input) from held-out, CC BY 4.0 footage.

Picks Voxel51 Safe_and_Unsafe_Behaviours TEST-split clips (never trained on) with clear hazards
(the head's test predictions in training/04-head/results/h3_wide.json + dataset class), puts safe
clips between them so incidents open and close, and adds a run of consecutive Zenodo table-saw
TEST frames (hand at the blade). Writes the MP4 plus ATTRIBUTION.json/.md with every segment.

  python -m sightline_pipeline.make_demo --out data/demo/sightline_demo.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import REPO
from .storage import ClipWriter

TR = REPO / "training"
ATTR_VOXEL = ("Önal, O., & Dandıl, E. (2024). Video dataset for the detection of safe and unsafe behaviours in "
              "workplaces. Data in Brief, 56, 110756. https://doi.org/10.1016/j.dib.2024.110756 "
              "(FiftyOne export by Voxel51). CC BY 4.0")
ATTR_ZENODO = ("Steckhan, N., Prajapati, K., Shao, W., Heinle, J., Vock, S. (2026). Dataset for Hand Detection Near a "
               "Sliding Table Saw in Safety-Critical Woodworking Scenarios. Fraunhofer IPA. Zenodo. "
               "https://doi.org/10.5281/zenodo.18936355. CC BY 4.0")
PLAN = [  # (dataset class, factor to rank by, take top-n, high=True means most hazardous first)
    ("Safe Walkway", "F8", 1, False),
    ("Carrying Overload with Forklift", "F7", 1, True),
    ("Safe Carrying", "F7", 1, False),
    ("Opened Panel Cover", "F9", 1, True),
    ("Closed Panel Cover", "F9", 1, False),
    ("Safe Walkway Violation", "F8", 1, True),
    ("Safe Walkway", "F8", 1, False),
]


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data/demo/sightline_demo.mp4"))
    ap.add_argument("--clip-seconds", type=float, default=9.0)
    ap.add_argument("--zenodo-frames", type=int, default=120)
    ap.add_argument("--fps", type=float, default=25.0)
    a = ap.parse_args(argv)
    W, H = 1280, 720
    # rank clips by out-of-fold head scores (5-fold CV of the same config: every clip scored while held out)
    preds = {}
    for rp in sorted((TR / "04-head/results").glob("h3_wide_cv5_f*.json")):
        for p in json.load(open(rp, encoding="utf-8")).get("test_predictions_voxel51", []):
            preds[p["id"]] = p["p"]
    factors = [f"F{i}" for i in range(1, 11)]
    rows = [json.loads(l) for l in open(TR / "02-labeling/outputs/weak_labels_v2_vllm.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["split"] == "test" and r["clip_id"] in preds]
    used, segs = set(), []
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = ClipWriter(out, a.fps, (W, H), crf=22)
    t = 0.0

    def add_video(r, why):
        nonlocal t
        cap = cv2.VideoCapture(str(TR / r["path"]))
        sfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = int(a.clip_seconds * sfps)
        start = max(0, (n - want) // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        k, t0, src_frames = 0, t, []
        while k < want:
            ok, im = cap.read()
            if not ok:
                break
            src_frames.append(im); k += 1
        cap.release()
        dur = len(src_frames) / sfps
        for j in range(int(dur * a.fps)):
            im = src_frames[min(len(src_frames) - 1, int(j * sfps / a.fps))]
            w.write(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA))
        t += int(dur * a.fps) / a.fps
        segs.append({"start_s": round(t0, 2), "end_s": round(t, 2), "source": "Voxel51 Safe_and_Unsafe_Behaviours",
                     "item": r["clip_id"], "file": r["path"], "split": r["split"], "class": r["source_label"],
                     "why": why, "license": "CC BY 4.0",
                     "head_test_scores": {f: preds[r["clip_id"]][i] for i, f in enumerate(factors) if f not in ("F3", "F4")}})

    for cls, f, n, high in PLAN:
        cand = [r for r in rows if r["source_label"] == cls and r["clip_id"] not in used and not r.get("duplicate_of")]
        fi = factors.index(f)
        cand.sort(key=lambda r: -preds[r["clip_id"]][fi] if high else preds[r["clip_id"]][fi])
        for r in cand[:n]:
            used.add(r["clip_id"])
            add_video(r, f"{'hazard' if high else 'safe contrast'} for {f}")
    # Zenodo table saw: consecutive test frames with hand boxes
    man = [json.loads(l) for l in open(TR / "data/manifest.jsonl", encoding="utf-8")]
    z = sorted((m for m in man if m["source"] == "zenodo_tablesaw" and m.get("split") == "test"
                and int(m.get("n_hand_boxes") or 0) > 0), key=lambda m: m["id"])
    mid = len(z) // 3
    run = z[mid: mid + a.zenodo_frames]
    t0 = t
    rep = max(1, int(round(a.fps / 12.5)))
    for m in run:
        im = cv2.imread(str(TR / m["path"]))
        if im is None:
            continue
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        for _ in range(rep):
            w.write(im)
        t += rep / a.fps
    segs.append({"start_s": round(t0, 2), "end_s": round(t, 2),
                 "source": "Fraunhofer IPA sliding table saw hand dataset (Zenodo)", "items": [run[0]["id"], run[-1]["id"]],
                 "n_frames": len(run), "split": "test", "why": "hazard for F1 (hand near the blade)",
                 "license": "CC BY 4.0", "note": "still frames shown at 12.5 fps"})
    codec = w.close()
    att = {"video": str(out.name), "fps": a.fps, "size": [W, H], "codec": codec, "duration_s": round(t, 2),
           "segments": segs,
           "attribution": [ATTR_VOXEL, ATTR_ZENODO],
           "note": "All footage is from held-out TEST splits (not used to train the head)."}
    json.dump(att, open(out.with_name("ATTRIBUTION.json"), "w", encoding="utf-8"), indent=1)
    md = ["# Sightline demo video - attribution", "", f"`{out.name}`: {t:.1f} s, {W}x{H} @ {a.fps:g} fps.", "",
          "| time (s) | source | item | class / note | license |", "|---|---|---|---|---|"]
    for s in segs:
        md.append(f"| {s['start_s']}-{s['end_s']} | {s['source']} | {s.get('item') or ' .. '.join(s['items'])} | "
                  f"{s.get('class', s.get('note', ''))} ({s['why']}) | {s['license']} |")
    md += ["", "All clips/frames come from the held-out test splits.", "", "## Sources", "", f"- {ATTR_VOXEL}", f"- {ATTR_ZENODO}"]
    out.with_name("ATTRIBUTION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(att, indent=1))


if __name__ == "__main__":
    main()
