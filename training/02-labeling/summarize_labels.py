#!/usr/bin/env python3
"""Phase 02 - summary of a full Glimmer weak-label run (no model calls).

Reads a weak_labels_*.jsonl written by label_glimmer.py and reports:
  - row / unique clip counts, splits, model / prompt / backend
  - per-factor clip-level labels (present / absent / uncertain) for F1-F10
  - per-factor frame-level answers (from the per-clip "votes")
  - agreement of Glimmer F7/F8/F9 with the Voxel51 class hints (confusion, precision, recall)
  - present counts per dataset class
  - malformed rows (missing fields, bad labels, votes that don't add up to the keyframes)

  python training/02-labeling/summarize_labels.py
  python training/02-labeling/summarize_labels.py --labels training/02-labeling/outputs/weak_labels_v2_vllm.jsonl \
      --out training/02-labeling/outputs/eval/full_run_summary_vllm.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from glimmer_schema import GLIMMER_FACTORS, STATES  # noqa: E402

HINT_FACTORS = ("F7", "F8", "F9")
REQUIRED = ("clip_id", "source", "split", "factor_hints", "keyframes", "factors")
DEFAULT_LABELS = HERE / "outputs" / "weak_labels_v2_vllm.jsonl"
DEFAULT_OUT = HERE / "outputs" / "eval" / "full_run_summary_vllm.md"


def read_rows(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (rows, unparseable) where unparseable lists lines that are not JSON objects."""
    rows, bad = [], []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                bad.append({"line": n, "clip_id": None, "problems": [f"invalid JSON: {e.msg}"]})
                continue
            if not isinstance(obj, dict):
                bad.append({"line": n, "clip_id": None, "problems": ["not a JSON object"]})
                continue
            obj["_line"] = n
            rows.append(obj)
    return rows, bad


def check_row(row: dict) -> list[str]:
    """Problems with one clip row; an empty list means the row is well formed."""
    problems = [f"missing {k}" for k in REQUIRED if k not in row]
    facs = row.get("factors")
    if not isinstance(facs, dict):
        return problems + ["factors is not an object"]
    missing = [f for f in GLIMMER_FACTORS if f not in facs]
    extra = sorted(set(facs) - set(GLIMMER_FACTORS))
    if missing:
        problems.append(f"missing factors {missing}")
    if extra:
        problems.append(f"unexpected factors {extra}")
    nk = len(row.get("keyframes") or [])
    if "keyframes" in row and nk == 0:
        problems.append("no keyframes")
    for fid in GLIMMER_FACTORS:
        f = facs.get(fid)
        if f is None:
            continue
        if not isinstance(f, dict):
            problems.append(f"{fid} is not an object")
            continue
        if f.get("label") not in STATES:
            problems.append(f"{fid}.label {f.get('label')!r} not in {list(STATES)}")
        c = f.get("confidence")
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0.0 <= c <= 1.0:
            problems.append(f"{fid}.confidence {c!r} not in [0, 1]")
        votes = f.get("votes")
        if not isinstance(votes, dict) or set(votes) - set(STATES) or \
                any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in votes.values()):
            problems.append(f"{fid}.votes malformed: {votes!r}")
        elif nk and sum(votes.values()) != nk:
            problems.append(f"{fid}.votes add up to {sum(votes.values())}, expected {nk} keyframes")
    return problems


def _ratio(a: int, b: int):
    return round(a / b, 3) if b else None


def summarize(rows: list[dict], unparseable: list[dict] | None = None) -> dict:
    malformed = list(unparseable or [])
    good = []
    for r in rows:
        p = check_row(r)
        if p:
            malformed.append({"line": r.get("_line"), "clip_id": r.get("clip_id"), "problems": p})
        else:
            good.append(r)
    ids = Counter(r.get("clip_id") for r in rows)
    clip = {f: Counter() for f in GLIMMER_FACTORS}
    frame = {f: Counter() for f in GLIMMER_FACTORS}
    by_class: dict[str, Counter] = {}
    frames_total = 0
    for r in good:
        frames_total += len(r["keyframes"])
        cls = r.get("source_label") or "(none)"
        bc = by_class.setdefault(cls, Counter())
        bc["_clips"] += 1
        for f in GLIMMER_FACTORS:
            lab = r["factors"][f]["label"]
            clip[f][lab] += 1
            bc[f] += lab == "present"
            for s in STATES:
                frame[f][s] += r["factors"][f]["votes"].get(s, 0)

    agreement = {}
    for f in HINT_FACTORS:
        conf = {"present": Counter(), "absent": Counter()}
        for r in good:
            hint = (r.get("factor_hints") or {}).get(f)
            if hint in conf:
                conf[hint][r["factors"][f]["label"]] += 1
        npos, nneg = sum(conf["present"].values()), sum(conf["absent"].values())
        tp, fp = conf["present"]["present"], conf["absent"]["present"]
        tn = conf["absent"]["absent"]
        unc = conf["present"]["uncertain"] + conf["absent"]["uncertain"]
        decided = npos + nneg - unc
        agreement[f] = {
            "hint_present": npos, "hint_absent": nneg,
            "confusion": {h: {s: conf[h][s] for s in STATES} for h in conf},
            "tp": tp, "fp": fp, "fn": npos - tp, "tn": tn, "uncertain": unc,
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, npos),
            "specificity": _ratio(tn, nneg), "uncertain_rate": _ratio(unc, npos + nneg),
            "agree_when_decided": _ratio(tp + tn, decided),
        }

    labels = [r["factors"][f]["label"] for r in good for f in GLIMMER_FACTORS]
    return {
        "rows": len(rows) + len(unparseable or []),
        "unique_clips": len(ids),
        "duplicate_clip_ids": sorted(str(k) for k, n in ids.items() if n > 1),
        "valid_rows": len(good),
        "malformed": malformed,
        "frames": frames_total,
        "splits": dict(Counter(r.get("split") for r in good)),
        "sources": dict(Counter(r.get("source") for r in good)),
        "models": dict(Counter(r.get("model") for r in good)),
        "prompts": dict(Counter(r.get("prompt_id") for r in good)),
        "reasoning": dict(Counter(r.get("reasoning_effort") for r in good)),
        "merged_duplicate_groups": sum(1 for r in good if r.get("duplicate_of")),
        "clip_level": {f: {s: clip[f][s] for s in STATES} for f in GLIMMER_FACTORS},
        "frame_level": {f: {s: frame[f][s] for s in STATES} for f in GLIMMER_FACTORS},
        "uncertain_rate_all": _ratio(labels.count("uncertain"), len(labels)),
        "present_rate_all": _ratio(labels.count("present"), len(labels)),
        "agreement": agreement,
        "present_by_class": {k: dict(v) for k, v in sorted(by_class.items(), key=lambda kv: -kv[1]["_clips"])},
    }


def _f(v) -> str:
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def _pct(a: int, n: int) -> str:
    return f"{a} ({100 * a / n:.0f}%)" if n else str(a)


def to_markdown(s: dict, title: str) -> str:
    n, nf = s["valid_rows"], s["frames"]
    out = [f"# {title}", ""]
    out += ["## Run", "",
            f"- Rows: {s['rows']}, unique clips: {s['unique_clips']}, valid rows: {n}, "
            f"malformed: {len(s['malformed'])}, duplicate clip ids: {len(s['duplicate_clip_ids'])}",
            f"- Frames (keyframes labeled): {nf}",
            f"- Splits: {s['splits']}; sources: {s['sources']}",
            f"- Model: {s['models']}; prompt: {s['prompts']}; reasoning: {s['reasoning']}",
            f"- Clips standing for a byte-identical duplicate group (hints merged): {s['merged_duplicate_groups']}",
            f"- All F1-F10 clip labels: present {_f(s['present_rate_all'])}, uncertain {_f(s['uncertain_rate_all'])}",
            ""]
    out += ["## Per-factor labels", "",
            "Clip level = MIL aggregation (present if any frame says present with conf >= 0.5; absent only if "
            "every frame says absent; otherwise uncertain). Frame level = the raw per-keyframe answers.", "",
            "| Factor | Clip present | Clip absent | Clip uncertain | Frame present | Frame absent | Frame uncertain |",
            "|---|---|---|---|---|---|---|"]
    for f in GLIMMER_FACTORS:
        c, fr = s["clip_level"][f], s["frame_level"][f]
        out.append(f"| {f} | {_pct(c['present'], n)} | {_pct(c['absent'], n)} | {_pct(c['uncertain'], n)} | "
                   f"{_pct(fr['present'], nf)} | {_pct(fr['absent'], nf)} | {_pct(fr['uncertain'], nf)} |")
    out += ["", "## Glimmer vs Voxel51 class hints (F7 / F8 / F9)", "",
            "Positive = Glimmer clip label present. Uncertain counts as a miss for recall and is not a false "
            "positive. Agree-when-decided ignores uncertain clips.", "",
            "| Factor | Hint present (n) | -> present / absent / unc | Hint absent (n) | -> present / absent / unc "
            "| Precision | Recall | Specificity | Uncertain | Agree when decided |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for f, a in s["agreement"].items():
        cp, ca = a["confusion"]["present"], a["confusion"]["absent"]
        out.append(f"| {f} | {a['hint_present']} | {cp['present']} / {cp['absent']} / {cp['uncertain']} | "
                   f"{a['hint_absent']} | {ca['present']} / {ca['absent']} / {ca['uncertain']} | "
                   f"{_f(a['precision'])} | {_f(a['recall'])} | {_f(a['specificity'])} | "
                   f"{_f(a['uncertain_rate'])} | {_f(a['agree_when_decided'])} |")
    out += ["", "## Clip-level present count by dataset class", "",
            "| Class | Clips | " + " | ".join(GLIMMER_FACTORS) + " |",
            "|---|---|" + "---|" * len(GLIMMER_FACTORS)]
    for cls, c in s["present_by_class"].items():
        out.append(f"| {cls} | {c['_clips']} | " + " | ".join(str(c.get(f, 0)) for f in GLIMMER_FACTORS) + " |")
    out += ["", "## Malformed rows", ""]
    if s["malformed"]:
        for m in s["malformed"][:50]:
            out.append(f"- line {m['line']} `{m['clip_id']}`: {'; '.join(m['problems'])}")
    else:
        out.append("None: every row has F1-F10 with a valid label, a confidence in [0, 1] and votes that add up "
                   "to its keyframes.")
    if s["duplicate_clip_ids"]:
        out += ["", f"Duplicate clip ids: {', '.join(s['duplicate_clip_ids'][:50])}"]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="markdown report; a .json twin is written next to it")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)
    rows, bad = read_rows(args.labels)
    s = summarize(rows, bad)
    s["labels_file"] = args.labels.name
    md = to_markdown(s, args.title or f"Glimmer weak-label summary: {args.labels.name}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    print(md)
    return 1 if s["malformed"] or s["duplicate_clip_ids"] else 0


if __name__ == "__main__":
    sys.exit(main())