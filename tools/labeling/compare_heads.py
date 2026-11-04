"""Compare head runs on Voxel51 F1, Zenodo patch F1, PPE F2/F5, and demo replay.

  python tools/labeling/compare_heads.py

Reads training/04-head/results/<run>.json and <run>_eval.json, plus
/mnt/f/models/ppe_eval/<run>/replay_summary.json when that replay exists.
Default runs are h3_wide and every results file whose name starts with ppe_.
Writes training/04-head/results/PPE_COMPARISON.md.

A candidate PASSes only when its Voxel51 F1 test AUROC and Zenodo F1 patch
AUROC are within 0.01 of h3_wide and both F2 and F5 pooled PPE-test clip
AUROCs are at least 0.75.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "training" / "04-head" / "results"
REPLAY_ROOT = Path("/mnt/f/models/ppe_eval")
TOL = 0.01
PPE_AUROC_MIN = 0.75


def _get(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def metric_pair(d):
    if not isinstance(d, dict) or "auroc" not in d:
        return None, None
    return d.get("auroc"), d.get("ap")


def fmt_pair(auroc, ap, pos=None, n=None):
    if auroc is None:
        return "-"
    cell = f"{auroc:.3f} / {ap:.3f}" if ap is not None else f"{auroc:.3f}"
    if n:
        cell += f" ({pos}/{n})"
    return cell


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_runs(results_dir: Path, explicit: list[str] | None, baseline: str) -> list[str]:
    if explicit:
        names = list(explicit)
        if baseline not in names:
            names.insert(0, baseline)
        return names
    names = {baseline}
    if results_dir.is_dir():
        for path in results_dir.glob("*.json"):
            stem = path.stem[:-5] if path.stem.endswith("_eval") else path.stem
            if stem == baseline or stem.startswith("ppe_"):
                names.add(stem)
    return sorted(names, key=lambda n: (n != baseline, n))


def load_run(results_dir: Path, name: str) -> dict:
    """Training json wins; an eval json fills a missing ppe block."""
    train = load_json(results_dir / f"{name}.json") or {}
    ev = load_json(results_dir / f"{name}_eval.json") or {}
    if not train:
        return ev
    if not train.get("ppe") and ev.get("ppe"):
        merged = dict(train)
        merged["ppe"] = ev["ppe"]
        return merged
    return train


def row_from_results(name: str, doc: dict, replay: dict | None) -> dict:
    voxel = _get(doc, "test_by_source", "voxel51", "F1") or _get(doc, "test", "F1") or {}
    zenodo = _get(doc, "test_by_source", "zenodo", "F1_patch_vs_boxes") or _get(doc, "test", "F1_patch_vs_boxes") or {}
    ppe = doc.get("ppe") or {}
    clip = (ppe.get("clip") or {})
    window = (ppe.get("window") or {})

    def auroc_of(block, source, factor):
        d = _get(block, source, factor) or {}
        return d.get("auroc"), d.get("ap"), d.get("pos"), d.get("n")

    ppe_clip = {}
    ppe_window = {}
    for source in ("all", "pexels", "voxel51"):
        ppe_clip[source] = {}
        ppe_window[source] = {}
        for factor in ("F2", "F5"):
            ppe_clip[source][factor] = auroc_of(clip, source, factor)
            ppe_window[source][factor] = auroc_of(window, source, factor)
    return {
        "name": name,
        "f1_auroc": voxel.get("auroc"),
        "f1_ap": voxel.get("ap"),
        "f1_pos": voxel.get("pos"),
        "f1_n": voxel.get("n"),
        "zenodo_auroc": zenodo.get("auroc"),
        "zenodo_ap": zenodo.get("ap"),
        "ppe_clip": ppe_clip,
        "ppe_window": ppe_window,
        "ppe_clip_all": {f: (ppe_clip["all"][f][0]) for f in ("F2", "F5")},
        "replay": replay_brief(replay),
        "missing": not bool(doc),
    }


def replay_brief(summary: dict | None) -> list[dict]:
    if not summary:
        return []
    out = []
    for clip in summary.get("clips") or []:
        scores = clip.get("scores") or {}
        out.append({
            "id": clip.get("id") or clip.get("camera_id") or clip.get("path"),
            "path": clip.get("path"),
            "skipped": bool(clip.get("skipped")),
            "f2_max": _get(scores, "F2", "max"),
            "f5_max": _get(scores, "F5", "max"),
            "f1_max": _get(scores, "F1", "max"),
            "incidents": clip.get("incidents") or {},
            "circle_distance_px": clip.get("circle_distance_px") or {},
        })
    return out


def passes(row: dict, baseline: dict, tol: float = TOL, floor: float = PPE_AUROC_MIN) -> bool:
    """F1 within tol of h3_wide, and both PPE test clip AUROCs at least `floor`."""
    if row.get("f1_auroc") is None or baseline.get("f1_auroc") is None:
        return False
    if row.get("zenodo_auroc") is None or baseline.get("zenodo_auroc") is None:
        return False
    if abs(float(row["f1_auroc"]) - float(baseline["f1_auroc"])) > tol + 1e-12:
        return False
    if abs(float(row["zenodo_auroc"]) - float(baseline["zenodo_auroc"])) > tol + 1e-12:
        return False
    for factor in ("F2", "F5"):
        val = (row.get("ppe_clip_all") or {}).get(factor)
        if val is None or float(val) < floor:
            return False
    return True


def _fmt_num(v, digits=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def render_markdown(rows: list[dict], baseline_name: str) -> str:
    base = next((r for r in rows if r["name"] == baseline_name), None)
    lines = [
        "# PPE head comparison",
        "",
        f"Baseline: `{baseline_name}`. A run is **PASS** only when Voxel51 test F1 AUROC and "
        f"Zenodo F1 patch AUROC are within {TOL:.2f} of `{baseline_name}`, and pooled PPE-test "
        f"clip AUROC is at least {PPE_AUROC_MIN:.2f} for both F2 and F5.",
        "",
        "Clip score is the MIL probability. Window score is the max patch evidence.",
        "",
        "| run | verdict | F1 voxel AUROC/AP | Zenodo F1 patch AUROC/AP | F2 clip all | F5 clip all | "
        "F2 clip pexels | F5 clip pexels | F2 clip voxel51 | F5 clip voxel51 | F2 window all | F5 window all |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["name"] == baseline_name:
            verdict = "baseline"
        elif base is None or row.get("missing"):
            verdict = "FAIL"
        else:
            verdict = "PASS" if passes(row, base) else "FAIL"
        def pair(block, source, factor):
            auroc, ap, pos, n = row[block][source][factor]
            return fmt_pair(auroc, ap, pos, n)
        lines.append(
            "| {name} | {verdict} | {f1} | {zen} | {c2} | {c5} | {p2} | {p5} | {v2} | {v5} | {w2} | {w5} |".format(
                name=row["name"], verdict=verdict,
                f1=fmt_pair(row["f1_auroc"], row["f1_ap"], row["f1_pos"], row["f1_n"]),
                zen=fmt_pair(row["zenodo_auroc"], row["zenodo_ap"]),
                c2=pair("ppe_clip", "all", "F2"), c5=pair("ppe_clip", "all", "F5"),
                p2=pair("ppe_clip", "pexels", "F2"), p5=pair("ppe_clip", "pexels", "F5"),
                v2=pair("ppe_clip", "voxel51", "F2"), v5=pair("ppe_clip", "voxel51", "F5"),
                w2=pair("ppe_window", "all", "F2"), w5=pair("ppe_window", "all", "F5"),
            )
        )
    lines += ["", "## Window metrics per source", "",
              "| run | F2 window pexels | F5 window pexels | F2 window voxel51 | F5 window voxel51 |",
              "|---|---|---|---|---|"]
    for row in rows:
        def wpair(source, factor):
            auroc, ap, pos, n = row["ppe_window"][source][factor]
            return fmt_pair(auroc, ap, pos, n)
        lines.append(
            f"| {row['name']} | {wpair('pexels', 'F2')} | {wpair('pexels', 'F5')} | "
            f"{wpair('voxel51', 'F2')} | {wpair('voxel51', 'F5')} |"
        )
    lines += ["", "## Demo replay", "",
              "| run | clip | F1 max | F2 max | F5 max | F1 incidents | F2 incidents | F5 incidents | "
              "F2 circle px | F5 circle px |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    any_replay = False
    for row in rows:
        if not row["replay"]:
            lines.append(f"| {row['name']} | - | - | - | - | - | - | - | - | - |")
            continue
        any_replay = True
        for clip in row["replay"]:
            inc = clip.get("incidents") or {}
            dist = clip.get("circle_distance_px") or {}
            label = clip.get("id") or clip.get("path")
            if clip.get("skipped"):
                label = f"{label} (missing)"
            lines.append(
                f"| {row['name']} | {label} | {_fmt_num(clip.get('f1_max'))} | {_fmt_num(clip.get('f2_max'))} | "
                f"{_fmt_num(clip.get('f5_max'))} | {_fmt_num(inc.get('F1'), 0)} | {_fmt_num(inc.get('F2'), 0)} | "
                f"{_fmt_num(inc.get('F5'), 0)} | {_fmt_num(dist.get('F2'))} | {_fmt_num(dist.get('F5'))} |"
            )
    if not any_replay:
        lines += ["", "No replay summaries were found under the replay root.", ""]
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write PPE_COMPARISON.md for h3_wide and ppe_* runs")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--replay-root", default=str(REPLAY_ROOT))
    ap.add_argument("--out", default="")
    ap.add_argument("--baseline", default="h3_wide")
    ap.add_argument("--runs", nargs="*", default=None)
    a = ap.parse_args(argv)
    results = Path(a.results)
    replay_root = Path(a.replay_root)
    names = discover_runs(results, a.runs, a.baseline)
    rows = []
    for name in names:
        doc = load_run(results, name)
        replay = load_json(replay_root / name / "replay_summary.json")
        rows.append(row_from_results(name, doc, replay))
    text = render_markdown(rows, a.baseline)
    out = Path(a.out) if a.out else results / "PPE_COMPARISON.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
