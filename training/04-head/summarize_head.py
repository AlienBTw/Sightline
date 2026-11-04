"""Summarize head runs in 04-head/results/*.json into results/SUMMARY.md.

Per run: Voxel51 test (official, de-duplicated) metrics per factor, Zenodo test F1
(patch-level vs hand boxes), and, for runs named <base>_cv<k>_f<i>, pooled
out-of-fold metrics over all 658 Voxel51 clips.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FACTORS = [f"F{i}" for i in range(1, 11)]


def fmt(v):
    return "-" if v is None else f"{v:.2f}"


def oof_metrics(preds):
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = {}
    for i, f in enumerate(FACTORS):
        yy = [p["y"][i] for p in preds if p["m"][i]]
        pp = [p["p"][i] for p in preds if p["m"][i]]
        d = {"n": len(yy), "pos": int(sum(yy))}
        if len(set(yy)) == 2:
            d["auroc"] = float(roc_auc_score(yy, pp)); d["ap"] = float(average_precision_score(yy, pp))
        out[f] = d
    return out


def table(rows, title):
    lines = [f"### {title}", "", "| run | " + " | ".join(FACTORS) + " |", "|---" * (len(FACTORS) + 1) + "|"]
    for name, res in rows:
        cells = []
        for f in FACTORS:
            d = res.get(f, {})
            if "auroc" in d:
                cells.append(f"{d['auroc']:.2f} / {d['ap']:.2f} ({d['pos']}/{d['n']})")
            elif d.get("n"):
                cells.append(f"one class ({d['pos']}/{d['n']})")
            else:
                cells.append("not trained")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines + [""]


def main(results_dir=HERE / "results"):
    runs = {Path(p).stem: json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(str(Path(results_dir) / "*.json")))}
    single = [(n, r) for n, r in runs.items() if not re.search(r"_cv\d+_f\d+$", n)]
    lines = ["# 04-head results summary", "",
             "Cells: AUROC / AP (positives/labeled clips). Uncertain labels are excluded; F3/F4 are masked out of training.", ""]
    lines += table([(n, r["test_by_source"]["voxel51"]) for n, r in single], "Voxel51 official test split (117 unique clips)")
    lines += ["### Zenodo test (3,591 images): F1 patch map vs hand boxes", "", "| run | patch AUROC | patch AP | image F1 score |", "|---|---|---|---|"]
    for n, r in single:
        z = r["test_by_source"].get("zenodo", {})
        pz = z.get("F1_patch_vs_boxes", {})
        f1 = r["test"].get("F1", {})
        lines.append(f"| {n} | {fmt(pz.get('auroc'))} | {fmt(pz.get('ap'))} | {fmt(f1.get('f1')) if z else '-'} |")
    lines.append("")
    shwd_rows = [(n, r["shwd"]) for n, r in single if r.get("shwd")]
    if shwd_rows:
        lines += ["### SHWD: F2 patch map vs bare-head boxes", "",
                  "| run | val patch AUROC | val patch AP | test patch AUROC | test patch AP |",
                  "|---|---|---|---|---|"]
        for n, block in shwd_rows:
            v = (block.get("validation") or {}).get("F2_patch_vs_boxes") or {}
            t = (block.get("test") or {}).get("F2_patch_vs_boxes") or {}
            lines.append(
                f"| {n} | {fmt(v.get('auroc'))} | {fmt(v.get('ap'))} | {fmt(t.get('auroc'))} | {fmt(t.get('ap'))} |"
            )
        lines.append("")
    ppe_rows = [(n, r["ppe"]) for n, r in single if isinstance(r.get("ppe"), dict)]
    if ppe_rows:
        lines += ["### PPE video: F2/F5 on the real-video test split", "",
                  "Clip score is the MIL probability. Window score is the max patch evidence in that window.",
                  "",
                  "| run | F2 clip AUROC / AP (pos/n) | F5 clip AUROC / AP (pos/n) | F2 window | F5 window | F2 patch AUROC | F5 patch AUROC |",
                  "|---|---|---|---|---|---|---|"]
        for n, block in ppe_rows:
            clip = (block.get("clip") or {}).get("all") or {}
            win = (block.get("window") or {}).get("all") or {}
            patch = block.get("patch") or {}

            def cell(d):
                if not isinstance(d, dict) or not d.get("n"):
                    return "-"
                if "auroc" in d:
                    return f"{d['auroc']:.2f} / {d['ap']:.2f} ({d.get('pos')}/{d.get('n')})"
                return f"one class ({d.get('pos')}/{d.get('n')})"

            lines.append(
                f"| {n} | {cell(clip.get('F2'))} | {cell(clip.get('F5'))} | {cell(win.get('F2'))} | "
                f"{cell(win.get('F5'))} | {fmt((patch.get('F2') or {}).get('auroc'))} | "
                f"{fmt((patch.get('F5') or {}).get('auroc'))} |"
            )
        lines.append("")
    cv = defaultdict(list)
    for n, r in runs.items():
        m = re.match(r"(.+_cv\d+)_f\d+$", n)
        if m:
            cv[m.group(1)].extend(r.get("test_predictions_voxel51", []))
    if cv:
        lines += table([(b, oof_metrics(p)) for b, p in cv.items()], "Voxel51 k-fold out-of-fold (all 658 unique clips)")
    (Path(results_dir) / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
