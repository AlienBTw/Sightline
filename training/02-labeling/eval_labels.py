#!/usr/bin/env python3
"""Phase 02 - repeatable comparison of Glimmer prompt / reasoning settings.

Builds (once) a fixed evaluation set of unique Voxel51 clips whose dataset class gives a
ground-truth hint for F7, F8 or F9 (present and absent), labels it with every requested
configuration through label_glimmer.py (cache-backed, so nothing is re-asked), and reports
per-factor precision / recall against the hints, the uncertain rate and seconds per call.

  python training/02-labeling/eval_labels.py                      # default: v1/v2 x none/low
  python training/02-labeling/eval_labels.py --configs v2:low     # one configuration
  python training/02-labeling/eval_labels.py --metrics-only       # recompute tables, no model calls
  python training/02-labeling/eval_labels.py --backend sglang --configs v2:none --concurrency 16
      # SGLang run: writes labels_<prompt>_<effort>_sglang.jsonl and eval_results_sglang.{json,md}

Outputs (small, committable): 02-labeling/outputs/eval/
  eval_set.json                        the fixed clip list
  labels_<prompt>_<effort>.jsonl       clip labels per configuration
  eval_results.json / eval_results.md  metrics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import label_glimmer as lg  # noqa: E402
from glimmer_schema import GLIMMER_FACTORS  # noqa: E402

EVAL_DIR = HERE / "outputs" / "eval"
GT_FACTORS = ("F7", "F8", "F9")
DEFAULT_BUCKETS = {("F8", "present"): 5, ("F8", "absent"): 4, ("F9", "present"): 4,
                   ("F9", "absent"): 3, ("F7", "present"): 4, ("F7", "absent"): 3}
DEFAULT_CONFIGS = "v1:none,v2:none,v1:low,v2:low"


def select_eval_set(rows: list[dict], buckets: dict = DEFAULT_BUCKETS, seed: str = "sightline-eval-1") -> list[dict]:
    """Deterministic pick of unique clips (no byte-identical copies, exactly one hinted factor)."""
    pool = [r for r in rows if r.get("source") == "voxel51_safe_unsafe" and not r.get("duplicate_of")
            and len(r.get("factor_hints") or {}) == 1]
    pool.sort(key=lambda r: hashlib.sha256((seed + r["id"]).encode()).hexdigest())
    chosen, taken = [], set()
    for (fid, state), n in buckets.items():
        got = [r for r in pool if r["factor_hints"].get(fid) == state and r["id"] not in taken][:n]
        if len(got) < n:
            raise SystemExit(f"ERROR: only {len(got)} unique clips for {fid}={state}, need {n}")
        for r in got:
            taken.add(r["id"])
            chosen.append({"id": r["id"], "split": r.get("split"), "source_label": r.get("source_label"),
                           "factor_hints": r["factor_hints"]})
    return chosen


def compute_metrics(records: list[dict], eval_set: list[dict], seconds: list[float]) -> dict:
    """Precision/recall per GT factor (prediction 'present' = positive; uncertain counts as not
    positive for recall), uncertain rates, and timing."""
    by_id = {r["clip_id"]: r for r in records}
    missing = [e["id"] for e in eval_set if e["id"] not in by_id]
    per = {}
    for fid in GT_FACTORS:
        tp = fp = fn = tn = unc = n = 0
        for e in eval_set:
            gt = e["factor_hints"].get(fid)
            if gt is None or e["id"] not in by_id:
                continue
            n += 1
            pred = by_id[e["id"]]["factors"][fid]["label"]
            unc += pred == "uncertain"
            if gt == "present":
                tp += pred == "present"
                fn += pred != "present"
            else:
                fp += pred == "present"
                tn += pred == "absent"
        per[fid] = {
            "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "uncertain": unc,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "uncertain_rate": round(unc / n, 3) if n else None,
        }
    labels = [r["factors"][f]["label"] for r in records if r["clip_id"] in {e["id"] for e in eval_set}
              for f in GLIMMER_FACTORS]
    secs = sorted(seconds)
    return {
        "clips": len(eval_set) - len(missing), "missing": missing, "per_factor": per,
        "uncertain_rate_all_factors": round(labels.count("uncertain") / len(labels), 3) if labels else None,
        "present_rate_all_factors": round(labels.count("present") / len(labels), 3) if labels else None,
        "calls_timed": len(secs),
        "sec_per_call_mean": round(sum(secs) / len(secs), 2) if secs else None,
        "sec_per_call_median": round(secs[len(secs) // 2], 2) if secs else None,
    }


def cached_seconds(cache_dir: Path, prompt_id: str, model: str, effort: str, max_side: int, ids: list[str]) -> list[float]:
    out = []
    for cid in ids:
        d = lg.cache_path(cache_dir, prompt_id, model, cid, 0, max_side, effort).parent
        for f in sorted(d.glob("f*.json")) if d.exists() else []:
            s = json.loads(f.read_text(encoding="utf-8")).get("seconds")
            if s is not None:
                out.append(float(s))
    return out


def fmt(v) -> str:
    return "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def to_markdown(results: dict) -> str:
    lines = ["| Config | " + " | ".join(f"{f} P / R / unc" for f in GT_FACTORS)
             + " | Uncertain (all F1–F10) | Present (all) | s/call (mean) | calls/min |",
             "|---|" + "---|" * (len(GT_FACTORS) + 4)]
    for name, m in results["configs"].items():
        cells = [f"{fmt(m['per_factor'][f]['precision'])} / {fmt(m['per_factor'][f]['recall'])} / "
                 f"{fmt(m['per_factor'][f]['uncertain_rate'])}" for f in GT_FACTORS]
        lines.append(f"| {name} | " + " | ".join(cells)
                     + f" | {fmt(m['uncertain_rate_all_factors'])} | {fmt(m['present_rate_all_factors'])}"
                     f" | {fmt(m['sec_per_call_mean'])} | {fmt(m.get('calls_per_min'))} |")
    n = {f: results["configs"][next(iter(results["configs"]))]["per_factor"][f]["n"] for f in GT_FACTORS} if results["configs"] else {}
    lines.append("")
    lines.append(f"Clips per GT factor: {n}. P = precision, R = recall (uncertain counts as a miss), "
                 "unc = share of hinted clips labeled uncertain.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=lg.DEFAULT_MANIFEST)
    ap.add_argument("--configs", default=DEFAULT_CONFIGS, help="comma list of <prompt version>:<reasoning effort>")
    ap.add_argument("--metrics-only", action="store_true", help="no model calls; use existing label files")
    ap.add_argument("--rebuild-set", action="store_true", help="re-pick the evaluation set")
    ap.add_argument("--keyframes", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--backend", default="ollama", choices=sorted(lg.BACKENDS))
    ap.add_argument("--base-url", default=None, help="default: from --backend")
    ap.add_argument("--model", default=None, help="default: from --backend")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--run-name", default=None,
                    help="suffix for output files (default: none for ollama, the backend name otherwise)")
    ap.add_argument("--cache-dir", type=Path, default=lg.DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    args = ap.parse_args(argv)
    args.base_url = args.base_url or lg.BACKENDS[args.backend][0]
    args.model = args.model or lg.BACKENDS[args.backend][1]
    run = args.run_name if args.run_name is not None else ("" if args.backend == "ollama" else args.backend)
    sfx = f"_{run}" if run else ""

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    set_path = out_dir / "eval_set.json"
    if set_path.exists() and not args.rebuild_set:
        eval_set = json.loads(set_path.read_text(encoding="utf-8"))["clips"]
    else:
        eval_set = select_eval_set(lg.read_jsonl(args.manifest))
        set_path.write_text(json.dumps({"created": lg.now_tunis(), "buckets": {f"{k[0]}={k[1]}": v for k, v in DEFAULT_BUCKETS.items()},
                                        "clips": eval_set}, indent=1, ensure_ascii=False), encoding="utf-8")
    ids = [e["id"] for e in eval_set]
    lg.log(f"eval set: {len(ids)} clips ({set_path})")

    results = {"created": lg.now_tunis(), "model": args.model, "backend": args.backend,
               "base_url": args.base_url, "concurrency": args.concurrency, "keyframes": args.keyframes,
               "max_side": args.max_side, "eval_set": str(set_path.name), "configs": {}}
    failed = False
    for cfg in [c.strip() for c in args.configs.split(",") if c.strip()]:
        version, _, effort = cfg.partition(":")
        prompt = HERE / "prompts" / f"factors_{version}.md"
        if not prompt.exists():
            raise SystemExit(f"ERROR: prompt {prompt} not found")
        _, _, prompt_id = lg.load_prompt(prompt)
        out = out_dir / f"labels_{version}_{effort}{sfx}.jsonl"
        run_stats = {}
        if not args.metrics_only:
            lg.log(f"== {cfg} ({prompt_id})")
            rc = lg.main(["--manifest", str(args.manifest), "--prompt", str(prompt), "--reasoning-effort", effort,
                          "--keyframes", str(args.keyframes), "--max-side", str(args.max_side), "--model", args.model,
                          "--cache-dir", str(args.cache_dir), "--out", str(out), "--keep-duplicates",
                          "--backend", args.backend, "--base-url", args.base_url,
                          "--concurrency", str(args.concurrency), "--ids", *ids])
            failed |= rc != 0
            run_stats = dict(lg.LAST_RUN)
        recs = lg.read_jsonl(out)
        secs = cached_seconds(args.cache_dir, prompt_id, args.model, effort, args.max_side, ids)
        m = compute_metrics(recs, eval_set, secs)
        m["prompt_id"] = prompt_id
        if run_stats.get("calls"):
            m["run"] = run_stats
            m["calls_per_min"] = round(run_stats["calls"] / run_stats["wall_s"] * 60, 1)
        results["configs"][cfg] = m
        if m["missing"]:
            failed = True
            lg.log(f"   {cfg}: {len(m['missing'])} clip(s) without labels")
    (out_dir / f"eval_results{sfx}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    md = to_markdown(results)
    (out_dir / f"eval_results{sfx}.md").write_text(f"# Glimmer label eval ({results['created']})\n\n" + md, encoding="utf-8")
    print(md)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
