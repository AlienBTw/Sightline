"""Pexels-only F2/F5 metrics that show a Voxel51 domain shortcut.

Voxel51 factory clips are almost all F2/F5 present, so a head can rank well on a
pooled set by recognising that domain. This scores the same clip logit and
window evidence as ``train_head.py`` eval, then reports Pexels on its own and
the gap between sources that share a present label.

Clip score: sigmoid of the factor logit pooled over every patch of every window.
Window score: max sigmoid(patch evidence) in that window (``collect_ppe_scores``).
Pexels held-out is test + demo_eval. The threshold is best F1 on Pexels val.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import train_head as th

HEAD = Path(__file__).resolve().parent
REPO = HEAD.parents[1]
WORK = Path(os.environ.get("SIGHTLINE_WORK_DIR", REPO / "work"))
FACTORS = ("F2", "F5")
PEXELS_SPLITS = ("train", "val", "test", "demo_eval")
HELDOUT_SPLITS = ("test", "demo_eval")
VOXEL_SPLITS = ("val", "test")
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
CLIP_SCORE = "sigmoid of the factor logit pooled over every patch of every window"
WINDOW_SCORE = "max sigmoid(patch evidence) inside the window"
DEFAULT_JSON = str(WORK / "ppe_eval" / "shortcut_eval.json")
DEFAULT_MD = HEAD / "results" / "PPE_SHORTCUT.md"


def discover_checkpoints(ckpt_dir: Path) -> list[Path]:
    """h3_wide.pt, then every ppe_*.pt. Other heads are not part of this eval."""
    found = []
    wide = ckpt_dir / "h3_wide.pt"
    if wide.is_file():
        found.append(wide)
    found.extend(sorted(p for p in ckpt_dir.glob("ppe_*.pt") if p.is_file()))
    uniq, seen = [], set()
    for path in found:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(path)
    return uniq


def checkpoints_from_args(text: str, ckpt_dir: Path) -> list[Path]:
    text = (text or "").strip()
    if not text:
        found = discover_checkpoints(ckpt_dir)
        if not found:
            raise SystemExit(
                f"no checkpoints in {ckpt_dir}: expected h3_wide.pt or ppe_*.pt, or pass --checkpoints"
            )
        return found
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(th.resolve_ckpt(part, "checkpoint"))
    if not out:
        raise SystemExit("--checkpoints is empty")
    return out


def canon_split(row) -> str:
    split = str(row.get("split") or "")
    return "val" if split == "validation" else split


def is_pexels_row(row) -> bool:
    if row.get("ppe_source") == "pexels" or row.get("source") in ("ppe_video", "pexels"):
        return True
    return str(row.get("id") or "").startswith("pexels/")


def ranking_metrics(y, p) -> dict:
    """AUROC and AP. One missing class is n/a; counts are still reported."""
    if len(y) != len(p):
        raise SystemExit(f"score length {len(p)} != label length {len(y)}")
    labels = [1 if float(v) > 0.5 else 0 for v in y]
    pos = int(sum(labels))
    out = {"n": len(labels), "pos": pos, "neg": len(labels) - pos, "auroc": "n/a", "ap": "n/a"}
    if pos == 0 or out["neg"] == 0:
        return out
    from sklearn.metrics import average_precision_score, roc_auc_score
    out["auroc"] = float(roc_auc_score(labels, p))
    out["ap"] = float(average_precision_score(labels, p))
    return out


def bootstrap_auroc(y, p, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Percentile 95% interval of AUROC. Resamples with one class are dropped."""
    point = ranking_metrics(y, p)
    if point["auroc"] == "n/a":
        return {"auroc_ci95": "n/a", "auroc_boot_used": 0}
    import numpy as np
    from sklearn.metrics import roc_auc_score
    labels = np.asarray([1 if float(v) > 0.5 else 0 for v in y])
    scores = np.asarray(p, dtype=np.float64)
    rng = np.random.default_rng(seed)
    kept = []
    n = int(labels.shape[0])
    for _ in range(int(resamples)):
        idx = rng.integers(0, n, n)
        if labels[idx].min() == labels[idx].max():
            continue
        kept.append(float(roc_auc_score(labels[idx], scores[idx])))
    if not kept:
        return {"auroc_ci95": "n/a", "auroc_boot_used": 0}
    lo, hi = np.quantile(np.asarray(kept, dtype=np.float64), [0.025, 0.975])
    return {"auroc_ci95": [float(lo), float(hi)], "auroc_boot_used": len(kept)}


def score_dist(scores) -> dict:
    if not scores:
        return {"n": 0, "mean": "n/a", "median": "n/a"}
    import numpy as np
    arr = np.asarray(list(scores), dtype=np.float64)
    return {"n": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr))}


def fraction_above(y, p, threshold, positive: bool):
    want = 1 if positive else 0
    chosen = [float(score) for lab, score in zip(y, p) if (1 if float(lab) > 0.5 else 0) == want]
    if not chosen:
        return "n/a"
    return float(sum(1 for score in chosen if score >= threshold) / len(chosen))


def val_threshold(y, p):
    """Best-F1 cut on Pexels val, the same rule as train_head eval. Else 0.5."""
    metrics = ranking_metrics(y, p)
    if metrics["pos"] == 0 or metrics["neg"] == 0:
        return 0.5, False
    return float(th._choose_threshold(y, p, 0.0)), True


def take(records, splits):
    y, p = [], []
    for rec in records:
        if rec["split"] in splits:
            y.append(rec["y"])
            p.append(rec["p"])
    return y, p


def domain_auroc(voxel_scores, pexels_scores) -> dict:
    """Voxel51 is the positive class. Above 0.5 means those clips score higher."""
    y = [1] * len(voxel_scores) + [0] * len(pexels_scores)
    p = list(voxel_scores) + list(pexels_scores)
    metrics = ranking_metrics(y, p)
    return {
        "domain_auroc": metrics["auroc"],
        "domain_n_voxel51": len(voxel_scores),
        "domain_n_pexels": len(pexels_scores),
    }


def factor_report(pexels, voxel, window_y, window_p, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED) -> dict:
    """One factor. ``pexels`` / ``voxel`` are labeled clip records ``{split, y, p}``."""
    splits = {}
    for name, which in (
        ("test", ("test",)),
        ("demo_eval", ("demo_eval",)),
        ("held_out", HELDOUT_SPLITS),
        ("val", ("val",)),
        ("train", ("train",)),
    ):
        y, p = take(pexels, which)
        block = ranking_metrics(y, p)
        if name == "held_out":
            block.update(bootstrap_auroc(y, p, resamples=resamples, seed=seed))
        splits[name] = block
    y_val, p_val = take(pexels, ("val",))
    threshold, fitted = val_threshold(y_val, p_val)
    y_hold, p_hold = take(pexels, HELDOUT_SPLITS)
    voxel_pos = [rec["p"] for rec in voxel if rec["split"] in VOXEL_SPLITS and rec["y"] == 1]
    pexels_pos = [rec["p"] for rec in pexels if rec["split"] in HELDOUT_SPLITS and rec["y"] == 1]
    pexels_neg = [rec["p"] for rec in pexels if rec["split"] in HELDOUT_SPLITS and rec["y"] == 0]
    shortcut = domain_auroc(voxel_pos, pexels_pos)
    shortcut.update({
        "val_threshold": threshold,
        "val_threshold_fit": fitted,
        "heldout_pos_above": fraction_above(y_hold, p_hold, threshold, True),
        "heldout_neg_above": fraction_above(y_hold, p_hold, threshold, False),
    })
    return {
        "splits": splits,
        "distributions": {
            "pexels_pos": score_dist(pexels_pos),
            "pexels_neg": score_dist(pexels_neg),
            "voxel51_pos": score_dist(voxel_pos),
        },
        "shortcut": shortcut,
        "window_held_out": ranking_metrics(window_y, window_p),
    }


def labeled_windows(rows, evid, factor, splits=None):
    """Window labels and max-evidence scores, chunked like ``collect_ppe_scores``.

    ``splits`` drops windows after the chunk is read, so a skipped row does not
    shift the evidence of a later row.
    """
    index = th.FACTORS.index(factor)
    if index not in evid:
        raise SystemExit(f"no window evidence for {factor}")
    maps = evid[index]
    y, p = [], []
    bi = mi = 0
    while bi < len(rows):
        chunk = rows[bi:bi + 32]
        if mi >= len(maps):
            raise SystemExit("window evidence chunks do not match the row list")
        pm = maps[mi]
        mi += 1
        for j, row in enumerate(chunk):
            if splits is not None and canon_split(row) not in splits:
                continue
            sig = pm[j]
            n = int(row["n"])
            p_count = int(row.get("n_patches") or (sig.shape[0] // max(n, 1)))
            wins = row.get("ppe_windows") or [None] * n
            for w in range(n):
                spec = wins[w] if w < len(wins) else None
                sl = sig[w * p_count:(w + 1) * p_count]
                if spec is None or spec.get(factor) is None or sl.numel() == 0:
                    continue
                y.append(float(spec[factor]))
                p.append(float(sl.max()))
        bi += 32
    if mi != len(maps):
        raise SystemExit("window evidence chunks do not match the row list")
    return y, p


def select_rows(meta):
    pexels, voxel = [], []
    for row in meta:
        if is_pexels_row(row):
            if canon_split(row) in PEXELS_SPLITS:
                pexels.append(row)
            continue
        if row.get("source") == "voxel51" and row.get("set") in VOXEL_SPLITS:
            voxel.append(row)
    return pexels, voxel


def _count(rows, split_of):
    counts = {}
    for row in rows:
        key = split_of(row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def missing_pexels(labels, meta) -> list[str]:
    have = {row["id"] for row in meta}
    missing = []
    for clip_id, lab in (labels or {}).items():
        if th.ppe_metric_source({"id": clip_id}, lab) != "pexels":
            continue
        split = canon_split(lab)
        if split not in PEXELS_SPLITS:
            continue
        if clip_id not in have:
            missing.append(clip_id)
    return missing


def clip_records(rows, probs, factor, split_of):
    index = th.FACTORS.index(factor)
    out = []
    for k, row in enumerate(rows):
        if not row["m"][index]:
            continue
        out.append({
            "split": split_of(row),
            "y": 1 if float(row["y"][index]) > 0.5 else 0,
            "p": float(probs[k, index]),
        })
    return out


def score_rows(rows, model, banks, dev):
    """Clip probabilities and per-factor evidence, in the eval batching of 32."""
    import numpy as np
    import torch
    evid = {0: [], 1: [], 4: []}
    if not rows:
        return np.zeros((0, len(th.FACTORS)), np.float32), evid
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            chunk = rows[start:start + 32]
            xs = []
            for row in chunk:
                bank = banks[row.get("bank", 0)]
                x = np.asarray(bank[row["off"]:row["off"] + row["n"]])
                if x.ndim != 3:
                    raise SystemExit(
                        f"{row.get('id')} tokens have shape {tuple(x.shape)}, expected (windows, patches, dim)"
                    )
                xs.append(x.reshape(-1, x.shape[-1]))
            width = max(arr.shape[0] for arr in xs)
            dim = xs[0].shape[-1]
            packed = np.zeros((len(xs), width, dim), np.float16)
            valid = np.zeros((len(xs), width), bool)
            for j, arr in enumerate(xs):
                packed[j, :arr.shape[0]] = arr
                valid[j, :arr.shape[0]] = True
            xt = torch.from_numpy(packed).to(dev).float()
            vt = torch.from_numpy(valid).to(dev)
            z, s, _w = model(xt, vt)
            out.append(torch.sigmoid(z).cpu())
            evid[0].append(torch.sigmoid(s[..., 0]).cpu())
            evid[1].append(torch.sigmoid(s[..., 1]).cpu())
            evid[4].append(torch.sigmoid(s[..., 4]).cpu())
    return torch.cat(out).numpy(), evid


def load_checkpoint(path: Path, dev):
    import torch
    blob = torch.load(path, map_location=dev, weights_only=False)
    cfg = blob.get("config") or {}
    if "hidden" not in cfg:
        raise SystemExit(f"checkpoint {path} has no config.hidden")
    hidden = int(cfg["hidden"])
    drop = float(cfg.get("dropout", 0.0))
    model = th.build_model(hidden=hidden, drop=drop).to(dev)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, hidden, drop


def prepare(cache, labels, ppe_labels, ppe_cache, gpu: int):
    cache = Path(os.path.expanduser(cache))
    ppe_cache = Path(os.path.expanduser(ppe_cache))
    for name, path in (("feature cache", cache), ("ppe cache", ppe_cache)):
        if not (path / "tokens.npy").is_file() or not (path / "meta.jsonl").is_file():
            raise SystemExit(f"{name} not found: {path} (need tokens.npy and meta.jsonl)")
    label_path = Path(os.path.expanduser(labels))
    if not label_path.is_file() and not (th.TRAINING / labels).is_file():
        raise SystemExit(f"weak labels not found: {labels}")
    ns = th.build_arg_parser().parse_args([
        "eval", "--checkpoint", "shortcut-eval", "--name", "shortcut",
        "--cache", str(cache), "--labels", labels,
        "--ppe-labels", ppe_labels, "--ppe-cache", str(ppe_cache),
        "--gpu", str(gpu),
    ])
    prep = th.load_meta(ns)
    stats = th.apply_voxel_ppe_policy(
        prep["meta"], prep["ppe_labels"], True, True,
        prep["gh"], prep["gw"], ns.f2_min_frac, ns.f2_hatneg_weight,
    )
    pexels, voxel = select_rows(prep["meta"])
    if not pexels:
        raise SystemExit("no Pexels clips in the loaded caches")
    if not voxel:
        raise SystemExit("no Voxel51 val/test clips in the feature cache")
    labeled = sum(1 for row in voxel if row.get("ppe_labeled"))
    if labeled == 0:
        raise SystemExit("no Voxel51 val/test clip matched the PPE label file")
    missing = missing_pexels(prep["ppe_labels"], prep["meta"])
    if missing:
        sample = ", ".join(missing[:5])
        raise SystemExit(f"{len(missing)} Pexels label rows have no features ({sample})")
    print(
        f"clips pexels={_count(pexels, canon_split)} voxel51={_count(voxel, lambda r: r.get('set'))} "
        f"ppe_override={stats} val_test_labeled={labeled}",
        flush=True,
    )
    data = {
        "pexels_clips": _count(pexels, canon_split),
        "voxel51_clips": _count(voxel, lambda row: row.get("set")),
        "voxel_ppe": {
            "overridden": stats["overridden"],
            "cleared": stats["cleared"],
            "val_test_labeled": labeled,
        },
    }
    return prep, pexels, voxel, data


def evaluate_checkpoint(path: Path, pexels, voxel, banks, dev) -> dict:
    import time
    import torch
    t0 = time.time()
    model, hidden, drop = load_checkpoint(path, dev)
    print(f"score {path.name} hidden={hidden} dropout={drop}", flush=True)
    pex_prob, pex_evid = score_rows(pexels, model, banks, dev)
    vox_prob, _vox_evid = score_rows(voxel, model, banks, dev)
    factors = {}
    for factor in FACTORS:
        window_y, window_p = labeled_windows(pexels, pex_evid, factor, HELDOUT_SPLITS)
        factors[factor] = factor_report(
            clip_records(pexels, pex_prob, factor, canon_split),
            clip_records(voxel, vox_prob, factor, lambda row: row.get("set")),
            window_y, window_p,
        )
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "name": path.stem,
        "path": str(path),
        "hidden": hidden,
        "dropout": drop,
        "seconds": round(time.time() - t0, 1),
        "factors": factors,
    }


def pin_visible_gpu(gpu: int) -> int:
    """Use only the requested physical GPU when the mask is not already set.

    ``--gpu 1`` with a clear environment hides GPU 0 and scores on cuda:0 of
    that mask, so a training job on GPU 0 is not initialised by this process.
    """
    if gpu < 0:
        raise SystemExit(f"--gpu must be >= 0, got {gpu}")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return 0
    return gpu


def json_safe(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def fmt_num(value) -> str:
    if value == "n/a" or value is None:
        return "n/a"
    return f"{float(value):.3f}"


def fmt_ci(value) -> str:
    if value == "n/a" or value is None:
        return "n/a"
    return f"[{float(value[0]):.3f}, {float(value[1]):.3f}]"


def _counts_phrase(counts: dict) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key} {counts[key]}" for key in sorted(counts))


def render_markdown(report: dict) -> str:
    data = report.get("data") or {}
    voxel_ppe = data.get("voxel_ppe") or {}
    lines = [
        "# PPE shortcut eval",
        "",
        f"Clip score: {report.get('clip_score', CLIP_SCORE)}.",
        f"Window score: {report.get('window_score', WINDOW_SCORE)}.",
        "Pexels held-out is test + demo_eval. Voxel51 scores are val and test.",
        "F2 and F5 labels are the Muse Spark clip verdicts. A null verdict is excluded.",
        "A split with only one class is n/a.",
        "",
        "Domain AUROC separates Voxel51 (positive class) from Pexels among held-out clips",
        "whose factor label is present. Above 0.5 means the Voxel51 clips score higher",
        "than Pexels clips with the same present label. The threshold is best F1 on",
        "Pexels validation. pos above and neg above are the fractions of Pexels held-out",
        f"positives and negatives at or above that cut. The interval is a {BOOTSTRAP_RESAMPLES}-draw",
        f"bootstrap of held-out AUROC, seed {BOOTSTRAP_SEED}, 2.5 and 97.5 percentiles.",
        "Resamples that contain only one class are dropped.",
        "",
        f"Pexels clips: {_counts_phrase(data.get('pexels_clips') or {})}.",
        f"Voxel51 clips: {_counts_phrase(data.get('voxel51_clips') or {})}.",
        (
            f"PPE labels replaced F2/F5 on {voxel_ppe.get('overridden', 'n/a')} Voxel51 clips; "
            f"{voxel_ppe.get('cleared', 'n/a')} clips had no PPE row and stay unlabeled; "
            f"{voxel_ppe.get('val_test_labeled', 'n/a')} val/test clips carry a PPE row."
        ),
        "",
        "## Summary",
        "",
        "| checkpoint | factor | held-out pos/neg | AUROC | AP | AUROC 95% CI | domain AUROC | pos above | neg above | window AUROC |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ckpt in report.get("checkpoints") or []:
        for factor in FACTORS:
            block = ckpt["factors"][factor]
            held = block["splits"]["held_out"]
            cut = block["shortcut"]
            window = block["window_held_out"]
            lines.append(
                "| {name} | {factor} | {pos}/{neg} | {auroc} | {ap} | {ci} | {domain} | {pos_a} | {neg_a} | {window} |".format(
                    name=ckpt["name"],
                    factor=factor,
                    pos=held["pos"],
                    neg=held["neg"],
                    auroc=fmt_num(held["auroc"]),
                    ap=fmt_num(held["ap"]),
                    ci=fmt_ci(held.get("auroc_ci95")),
                    domain=fmt_num(cut["domain_auroc"]),
                    pos_a=fmt_num(cut["heldout_pos_above"]),
                    neg_a=fmt_num(cut["heldout_neg_above"]),
                    window=fmt_num(window["auroc"]),
                )
            )
    for ckpt in report.get("checkpoints") or []:
        lines.extend([
            "",
            f"## {ckpt['name']}",
            "",
            f"Checkpoint `{ckpt.get('path', '')}`, hidden {ckpt.get('hidden', '')}, dropout {ckpt.get('dropout', '')}.",
            "",
            "### Pexels clip metrics",
            "",
            "| factor | split | pos | neg | AUROC | AP | AUROC 95% CI |",
            "|---|---|---|---|---|---|---|",
        ])
        for factor in FACTORS:
            for split in ("test", "demo_eval", "held_out", "val", "train"):
                block = ckpt["factors"][factor]["splits"][split]
                ci = fmt_ci(block["auroc_ci95"]) if split == "held_out" else ""
                lines.append(
                    f"| {factor} | {split} | {block['pos']} | {block['neg']} | "
                    f"{fmt_num(block['auroc'])} | {fmt_num(block['ap'])} | {ci} |"
                )
        lines.extend([
            "",
            "### Held-out score distributions",
            "",
            "| factor | group | n | mean | median |",
            "|---|---|---|---|---|",
        ])
        groups = (
            ("pexels_pos", "Pexels present (test+demo_eval)"),
            ("pexels_neg", "Pexels absent (test+demo_eval)"),
            ("voxel51_pos", "Voxel51 present (val+test)"),
        )
        for factor in FACTORS:
            dist = ckpt["factors"][factor]["distributions"]
            for key, label in groups:
                row = dist[key]
                lines.append(
                    f"| {factor} | {label} | {row['n']} | {fmt_num(row['mean'])} | {fmt_num(row['median'])} |"
                )
        lines.extend([
            "",
            "### Shortcut",
            "",
            "| factor | domain AUROC | n Voxel51 | n Pexels | val threshold | pos above | neg above | window AUROC | window pos/neg |",
            "|---|---|---|---|---|---|---|---|---|",
        ])
        for factor in FACTORS:
            block = ckpt["factors"][factor]
            cut = block["shortcut"]
            window = block["window_held_out"]
            thr = fmt_num(cut["val_threshold"])
            if not cut.get("val_threshold_fit"):
                thr += " (default, val has one class)"
            lines.append(
                f"| {factor} | {fmt_num(cut['domain_auroc'])} | {cut['domain_n_voxel51']} | "
                f"{cut['domain_n_pexels']} | {thr} | "
                f"{fmt_num(cut['heldout_pos_above'])} | {fmt_num(cut['heldout_neg_above'])} | "
                f"{fmt_num(window['auroc'])} | {window['pos']}/{window['neg']} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_outputs(report, json_path, md_path):
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(report)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Pexels-only F2/F5 shortcut metrics for a saved head.")
    parser.add_argument("--checkpoints", default="", help="comma-separated checkpoints; default is h3_wide.pt plus checkpoints/ppe_*.pt")
    parser.add_argument("--ppe-labels", default="data/ppe_video_labels.jsonl")
    parser.add_argument("--ppe-cache", default=str(WORK / "cache" / "ppe_video_p2"))
    parser.add_argument("--cache", default=str(WORK / "cache" / "vjepa2_1_vitl_384_p2"), help="main feature cache used by train_head eval")
    parser.add_argument("--labels", default="02-labeling/outputs/weak_labels_v2_vllm.jsonl", help="weak labels used to build Voxel51 splits")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--out-json", default=DEFAULT_JSON)
    parser.add_argument("--out-md", default=str(DEFAULT_MD))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    device_index = pin_visible_gpu(args.gpu)
    ckpts = checkpoints_from_args(args.checkpoints, HEAD / "checkpoints")
    print("checkpoints " + ", ".join(path.name for path in ckpts), flush=True)
    _prep, pexels, voxel, data = prepare(args.cache, args.labels, args.ppe_labels, args.ppe_cache, args.gpu)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; refusing to score checkpoints on CPU")
    visible = torch.cuda.device_count()
    if device_index >= visible:
        raise SystemExit(
            f"cuda:{device_index} is not available (visible devices: {visible}, --gpu {args.gpu}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})"
        )
    dev = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(dev)
    print(
        f"device physical GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} as cuda:{device_index} "
        f"({torch.cuda.get_device_name(dev)})",
        flush=True,
    )
    banks = _prep["banks"]
    scored = [evaluate_checkpoint(path, pexels, voxel, banks, dev) for path in ckpts]
    report = {
        "clip_score": CLIP_SCORE,
        "window_score": WINDOW_SCORE,
        "threshold_rule": "best F1 on Pexels validation clip scores",
        "domain_positive_class": "voxel51",
        "held_out": "pexels test+demo_eval; voxel51 val+test",
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "level": 0.95},
        "data": data,
        "checkpoints": scored,
    }
    write_outputs(report, args.out_json, args.out_md)
    print(render_markdown(json_safe(report)), flush=True)


if __name__ == "__main__":
    main()
