#!/usr/bin/env python3
"""Phase 02 - weak labels for F1-F10 from local Muse Glimmer 30B 4-bit (RedHatAI/Muse-Glimmer-30B-INT4 on vLLM).

For every clip in data/manifest.jsonl:
  1. sample K keyframes (uniform, centre of K equal segments; images = 1 frame)
  2. send each frame to Glimmer through the OpenAI-compatible endpoint with the
     versioned prompt in prompts/, temperature 0, JSON mode
  3. parse + validate the answer against the schema (glimmer_schema.py); on bad JSON
     retry with the validation error fed back to the model
  4. cache every valid frame answer under cache/glimmer/<prompt-id>/<model>/ (resumable:
     a rerun skips cached frames)
  5. aggregate frames to clip-level weak labels (multiple-instance view) and write
     02-labeling/outputs/weak_labels_<prompt-version>.jsonl
F11 is not labeled here (it comes from V-JEPA prediction error in 04-head).

Usage (from the repo root, Ollama running with muse-glimmer:30b):
  python training/02-labeling/label_glimmer.py --limit 3            # dry run on 3 clips
  python training/02-labeling/label_glimmer.py --plan-only --limit 3 # no model calls, saves the keyframes
  python training/02-labeling/label_glimmer.py --sources voxel51_safe_unsafe
  python training/02-labeling/label_glimmer.py --backend sglang --concurrency 16 --limit 3   # SGLang in WSL

--backend picks the default endpoint + model name (override with --base-url / --model). The model
name is part of the cache key, so Ollama and SGLang answers never mix. --concurrency N keeps up to N
requests in flight (clips are processed in parallel, frames of one clip in order).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINING_DIR = HERE.parent
sys.path.insert(0, str(HERE))
from glimmer_schema import (  # noqa: E402
    GLIMMER_FACTORS, LabelValidationError, aggregate_clip, hint_agreement, parse_and_validate,
)

TUNIS = timezone(timedelta(hours=1), "Africa/Tunis")
# backend -> (base url, served model name). Env vars override the Ollama defaults as before.
BACKENDS = {
    "ollama": (os.environ.get("GLIMMER_BASE_URL", "http://localhost:11434/v1"),
               os.environ.get("GLIMMER_MODEL", "muse-glimmer:30b")),
    # SGLang in WSL serving RedHatAI/Muse-Glimmer-30B-INT4 with --served-model-name muse-glimmer-30b-int4
    "sglang": (os.environ.get("GLIMMER_SGLANG_URL", "http://localhost:30000/v1"), "muse-glimmer-30b-int4"),
    # vLLM in WSL serving the same INT4 checkpoint with --served-model-name muse-glimmer-30b-int4-vllm
    # (own model name = own cache key, so it never mixes with the SGLang answers)
    "vllm": (os.environ.get("GLIMMER_VLLM_URL", "http://localhost:30001/v1"), "muse-glimmer-30b-int4-vllm"),
}
DEFAULT_BASE_URL, DEFAULT_MODEL = BACKENDS["ollama"]
LAST_RUN: dict = {}  # filled by main(): model calls, wall seconds, concurrency (used by eval_labels.py)
DEFAULT_PROMPT = HERE / "prompts" / "factors_v1.md"
DEFAULT_MANIFEST = TRAINING_DIR / "data" / "manifest.jsonl"
DEFAULT_CACHE = TRAINING_DIR / "cache" / "glimmer"
DEFAULT_OUT_DIR = HERE / "outputs"


def now_tunis() -> str:
    return datetime.now(TUNIS).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{datetime.now(TUNIS).strftime('%H:%M:%S')} Tunis] {msg}", flush=True)


# --------------------------------------------------------------------------- prompt

def load_factor_defs() -> dict[str, dict]:
    import yaml  # PyYAML is a repo dependency
    cfg = yaml.safe_load((TRAINING_DIR / "configs" / "factors.yaml").read_text(encoding="utf-8"))
    return {f["id"]: f for f in cfg["factors"]}


def load_prompt(path: Path, factor_ids=GLIMMER_FACTORS) -> tuple[str, str, str]:
    """Returns (system, user, prompt_id). prompt_id = <version>-<sha8 of rendered text>."""
    raw = path.read_text(encoding="utf-8")
    m = re.search(r"prompt_version:\s*(\S+)", raw)
    version = m.group(1) if m else path.stem
    body = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL).strip()
    if "SYSTEM:" not in body or "USER:" not in body:
        raise SystemExit(f"prompt {path} must contain SYSTEM: and USER: sections")
    system = body.split("SYSTEM:", 1)[1].split("USER:", 1)[0].strip()
    user = body.split("USER:", 1)[1].strip()
    defs = load_factor_defs()
    block = "\n".join(f"- {fid} ({defs[fid]['name']}): {defs[fid]['description']}" for fid in factor_ids)
    system = system.replace("{factor_block}", block).replace("{factor_ids}", ", ".join(factor_ids))
    digest = hashlib.sha256((system + "\n---\n" + user).encode("utf-8")).hexdigest()[:8]
    return system, user, f"{version}-{digest}"


# --------------------------------------------------------------------------- frames

def keyframe_indices(n_frames: int, k: int) -> list[int]:
    if n_frames <= 0:
        return []
    k = max(1, min(k, n_frames))
    return sorted({min(n_frames - 1, int((i + 0.5) * n_frames / k)) for i in range(k)})


def _resize(img, max_side: int):
    import cv2
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img


def sample_frames(path: Path, kind: str, k: int, max_side: int) -> list[tuple[int, float | None, bytes]]:
    """Returns [(frame_index, t_seconds, jpeg_bytes)]. Raises RuntimeError if unreadable."""
    import cv2
    import numpy as np
    if kind == "image":
        buf = np.fromfile(str(path), dtype=np.uint8)  # imdecode handles non-ASCII Windows paths
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cannot read image {path}")
        ok, jpg = cv2.imencode(".jpg", _resize(img, max_side), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return [(0, None, jpg.tobytes())]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        out = []
        for idx in keyframe_indices(n, k):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if not ok or img is None:
                continue
            ok, jpg = cv2.imencode(".jpg", _resize(img, max_side), [cv2.IMWRITE_JPEG_QUALITY, 88])
            out.append((idx, round(idx / fps, 3) if fps else None, jpg.tobytes()))
        if not out:
            raise RuntimeError(f"no decodable frames in {path} (frame count {n})")
        return out
    finally:
        cap.release()


# --------------------------------------------------------------------------- model client

class GlimmerClient:
    def __init__(self, base_url: str, model: str, timeout: float = 300.0, json_mode: bool = True,
                 max_tokens: int = 2048, reasoning_effort: str = "none", backend: str = "ollama"):
        self.base = base_url.rstrip("/")
        self.backend = backend
        self._tls = threading.local()  # per-thread finish_reason / usage (safe with --concurrency)
        self.reasoning_effort = reasoning_effort  # Glimmer is a reasoning model; reasoning tokens count against max_tokens
        self.model = model
        self.timeout = timeout
        self.json_mode = json_mode
        self.max_tokens = max_tokens

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer ollama"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def check(self) -> None:
        try:
            with urllib.request.urlopen(self.base + "/models", timeout=10) as r:
                models = [m.get("id") for m in json.loads(r.read().decode("utf-8")).get("data", [])]
        except (urllib.error.URLError, OSError) as e:
            raise SystemExit(f"ERROR: cannot reach {self.base}/models ({e}). Is the {self.backend} server running?")
        if self.model not in models:
            raise SystemExit(f"ERROR: model '{self.model}' not served by {self.base}. Available: {models}")

    def chat(self, messages: list[dict]) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": 0,
                   "max_tokens": self.max_tokens, "stream": False}
        prefill = ""
        if self.backend == "vllm" and self.reasoning_effort == "none":
            # Glimmer's HF chat template has no "off" value (anything unknown is written literally as
            # "Reasoning strength: <x>." and the model still reasons). To really skip reasoning, end the
            # prompt inside the to=user channel with "{" and let the model continue the JSON.
            prefill = "{"
            payload["messages"] = messages + [{"role": "assistant", "content": prefill}]
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False
        elif self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if prefill:
            pass
        elif self.reasoning_effort != "default":
            if self.backend in ("sglang", "vllm"):
                # Glimmer's chat template reads `reasoning_strength` (default 'high'); SGLang passes it through
                payload["chat_template_kwargs"] = {"reasoning_strength": self.reasoning_effort}
            else:
                payload["reasoning_effort"] = self.reasoning_effort
        data = self._post("/chat/completions", payload)
        try:
            choice = data["choices"][0]
            content = prefill + (choice["message"].get("content") or "")
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"unexpected response: {str(data)[:300]}")
        self._tls.finish_reason = choice.get("finish_reason")
        self._tls.usage = data.get("usage") or {}
        return content

    @property
    def last_finish_reason(self):
        return getattr(self._tls, "finish_reason", None)

    @property
    def last_usage(self):
        return getattr(self._tls, "usage", {})


def label_frame(client, system: str, user: str, jpeg: bytes, retries: int) -> tuple[dict, str, int]:
    """Ask the model about one frame; retry on invalid JSON. Returns (parsed, raw, attempts)."""
    b64 = base64.b64encode(jpeg).decode("ascii")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]},
    ]
    last_err = "no attempt"
    raw = ""
    for attempt in range(1, retries + 2):
        try:
            raw = client.chat(messages)
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as e:
            last_err = f"request failed: {e}"
            time.sleep(min(30, 2 ** attempt))
            continue
        try:
            return parse_and_validate(raw), raw, attempt
        except LabelValidationError as e:
            last_err = str(e)
            if getattr(client, "last_finish_reason", None) == "length":
                last_err += " (reply cut off at max_tokens - raise --max-tokens or lower --reasoning-effort)"
            messages = messages[:2] + [
                {"role": "assistant", "content": raw[:4000]},
                {"role": "user", "content": f"That answer was invalid: {e}. Reply again with ONLY the JSON "
                                            f"object, keys {', '.join(GLIMMER_FACTORS)} under \"factors\", "
                                            "each with state (present|absent|uncertain) and confidence 0-1."},
            ]
    raise LabelValidationError(f"gave up after {retries + 1} attempts: {last_err} | last reply: {raw[:300]!r}")


# --------------------------------------------------------------------------- cache / io

def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120]


def cache_path(cache_root: Path, prompt_id: str, model: str, clip_id: str, frame_idx: int, max_side: int,
               effort: str = "none") -> Path:
    h = hashlib.sha1(clip_id.encode("utf-8")).hexdigest()[:8]  # keeps long ids unique and paths short
    return (cache_root / prompt_id / slug(model) / f"s{max_side}-r{slug(effort)}"
            / f"{slug(clip_id)[-60:]}-{h}" / f"f{frame_idx:06d}.json")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def resolve_clip_path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else TRAINING_DIR / q


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--sources", nargs="*", default=None, help="manifest 'source' values to include")
    ap.add_argument("--split", default=None, help="only this split (train/test/validation)")
    ap.add_argument("--ids", nargs="*", default=None, help="only these clip ids")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (dry run)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="also label byte-identical copies (by default only the first copy is labeled)")
    ap.add_argument("--keyframes", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768, help="resize frames so the long side is at most this")
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    ap.add_argument("--backend", default="ollama", choices=sorted(BACKENDS),
                    help="sets the default --base-url and --model (ollama: GGUF Q4; sglang: INT4 W4A16 in WSL)")
    ap.add_argument("--base-url", default=None, help="default: from --backend")
    ap.add_argument("--model", default=None, help="default: from --backend (part of the cache key)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="requests in flight (clips in parallel). Ollama serves one at a time; SGLang batches")
    ap.add_argument("--retries", type=int, default=2, help="extra attempts after an invalid answer")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--reasoning-effort", default="none", choices=["none", "low", "medium", "high", "default"],
                    help="Glimmer thinking budget; 'none' measured ~12 s/frame vs ~36 s with default thinking "
                         "(Oct 30). 'default' sends nothing.")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="includes reasoning tokens; 900 was too small with thinking on")
    ap.add_argument("--no-json-mode", action="store_true", help="do not send response_format=json_object")
    ap.add_argument("--present-threshold", type=float, default=0.5)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=None, help="default: outputs/weak_labels_<version>.jsonl")
    ap.add_argument("--plan-only", action="store_true", help="sample + save keyframes, no model calls")
    args = ap.parse_args(argv)
    if args.concurrency < 1:
        raise SystemExit("ERROR: --concurrency must be >= 1")
    args.base_url = args.base_url or BACKENDS[args.backend][0]
    args.model = args.model or BACKENDS[args.backend][1]

    system, user, prompt_id = load_prompt(args.prompt)
    version = prompt_id.split("-")[0]
    out_path = args.out or (DEFAULT_OUT_DIR / f"weak_labels_{version}.jsonl")
    err_path = out_path.with_name(out_path.stem + ".errors.jsonl")

    rows = read_jsonl(args.manifest)
    if not rows:
        raise SystemExit(f"ERROR: manifest {args.manifest} missing or empty - run training/01-data/download.py first")
    if args.sources:
        rows = [r for r in rows if r.get("source") in args.sources]
    if args.split:
        rows = [r for r in rows if r.get("split") == args.split]
    if args.ids:
        wanted = set(args.ids)
        rows = [r for r in rows if r["id"] in wanted]
    if not args.keep_duplicates:
        seen, kept = set(), []
        for r in rows:
            key = r.get("sha256") or r["id"]
            if key in seen:
                continue
            seen.add(key)
            kept.append(r)
        rows = kept
    if args.limit is not None:
        rows = rows[: args.limit]
    log(f"{len(rows)} clip(s), prompt {prompt_id}, model {args.model} ({args.backend}, {args.base_url}), "
        f"{args.keyframes} keyframes, max side {args.max_side}, concurrency {args.concurrency}")

    client = None
    if not args.plan_only:
        client = GlimmerClient(args.base_url, args.model, args.timeout, not args.no_json_mode,
                               args.max_tokens, args.reasoning_effort, backend=args.backend)
        client.check()

    existing = {r["clip_id"]: r for r in read_jsonl(out_path)}
    errors: list[dict] = []
    done = called = 0
    call_secs: list[float] = []
    t0 = time.time()

    def work(clip: dict):
        """Runs in a worker thread: decode keyframes, ask the model (cache first). Returns
        (kind, n_frames, answers, errors, call_seconds, log_lines)."""
        path = resolve_clip_path(clip["path"])
        kind = clip.get("kind", "video")
        errs, secs, notes = [], [], []
        try:
            frames = sample_frames(path, kind, 1 if kind == "image" else args.keyframes, args.max_side)
        except RuntimeError as e:
            return kind, 0, [], [{"clip_id": clip["id"], "frame_index": None, "error": str(e), "at": now_tunis()}], [], [str(e)]
        if args.plan_only:
            for idx, _, jpg in frames:
                p = cache_path(args.cache_dir / "_preview", prompt_id, "none", clip["id"], idx, args.max_side).with_suffix(".jpg")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(jpg)
            return kind, len(frames), [{"frame_index": f[0]} for f in frames], [], [], []
        answers = []
        for idx, t, jpg in frames:
            cp = cache_path(args.cache_dir, prompt_id, args.model, clip["id"], idx, args.max_side,
                            args.reasoning_effort)
            if cp.exists():
                answers.append(json.loads(cp.read_text(encoding="utf-8")))
                continue
            t_call = time.time()
            try:
                parsed, raw, attempts = label_frame(client, system, user, jpg, args.retries)
            except LabelValidationError as e:
                errs.append({"clip_id": clip["id"], "frame_index": idx, "error": str(e), "at": now_tunis()})
                notes.append(f"frame {idx}: {e}")
                continue
            secs.append(time.time() - t_call)
            entry = {"clip_id": clip["id"], "frame_index": idx, "t_sec": t, "model": args.model,
                     "backend": args.backend, "prompt_id": prompt_id, "reasoning_effort": args.reasoning_effort,
                     "attempts": attempts, "seconds": round(secs[-1], 2), "concurrency": args.concurrency,
                     "usage": getattr(client, "last_usage", {}), "created_at": now_tunis(),
                     "factors": parsed["factors"], "raw": raw}
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cp)
            answers.append(entry)
        return kind, len(frames), answers, errs, secs, notes

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(work, clip): (n, clip) for n, clip in enumerate(rows, 1)}
        for fut in as_completed(futs):
            n, clip = futs[fut]
            kind, n_frames, answers, errs, secs, notes = fut.result()  # unexpected exceptions propagate (fail loud)
            errors.extend(errs)
            called += len(secs)
            call_secs.extend(secs)
            for note in notes:
                log(f"  [{n}/{len(rows)}] {clip['id']}: {note}")
            if args.plan_only:
                if n_frames:
                    log(f"  [{n}/{len(rows)}] {clip['id']}: frames {[a['frame_index'] for a in answers]} saved")
                continue
            if n_frames == 0:
                continue
            if len(answers) < n_frames:
                log(f"  [{n}/{len(rows)}] {clip['id']}: {n_frames - len(answers)} frame(s) failed - clip skipped, rerun to retry")
                continue
            answers.sort(key=lambda a: a["frame_index"])
            factors = aggregate_clip(answers, GLIMMER_FACTORS, args.present_threshold)
            existing[clip["id"]] = {
                "clip_id": clip["id"], "source": clip.get("source"), "path": clip["path"],
                "split": clip.get("split"), "kind": kind, "license": clip.get("license"),
                "source_label": clip.get("source_label"), "factor_hints": clip.get("factor_hints", {}),
                "hint_agreement": hint_agreement(factors, clip.get("factor_hints", {})),
                "duplicate_of": clip.get("duplicate_of", []),
                "model": args.model, "backend": args.backend, "prompt_id": prompt_id,
                "reasoning_effort": args.reasoning_effort, "keyframes": [a["frame_index"] for a in answers],
                "present_threshold": args.present_threshold, "factors": factors, "labeled_at": now_tunis(),
            }
            done += 1
            present = [f for f, v in factors.items() if v["label"] == "present"]
            log(f"  [{n}/{len(rows)}] {clip['id']}: present={present or '-'}")
            if done % 10 == 0:
                write_jsonl(out_path, sorted(existing.values(), key=lambda r: r["clip_id"]))

    if args.plan_only:
        log(f"plan only - keyframes in {args.cache_dir / '_preview'}")
        return 1 if errors else 0
    write_jsonl(out_path, sorted(existing.values(), key=lambda r: r["clip_id"]))
    if errors:
        with open(err_path, "a", encoding="utf-8", newline="\n") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    LAST_RUN.clear()
    LAST_RUN.update({"calls": called, "wall_s": round(dt, 1), "concurrency": args.concurrency,
                     "backend": args.backend, "model": args.model})
    log(f"labeled {done}/{len(rows)} clip(s), {called} model call(s) in {dt:.0f}s -> {out_path}")
    if called and dt > 0:
        log(f"throughput: {called / dt * 60:.1f} calls/min at concurrency {args.concurrency}")
    if call_secs:
        srt = sorted(call_secs)
        log(f"seconds per frame (incl. retries): mean {sum(srt) / len(srt):.1f}, median {srt[len(srt) // 2]:.1f}, "
            f"max {srt[-1]:.1f}")
    if errors:
        log(f"FAILED: {len(errors)} error(s), see {err_path}. Re-run to retry (cached frames are skipped).")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted - re-run the same command to resume (cache keeps finished frames)", file=sys.stderr)
        sys.exit(130)
