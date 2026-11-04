"""Muse Spark check of F2 and F5 on one clip (three frames, one call).

The request uses the same chat-completions path, temperature 0, and default
``max_tokens`` 4000 as ``sightline_pipeline.verify_gate.VerifyClient``. The
model is ``--model``, then ``MUSE_SPARK_MODEL``, then Muse Spark 1.3
Contributor (``muse-spark-1.3-contributor``). The id actually used is printed.
``--dry-run`` prints the prompt and frame paths and does not open the network,
the .env file, or the budget.

A real call refuses when this run's ``--max-calls`` (default 0) is exhausted or
when the persistent counter ``<SIGHTLINE_WORK_DIR>/vidlabel/muse_budget.json`` has
already reached ``--budget-total`` (default 400). The counter is shared across
runs and is never raised past that total. Every real call, including a parse
failure and a ``--redo-parse-errors`` retry, is appended to ``muse_calls.jsonl``
(with ``finish_reason`` and a 300-character answer snippet, never the API key)
and stored as uncertain when the JSON is unusable.

  python tools/labeling/muse_verify.py --dry-run --max-calls 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("SIGHTLINE_WORK_DIR", REPO / "work"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training" / "01-data"))
sys.path.insert(0, str(REPO / "tools" / "labeling"))

import ppe_rules  # noqa: E402
import ppe_video as pv  # noqa: E402

try:
    from sightline_pipeline.escalation import DEFAULT_MUSE_MODEL
except Exception:
    DEFAULT_MUSE_MODEL = "muse-spark-1.3-contributor"

SNIPPET_CHARS = 300
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)

DEFAULT_MANIFEST = REPO / "training" / "data" / "ppe_video_manifest.jsonl"
DEFAULT_VOXEL = REPO / "training" / "data" / "manifest.jsonl"
DEFAULT_WEAK = REPO / "training" / "02-labeling" / "outputs" / "weak_labels_v2_vllm.jsonl"
_VID = WORK / "vidlabel"
DEFAULT_INDEX = _VID / "frames" / "index.jsonl"
DEFAULT_SAM = _VID / "sam_windows.jsonl"
DEFAULT_BUDGET = _VID / "muse_budget.json"
DEFAULT_LOG = _VID / "muse_calls.jsonl"
DEFAULT_OUT = _VID / "muse_verdicts.jsonl"
BUDGET_TOTAL = 400

PROMPT = """You check a short workshop video for two hazards. Three frames from the same clip are attached, in time order.
For any visible worker:
- F2 is "present" if at least one person has no hard hat and no safety helmet. A cap, a hood, a hairnet, or a bare head is not head protection. "absent" only if every visible person is wearing a hard hat or helmet. "uncertain" if no person's head is visible or you cannot tell.
- F5 is "present" if at least one person is not wearing a high-visibility safety vest (fluorescent yellow, orange, or lime on the torso, with or without reflective strips). "absent" only if every visible person is wearing one. "uncertain" if no person's torso is visible or you cannot tell.
If no person is visible, both are "uncertain".
Reply with JSON only, no markdown, exactly:
{"F2":"present|absent|uncertain","F5":"present|absent|uncertain","reason":"<one short sentence>"}"""


def resolve_model(explicit: str | None) -> str:
    """``--model``, then ``MUSE_SPARK_MODEL``, then Muse Spark 1.3 Contributor."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    env = os.environ.get("MUSE_SPARK_MODEL")
    if env is not None and str(env).strip():
        return str(env).strip()
    return DEFAULT_MUSE_MODEL


def public_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """First ``limit`` characters of model text. Environment keys are never kept."""
    snippet = text or ""
    for name in ("MODEL_API_KEY", "META_API_KEY"):
        secret = os.environ.get(name) or ""
        if len(secret) >= 8 and secret in snippet:
            snippet = snippet.replace(secret, "[redacted]")
    return snippet[:limit]


def _message_text_local(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        return "".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content", "reasoning", "reasoning_content"):
            val = content.get(key)
            if isinstance(val, str):
                return val
        return ""
    return "" if content is None else str(content)


def _assistant_text_local(message) -> str:
    if not isinstance(message, dict):
        return _message_text_local(message).strip()
    content = _message_text_local(message.get("content")).strip()
    if content:
        return content
    for key in ("reasoning", "reasoning_content"):
        alt = _message_text_local(message.get(key)).strip()
        if alt:
            return alt
    return ""


def _reasoning_text_local(message, choice) -> str:
    parts = []
    sources = []
    if isinstance(message, dict):
        sources.append(message)
    if isinstance(choice, dict):
        sources.append(choice)
    for source in sources:
        for key in ("reasoning", "reasoning_content"):
            text = _message_text_local(source.get(key)).strip()
            if text and text not in parts:
                parts.append(text)
    return "\n".join(parts)


def _text_helpers():
    """verify_gate parsers when that module imports; the same logic otherwise."""
    try:
        from sightline_pipeline.verify_gate import _assistant_text, _message_text, _reasoning_text
        return _message_text, _assistant_text, _reasoning_text
    except Exception:
        return _message_text_local, _assistant_text_local, _reasoning_text_local


def answer_text(message, choice=None) -> str:
    """Message content, or reasoning / reasoning_content when it is empty or has no JSON."""
    message_text, assistant_text, reasoning_text = _text_helpers()
    if not isinstance(message, dict):
        return assistant_text(message).strip()
    content = message_text(message.get("content")).strip()
    if content and _JSON_OBJECT.search(content):
        return content
    choice = choice if isinstance(choice, dict) else None
    if not content:
        fallback = assistant_text(message).strip()
        extra = reasoning_text(message, choice).strip()
        return extra or fallback
    reasoning = reasoning_text(message, choice).strip()
    if reasoning:
        return reasoning
    return content


def _finish_and_text(payload) -> tuple:
    choice = None
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
    finish = choice.get("finish_reason") if choice else None
    message = choice.get("message") if choice else None
    if not isinstance(message, dict):
        message = {"content": "" if message is None else message}
    return finish, answer_text(message, choice)


def load_model_env() -> None:
    """Same .env load as sightline_pipeline.run. The file and the key are never printed."""
    if os.environ.get("SIGHTLINE_NO_DOTENV"):
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO / ".env", override=False)


def load_budget(path: Path) -> int:
    if not path.is_file():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data.get("total") or 0)


def save_budget(path: Path, total: int, budget_total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"total": int(total), "budget_total": int(budget_total)}) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def reserve(path: Path, budget_total: int, used_this_run: int, max_calls: int) -> tuple[bool, int, str]:
    """Increment the persistent counter only when the call is allowed.

    A file lock keeps two processes from both passing the cap. Dry-runs must
    not call this.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    try:
        import fcntl
    except ImportError:
        fcntl = None
    with lock_path.open("a+") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        total = load_budget(path)
        try:
            updated = ppe_rules.apply_claim(total, budget_total, used_this_run, max_calls)
        except ppe_rules.BudgetRefusal as refusal:
            return False, total, refusal.reason
        save_budget(path, updated, budget_total)
        return True, updated, "ok"


def group_index(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(row["id"], []).append(row)
    for windows in grouped.values():
        windows.sort(key=lambda row: int(row["window"]))
    return grouped


def sam_by_clip(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: {int(w["w"]): w for w in row.get("windows") or []} for row in rows}


def _source_rows(manifest: Path, source: str, voxel_manifest: Path, weak_labels: Path) -> list[dict]:
    if source == pv.SOURCE:
        return [row for row in pv.read_jsonl(manifest) if row.get("source") == pv.SOURCE]
    if source in (pv.VOXEL_SOURCE, "voxel51"):
        weak = pv.read_jsonl(weak_labels) if weak_labels.is_file() else None
        return pv.voxel_rows_for_labeling(pv.read_jsonl(voxel_manifest), weak)
    raise SystemExit(f"unknown --source {source}")


def _order_key(order: str):
    if order == "priority":
        return ppe_rules.priority_key
    if order == "id":
        return lambda row: str(row.get("id"))
    raise SystemExit(f"unknown --order {order}")


def parse_source_splits(spec: str | None) -> set[tuple[str, str]] | None:
    """``voxel51_safe_unsafe:test`` or a comma-separated list of ``source:split``."""
    text = (spec or "").strip()
    if not text:
        return None
    pairs: set[tuple[str, str]] = set()
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if item.count(":") != 1:
            raise SystemExit(f"--limit-source-split expects source:split, got {item}")
        source, split = (piece.strip() for piece in item.split(":", 1))
        if not source or not split:
            raise SystemExit(f"--limit-source-split expects source:split, got {item}")
        pairs.add((source, split))
    return pairs or None


def select_clips(manifest: Path, source: str, ids: set[str] | None, voxel_manifest: Path,
                 weak_labels: Path, source_splits: set[tuple[str, str]] | None = None,
                 order: str = "priority") -> list[dict]:
    if source == "all":
        rows = _source_rows(manifest, pv.SOURCE, voxel_manifest, weak_labels)
        rows += _source_rows(manifest, pv.VOXEL_SOURCE, voxel_manifest, weak_labels)
    else:
        rows = _source_rows(manifest, source, voxel_manifest, weak_labels)
    if source_splits:
        rows = [row for row in rows if (row.get("source"), row.get("split")) in source_splits]
    if ids is not None:
        rows = [row for row in rows if row["id"] in ids]
    return sorted(rows, key=_order_key(order))


def clip_is_pending(previous: dict | None, redo_parse_errors: bool) -> bool:
    """Existing verdicts are skipped. ``parse_error`` is pending again only with the flag."""
    if not previous:
        return True
    return bool(redo_parse_errors and previous.get("reason") == "parse_error")


def frames_for(clip: dict, index_by_id: dict, sam: dict) -> list[str]:
    windows = index_by_id.get(clip["id"]) or []
    return ppe_rules.choose_muse_paths(windows, sam.get(clip["id"]) or {})


def ask_muse(frame_paths: list[str], model: str, timeout: float, max_tokens: int = 4000) -> dict:
    """One multi-frame chat completion. Returns status, verdict, latency, usage.

    Temperature is 0 and ``max_tokens`` defaults to 4000, matching
    ``VerifyClient``. The JSON is taken from message content, then from
    reasoning / reasoning_content when content is empty or has no JSON object.
    """
    import cv2
    load_model_env()
    from sightline_pipeline.escalation import api_base, api_key, jpeg_data_url
    key = api_key()
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set")
    content = [{"type": "text", "text": PROMPT}]
    for path in frame_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read frame {path}")
        content.append({"type": "image_url", "image_url": {"url": jpeg_data_url(image, long_side=768)}})
    body = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    import requests
    started = time.time()
    response = requests.post(api_base() + "/chat/completions", json=body, timeout=timeout, headers=headers)
    latency = round(time.time() - started, 3)
    status = int(response.status_code)
    payload = None
    decode_error = None
    try:
        payload = response.json()
    except Exception as exc:
        decode_error = repr(exc)[:300]
    finish, text = _finish_and_text(payload)
    snippet = public_snippet(text)
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if status >= 400:
        verdict = {"F2": "uncertain", "F5": "uncertain", "reason": "http_error", "parsed": False}
        return {"http_status": status, "latency_s": latency, "usage": usage, "verdict": verdict,
                "error": public_snippet(getattr(response, "text", "") or "", 200),
                "finish_reason": finish, "raw_snippet": snippet}
    if decode_error is not None or payload is None:
        verdict = {"F2": "uncertain", "F5": "uncertain", "reason": "parse_error", "parsed": False}
        return {"http_status": status, "latency_s": latency, "usage": usage, "verdict": verdict,
                "error": public_snippet(decode_error or "response was not JSON", 300),
                "finish_reason": finish, "raw_snippet": snippet}
    verdict = ppe_rules.parse_muse_json(text)
    return {"http_status": status, "latency_s": latency, "usage": usage, "verdict": verdict,
            "error": None, "finish_reason": finish, "raw_snippet": snippet}


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_verdicts(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Muse Spark clip check for F2 and F5")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--voxel-manifest", type=Path, default=DEFAULT_VOXEL)
    parser.add_argument("--weak-labels", type=Path, default=DEFAULT_WEAK)
    parser.add_argument("--source", default=pv.SOURCE, help="ppe_video, voxel51_safe_unsafe, or all")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--sam", type=Path, default=DEFAULT_SAM)
    parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument("--budget-total", type=int, default=BUDGET_TOTAL)
    parser.add_argument("--max-calls", type=int, default=0, help="cap for this run; 0 sends nothing")
    parser.add_argument("--call-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ids", type=Path, default=None, help="clip ids, one per line")
    parser.add_argument("--limit-source-split", default="",
                        help="comma-separated source:split filters, e.g. voxel51_safe_unsafe:test")
    parser.add_argument("--order", default="priority", choices=("priority", "id"),
                        help="priority: test, demo_eval, val, train, then id (default); id: clip id only")
    parser.add_argument("--model", default=None,
                        help="default: MUSE_SPARK_MODEL or muse-spark-1.3-contributor")
    parser.add_argument("--max-tokens", type=int, default=4000,
                        help="completion cap (default 4000, same as VerifyClient)")
    parser.add_argument("--redo-parse-errors", action="store_true",
                        help="call again when the stored reason is parse_error; still spends the budget")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true", help="print the request; no HTTP and no budget change")
    return parser


def read_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(line.split()[0])
    return ids


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.dry_run:
        load_model_env()
    model = resolve_model(args.model)
    clips = select_clips(
        args.manifest, args.source, read_ids(args.ids), args.voxel_manifest, args.weak_labels,
        parse_source_splits(args.limit_source_split), args.order,
    )
    index_by_id = group_index(pv.read_jsonl(args.index))
    sam = sam_by_clip(pv.read_jsonl(args.sam))
    done = {row["id"]: row for row in pv.read_jsonl(args.out)}
    pending = [clip for clip in clips if clip_is_pending(done.get(clip["id"]), args.redo_parse_errors)]
    print(
        f"muse model={model} max_tokens={args.max_tokens} clips={len(clips)} pending={len(pending)} "
        f"max_calls={args.max_calls} budget_total={args.budget_total} dry_run={args.dry_run}",
        flush=True,
    )
    if not args.dry_run and args.max_calls <= 0:
        print("refusing Muse calls: --max-calls is 0 (pass a positive cap to spend the Meta API)", flush=True)
        return 2
    sent = 0
    written = dict(done)
    for clip in pending:
        frames = frames_for(clip, index_by_id, sam)
        if len(frames) < 3:
            print(f"skip {clip['id']}: need 3 frames, have {len(frames)} (run sample_frames.py)", flush=True)
            continue
        if args.dry_run:
            if sent >= args.max_calls:
                break
            sent += 1
            print(
                f"dry-run {sent}: POST /chat/completions model={model} max_tokens={args.max_tokens} "
                f"id={clip['id']} split={clip.get('split')}"
            )
            print(f"  frames: {frames[0]}")
            print(f"          {frames[1]}")
            print(f"          {frames[2]}")
            print("  prompt: F2 present|absent|uncertain, F5 present|absent|uncertain, reason")
            continue
        allowed, total, reason = reserve(args.budget, args.budget_total, sent, args.max_calls)
        if not allowed:
            print(f"refusing further calls: {reason} (budget total={total}, this run sent={sent})", flush=True)
            break
        sent += 1
        try:
            result = ask_muse(frames, model, args.timeout, args.max_tokens)
        except Exception as exc:
            # The reservation already counted. Record uncertain and keep going.
            result = {
                "http_status": 0,
                "latency_s": None,
                "usage": None,
                "verdict": {"F2": "uncertain", "F5": "uncertain", "reason": "request_error", "parsed": False},
                "error": public_snippet(repr(exc), 300),
                "finish_reason": None,
                "raw_snippet": "",
            }
        verdict = result["verdict"]
        if not verdict.get("parsed"):
            verdict = {"F2": "uncertain", "F5": "uncertain", "reason": verdict.get("reason") or "parse_error", "parsed": False}
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "id": clip["id"],
            "frames": frames,
            "verdict": {key: verdict[key] for key in ("F2", "F5", "reason")},
            "parsed": bool(verdict.get("parsed")),
            "latency_s": result.get("latency_s"),
            "http_status": result.get("http_status"),
            "usage": result.get("usage"),
            "error": result.get("error"),
            "model": model,
            "budget_total": total,
            "finish_reason": result.get("finish_reason"),
            "raw_snippet": result.get("raw_snippet") or "",
        }
        append_jsonl(args.call_log, record)
        written[clip["id"]] = {
            "id": clip["id"],
            "source": clip.get("source"),
            "split": clip.get("split"),
            "F2": verdict["F2"],
            "F5": verdict["F5"],
            "reason": verdict["reason"],
            "parsed": bool(verdict.get("parsed")),
            "model": model,
        }
        print(
            f"call {sent} {clip['id']} http={record['http_status']} finish={record.get('finish_reason')} "
            f"F2={verdict['F2']} F5={verdict['F5']} budget={total}",
            flush=True,
        )
    if not args.dry_run and sent:
        ordered = [written[key] for key in sorted(written)]
        write_verdicts(args.out, ordered)
        print(f"wrote {len(ordered)} verdicts -> {args.out}", flush=True)
    elif args.dry_run:
        print(f"dry-run planned {sent} call(s); budget file not touched", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
