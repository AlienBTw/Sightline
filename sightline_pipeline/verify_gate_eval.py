"""Score an F2 or F5 verify prompt on a folder of still frames. No live pipeline.

    python -m sightline_pipeline.verify_gate_eval --frames <dir> --labels <csv> [--dry-run]
        [--factor F2|F5] [--batch-size 1|3] [--max-tokens N] [--model ID] [--show-raw]

The CSV is `file,vest` for F5 (1 = wearing a hi-vis vest) or `file,head` for F2
(1 = wearing head protection). Images are sent with the same Muse Spark prompt as the
live gate, either one per request (`--batch-size 1`) or three per request
(`--batch-size 3`, the default).

The model is ``--model`` if you pass it, otherwise ``MUSE_SPARK_MODEL``, otherwise
``escalation.muse_model``, otherwise ``muse-spark-1.3-contributor`` (Muse Spark 1.3 Contributor).

`--dry-run` checks the files and prints how many calls would be made. It does not read an API
key and does not use the network. A real run needs MODEL_API_KEY in the environment.

A real run prints one line per frame: file, label, person_visible, the protection field, correct.
`--show-raw` also prints, per batch, the status, the error, and the first 1500 characters of the
model answer. Request headers and the API key are not printed.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from .config import load_config
from .escalation import api_key
from .verify_gate import FACTOR_FIELD, VerifyClient


def load_labels(path: Path) -> list[tuple[str, int]]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = [p.strip() for p in text.split(",")]
        if len(parts) < 2:
            raise ValueError(f"{path}:{i + 1}: expected file,label")
        header = parts[1].lower()
        if i == 0 and parts[0].lower() == "file" and header.startswith(("vest", "head", "label", "protected")):
            continue
        if parts[1] not in ("0", "1"):
            raise ValueError(f"{path}:{i + 1}: label must be 0 or 1, got {parts[1]!r}")
        rows.append((parts[0], int(parts[1])))
    if not rows:
        raise ValueError(f"{path}: no labelled frames")
    return rows


def read_bgr(path: Path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _batches(rows, size: int):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def _bool_word(value) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "n/a"


def _prediction_line(name: str, label: int, ans, field: str) -> str:
    if not isinstance(ans, dict):
        return (f"file: {name} label: {label} person_visible: n/a "
                f"{field}: n/a correct: n/a")
    wearing = ans.get(field)
    pred = 1 if wearing is True else 0
    return (f"file: {name} label: {label} person_visible: {_bool_word(ans.get('person_visible'))} "
            f"{field}: {_bool_word(wearing)} correct: {int(pred == label)}")


def _print_raw(batch_i: int, asked: dict) -> None:
    """Model answer only. Request headers and the API key are not on `asked`."""
    raw = asked.get("raw") or ""
    if not isinstance(raw, str):
        raw = str(raw)
    print(f"batch: {batch_i}")
    print(f"status: {asked.get('status')}")
    print(f"error: {asked.get('error') or ''}")
    print(f"raw: {raw[:1500]}")
    diag = asked.get("diagnostic")
    if isinstance(diag, dict):
        print(f"http_status: {diag.get('http_status')}")
        print(f"finish_reason: {diag.get('finish_reason')}")
        usage = diag.get("usage")
        if isinstance(usage, dict):
            print(f"usage: {usage}")
        print(f"content: {(diag.get('content') or '')[:1500]}")
        print(f"reasoning: {(diag.get('reasoning') or '')[:1500]}")


def _print_predictions(batch, answers, aligned: bool, field: str) -> None:
    for i, (name, label, _) in enumerate(batch):
        ans = answers[i] if aligned and isinstance(answers, list) else None
        print(_prediction_line(name, label, ans, field))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sightline_pipeline.verify_gate_eval")
    ap.add_argument("--frames", required=True, help="directory of still frames")
    ap.add_argument("--labels", required=True, help="CSV: file,label with 1 = wearing the protective item")
    ap.add_argument("--factor", default="F5", choices=("F2", "F5"),
                    help="F5 asks about a hi-vis vest, F2 about head protection (default F5)")
    ap.add_argument("--dry-run", action="store_true", help="do not call the API")
    ap.add_argument("--batch-size", type=int, default=3, choices=(1, 3))
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="reply token budget sent to the model (default 4000)")
    ap.add_argument("--model", default=None,
                    help="Muse Spark model id. Default: MUSE_SPARK_MODEL if set, else "
                         "escalation.muse_model, else muse-spark-1.3-contributor")
    ap.add_argument("--show-raw", action="store_true",
                    help="print each batch status, error, and the first 1500 characters of the model answer")
    args = ap.parse_args(argv)
    field = FACTOR_FIELD[args.factor]
    root = Path(args.frames)
    try:
        rows = load_labels(Path(args.labels))
    except (OSError, ValueError) as e:
        print(f"error_code: bad_labels\nmessage: {e}", file=sys.stderr)
        return 1
    images = []
    for name, label in rows:
        path = Path(name)
        if not path.is_file():
            path = root / name
        try:
            images.append((name, label, read_bgr(path)))
        except (OSError, FileNotFoundError) as e:
            print(f"error_code: missing_frame\nmessage: {e}", file=sys.stderr)
            return 1
    n_calls = math.ceil(len(images) / args.batch_size) if images else 0
    print(f"factor: {args.factor}")
    print(f"frames: {len(images)}")
    print(f"batch_size: {args.batch_size}")
    if args.dry_run:
        print("dry_run: true")
        print("api_calls: 0")
        print(f"would_call: {n_calls}")
        print("accuracy: n/a")
        return 0
    if not api_key():
        print("error_code: skipped_no_key\nmessage: MODEL_API_KEY is not set", file=sys.stderr)
        print("api_calls: 0")
        return 1
    configured = (load_config().get("escalation") or {}).get("muse_model")
    client = VerifyClient(max_tokens=args.max_tokens, model=configured, model_override=args.model)
    correct = 0
    api_calls = 0
    for batch_i, batch in enumerate(_batches(images, args.batch_size), start=1):
        api_calls += 1
        asked = client.ask([img for _, _, img in batch], factors=(args.factor,))
        answers = asked.get("answers")
        aligned = asked.get("status") == "ok" and isinstance(answers, list) and len(answers) == len(batch)
        if args.show_raw:
            _print_raw(batch_i, asked)
        _print_predictions(batch, answers, aligned, field)
        if not aligned:
            print(f"batch_error: {asked.get('status')} {asked.get('error') or ''}".rstrip(), file=sys.stderr)
            continue
        for (_, label, _), ans in zip(batch, answers):
            pred = 1 if ans.get(field) is True else 0
            correct += int(pred == label)
    accuracy = correct / len(images) if images else 0.0
    print(f"api_calls: {api_calls}")
    print(f"correct: {correct}")
    print(f"accuracy: {accuracy:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
