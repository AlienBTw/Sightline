"""SAM 3.1 boxes on PPE keyframes, then F2/F5 window labels.

Prompts: head, hard hat, helmet, safety vest, high-visibility vest, person.
``--backend local`` encodes each keyframe once and reuses that embedding for
every prompt. A ``.pt`` checkpoint (the default path
``/mnt/f/models/sam3.1/sam3.1_multiplex.pt``) loads through
``build_sam3_image_model(..., load_from_HF=False, checkpoint_path=...)``.
The ungated unsloth/sam3.1 mirror is a transformers ``Sam3Model`` directory
(``model.safetensors``); pass that directory as ``--checkpoint``. Default GPU
is 1. Before torch is imported, that physical GPU is the only visible device,
so the live demo on GPU 0 is not touched. ``--backend api`` calls Meta
``sam-3.1`` the same way as ``SamRefiner._api``, with its own
``--max-api-calls`` (default 0) and an append-only log. It never reads
``data/live/escalation_budget.json``.

``--dry-run`` prints the plan and does not load weights or send HTTP.

  CUDA_VISIBLE_DEVICES=1 /home/user/sam3-venv/bin/python tools/labeling/sam_label.py --backend local --gpu 1 --checkpoint /mnt/f/models/sam3.1
  python tools/labeling/sam_label.py --backend api --dry-run --max-api-calls 0
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "labeling"))

import ppe_rules  # noqa: E402

DEFAULT_INDEX = Path("/mnt/f/models/vidlabel/frames/index.jsonl")
DEFAULT_CACHE = Path("/mnt/f/models/vidlabel/sam_cache")
DEFAULT_OUT = Path("/mnt/f/models/vidlabel/sam_windows.jsonl")
DEFAULT_LOG = Path("/mnt/f/models/vidlabel/sam_api_calls.jsonl")
DEFAULT_CKPT = Path("/mnt/f/models/sam3.1/sam3.1_multiplex.pt")
RULE_VERSION = 2


def load_model_env() -> None:
    """Same .env load as sightline_pipeline.run. The file and the key are never printed."""
    if os.environ.get("SIGHTLINE_NO_DOTENV"):
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO / ".env", override=False)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def cache_file(cache_dir: Path, key: str, window: int, j: int, threshold: float) -> Path:
    thr = f"{float(threshold):.2f}".replace(".", "p")
    return cache_dir / key / f"w{window}_{j}_t{thr}.json"


def jpeg_data_url(path: Path) -> str:
    payload = path.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


def parse_api_stream(response) -> str:
    text = ""
    ctype = response.headers.get("content-type", "")
    if "event-stream" in ctype:
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "response.output_text.delta":
                text += event.get("delta", "")
            elif event.get("type") == "response.completed" and not text:
                text = json.dumps(event.get("response", {}))
    else:
        text = response.text or ""
    return text


def api_boxes(image_path: Path, prompt: str, model: str, timeout: float) -> tuple[list[dict], int]:
    """One SAM 3.1 /responses call. Returns (boxes, http status)."""
    import requests
    load_model_env()
    from sightline_pipeline.escalation import api_base, api_key, parse_sam_boxes
    key = api_key()
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set")
    body = {
        "model": model,
        "stream": True,
        "metadata": {"include_confidence": "true"},
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": jpeg_data_url(image_path)},
        ]}],
    }
    response = requests.post(
        api_base() + "/responses",
        json=body,
        stream=True,
        timeout=timeout,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    status = int(response.status_code)
    if status >= 400:
        raise RuntimeError(f"HTTP {status}: {(response.text or '')[:200]}")
    return parse_sam_boxes(parse_api_stream(response)), status


def boxes_to_records(boxes: list[dict], width: int, height: int) -> list[dict]:
    out = []
    for box in boxes:
        if "x1" not in box:
            continue
        # API boxes are in the image SAM saw. ``w``/``h`` are that image; our JPEG is that image.
        scale_w = width / float(box["w"]) if box.get("w") else 1.0
        scale_h = height / float(box["h"]) if box.get("h") else 1.0
        out.append({
            "x1": float(box["x1"]) * scale_w,
            "y1": float(box["y1"]) * scale_h,
            "x2": float(box["x2"]) * scale_w,
            "y2": float(box["y2"]) * scale_h,
            "c": box.get("c"),
        })
    return out


def resolve_local_gpu(gpu: int, visible: str | None) -> tuple[int, str | None]:
    """Pin local SAM to one physical GPU before torch is imported.

    Returns ``(index_inside_this_process, CUDA_VISIBLE_DEVICES to set or None)``.
    No visible-device list and ``--gpu 1`` becomes ``CUDA_VISIBLE_DEVICES=1``
    and process index 0, so GPU 0 is not visible. ``CUDA_VISIBLE_DEVICES=1``
    together with ``--gpu 1`` is already that physical GPU, so the process
    index is 0.
    """
    if gpu < 0:
        return gpu, None
    if visible is None or str(visible).strip() == "":
        return 0, str(int(gpu))
    devices = [part.strip() for part in str(visible).split(",") if part.strip() != ""]
    if len(devices) == 1 and devices[0] == str(int(gpu)):
        return 0, None
    return int(gpu), None


def checkpoint_kind(path: Path) -> str:
    """``transformers`` (unsloth directory or safetensors), ``sam3_pt``, or ``missing``."""
    if path.is_dir() and (path / "model.safetensors").is_file() and (path / "config.json").is_file():
        return "transformers"
    if path.is_file() and path.suffix == ".safetensors":
        return "transformers"
    if path.is_file() and path.suffix == ".pt":
        return "sam3_pt"
    return "missing"


def _dtype_and_autocast(dtype_name: str, device):
    import torch
    if dtype_name == "fp32" or device.type != "cuda":
        return torch.float32, None
    if dtype_name == "fp16":
        return torch.float16, torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16, torch.bfloat16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, torch.bfloat16
    return torch.float16, torch.float16


def _records_from_tensors(boxes, scores) -> list[dict]:
    coords = boxes.detach().float().cpu().tolist() if boxes is not None else []
    score_list = scores.detach().float().cpu().tolist() if scores is not None else [None] * len(coords)
    out = []
    for coord, score in zip(coords, score_list):
        out.append({
            "x1": float(coord[0]), "y1": float(coord[1]),
            "x2": float(coord[2]), "y2": float(coord[3]),
            "c": None if score is None else float(score),
        })
    return out


class _TransformersGrounder:
    """unsloth/sam3.1 ``Sam3Model``. The vision tower runs once per image."""

    def __init__(self, model, processor, device, autocast, threshold: float):
        self.model = model
        self.processor = processor
        self.device = device
        self.autocast = autocast
        self.threshold = float(threshold)
        self._text = {}

    def _text_embeds(self, prompt: str):
        import torch
        cached = self._text.get(prompt)
        if cached is not None:
            return cached
        text = self.processor(text=prompt, return_tensors="pt")
        ids = text["input_ids"].to(self.device)
        mask = text["attention_mask"].to(self.device)
        with torch.inference_mode():
            embeds = self._forward(lambda: self.model.get_text_features(
                input_ids=ids, attention_mask=mask, return_dict=True,
            ))
        self._text[prompt] = (embeds, mask)
        return embeds, mask

    def _forward(self, call):
        import torch
        if self.autocast is not None and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.autocast):
                return call()
        return call()

    def infer(self, image, prompts) -> dict[str, list[dict]]:
        import torch
        encoded = self.processor(images=image, return_tensors="pt")
        pixels = encoded["pixel_values"].to(self.device)
        found = {}
        timing = []
        with torch.inference_mode():
            t_vision = time.time()
            vision = self._forward(lambda: self.model.get_vision_features(pixel_values=pixels))
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            vision_s = time.time() - t_vision
            for prompt in prompts:
                t_prompt = time.time()
                embeds, mask = self._text_embeds(prompt)
                outputs = self._forward(lambda embeds=embeds, mask=mask: self.model(
                    vision_embeds=vision, text_embeds=embeds, attention_mask=mask,
                ))
                if outputs.pred_logits is not None:
                    outputs.pred_logits = outputs.pred_logits.float()
                if outputs.pred_boxes is not None:
                    outputs.pred_boxes = outputs.pred_boxes.float()
                if getattr(outputs, "presence_logits", None) is not None:
                    outputs.presence_logits = outputs.presence_logits.float()
                parsed = self.processor.post_process_object_detection(
                    outputs,
                    threshold=0.0,
                    target_sizes=[(image.height, image.width)],
                )[0]
                boxes = _records_from_tensors(parsed["boxes"], parsed["scores"])
                found[prompt] = [box for box in boxes if box["c"] is None or box["c"] >= self.threshold]
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                timing.append((prompt, time.time() - t_prompt, len(found[prompt])))
                del outputs
        self.last_timing = {"vision_s": vision_s, "prompts": timing}
        return found


class _Sam3PtGrounder:
    """facebookresearch/sam3 image model. The backbone runs once per image."""

    def __init__(self, processor, device, autocast):
        self.processor = processor
        self.device = device
        self.autocast = autocast

    def infer(self, image, prompts) -> dict[str, list[dict]]:
        return _one_image(self.processor, image, prompts, self.device, self.autocast)


def _missing_checkpoint(checkpoint: Path) -> SystemExit:
    sibling = checkpoint / "model.safetensors" if checkpoint.suffix != ".safetensors" else checkpoint
    if checkpoint.suffix == ".pt":
        sibling = checkpoint.parent / "model.safetensors"
    hint = ""
    if sibling.is_file():
        hint = (
            f" unsloth/sam3.1 does not ship sam3.1_multiplex.pt; "
            f"pass --checkpoint {sibling.parent}"
        )
    return SystemExit(f"SAM checkpoint missing: {checkpoint}.{hint}")


def load_image_model(checkpoint: Path, gpu: int, threshold: float, dtype_name: str):
    import torch
    kind = checkpoint_kind(checkpoint)
    if kind == "missing":
        raise _missing_checkpoint(checkpoint)
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
    if gpu >= 0 and device.type != "cuda":
        raise SystemExit(f"--gpu {gpu} requested but CUDA is not available in this process")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _dtype, autocast = _dtype_and_autocast(dtype_name, device)
    if kind == "transformers":
        grounder = _load_transformers(checkpoint, device, autocast, _dtype, threshold)
    else:
        grounder = _load_sam3_pt(checkpoint, device, autocast, threshold)
    return grounder, device, autocast


def _load_transformers(checkpoint: Path, device, autocast, dtype, threshold: float):
    import torch
    from transformers import Sam3Model, Sam3Processor
    folder = checkpoint if checkpoint.is_dir() else checkpoint.parent
    weight = folder / "model.safetensors"
    if not weight.is_file():
        raise SystemExit(f"SAM transformers weights missing: {weight}")
    print(
        f"loading transformers Sam3Model from {folder} dtype={dtype} device={device} "
        f"file={weight.name} bytes={weight.stat().st_size}",
        flush=True,
    )
    try:
        model = Sam3Model.from_pretrained(str(folder), dtype=dtype, local_files_only=True)
    except TypeError:
        model = Sam3Model.from_pretrained(str(folder), torch_dtype=dtype, local_files_only=True)
    model = model.to(device).eval()
    processor = Sam3Processor.from_pretrained(str(folder), local_files_only=True)
    print(f"loaded {weight.name} local_files_only transformers Sam3Model", flush=True)
    return _TransformersGrounder(model, processor, device, autocast, threshold)


def _load_sam3_pt(checkpoint: Path, device, autocast, threshold: float):
    import torch
    import sam3.model_builder as model_builder
    from sam3.model.sam3_image_processor import Sam3Processor
    model_builder._load_checkpoint = _load_sam3_checkpoint
    # device="cpu" so build_sam3_image_model cannot call model.cuda() on GPU 0.
    model = model_builder.build_sam3_image_model(
        device="cpu",
        eval_mode=True,
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        enable_segmentation=False,
        enable_inst_interactivity=False,
    )
    if device.type == "cuda":
        model = model.to(device)
    processor = _BoxProcessor(model, device=str(device), confidence_threshold=float(threshold))
    return _Sam3PtGrounder(processor, device, autocast)


def _load_sam3_checkpoint(model, checkpoint_path) -> None:
    """Loader used by ``build_sam3_image_model``. Remaps multiplex key prefixes.

    Official image checkpoints use ``detector.*``. Some multiplex exports use
    ``sam3_model.*`` / ``sam2_predictor.*``. ``load_from_HF`` stays false; the
    path passed to the builder is the local file.
    """
    import torch
    path = Path(checkpoint_path)
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and isinstance(blob.get("model"), dict):
        blob = blob["model"]
    if not isinstance(blob, dict):
        raise RuntimeError(f"unrecognized checkpoint {path}")
    if any(str(key).startswith("sam3_model.") or str(key).startswith("sam2_predictor.") for key in blob):
        remapped = {}
        for key, value in blob.items():
            name = str(key)
            if name.startswith("sam3_model."):
                remapped["detector." + name[len("sam3_model."):]] = value
            elif name.startswith("sam2_predictor."):
                remapped["tracker." + name[len("sam2_predictor."):]] = value
            else:
                remapped[key] = value
        blob = remapped
    state = {
        str(key).replace("detector.", "", 1): value
        for key, value in blob.items()
        if "detector" in str(key)
    }
    if not state:
        raise RuntimeError(f"no detector weights in {path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"loaded {path.name} missing={len(missing)} unexpected={len(unexpected)} "
        f"detector_tensors={len(state)}",
        flush=True,
    )
    if len(missing) > max(8, int(0.5 * max(1, len(state)))):
        raise RuntimeError(
            f"checkpoint {path.name} did not match the SAM image model "
            f"({len(missing)} missing keys, {len(unexpected)} unexpected)"
        )


class _BoxProcessor:
    """Sam3Processor that keeps boxes and skips full-resolution mask upsampling."""

    def __init__(self, model, device: str, confidence_threshold: float):
        from sam3.model.sam3_image_processor import Sam3Processor
        self._processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)
        self.confidence_threshold = float(confidence_threshold)

    def set_image(self, image):
        return self._processor.set_image(image)

    def reset_all_prompts(self, state):
        return self._processor.reset_all_prompts(state)

    def set_text_prompt(self, prompt, state):
        import torch
        from sam3.model import box_ops
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")
        text_outputs = self._processor.model.backbone.forward_text([prompt], device=self._processor.device)
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self._processor.model._get_dummy_prompt()
        outputs = self._processor.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self._processor.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )
        out_probs = outputs["pred_logits"].sigmoid()
        presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence).squeeze(-1)
        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_bbox = outputs["pred_boxes"][keep]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale = torch.tensor(
            [state["original_width"], state["original_height"], state["original_width"], state["original_height"]],
            device=boxes.device,
        )
        state["boxes"] = boxes * scale
        state["scores"] = out_probs
        state["masks"] = None
        return state


def _prompt_boxes(processor, state, prompts, device, autocast) -> dict[str, list[dict]]:
    import torch
    found = {}
    for prompt in prompts:
        processor.reset_all_prompts(state)
        if autocast is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast):
                result = processor.set_text_prompt(prompt, state)
        else:
            result = processor.set_text_prompt(prompt, state)
        boxes = []
        tensor = result.get("boxes")
        scores = result.get("scores")
        if tensor is not None and len(tensor) > 0:
            coords = tensor.detach().float().cpu().tolist()
            score_list = scores.detach().float().cpu().tolist() if scores is not None else [None] * len(coords)
            for coord, score in zip(coords, score_list):
                boxes.append({
                    "x1": float(coord[0]), "y1": float(coord[1]),
                    "x2": float(coord[2]), "y2": float(coord[3]),
                    "c": None if score is None else float(score),
                })
        found[prompt] = boxes
    return found


def local_boxes_batch(grounder, images, prompts) -> list[dict]:
    """One vision encode per image. Every prompt reuses that encode."""
    from PIL import Image
    out = []
    for path in images:
        image = Image.open(path).convert("RGB")
        out.append({
            "width": image.width,
            "height": image.height,
            "prompt_boxes": grounder.infer(image, prompts),
        })
    return out


def _one_image(processor, image, prompts, device, autocast) -> dict:
    import torch
    if autocast is not None and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=autocast):
            state = processor.set_image(image)
    else:
        state = processor.set_image(image)
    return _prompt_boxes(processor, state, prompts, device, autocast)


def derive(prompt_boxes: dict, width: int, height: int, threshold: float) -> dict:
    """``prompt_boxes`` are normalised to the keyframe. Geometry runs in pixels."""
    scaled = {
        name: ppe_rules.scale_box_records(boxes, width, height)
        for name, boxes in (prompt_boxes or {}).items()
    }
    persons, heads, hats, vests = ppe_rules.collect_detections(scaled)
    return ppe_rules.label_frame(persons, hats, vests, width, height, threshold, heads=heads)


def clip_sort_key(row: dict):
    """Pexels before Voxel51, then the existing split rank."""
    ident = str(row.get("id") or "")
    source = str(row.get("source") or "")
    pexels = ident.startswith("pexels/") or ident.startswith("pexels_") or source in {"ppe_video", "pexels"}
    return (0 if pexels else 1,) + tuple(ppe_rules.priority_key(row))


def jobs_from_index(rows: list[dict], ids: set[str] | None, splits: set[str] | None) -> list[dict]:
    grouped: dict[str, list] = {}
    for row in rows:
        if ids and row["id"] not in ids:
            continue
        if splits and row.get("split") not in splits:
            continue
        grouped.setdefault(row["id"], []).append(row)
    clips = []
    for clip_id, windows in grouped.items():
        windows.sort(key=lambda row: int(row["window"]))
        clips.append({
            "id": clip_id,
            "windows": windows,
            "key": windows[0].get("key"),
            "split": windows[0].get("split"),
            "source": windows[0].get("source"),
        })
    clips.sort(key=clip_sort_key)
    return clips


def cache_is_current(slot: Path) -> bool:
    if not slot.is_file():
        return False
    try:
        cached = json.loads(slot.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        cached.get("rule_version") == RULE_VERSION
        and cached.get("coord_space") == "norm"
        and bool(cached.get("prompt_boxes"))
    )


def planned_calls(clips: list[dict]) -> list[dict]:
    calls = []
    for clip in clips:
        for window in clip["windows"]:
            for j, path in enumerate(window.get("paths") or []):
                for prompt in ppe_rules.SAM_PROMPTS:
                    calls.append({
                        "id": clip["id"], "window": window["window"], "j": j,
                        "frame": path, "prompt": prompt,
                    })
    return calls


def label_cached_or_compute(window_rows, cache_dir, threshold, predict, log_path, max_api_calls, dry_run, backend):
    """``predict(path, width, height) -> prompt->boxes`` or None when the cap stops the run."""
    sent = 0
    window_out = []
    stopped = False
    for window in window_rows:
        frame_labels = []
        width = int(window["width"])
        height = int(window["height"])
        paths = window.get("paths") or []
        indices = window.get("frame_indices") or []
        for j, path in enumerate(paths):
            if stopped:
                break
            slot = cache_file(cache_dir, window["key"], int(window["window"]), j, threshold)
            cached = None
            if slot.is_file() and not dry_run and cache_is_current(slot):
                try:
                    cached = json.loads(slot.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    cached = None
            if cached and cached.get("prompt_boxes"):
                frame_labels.append(derive(
                    cached["prompt_boxes"],
                    int(cached.get("width") or width),
                    int(cached.get("height") or height),
                    threshold,
                ))
                continue
            if dry_run:
                frame_labels.append(None)
                continue
            if backend == "api":
                prompt_boxes = {}
                failed = False
                for prompt in ppe_rules.SAM_PROMPTS:
                    if not ppe_rules.api_calls_allowed(sent, max_api_calls):
                        stopped = True
                        break
                    started = time.time()
                    status = 0
                    error = None
                    try:
                        boxes, status = api_boxes(Path(path), prompt, os.environ.get("SAM_MODEL") or "sam-3.1", 60)
                        prompt_boxes[prompt] = boxes_to_records(boxes, width, height)
                    except Exception as exc:
                        error = str(exc)[:300]
                        failed = True
                    sent += 1
                    append_jsonl(log_path, {
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "id": window["id"],
                        "window": window["window"],
                        "j": j,
                        "frame": path,
                        "prompt": prompt,
                        "http_status": status,
                        "latency_s": round(time.time() - started, 3),
                        "n_boxes": len(prompt_boxes.get(prompt) or []),
                        "error": error,
                    })
                    if failed:
                        print(f"ERROR {window['id']} w{window['window']} j{j} {prompt}: {error}", flush=True)
                        break
                # A failed or unfinished keyframe is not cached, so a later run can retry it.
                if failed or len(prompt_boxes) < len(ppe_rules.SAM_PROMPTS):
                    stopped = True
                    break
            else:
                prompt_boxes = predict(Path(path), width, height)
            prompt_boxes = {
                name: ppe_rules.normalize_box_records(rows, width, height)
                for name, rows in prompt_boxes.items()
            }
            payload = {
                "rule_version": RULE_VERSION,
                "coord_space": "norm",
                "id": window["id"],
                "window": window["window"],
                "j": j,
                "frame_index": indices[j] if j < len(indices) else None,
                "path": path,
                "width": width,
                "height": height,
                "prompt_boxes": prompt_boxes,
            }
            slot.parent.mkdir(parents=True, exist_ok=True)
            tmp = slot.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, slot)
            frame_labels.append(derive(prompt_boxes, width, height, threshold))
        if stopped or len(frame_labels) != len(paths) or any(label is None for label in frame_labels):
            if dry_run:
                window_out.append({
                    "w": int(window["window"]),
                    "start": int(window["start"]),
                    "F2": None,
                    "F5": None,
                    "boxes": ppe_rules.empty_boxes(),
                    "dry_run": True,
                })
            break
        summary = ppe_rules.aggregate_window(frame_labels)
        window_out.append({
            "w": int(window["window"]),
            "start": int(window["start"]),
            **summary,
        })
    return window_out, sent, stopped


def write_windows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="SAM 3.1 PPE labels on keyframes")
    parser.add_argument("--backend", choices=("local", "api"), default="local")
    parser.add_argument("--gpu", type=int, default=1, help="local backend device; default 1 (live pipeline uses GPU 0)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--batch", type=int, default=1, help="kept for compatibility; each image is encoded once")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--call-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--max-api-calls", type=int, default=0, help="API backend hard cap for this run; 0 sends nothing")
    parser.add_argument("--split", default="")
    parser.add_argument("--ids", type=Path, default=None, help="text file of clip ids, one per line")
    parser.add_argument("--limit", type=int, default=0, help="max clips after the priority sort")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; no weights and no HTTP")
    return parser


def _read_ids(path: Path | None) -> set[str] | None:
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
    rows = read_jsonl(args.index)
    if not rows:
        raise SystemExit(f"frame index is empty or missing: {args.index}")
    splits = {part.strip() for part in args.split.split(",") if part.strip()} or None
    clips = jobs_from_index(rows, _read_ids(args.ids), splits)
    if args.limit:
        clips = clips[: args.limit]
    calls = planned_calls(clips)
    print(
        f"sam backend={args.backend} clips={len(clips)} keyframes={len(calls) // max(1, len(ppe_rules.SAM_PROMPTS))} "
        f"api_calls={len(calls)} max_api_calls={args.max_api_calls} dry_run={args.dry_run}",
        flush=True,
    )
    if args.dry_run:
        show = calls if args.backend == "local" else calls[: max(args.max_api_calls, 0) or 8]
        for call in show[:40]:
            print(f"  {call['id']} w{call['window']} j{call['j']} prompt={call['prompt']} frame={call['frame']}")
        if len(show) > 40:
            print(f"  ... {len(show) - 40} more")
        if args.backend == "api" and args.max_api_calls <= 0:
            print("dry-run: a real API run would send nothing because --max-api-calls is 0")
        else:
            print("dry-run: no HTTP and no checkpoint load")
        return 0
    if args.backend == "api" and args.max_api_calls <= 0:
        print("refusing API calls: --max-api-calls is 0 (pass a positive cap to spend the Meta API)", flush=True)
        return 2

    if args.backend == "local":
        local_gpu, pin = resolve_local_gpu(args.gpu, os.environ.get("CUDA_VISIBLE_DEVICES"))
        if pin is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = pin
            print(
                f"CUDA_VISIBLE_DEVICES={pin} physical_gpu={args.gpu} process_index={local_gpu}",
                flush=True,
            )
        elif local_gpu != args.gpu:
            print(
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"physical_gpu={args.gpu} process_index={local_gpu}",
                flush=True,
            )
        args.gpu = local_gpu
        grounder, _device, _autocast = load_image_model(
            args.checkpoint, args.gpu, args.score_threshold, args.dtype,
        )
        _run_local_batched(clips, grounder, args)
        out_rows = _windows_from_cache(clips, args.cache_dir, args.score_threshold)
        write_windows(args.out, out_rows)
        print(f"wrote {len(out_rows)} clips -> {args.out}", flush=True)
        return 0

    def predict(_path, _width, _height):
        raise RuntimeError("api backend does not use the local predict hook")

    out_rows = []
    sent_total = 0
    for clip in clips:
        windows, sent, stopped = label_cached_or_compute(
            clip["windows"], args.cache_dir, args.score_threshold, predict,
            args.call_log, args.max_api_calls - sent_total, False, "api",
        )
        sent_total += sent
        if windows:
            out_rows.append({"id": clip["id"], "windows": windows})
        print(f"{clip['id']} windows={len(windows)} api_calls_run={sent_total}", flush=True)
        if stopped or sent_total >= args.max_api_calls:
            break
    write_windows(args.out, out_rows)
    print(f"wrote {len(out_rows)} clips -> {args.out}  api_calls={sent_total}", flush=True)
    return 0


def _clip_record(clip: dict, cache_dir: Path, threshold: float) -> dict | None:
    windows = []
    for window in clip["windows"]:
        labels = []
        width, height = int(window["width"]), int(window["height"])
        paths = window.get("paths") or []
        if not paths:
            return None
        for j in range(len(paths)):
            slot = cache_file(cache_dir, window["key"], int(window["window"]), j, threshold)
            if not cache_is_current(slot):
                return None
            cached = json.loads(slot.read_text(encoding="utf-8"))
            labels.append(derive(
                cached.get("prompt_boxes") or {},
                int(cached.get("width") or width),
                int(cached.get("height") or height),
                threshold,
            ))
        summary = ppe_rules.aggregate_window(labels)
        windows.append({"w": int(window["window"]), "start": int(window["start"]), **summary})
    if not windows:
        return None
    return {"id": clip["id"], "windows": windows}


def _windows_from_cache(clips, cache_dir: Path, threshold: float) -> list[dict]:
    out = []
    for clip in clips:
        record = _clip_record(clip, cache_dir, threshold)
        if record:
            out.append(record)
    return out


def _gpu_mem() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f" gpu_alloc_gb={alloc:.2f} gpu_reserved_gb={reserved:.2f}"
    except Exception:
        return ""


def _run_local_batched(clips, grounder, args) -> None:
    """Fill uncached keyframes. Each image is encoded once; prompts reuse it.

    A finished clip is appended to ``--out`` so a stopped run keeps its labels.
    Cache files are the resume point. A cache from an older rule version is redone.
    """
    by_id = {clip["id"]: clip for clip in clips}
    remaining = {clip["id"]: 0 for clip in clips}
    pending = []
    frames_total = 0
    for clip in clips:
        for window in clip["windows"]:
            width, height = int(window["width"]), int(window["height"])
            indices = window.get("frame_indices") or []
            for j, path in enumerate(window.get("paths") or []):
                frames_total += 1
                slot = cache_file(args.cache_dir, window["key"], int(window["window"]), j, args.score_threshold)
                if cache_is_current(slot):
                    continue
                remaining[clip["id"]] += 1
                pending.append({
                    "slot": slot,
                    "path": Path(path),
                    "clip_id": clip["id"],
                    "meta": {
                        "id": window["id"],
                        "window": int(window["window"]),
                        "j": j,
                        "frame_index": indices[j] if j < len(indices) else None,
                        "path": path,
                        "width": width,
                        "height": height,
                    },
                })
    cached = frames_total - len(pending)
    print(
        f"local pending={len(pending)} cached={cached} frames={frames_total} "
        f"prompts={len(ppe_rules.SAM_PROMPTS)}",
        flush=True,
    )
    ready = []
    ready_ids = set()

    def publish(clip_id: str) -> None:
        if clip_id in ready_ids or remaining.get(clip_id, 1) > 0:
            return
        record = _clip_record(by_id[clip_id], args.cache_dir, args.score_threshold)
        if record is None:
            return
        ready.append(record)
        ready_ids.add(clip_id)
        write_windows(args.out, ready)
        print(f"clip {clip_id} windows={len(record['windows'])} out={args.out}", flush=True)

    for clip in clips:
        if remaining[clip["id"]] == 0:
            publish(clip["id"])
    started = time.time()
    errors = 0
    for index, item in enumerate(pending, start=1):
        try:
            found = local_boxes_batch(grounder, [item["path"]], ppe_rules.SAM_PROMPTS)[0]
        except Exception as exc:
            errors += 1
            print(f"ERROR {item['meta']['id']} {item['path']}: {exc}", flush=True)
            if errors > 20:
                raise
            continue
        norm = {
            name: ppe_rules.normalize_box_records(rows, found["width"], found["height"])
            for name, rows in found["prompt_boxes"].items()
        }
        payload = {
            "rule_version": RULE_VERSION,
            "coord_space": "norm",
            "prompt_boxes": norm,
            **item["meta"],
        }
        payload["width"] = int(found["width"])
        payload["height"] = int(found["height"])
        item["slot"].parent.mkdir(parents=True, exist_ok=True)
        tmp = item["slot"].with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, item["slot"])
        remaining[item["clip_id"]] -= 1
        elapsed = max(time.time() - started, 1e-6)
        rate = index / elapsed
        eta_min = (len(pending) - index) / rate / 60.0
        counts = " ".join(f"{name}={len(rows)}" for name, rows in found["prompt_boxes"].items())
        print(
            f"local {index}/{len(pending)} {rate:.3f} frames/s eta_min={eta_min:.1f} "
            f"{item['meta']['id']} w{item['meta']['window']} j{item['meta']['j']} {counts}{_gpu_mem()}",
            flush=True,
        )
        publish(item["clip_id"])


if __name__ == "__main__":
    raise SystemExit(main())
