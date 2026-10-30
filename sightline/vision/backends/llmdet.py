"""LLMDet open-vocabulary detector via Hugging Face transformers."""

from __future__ import annotations

from typing import Any

import numpy as np

from sightline.config import load_yaml
from sightline.errors import SightlineError
from sightline.vision.backends.base import Detection


class LLMDetBackend:
    name = "llmdet"

    def __init__(self, device: str = "cuda", model_id: str = "iSEE-Laboratory/llmdet_large") -> None:
        self.device = device
        self.model_id = model_id
        self._model: Any = None
        self._processor: Any = None
        self.prompts = load_yaml("classes.yaml").get("detector_prompts") or []
        if not self.prompts:
            raise SightlineError(
                error_code="detector_prompts_missing",
                component="vision.llmdet",
                message="config/classes.yaml has no detector_prompts",
                hint="Add prompt class strings under detector_prompts",
            )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForObjectDetection, AutoProcessor
        except ImportError as e:
            raise SightlineError(
                error_code="detector_deps_missing",
                component="vision.llmdet",
                message=f"Missing vision deps: {e}",
                hint="pip install -e '.[vision]'",
            ) from e
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForObjectDetection.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()
            self._torch = torch
        except Exception as e:
            raise SightlineError(
                error_code="detector_load_failed",
                component="vision.llmdet",
                message=f"Failed to load {self.model_id}: {e}",
                hint="Check network, HF access, CUDA, or set DETECTOR_BACKEND explicitly to another installed backend",
            ) from e

    def detect(
        self,
        frame: np.ndarray,
        *,
        camera_id: str,
        frame_id: str,
        timestamp: float,
    ) -> list[Detection]:
        self._ensure_loaded()
        assert self._model is not None and self._processor is not None
        torch = self._torch
        try:
            from PIL import Image

            rgb = frame[:, :, ::-1] if frame.shape[-1] == 3 else frame
            image = Image.fromarray(rgb.astype(np.uint8))
            text = ". ".join(self.prompts)
            inputs = self._processor(images=image, text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            # Generic post-process — API varies by checkpoint; fail loud if unexpected
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            if hasattr(self._processor, "post_process_object_detection"):
                results = self._processor.post_process_object_detection(
                    outputs, threshold=0.35, target_sizes=target_sizes
                )[0]
            else:
                raise SightlineError(
                    error_code="detector_postprocess_unsupported",
                    component="vision.llmdet",
                    message="Processor has no post_process_object_detection",
                    hint="Pin a compatible transformers + LLMDet release, or switch DETECTOR_BACKEND after verifying smoke",
                )
            dets: list[Detection] = []
            for score, label_id, box in zip(
                results["scores"].tolist(),
                results["labels"].tolist(),
                results["boxes"].tolist(),
                strict=False,
            ):
                label = (
                    self._model.config.id2label.get(int(label_id), str(label_id))
                    if hasattr(self._model.config, "id2label")
                    else str(label_id)
                )
                x1, y1, x2, y2 = [int(v) for v in box]
                dets.append(
                    Detection(
                        camera_id=camera_id,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        label=str(label).lower(),
                        score=float(score),
                        bbox_xyxy=(x1, y1, x2, y2),
                        metadata={"backend": self.name},
                    )
                )
            return dets
        except SightlineError:
            raise
        except Exception as e:
            raise SightlineError(
                error_code="detector_infer_failed",
                component="vision.llmdet",
                message=str(e),
                hint="Check TORCH_DEVICE, frame shape, and model compatibility",
            ) from e
