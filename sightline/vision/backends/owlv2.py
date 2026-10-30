"""OWLv2 open-vocabulary detector (explicit config backend, not an auto-fallback)."""

from __future__ import annotations

from typing import Any

import numpy as np

from sightline.config import load_yaml
from sightline.errors import SightlineError
from sightline.vision.backends.base import Detection


class OWLv2Backend:
    name = "owlv2"

    def __init__(self, device: str = "cuda", model_id: str = "google/owlv2-base-patch16-ensemble") -> None:
        self.device = device
        self.model_id = model_id
        self._model: Any = None
        self._processor: Any = None
        self.prompts = load_yaml("classes.yaml").get("detector_prompts") or []

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as e:
            raise SightlineError(
                error_code="detector_deps_missing",
                component="vision.owlv2",
                message=str(e),
                hint="pip install -e '.[vision]'",
            ) from e
        try:
            self._processor = Owlv2Processor.from_pretrained(self.model_id)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()
            self._torch = torch
        except Exception as e:
            raise SightlineError(
                error_code="detector_load_failed",
                component="vision.owlv2",
                message=str(e),
                hint="Check HF access and TORCH_DEVICE",
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
        torch = self._torch
        try:
            from PIL import Image

            rgb = frame[:, :, ::-1] if frame.shape[-1] == 3 else frame
            image = Image.fromarray(rgb.astype(np.uint8))
            texts = [self.prompts]
            inputs = self._processor(text=texts, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self._processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.35
            )[0]
            dets: list[Detection] = []
            for score, label_id, box in zip(
                results["scores"].tolist(),
                results["labels"].tolist(),
                results["boxes"].tolist(),
                strict=False,
            ):
                label = self.prompts[int(label_id)] if int(label_id) < len(self.prompts) else str(label_id)
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
                component="vision.owlv2",
                message=str(e),
                hint="Check frame and model",
            ) from e
