"""Replay detector — only when DETECTOR_BACKEND=replay (and MODE=replay for product paths)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sightline.config import ROOT
from sightline.errors import SightlineError
from sightline.vision.backends.base import Detection


class ReplayBackend:
    name = "replay"

    def __init__(self, pack_dir: str | Path | None = None) -> None:
        self.pack_dir = Path(pack_dir) if pack_dir else ROOT / "replay" / "demo"
        self._by_frame: dict[str, list[Detection]] = {}
        self._ordered: list[list[Detection]] = []
        self._cursor = 0
        self._load()

    def _load(self) -> None:
        path = self.pack_dir / "detections.jsonl"
        if not path.exists():
            raise SightlineError(
                error_code="replay_pack_missing",
                component="vision.replay",
                message=f"Missing {path}",
                hint="Generate a replay pack or set SIGHTLINE_MODE=live with a real DETECTOR_BACKEND",
            )
        order_keys: list[str] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                fid = str(row["frame_id"])
                det = Detection(
                    camera_id=row["camera_id"],
                    frame_id=fid,
                    timestamp=float(row["timestamp"]),
                    label=row["label"],
                    score=float(row["score"]),
                    bbox_xyxy=tuple(row["bbox_xyxy"]),  # type: ignore[arg-type]
                    metadata=row.get("metadata") or {"backend": "replay"},
                )
                if fid not in self._by_frame:
                    order_keys.append(fid)
                self._by_frame.setdefault(fid, []).append(det)
        self._ordered = [self._by_frame[k] for k in order_keys]
        if not self._ordered:
            raise SightlineError(
                error_code="replay_pack_empty",
                component="vision.replay",
                message=f"No detections in {path}",
                hint="Add detections.jsonl rows",
            )

    def detect(
        self,
        frame: np.ndarray,
        *,
        camera_id: str,
        frame_id: str,
        timestamp: float,
    ) -> list[Detection]:
        if frame_id in self._by_frame:
            return list(self._by_frame[frame_id])
        # Sequential fallback for live analyze_video path that mints new frame_ids
        if self._cursor >= len(self._ordered):
            raise SightlineError(
                error_code="replay_exhausted",
                component="vision.replay",
                message="Replay pack has no more frames",
                hint="Rebuild pack or shorten the video sample",
            )
        dets = list(self._ordered[self._cursor])
        self._cursor += 1
        # Re-stamp camera/frame for the caller
        return [
            Detection(
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp=timestamp,
                label=d.label,
                score=d.score,
                bbox_xyxy=d.bbox_xyxy,
                metadata={**d.metadata, "replay_source_frame": d.frame_id},
            )
            for d in dets
        ]
