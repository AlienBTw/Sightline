"""HistGradientBoosting + isotonic calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sightline.config import ROOT, load_yaml
from sightline.decision.features import feature_names, row_from_candidate, vectorize
from sightline.errors import SightlineError

MODEL_DIR = ROOT / "data" / "models"


class HistGBDecision:
    name = "histgb_isotonic"

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or (MODEL_DIR / "histgb_isotonic.joblib")
        self.meta_path = self.model_path.with_suffix(".meta.json")
        self._bundle: dict[str, Any] | None = None
        self.thresholds = load_yaml("thresholds.yaml").get("decide") or {}

    def load(self) -> None:
        if not self.model_path.exists():
            raise SightlineError(
                error_code="decide_model_missing",
                component="decision.histgb",
                message=f"Missing model artifact {self.model_path}",
                hint="Run scripts/train_histgb.py after labeling candidates",
            )
        try:
            self._bundle = joblib.load(self.model_path)
        except Exception as e:
            raise SightlineError(
                error_code="decide_model_load_failed",
                component="decision.histgb",
                message=str(e),
                hint="Retrain or fix the joblib artifact",
            ) from e

    def predict(self, candidate: dict[str, Any]) -> dict[str, Any]:
        if self._bundle is None:
            self.load()
        assert self._bundle is not None
        row = row_from_candidate(candidate)
        X = np.array(vectorize([row]), dtype=np.float64)
        clf = self._bundle["classifier"]
        iso = self._bundle["calibrator"]
        raw = float(clf.predict_proba(X)[0, 1])
        cal = float(iso.transform([raw])[0])
        lo = float(self.thresholds.get("abstain_low", 0.35))
        hi = float(self.thresholds.get("abstain_high", 0.65))
        abstain = lo <= cal <= hi
        severity = float((candidate.get("features") or {}).get("severity_prior", 0.5))
        # VLM multipliers
        feats = candidate.get("features") or {}
        if feats.get("vlm_source_active") is True:
            severity = min(1.0, severity * 1.4)
        if feats.get("vlm_liquid_present") is True:
            severity = min(1.0, severity * 1.4)
        risk = cal * severity
        return {
            "model_name": self.name,
            "raw_probability": raw,
            "calibrated_probability": cal,
            "severity": severity,
            "risk_score": risk,
            "abstain": abstain,
            "needs_human": abstain or candidate.get("hazard_family") == "other",
            "hazard_family": candidate.get("hazard_family"),
        }


def train_histgb(
    X: np.ndarray,
    y: np.ndarray,
    *,
    out_path: Path | None = None,
) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import brier_score_loss, log_loss
    from sklearn.model_selection import train_test_split

    if len(y) < 10:
        raise SightlineError(
            error_code="decide_insufficient_labels",
            component="decision.histgb",
            message=f"Need at least 10 labeled candidates, got {len(y)}",
            hint="Label more candidates before training",
        )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if len(set(y)) > 1 else None)
    from sklearn.isotonic import IsotonicRegression

    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, random_state=42)
    clf.fit(X_train, y_train)
    raw_val = clf.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_val)
    proba = iso.transform(raw_val)
    metrics = {
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "brier": float(brier_score_loss(y_val, proba)),
        "log_loss": float(log_loss(y_val, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "ece": float(_ece(y_val, proba)),
        "feature_names": feature_names(),
    }
    bundle = {"classifier": clf, "calibrator": iso, "metrics": metrics}
    path = out_path or (MODEL_DIR / "histgb_isotonic.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    path.with_suffix(".meta.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if not np.any(mask):
            continue
        ece += abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.mean()
    return float(ece)
