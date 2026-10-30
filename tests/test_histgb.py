from pathlib import Path

import numpy as np

from sightline.decision.features import feature_names, vectorize
from sightline.decision.histgb import HistGBDecision, train_histgb


def test_train_and_predict(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 80
    rows = []
    y = []
    for i in range(n):
        score = float(rng.uniform(0.1, 0.95))
        row = {k: -1.0 for k in feature_names()}
        row["detector_score"] = score
        row["rule_score"] = score * 0.7
        row["severity_prior"] = 0.5
        row["duration_seconds"] = float(rng.uniform(0, 20))
        row["family_slip_trip"] = 1.0
        rows.append(row)
        y.append(1 if score > 0.55 else 0)
    X = np.array(vectorize(rows), dtype=np.float64)
    out = tmp_path / "histgb_isotonic.joblib"
    metrics = train_histgb(X, np.array(y), out_path=out)
    assert "ece" in metrics
    assert out.exists()
    model = HistGBDecision(model_path=out)
    pred = model.predict(
        {
            "hazard_family": "slip_trip",
            "features": {
                "detector_score": 0.9,
                "rule_score": 0.7,
                "severity_prior": 0.6,
                "duration_seconds": 10,
            },
        }
    )
    assert 0.0 <= pred["calibrated_probability"] <= 1.0
    assert "risk_score" in pred
