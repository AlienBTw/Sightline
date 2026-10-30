"""Train HistGB + isotonic from SQLite labels or a bootstrap CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from sightline.config import ROOT, get_settings
from sightline.decision.features import FEATURE_KEYS, FAMILY_ONEHOT, feature_names, vectorize
from sightline.decision.histgb import train_histgb
from sightline.errors import SightlineError
from sightline.hazards.store import Store


def _rows_from_store(db: str) -> tuple[np.ndarray, np.ndarray]:
    store = Store(db)
    labels = store.fetchall("SELECT * FROM labels")
    if not labels:
        raise SightlineError(
            error_code="labels_empty",
            component="scripts.train_histgb",
            message="No rows in labels table",
            hint="Run scripts/label_candidates.py or scripts/bootstrap_labels.py",
        )
    rows = []
    y = []
    for lab in labels:
        cand = store.fetchone("SELECT * FROM candidates WHERE candidate_id=?", (lab["candidate_id"],))
        if not cand:
            continue
        feats = json.loads(cand.get("features_json") or "{}")
        row = {k: feats.get(k, -1.0) for k in FEATURE_KEYS}
        fam = lab.get("hazard_family") or cand.get("hazard_family") or "other"
        for name in FAMILY_ONEHOT:
            row[f"family_{name}"] = 1.0 if fam == name else 0.0
        # coerce
        for k, v in list(row.items()):
            if isinstance(v, bool):
                row[k] = 1.0 if v else 0.0
            elif v is None:
                row[k] = -1.0
            else:
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = -1.0
        rows.append(row)
        y.append(int(lab["is_real_hazard"]))
    if len(y) < 10:
        raise SightlineError(
            error_code="decide_insufficient_labels",
            component="scripts.train_histgb",
            message=f"Need ≥10 labeled candidates, got {len(y)}",
            hint="Label more or use --bootstrap",
        )
    X = np.array(vectorize(rows), dtype=np.float64)
    return X, np.array(y, dtype=np.int64)


def _rows_from_bootstrap_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        y = []
        for rec in reader:
            row = {k: float(rec[k]) for k in feature_names()}
            rows.append(row)
            y.append(int(rec["is_real_hazard"]))
    X = np.array(vectorize(rows), dtype=np.float64)
    return X, np.array(y, dtype=np.int64)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", type=Path, help="CSV from bootstrap_labels.py")
    p.add_argument("--db", default=None)
    args = p.parse_args()
    try:
        if args.bootstrap:
            X, y = _rows_from_bootstrap_csv(args.bootstrap)
            source = str(args.bootstrap)
        else:
            settings = get_settings()
            db = args.db or settings.sightline_db
            X, y = _rows_from_store(db)
            source = db
        metrics = train_histgb(X, y)
        metrics["label_source"] = source
        meta_path = ROOT / "data" / "models" / "histgb_isotonic.meta.json"
        meta_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        print("OK: wrote data/models/histgb_isotonic.joblib")
        return 0
    except SightlineError as e:
        print(f"FAIL: {e.to_dict()}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
