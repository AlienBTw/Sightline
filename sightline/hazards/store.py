"""SQLite persistence for cameras, detections, candidates, hazards, labels, metrics."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from sightline.errors import SightlineError

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
  camera_id TEXT PRIMARY KEY,
  name TEXT,
  source TEXT,
  plan_id TEXT,
  config_json TEXT
);

CREATE TABLE IF NOT EXISTS frames (
  frame_id TEXT PRIMARY KEY,
  camera_id TEXT,
  timestamp REAL,
  source_path TEXT
);

CREATE TABLE IF NOT EXISTS detections (
  detection_id TEXT PRIMARY KEY,
  frame_id TEXT,
  camera_id TEXT,
  timestamp REAL,
  label TEXT,
  score REAL,
  bbox_x1 INTEGER,
  bbox_y1 INTEGER,
  bbox_x2 INTEGER,
  bbox_y2 INTEGER,
  floor_x REAL,
  floor_y REAL,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
  candidate_id TEXT PRIMARY KEY,
  camera_id TEXT,
  hazard_family TEXT,
  rule_name TEXT,
  first_seen REAL,
  last_seen REAL,
  duration_seconds REAL,
  detector_label TEXT,
  detector_score REAL,
  zone_id TEXT,
  floor_x REAL,
  floor_y REAL,
  features_json TEXT,
  status TEXT
);

CREATE TABLE IF NOT EXISTS vlm_calls (
  vlm_call_id TEXT PRIMARY KEY,
  candidate_id TEXT,
  timestamp REAL,
  model TEXT,
  prompt_name TEXT,
  response_json TEXT,
  valid_json INTEGER,
  latency_ms REAL
);

CREATE TABLE IF NOT EXISTS decisions (
  decision_id TEXT PRIMARY KEY,
  candidate_id TEXT,
  model_name TEXT,
  model_version TEXT,
  timestamp REAL,
  raw_probability REAL,
  calibrated_probability REAL,
  severity REAL,
  risk_score REAL,
  abstain INTEGER,
  needs_human INTEGER
);

CREATE TABLE IF NOT EXISTS hazards (
  hazard_id TEXT PRIMARY KEY,
  candidate_id TEXT,
  hazard_family TEXT,
  status TEXT,
  severity REAL,
  risk_score REAL,
  first_seen REAL,
  confirmed_at REAL,
  fixed_at REAL,
  verified_at REAL,
  time_to_fix_seconds REAL,
  location_json TEXT,
  evidence_json TEXT
);

CREATE TABLE IF NOT EXISTS labels (
  candidate_id TEXT PRIMARY KEY,
  is_real_hazard INTEGER,
  hazard_family TEXT,
  severity REAL,
  needs_human INTEGER,
  label_source TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
  metric_id TEXT PRIMARY KEY,
  model_name TEXT,
  model_version TEXT,
  dataset TEXT,
  ece REAL,
  brier REAL,
  log_loss REAL,
  precision REAL,
  recall REAL,
  created_at REAL,
  details_json TEXT
);
"""


class Store:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        try:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur
        except sqlite3.Error as e:
            raise SightlineError(
                error_code="db_error",
                component="hazards.store",
                message=str(e),
                hint="Check SIGHTLINE_DB path and schema",
            ) from e

    def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        cur = self.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        cur = self.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def upsert_hazard(self, hazard: dict[str, Any]) -> None:
        self.execute(
            """
            INSERT INTO hazards (
              hazard_id, candidate_id, hazard_family, status, severity, risk_score,
              first_seen, confirmed_at, fixed_at, verified_at, time_to_fix_seconds,
              location_json, evidence_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(hazard_id) DO UPDATE SET
              status=excluded.status,
              severity=excluded.severity,
              risk_score=excluded.risk_score,
              confirmed_at=excluded.confirmed_at,
              fixed_at=excluded.fixed_at,
              verified_at=excluded.verified_at,
              time_to_fix_seconds=excluded.time_to_fix_seconds,
              evidence_json=excluded.evidence_json
            """,
            (
                hazard["hazard_id"],
                hazard.get("candidate_id"),
                hazard.get("hazard_family"),
                hazard.get("status"),
                hazard.get("severity"),
                hazard.get("risk_score"),
                hazard.get("first_seen"),
                hazard.get("confirmed_at"),
                hazard.get("fixed_at"),
                hazard.get("verified_at"),
                hazard.get("time_to_fix_seconds"),
                json.dumps(hazard.get("location") or {}),
                json.dumps(hazard.get("evidence") or {}),
            ),
        )

    def list_hazards(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self.fetchall(
                "SELECT * FROM hazards WHERE status=? ORDER BY risk_score DESC",
                (status,),
            )
        else:
            rows = self.fetchall("SELECT * FROM hazards ORDER BY risk_score DESC")
        for r in rows:
            r["location"] = json.loads(r.pop("location_json") or "{}")
            r["evidence"] = json.loads(r.pop("evidence_json") or "{}")
        return rows

    def close(self) -> None:
        self._conn.close()
