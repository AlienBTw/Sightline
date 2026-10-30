"""CLI to insert labels for candidates already in SQLite."""

from __future__ import annotations

import argparse
import sys

from sightline.config import get_settings
from sightline.errors import SightlineError
from sightline.hazards.store import Store


def main() -> int:
    p = argparse.ArgumentParser(description="Label a candidate for HistGB training")
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--is-real", type=int, choices=[0, 1], required=True)
    p.add_argument("--family", default=None)
    p.add_argument("--severity", type=float, default=0.5)
    p.add_argument("--needs-human", type=int, choices=[0, 1], default=0)
    p.add_argument("--notes", default="")
    args = p.parse_args()
    try:
        settings = get_settings()
        store = Store(settings.sightline_db)
        cand = store.fetchone("SELECT * FROM candidates WHERE candidate_id=?", (args.candidate_id,))
        if not cand:
            # Allow labeling from hazard evidence path: store candidate stub
            raise SightlineError(
                error_code="candidate_not_found",
                component="scripts.label_candidates",
                message=f"No candidate {args.candidate_id}",
                hint="Analyze a clip first so candidates land in SQLite",
            )
        store.execute(
            """
            INSERT OR REPLACE INTO labels (
              candidate_id, is_real_hazard, hazard_family, severity, needs_human, label_source, notes
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                args.candidate_id,
                args.is_real,
                args.family or cand.get("hazard_family"),
                args.severity,
                args.needs_human,
                "cli",
                args.notes,
            ),
        )
        print(f"OK: labeled {args.candidate_id} is_real={args.is_real}")
        return 0
    except SightlineError as e:
        print(f"FAIL: {e.to_dict()}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
