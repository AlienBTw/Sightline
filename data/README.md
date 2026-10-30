# Data layout

| Path | Git? | Purpose |
|---|---|---|
| `clips/` | No | Raw MP4 / webcam captures (gitignored) |
| `models/` | Artifacts optional | HistGB joblib + meta JSON |
| `sightline.db` | No | SQLite runtime DB |

See [`docs/capture_checklist.md`](../docs/capture_checklist.md) for staged demo scenes.

Validate metres calibration:

```bash
python scripts/validate_calibration.py
```
