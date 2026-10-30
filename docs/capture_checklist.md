# Demo clip capture checklist

Clips stay under `data/clips/` (**gitignored**). Do not commit raw MP4s or faces.

## Safety

- Stage props only — no real unsafe setups (no live sparks, no blocking a real exit during work hours).
- Tape-measure ground truth for distances and clear widths.
- Avoid faces / PII; blur before storage or VLM if people appear.
- Fixed camera after calibration; do not move the lens between clips.

## Scenes (minimum)

| # | Family | Scene | Notes |
|---|---|---|---|
| 1 | slip_trip | Object / cable on walkway | Measure distance to walkway centreline |
| 2 | fire_heat | Combustible near heater (cold prop) | VLM should see heater-on attribute later |
| 3 | egress | Prop narrowing exit clearance | Known clear width vs `min_clear_width_m` |
| 4 | electrical | Cord near power strip (unpowered demo) | Distance in metres |
| 5 | fix_verify | Same hazard removed after “fix” | Second clip for verify path |

## Capture commands

```bash
# Webcam → data/clips/scene_01_slip.mp4 (Ctrl+C to stop)
python scripts/capture_clip.py --out data/clips/scene_01_slip.mp4 --source 0 --seconds 20

# Or copy an existing MP4 into data/clips/
```

## After capture

1. Confirm `cameras.json` has 4 pixel ↔ 4 plan points and `known_length_m`.
2. Run `python scripts/validate_calibration.py`.
3. Analyze via API: `POST /analyze/video` with the clip path.
