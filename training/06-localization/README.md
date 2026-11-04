# 06 · Localization (circles)

**Goal:** turn the head's per-patch factor activations (24×24 grid) into a circle around the flagged area on the replayed clip. The circle is the explanation; there's no heatmap and no text.

## Planned
- Upsample the patch activations for the flagged factor, threshold them, take the largest component, and fit a circle or ellipse in frame coordinates
- Smooth the circle over time across windows
- Check it against the Zenodo hand boxes for F1 (IoU or centre-in-circle rate)
- On a live circle, SAM 3.1 refines the area only on escalation. Offline, SAM 3.1 boxes are also stage 2 of the PPE labels (helmet/vest), with Muse Spark 1.3 Contributor; see [`08-video-ppe`](../08-video-ppe/README.md)

## Where the code is
Localization is implemented in the live pipeline: [`sightline_pipeline/localize.py`](../../sightline_pipeline/localize.py) (map to circle, geometry) and [`sightline_pipeline/eval_circles.py`](../../sightline_pipeline/eval_circles.py) (evaluation). The full description is in [`sightline_pipeline/README.md`](../../sightline_pipeline/README.md).

## Method (as built)
- The head's 12 x 21 per-factor map (pooled 24 x 42 patch grid) is upsampled to the frame aspect. The connected region of at least 0.7 x peak around the peak gives a mass-weighted centre and a radius from its area (scale 1.25, clamped to 4-35% of the frame diagonal). Coordinates are stored normalised for the app.
- F1 uses the patch-supervised evidence map `sigmoid(s_F1)`, computed from a single-frame image pass, which matches how the patch loss was trained. Other factors use the MIL attention map.
- Escalations refine the circle with SAM 3.1 (snap to the matching SAM box). See the pipeline README.

## Results (2026-10-31)
3,590 Zenodo test images with hands (held out), 1920 x 1080:

| Map, threshold | Centre in a hand box | Box centres in circle | IoU vs box | IoU vs box circle |
|---|---|---|---|---|
| **evidence, 0.7 (default)** | **0.937** | **0.838** | **0.466** | **0.669** |
| evidence, 0.55 | 0.895 | 0.871 | 0.364 | 0.584 |
| attention, 0.55 | 0.994 | 0.526 | 0.359 | 0.281 |
| baseline (mean training hand position) | 0.140 | 0.260 | 0.109 | - |

## Known issues
- Only F1 has ground-truth boxes. Circles for F6-F10 come from MIL attention trained with clip-level labels and are not validated. In the demo, F7 circled a floor reflection instead of the forklift load.
