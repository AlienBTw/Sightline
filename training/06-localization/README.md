# 06 · Localization (circles)

**Goal:** turn the head's per-patch factor activations (24×24 grid) into a circle around the flagged area on the replayed clip. The circle is the explanation; there's no heatmap and no text.

## Planned
- Upsample the patch activations for the flagged factor, threshold them, take the largest component, and fit a circle or ellipse in frame coordinates
- Smooth the circle over time across windows
- Check it against the Zenodo hand boxes for F1 (IoU or centre-in-circle rate)
- SAM 3.1 is used only on escalations, to refine the area

## Results
Not run yet.
