# 07 · Export / integration

**Goal:** package the trained head and its config so the backend can run it on live and replayed streams. The mobile app then gets the factors, the risk and the circle for each flagged clip.

## Planned
- Export the head weights plus `factors.yaml` plus the thresholds as one versioned bundle
- An inference wrapper used by the API (ViT-B live, stride windows)
- A small, replayable demo set for judging

## Results
Not run yet.
