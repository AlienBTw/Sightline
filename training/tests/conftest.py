import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
for sub in ("01-data", "02-labeling", "03-features", "04-head"):
    p = str(TRAINING / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
