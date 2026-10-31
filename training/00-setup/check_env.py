#!/usr/bin/env python3
"""Phase 00 - read-only environment check for the training pipeline.

Prints Python, key packages, GPUs, Ollama models and free disk. Changes nothing.
  python training/00-setup/check_env.py
"""
from __future__ import annotations

import importlib
import json
import platform
import shutil
import sys
import urllib.request
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
OLLAMA = "http://localhost:11434"
NEED_MODEL = "muse-glimmer:30b"


def main() -> int:
    problems = []
    print(f"python   {platform.python_version()} ({sys.executable})")
    if sys.version_info < (3, 11):
        problems.append("Python >= 3.11 required")
    for mod in ("numpy", "cv2", "yaml", "pytest", "torch"):
        try:
            m = importlib.import_module(mod)
            print(f"{mod:<8} {getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"{mod:<8} MISSING")
            if mod != "torch":
                problems.append(f"missing package: {mod}")
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"gpu{i}     {p.name}, {p.total_memory / 2**30:.0f} GiB")
        else:
            print("gpu      CUDA not available to torch")
    except ImportError:
        pass
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.loads(r.read())["models"]]
        print(f"ollama   {', '.join(names)}")
        if NEED_MODEL not in names:
            problems.append(f"Ollama does not have {NEED_MODEL} (ollama pull {NEED_MODEL})")
    except OSError as e:
        print(f"ollama   not reachable ({e})")
        problems.append("Ollama not reachable on localhost:11434 (needed for 02-labeling)")
    free = shutil.disk_usage(TRAINING_DIR).free / 2**30
    print(f"disk     {free:.0f} GiB free at {TRAINING_DIR}")
    if free < 60:
        problems.append("less than 60 GiB free (Voxel51 ~9.3 GiB, Zenodo ~38.6 GiB + extraction)")
    print("\n" + ("OK" if not problems else "PROBLEMS:\n  - " + "\n  - ".join(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
