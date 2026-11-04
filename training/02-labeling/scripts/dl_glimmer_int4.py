import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WORK = Path(os.environ.get("SIGHTLINE_WORK_DIR", REPO / "work"))
os.environ.setdefault("HF_HOME", str(WORK / "hf-cache"))
os.environ["HF_HUB_DISABLE_XET"] = "1"   # xet stalled here (20 min, ~0 bytes on disk); plain HTTP resumes .incomplete files
from huggingface_hub import snapshot_download
t = time.time()
p = snapshot_download("RedHatAI/Muse-Glimmer-30B-INT4", revision="397b1ee1eb83f09b9675e66e49d5ec3a522c509a",
                      local_dir=os.environ.get("MODEL_PATH", str(WORK / "models" / "muse-glimmer-30b-int4-w4a16")),
                      max_workers=8)
print(f"done {p} in {time.time()-t:.0f}s", flush=True)
