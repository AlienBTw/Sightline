"""Strip Meta's V-JEPA 2.1 training checkpoint down to the fp16 encoder weights.

The released vjepa2_1_vitl_dist_vitG_384.pt (5.2 GB) also holds the predictor and
other training state. The hub loader uses its 'ema_encoder' entry; we keep only
that, cleaned of 'module.'/'backbone.' prefixes and cast to fp16 (~0.6 GB).

usage: python convert_checkpoint.py SRC.pt DST.pt
"""
import sys

import torch


def main(src, dst):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    print("top-level keys:", list(ck.keys()))
    sd = ck["ema_encoder"]
    out = {}
    for k, v in sd.items():
        k = k.replace("module.", "").replace("backbone.", "")
        out[k] = v.half() if v.is_floating_point() else v
    n = sum(v.numel() for v in out.values())
    print(f"encoder tensors={len(out)} params={n/1e6:.1f}M")
    torch.save(out, dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
