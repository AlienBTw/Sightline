"""V-JEPA 2.1 ViT-L/16 384 encoder + trained concept-bottleneck head, for live windows.

Reuses the training code so inference matches training exactly:
  training/03-features/extract_features.py  load_encoder, to_input, tokens_to_outputs, sizes
  training/04-head/train_head.py            build_model (head), POOL
Window: 16 frames sampled every ~0.16 s (frame step 4 at 25 fps), squashed to 384 x 672 RGB.
Encoder tokens (8 x 24 x 42) -> time mean -> 2x2 pool -> 12 x 21 = 252 tokens x 1024.
The head runs MIL over the tokens of the last `mil_windows` windows (training pooled whole clips).
"""
from __future__ import annotations

import importlib.util
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

from .config import REPO, factor_ids, resolve_path


def _load(name: str, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def training_modules():
    ef = _load("sl_extract_features", REPO / "training/03-features/extract_features.py")
    th = _load("sl_train_head", REPO / "training/04-head/train_head.py")
    return ef, th


def _with_vjepa_path(repo: str, fn, *args):
    """The Sightline repo has its own `app/` package (Streamlit) and ~/vjepa2/app is a namespace
    package, so a regular package on sys.path would shadow it. Import the encoder with the repo
    root temporarily removed from sys.path."""
    import sys
    saved = list(sys.path)
    for k in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
        del sys.modules[k]
    sys.path[:] = [repo] + [p for p in saved if p not in ("", ".", str(REPO)) and Path(p).resolve() != REPO.resolve()]
    try:
        return fn(repo, *args)
    finally:
        sys.path[:] = saved


MAP_MODES = ("evidence", "attention", "attention_x_evidence")


class HazardModel:
    def __init__(self, cfg: dict):
        import torch
        self.torch = torch
        self.cfg = cfg
        m = cfg["model"]
        self.ef, self.th = training_modules()
        self.device = torch.device(m["device"] if torch.cuda.is_available() else "cpu")
        torch.backends.cuda.matmul.allow_tf32 = True
        t0 = time.time()
        self.encoder = _with_vjepa_path(os.path.expanduser(m["vjepa_repo"]), self.ef.load_encoder,
                                        str(resolve_path(m["encoder_weights"])), self.device)
        ck = torch.load(resolve_path(m["head_checkpoint"]), map_location="cpu", weights_only=False)
        hc = ck["config"]
        self.factors = list(ck.get("factors") or factor_ids())
        self.head = self.th.build_model(hidden=hc["hidden"], drop=hc["dropout"], nf=len(self.factors))
        self.head.load_state_dict(ck["state_dict"])
        self.head.to(self.device).eval()
        self.ckpt_thresholds = {f: float(v) for f, v in ck["thresholds"].items()}
        self.pool = int(ck.get("pool", 2))
        self.grid = tuple(ck.get("grid", (24, 42)))
        self.gh, self.gw = self.grid[0] // self.pool, self.grid[1] // self.pool
        self.in_h, self.in_w = self.ef.IN_H, self.ef.IN_W
        self.n_frames = self.ef.FRAMES_PER_WINDOW
        self.tubelet = self.ef.TUBELET
        self.history: deque = deque(maxlen=max(1, int(cfg["sampling"]["mil_windows"])))
        self.load_s = time.time() - t0

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        return self.ef._resize_rgb(bgr)        # same resize/colour path as training

    def encode(self, window_rgb: np.ndarray, t_tokens: int | None = None) -> np.ndarray:
        """(T, 384, 672, 3) uint8 -> pooled tokens (252, 1024) float32. T=1 for still images."""
        torch = self.torch
        T = window_rgb.shape[0]
        t_tokens = t_tokens or max(1, T // self.tubelet)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            x = self.ef.to_input(window_rgb[None] if T > 1 else window_rgb, self.device)
            tok = self.encoder(x)
            grid, _ = self.ef.tokens_to_outputs(tok, t_tokens)      # (1, 24, 42, D) fp16
        g = grid[0].float()
        D = g.shape[-1]
        g = g.reshape(self.gh, self.pool, self.gw, self.pool, D).mean((1, 3)).reshape(self.gh * self.gw, D)
        return g

    def head_maps(self, token_list: list) -> dict:
        """MIL head over the given windows' tokens; maps are for the LAST window."""
        torch = self.torch
        with torch.inference_mode():
            X = torch.cat(token_list, 0)[None]                      # (1, N, D)
            V = torch.ones(X.shape[:2], dtype=torch.bool, device=X.device)
            z, s, w = self.head(X, V)
            p = torch.sigmoid(z)[0].float().cpu().numpy()
            P = self.gh * self.gw
            s_last = torch.sigmoid(s[0, -P:]).float().cpu().numpy()          # (P, F)
            w_last = w[0, -P:].float()
            w_last = (w_last / w_last.sum(0, keepdim=True).clamp(min=1e-12)).cpu().numpy()
        return {"probs": {f: float(p[i]) for i, f in enumerate(self.factors)},
                "evidence": s_last.T.reshape(len(self.factors), self.gh, self.gw),
                "attention": w_last.T.reshape(len(self.factors), self.gh, self.gw)}

    def step(self, window_rgb: np.ndarray) -> dict:
        """One live step: encode the newest window, run the head over the last mil_windows."""
        torch = self.torch
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        tok = self.encode(window_rgb)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        self.history.append(tok)
        out = self.head_maps(list(self.history))
        t2 = time.perf_counter()
        out["latency_ms"] = {"encoder": (t1 - t0) * 1e3, "head": (t2 - t1) * 1e3}
        img_f = [f for f in self.cfg["localization"].get("image_pass_factors", []) if f in self.factors]
        if img_f:
            # The F1 patch loss was trained on single Zenodo images (T=1), so its map is computed from a
            # still-image pass over the newest frame (validated on the Zenodo test boxes).
            im = self.head_maps([self.encode(window_rgb[-1:], t_tokens=1)])
            for f in img_f:
                i = self.factors.index(f)
                out["evidence"][i] = im["evidence"][i]
                out["attention"][i] = im["attention"][i]
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            out["latency_ms"]["image_pass"] = (time.perf_counter() - t2) * 1e3
        return out

    def factor_map(self, out: dict, factor: str, mode: str) -> np.ndarray:
        i = self.factors.index(factor)
        ev, at = out["evidence"][i], out["attention"][i]
        if mode == "evidence":
            return ev
        if mode == "attention":
            return at
        return at * ev
