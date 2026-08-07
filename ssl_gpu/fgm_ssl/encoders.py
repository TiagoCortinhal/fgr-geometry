"""Pluggable frozen encoders — extract features online instead of from a npz.

Why this exists: swapping the frozen baseline (USFM layer 5 -> layer 11 -> a
different foundation model -> ImageNet ResNet) should be a flag, not a
regenerate-the-npz round trip through a laptop.

    python run_ssl.py --arm frozen --encoder usfm:5
    python run_ssl.py --arm frozen --encoder usfm:11
    python run_ssl.py --arm frozen --encoder resnet50

REGISTER A NEW ENCODER by adding one entry to ENCODERS below: a callable
returning (model, preprocess_fn, dim). Everything downstream — per-fetus mean
pooling, the evaluation protocol, the stop rule — is unchanged, which is the
point: only the representation varies.

THE ONE THING THAT MUST NOT DRIFT: `usfm:5` here has to reproduce the stored
`frozen_usfm.npz` numbers, or the incumbent moves and every comparison in the
project shifts under it. `verify_against_npz()` checks that, and RUN.md tells
you to run it once before trusting a swapped encoder.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

# The stored embeddings were built with these constants. Changing them changes
# the incumbent, so they are module-level and explicit rather than arguments.
USFM_MEAN = (0.485, 0.456, 0.406)
USFM_STD = (0.229, 0.224, 0.225)
USFM_SIZE = 224


def _usfm_preprocess(a):
    """a: (H,W) float32 in [0,1] -> (3,224,224) normalised.

    Matches extract_usfm_multilayer.py: Resize(224,224) -> ToTensor ->
    Normalize(ImageNet stats), with the grayscale frame repeated to 3 channels.
    """
    import torch.nn.functional as F
    x = torch.as_tensor(a, dtype=torch.float32)[None, None]
    x = F.interpolate(x, size=(USFM_SIZE, USFM_SIZE), mode="bilinear",
                      align_corners=False)
    x = x.repeat(1, 3, 1, 1)[0]
    m = torch.tensor(USFM_MEAN).view(3, 1, 1)
    s = torch.tensor(USFM_STD).view(3, 1, 1)
    return (x - m) / s


def load_usfm(weights, layer=5, device="cpu"):
    """USFM ViT-B/16 with a forward hook on `layer`, mean over patch tokens.

    Reproduces the stored extraction exactly: out[:, 1:, :].mean(1) — patch
    tokens only, CLS dropped. `strict=False` mirrors the original load.
    """
    from functools import partial
    try:
        from usfm.models.vision_transformer import VisionTransformer
    except ImportError:
        try:
            from timm.models.beit import Beit as VisionTransformer  # fallback
        except ImportError as e:
            raise ImportError(
                "USFM's VisionTransformer is not importable. Put the USFM repo "
                "on PYTHONPATH, or use --encoder resnet50 for a quick check."
            ) from e
    model = VisionTransformer(
        img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768,
        depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.0,
        init_values=0.1, use_abs_pos_emb=False, use_rel_pos_bias=False,
        use_shared_rel_pos_bias=True, use_mean_pooling=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    ck = torch.load(weights, map_location="cpu")
    model.load_state_dict(ck.get("model", ck.get("state_dict", ck)), strict=False)
    model.eval().to(device)

    grabbed = {}

    def hook(mod, inp, out):
        grabbed["f"] = out[:, 1:, :].mean(1).detach()

    model.blocks[layer].register_forward_hook(hook)

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        @torch.no_grad()
        def forward(self, x):
            self.m.forward_features(x)
            return grabbed["f"]

    return Wrapped().to(device), _usfm_preprocess, 768


def load_resnet50(device="cpu", pretrained=True):
    """ImageNet ResNet-50, global-pooled. A cheap sanity encoder: it should
    reproduce the maternal-BMI confound but little else."""
    import torchvision
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = torchvision.models.resnet50(weights=w)
    m.fc = nn.Identity()
    m.eval().to(device)

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = m

        @torch.no_grad()
        def forward(self, x):
            return self.m(x)

    return Wrapped().to(device), _usfm_preprocess, 2048


ENCODERS = {
    "usfm": lambda spec, device, weights: load_usfm(
        weights, layer=int(spec) if spec else 5, device=device),
    "resnet50": lambda spec, device, weights: load_resnet50(device=device),
}


def build_encoder(name, device="cpu", weights=None):
    """name is 'family' or 'family:spec', e.g. 'usfm:11'."""
    fam, _, spec = name.partition(":")
    if fam not in ENCODERS:
        raise SystemExit(f"unknown encoder '{fam}'. Available: {sorted(ENCODERS)}")
    if fam == "usfm" and not (weights and os.path.exists(weights)):
        raise SystemExit(f"--usfm-weights not found: {weights}")
    return ENCODERS[fam](spec, device, weights)


@torch.no_grad()
def embed_fetuses(model, preprocess, paths_by_fetus, keys, device="cpu",
                  batch=64, size=224, log_every=100):
    """Per-fetus representation = mean over that fetus's frames.

    Same pooling as the stored npz and as the trained arms, so a swapped encoder
    changes only the representation.
    """
    from .data import load_gray
    out = {}
    for n, k in enumerate(keys):
        ps = paths_by_fetus.get(k, [])
        if not ps:
            continue
        acc = []
        for i in range(0, len(ps), batch):
            xb = torch.stack([preprocess(load_gray(p, size)) for p in ps[i:i + batch]])
            acc.append(model(xb.to(device)).float().cpu().numpy())
        out[k] = np.concatenate(acc).mean(0)
        if log_every and n % log_every == 0:
            print(f"  [embed] {n}/{len(keys)} fetuses", flush=True)
    return out


def verify_against_npz(E_new, npz_path, fids, tol=0.02):
    """Does an online-extracted representation reproduce the stored incumbent?

    Compares per-dimension correlation on the overlapping fetuses. Run this once
    after switching to online extraction: if `usfm:5` does not reproduce
    frozen_usfm.npz, the incumbent has silently moved and every comparison in
    the project is against a different baseline.
    """
    z = np.load(npz_path, allow_pickle=True)
    E_old = z["E"]
    m = np.isfinite(E_old).all(1) & np.isfinite(E_new).all(1)
    if m.sum() < 20 or E_old.shape[1] != E_new.shape[1]:
        return dict(comparable=False, n=int(m.sum()),
                    reason=f"shapes {E_old.shape} vs {E_new.shape}")
    cors = [float(np.corrcoef(E_old[m, j], E_new[m, j])[0, 1])
            for j in range(E_old.shape[1])]
    cors = np.array([c for c in cors if np.isfinite(c)])
    return dict(comparable=True, n=int(m.sum()), mean_r=float(cors.mean()),
                min_r=float(cors.min()), frac_above_0p99=float((cors > 0.99).mean()),
                reproduces=bool(cors.mean() > 1 - tol))
