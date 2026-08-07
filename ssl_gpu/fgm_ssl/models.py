"""Encoders and heads for the four arms.

Deliberately small: n=951 fetuses. A ViT-L would memorise the cohort. The
supervised arm (A3) is the decisive one, and it must be capacity-matched to the
SSL arms so a difference between them is about the objective, not the model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout, stride=2):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.GELU(),
    )


class Encoder(nn.Module):
    """Shared trunk: 224 -> 7 spatial, then global average pool to `dim`."""

    def __init__(self, dim=256, width=32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(1, w, 7, 2, 3, bias=False),
                                  nn.BatchNorm2d(w), nn.GELU())
        self.b1 = conv_block(w, w * 2)
        self.b2 = conv_block(w * 2, w * 4)
        self.b3 = conv_block(w * 4, w * 8)
        self.b4 = conv_block(w * 8, dim, stride=1)
        self.dim = dim

    def forward(self, x, spatial=False):
        h = self.b4(self.b3(self.b2(self.b1(self.stem(x)))))
        if spatial:
            return h
        return h.mean(dim=(2, 3))


class MAE(nn.Module):
    """Masked autoencoder: blank out patches, reconstruct them.

    Loss is computed ON THE MASKED REGIONS ONLY. Averaging over all pixels lets
    the model score well by copying the visible ones.
    """

    def __init__(self, dim=256, width=32, patch=16, mask_ratio=0.6):
        super().__init__()
        self.enc = Encoder(dim, width)
        self.patch, self.mask_ratio = patch, mask_ratio
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim, 128, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
        )

    def make_mask(self, x):
        b, _, h, w = x.shape
        gh, gw = h // self.patch, w // self.patch
        m = (torch.rand(b, 1, gh, gw, device=x.device) > self.mask_ratio).float()
        return F.interpolate(m, size=(h, w), mode="nearest")

    def forward(self, x):
        m = self.make_mask(x)
        rec = self.dec(self.enc(x * m, spatial=True))
        if rec.shape[-2:] != x.shape[-2:]:
            rec = F.interpolate(rec, size=x.shape[-2:], mode="bilinear",
                                align_corners=False)
        hidden = 1.0 - m
        denom = hidden.sum().clamp(min=1.0)
        loss = (((rec - x) ** 2) * hidden).sum() / denom
        return loss, rec, m


class ContrastiveNet(nn.Module):
    """Same-fetus positives, different-fetus negatives (NT-Xent).

    NOTE what this can and cannot learn. Two frames of one fetus share fetal
    anatomy AND the acquisition session -- machine settings, operator, maternal
    habitus. The objective cannot separate those, so a strong maternal-BMI
    readout from this arm is EXPECTED and is not evidence of physiology.
    """

    def __init__(self, dim=256, width=32, proj=128, temp=0.2):
        super().__init__()
        self.enc = Encoder(dim, width)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, proj))
        self.temp = temp

    def forward(self, a, b):
        za = F.normalize(self.proj(self.enc(a)), dim=1)
        zb = F.normalize(self.proj(self.enc(b)), dim=1)
        z = torch.cat([za, zb], 0)
        n = za.shape[0]
        sim = (z @ z.t()) / self.temp
        sim.fill_diagonal_(-1e4)
        tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
        return F.cross_entropy(sim, tgt), za


class SupervisedNet(nn.Module):
    """A3 -- end-to-end pixels -> tabular target. The decisive arm.

    Frames are encoded independently and MEAN-POOLED PER FETUS inside the
    forward pass, so the supervision reaches the encoder through the same
    pooling the frozen-feature analyses used. If this finds nothing, no
    unsupervised proxy will.
    """

    def __init__(self, n_out, dim=256, width=32, dropout=0.3):
        super().__init__()
        self.enc = Encoder(dim, width)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 128),
                                  nn.GELU(), nn.Linear(128, n_out))

    def forward(self, x, group):
        """x: (N frames, 1, H, W); group: (N,) fetus index into 0..G-1."""
        h = self.enc(x)
        g = int(group.max().item()) + 1
        pooled = torch.zeros(g, h.shape[1], device=h.device, dtype=h.dtype)
        cnt = torch.zeros(g, 1, device=h.device, dtype=h.dtype)
        pooled.index_add_(0, group, h)
        cnt.index_add_(0, group, torch.ones_like(h[:, :1]))
        pooled = pooled / cnt.clamp(min=1)
        return self.head(pooled), pooled
