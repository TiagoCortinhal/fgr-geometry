"""Frame dataset + fetus-level splitting for the fetal-ultrasound SSL runs.

THE CRITICAL INVARIANT: every split in this package is BY FETUS, never by frame.
A fetus contributes ~22 frames; splitting at frame level puts near-duplicate
images of the same fetus in train and test and manufactures a positive result.
`fetus_level_folds` is the only splitter used anywhere in this codebase.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FrameManifest:
    """Resolves fetus_id -> list of image paths, with an optional cohort filter.

    manifest_csv needs columns: new_filename, fetus_id, dataset_type.
    image_root is where the PNGs live; filenames are joined onto it directly.
    """

    def __init__(self, manifest_csv, image_root, dataset_type="impact",
                 keep_csv=None, keep_col="keep_fetal"):
        m = pd.read_csv(manifest_csv)
        if dataset_type is not None:
            m = m[m.dataset_type.astype(str) == dataset_type]
        if keep_csv is not None and os.path.exists(keep_csv):
            k = pd.read_csv(keep_csv)
            if keep_col in k.columns and "new_filename" in k.columns:
                ok = set(k.loc[k[keep_col].astype(bool), "new_filename"].astype(str))
                m = m[m.new_filename.astype(str).isin(ok)]
        m = m.copy()
        m["fid"] = pd.to_numeric(m.fetus_id, errors="coerce")
        m = m[np.isfinite(m.fid)]
        m["path"] = [os.path.join(image_root, str(f)) for f in m.new_filename]
        self.df = m.reset_index(drop=True)

    def existing(self):
        """Drop rows whose file is not on disk. Run once; it stats every path."""
        keep = [os.path.exists(p) for p in self.df.path]
        self.df = self.df[np.array(keep)].reset_index(drop=True)
        return self

    def by_fetus(self):
        return {int(f): list(g.path) for f, g in self.df.groupby("fid")}


def fetus_level_folds(fids, n_folds=5, seed=0):
    """Yield (train_fids, test_fids). THE ONLY SPLITTER IN THIS PACKAGE.

    Splitting by frame instead of fetus is the one bug that would invalidate
    every number this code produces -- near-duplicate frames of the same fetus
    would appear on both sides.
    """
    u = np.array(sorted({int(f) for f in fids}))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(u))
    parts = np.array_split(perm, n_folds)
    for i in range(n_folds):
        te = set(u[parts[i]].tolist())
        tr = [int(x) for x in u if int(x) not in te]
        yield tr, sorted(te)


def load_gray(path, size=224):
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


class FrameDataset(Dataset):
    """Single frames. mode='mae' returns one view; mode='pair' returns two frames
    of the SAME fetus (contrastive positives)."""

    def __init__(self, paths_by_fetus, fids, size=224, mode="mae", augment=True):
        self.size, self.mode, self.augment = size, mode, augment
        self.items = []
        self.byf = {}
        for f in fids:
            ps = paths_by_fetus.get(int(f), [])
            if not ps:
                continue
            self.byf[int(f)] = ps
            for p in ps:
                self.items.append((int(f), p))
        self.fids = sorted(self.byf)

    def __len__(self):
        return len(self.items) if self.mode == "mae" else len(self.fids)

    def _aug(self, a):
        if not self.augment:
            return a
        if np.random.rand() < 0.5:
            a = a[:, ::-1].copy()
        g = 0.8 + 0.4 * np.random.rand()
        a = np.clip(a ** g, 0, 1)
        s = int(0.9 * self.size)
        if s < self.size:
            i = np.random.randint(0, self.size - s)
            j = np.random.randint(0, self.size - s)
            a = np.asarray(Image.fromarray((a[i:i + s, j:j + s] * 255).astype(np.uint8))
                           .resize((self.size, self.size), Image.BILINEAR),
                           dtype=np.float32) / 255.0
        return a

    def __getitem__(self, i):
        if self.mode == "mae":
            f, p = self.items[i]
            a = self._aug(load_gray(p, self.size))
            return torch.from_numpy(a)[None], f
        f = self.fids[i]
        ps = self.byf[f]
        j, k = np.random.choice(len(ps), 2, replace=len(ps) < 2)
        a = self._aug(load_gray(ps[j], self.size))
        b = self._aug(load_gray(ps[k], self.size))
        return torch.from_numpy(a)[None], torch.from_numpy(b)[None], f
